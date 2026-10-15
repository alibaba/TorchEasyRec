# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Offline best-effort SID collision resolution with TorchEasyRec-native I/O.

The input is a Semantic-ID table from ``tzrec.predict``: an ``item_id`` column,
a ``codes`` ``list<int>`` column and, for ``--strategy candidate`` or
``iterative``, a flat ``candidate_codes`` ``list<int>`` column holding
``topk * n_layers`` codes per item, split by the ``--codebook`` length. CSV
encodes each SID and candidate column as comma-separated codes because Arrow's
CSV writer cannot serialize list columns.

Item IDs are expected to be unique in the upstream prediction output. To avoid
blocking a large job on rare upstream violations, this tool does not perform a
full-input uniqueness check or remove duplicate rows. Duplicate rows remain
independent items in collision accounting and outputs. When candidate-strategy
overflow rows share an item ID, they reuse one candidate list matched for that
ID. Duplicate data should still be fixed upstream.

The runner retains the first capacity items in each SID and relocates
overflow items within their SID prefix through the pure NumPy core:

- ``candidate`` (the default) moves each item to its first model candidate
  with room.
- ``iterative`` places the same candidates in synchronous rounds that give a
  contested SID to the item ranking it earliest, so it can leave more items
  unresolved than ``candidate``.
- ``uniform`` reads no candidates and spreads the items over the least-loaded
  last-layer codes of their prefix, so a prefix with room leaves nothing over
  capacity.

It writes the ``item_to_sid`` and ``sid_to_items`` artifacts through
TorchEasyRec readers and writers; ``tzrec/utils/sid/bundle.py`` describes where
they land. ``sid_to_items`` is always parquet, so it keeps native list columns
whatever ``--item_to_sid_writer_type`` says.

Both item_to_sid and sid_to_items carry an ``offset_codebook`` column
alongside ``codebook``: the same SID with each layer shifted into one
contiguous vocabulary, so layer ``i`` is offset by ``sum(codebook[:i])``. With
``--codebook 64,64,64`` the SID ``[1, 2, 3]`` is written as ``[1, 66, 131]``.
It is derived from the resolved SID, so a relocated item's offset SID follows
where it landed, not what the model predicted.

Launch with ``torchrun --nproc_per_node`` to assign complete SID prefix ranges
to ranks. Each rank writes its own part files, and rank 0 publishes one manifest
after all ranks finish. All ranks must see the same input and output locations.

Example::

    torchrun --standalone --nproc_per_node=8 \
        -m tzrec.tools.sid.resolve_sid_collisions \
        --input_path 'sid_predict_output/*.parquet' \
        --codebook 256,256,256 --max_items_per_codebook 5 \
        --strategy candidate \
        --output_path sid_collision \
        --generation v1

Append mode
-----------

Supplying ``--from_generation`` switches the tool to appending new items onto an
already-published generation: ``--input_path`` then holds only the new items,
and **no published SID is ever moved**. Published items are never rows in the
plan, so they cannot be re-ranked; they are read only to be copied into the
merged outputs. A new item continues its SID's slot numbering from the
published occupancy instead of restarting at one::

    torchrun --standalone --nproc_per_node=8 \
        -m tzrec.tools.sid.resolve_sid_collisions \
        --input_path 'sid_predict_new_20260731/*.parquet' \
        --codebook 256,256,256 --max_items_per_codebook 5 \
        --strategy candidate \
        --output_path sid_collision \
        --generation v2 --from_generation v1

An append also writes ``delta_item_to_sid`` and ``delta_sid_to_items``, holding
this run's rows and every SID that gained them. Both corpus outputs stay complete,
so generation *n* is exactly what generation *n+1* reads at
``--from_generation``.

"""

from __future__ import annotations

import argparse
import glob
import math
import os
from dataclasses import asdict, dataclass
from functools import cached_property
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from torch import distributed as dist

from tzrec.datasets.dataset import (
    BaseReader,
    BaseWriter,
    create_reader,
    create_writer,
)
from tzrec.utils.logging_util import ProgressLogger, logger
from tzrec.utils.path_util import check_path_conflict, is_odps_path
from tzrec.utils.sid.bundle import (
    DELTA_ITEM_TO_SID_ARTIFACT,
    DELTA_SID_TO_ITEMS_ARTIFACT,
    ITEM_TO_SID_ARTIFACT,
    SID_TO_ITEMS_ARTIFACT,
    Bundle,
)
from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionResult,
    CollisionResolutionStats,
    CollisionResolver,
    ExistingSidState,
    SidItemRowIndex,
    build_origin_sid_item_row_index,
    build_resolved_sid_item_row_index,
    find_in_sorted_array,
    get_same_value_start_indices,
    prepare_collision_plan,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.sid.knn_collision import KnnCollisionResolver
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver

_RESOLVERS: Dict[str, Type[CollisionResolver]] = {
    "candidate": KnnCollisionResolver,
    "iterative": IterativeCollisionResolver,
    "uniform": UniformCollisionResolver,
}
_INT32_MAX = int(np.iinfo(np.int32).max)


def _is_list_column(values: pa.Array) -> bool:
    """Whether an Arrow column is one of the native list types."""
    return (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    )


class _ItemIdLookup:
    """Map streamed item IDs to positions in a fixed requested-ID array."""

    def __init__(self, item_ids: np.ndarray) -> None:
        self._sorted_to_requested = np.argsort(item_ids, kind="stable")
        self._sorted_ids = item_ids[self._sorted_to_requested]
        duplicate_sorted_rows = (
            np.flatnonzero(self._sorted_ids[1:] == self._sorted_ids[:-1]) + 1
        )
        self._duplicate_requested_rows = self._sorted_to_requested[
            duplicate_sorted_rows
        ]
        self._representative_requested_rows = self._sorted_to_requested[
            np.searchsorted(
                self._sorted_ids,
                self._sorted_ids[duplicate_sorted_rows],
                side="left",
            )
        ]

    def match(self, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return matching batch indices and their requested-ID indices."""
        positions, found = find_in_sorted_array(self._sorted_ids, item_ids)
        matching_batch_indices = np.flatnonzero(found)
        return (
            matching_batch_indices,
            self._sorted_to_requested[positions[matching_batch_indices]],
        )

    def broadcast_duplicate_targets(self, values: np.ndarray) -> None:
        """Copy representative values to duplicate requested-ID positions."""
        values[self._duplicate_requested_rows] = values[
            self._representative_requested_rows
        ]


@dataclass(frozen=True)
class ResolveSidCollisionsConfig:
    """Validated configuration for SID collision-resolution orchestration.

    ``codebook`` contains per-layer codebook sizes, matching the model config.
    """

    input_path: str
    output_path: Optional[str]
    item_to_sid_root: Optional[str]
    generation: Optional[str]
    reader_type: Optional[str]
    item_to_sid_writer_type: Optional[str]
    batch_size: int
    progress_interval: int
    item_id_field: str
    code_field: str
    candidate_codes_field: str
    codebook: Tuple[int, ...]
    max_items_per_codebook: int
    strategy: str
    rate_only: bool
    odps_data_quota_name: str
    from_generation: Optional[str] = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if not self.rate_only and not self.generation:
            raise ValueError("generation is required unless rate_only is set.")
        if not self.output_path and (not self.rate_only or self.is_append):
            raise ValueError(
                "output_path is required unless rate_only is set without "
                "from_generation; an append reads the published sid_to_items and "
                "manifest from it."
            )
        if self.from_generation is not None and self.from_generation == self.generation:
            raise ValueError(
                f"from_generation and generation are both {self.generation!r}; "
                "writing over the artifacts being read destroys them mid-run."
            )

        # Validate here, not lazily in run(), and before output_path is
        # treated as a local path below.
        _ = self.collision_config
        _ = self.bundle

        # The two roots hold disjoint artifacts, so only the input can collide.
        roots = (self.output_path, self.item_to_sid_root) if self.output_path else ()
        for root in filter(None, roots):
            has_conflict, conflict_message = check_path_conflict(
                [self.input_path, root]
            )
            if has_conflict:
                raise ValueError(conflict_message)

    @property
    def is_append(self) -> bool:
        """Whether this run appends onto an already-published generation."""
        return self.from_generation is not None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ResolveSidCollisionsConfig":
        """Build a validated configuration from parsed CLI arguments."""
        try:
            codebook = tuple(int(value) for value in args.codebook.split(","))
        except ValueError as err:
            raise ValueError(
                "--codebook must be 'int,int,...' with no empty fields, got "
                f"{args.codebook!r}."
            ) from err
        return cls(
            input_path=args.input_path,
            output_path=args.output_path,
            item_to_sid_root=args.item_to_sid_root,
            generation=args.generation,
            from_generation=args.from_generation,
            reader_type=args.reader_type,
            item_to_sid_writer_type=args.item_to_sid_writer_type,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            item_id_field=args.item_id_field,
            code_field=args.code_field,
            candidate_codes_field=args.candidate_codes_field,
            codebook=codebook,
            max_items_per_codebook=args.max_items_per_codebook,
            strategy=args.strategy,
            rate_only=args.rate_only,
            odps_data_quota_name=args.odps_data_quota_name,
        )

    @cached_property
    def bundle(self) -> Bundle:
        """Every artifact location this run reads and writes, and its manifest."""
        return Bundle(
            output_path=self.output_path,
            item_to_sid_root=self.item_to_sid_root or self.output_path,
            generation=self.generation,
            from_generation=self.from_generation,
        )

    @cached_property
    def collision_config(self) -> CollisionResolutionConfig:
        """Return the SID collision algorithm configuration."""
        return CollisionResolutionConfig(
            codebook=self.codebook,
            capacity=self.max_items_per_codebook,
        )


class CollisionResolutionRunner:
    """Run best-effort SID collision resolution over repository I/O."""

    def __init__(
        self,
        config: ResolveSidCollisionsConfig,
    ) -> None:
        self._config = config
        self._resolver = _RESOLVERS[config.strategy](
            progress_interval=config.progress_interval
        )
        self._resolved_writer_type: Optional[str] = None
        self._item_id_type: Optional[pa.DataType] = None
        self._bundle = config.bundle
        self._rank = int(os.environ.get("RANK", "0"))
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        prefixes_per_rank = (
            math.prod(config.codebook[:-1]) + self._world_size - 1
        ) // self._world_size
        self._prefix_start = self._rank * prefixes_per_rank
        self._prefix_stop = self._prefix_start + prefixes_per_rank
        self._check_input_locations()
        self._bundle.check_prior_compatible(
            list(config.codebook), config.max_items_per_codebook
        )

    def run(self) -> CollisionResolutionStats:
        """Read, resolve collisions, and publish the resulting generation."""
        if not self._config.rate_only:
            self._check_output_locations()
        item_ids, codes = self._load_codes()
        existing_sid_state, existing_item_count = self._load_existing_sid_state(
            item_ids, codes
        )
        plan = prepare_collision_plan(
            item_ids,
            codes,
            self._config.collision_config,
            existing_sid_state=existing_sid_state,
        )
        has_overflow = bool(plan.overflow_rows.size)
        if self._world_size > 1 and self._resolver.requires_candidates():
            overflow_by_rank = [None] * self._world_size
            dist.all_gather_object(overflow_by_rank, has_overflow)
            has_overflow = any(overflow_by_rank)
        result = self._resolver.resolve(
            plan,
            (lambda: self._load_candidate_last_codes(plan.overflow_item_ids))
            if has_overflow and self._resolver.requires_candidates()
            else None,
            collect_sid_metadata=not self._config.rate_only
            and bool(plan.overflow_rows.size),
        )

        if self._config.rate_only:
            logger.info("rate_only: skipping item_to_sid and sid_to_items writes")
            del plan
        else:
            self._write_sid_to_items_outputs(item_ids, plan, result)
            del plan
            self._write_item_to_sid(item_ids, codes, result, existing_item_count)
        stats = self._gather_stats(result.stats)
        if not self._config.rate_only and self._rank == 0:
            self._bundle.write_manifest(
                codebook=list(self._config.codebook),
                capacity=self._config.max_items_per_codebook,
                observed=stats.max_final_items_per_sid,
                item_id_type=str(self._item_id_type),
                writer_type=self._resolved_writer_type,
            )

        if self._rank == 0:
            logger.info("SID collision resolution finished: %s", stats)
        return stats

    def _get_prefix_mask(self, codes: np.ndarray) -> np.ndarray:
        """Return a boolean mask for rows in this rank's SID prefix range."""
        if self._world_size == 1:
            return np.ones(codes.shape[0], dtype=bool)
        prefixes = self._config.collision_config.prefix_ids(codes)
        # NOTE: Keep invalid prefixes so the core can reject out-of-range codes.
        prefixes = np.clip(prefixes, 0, math.prod(self._config.codebook[:-1]) - 1)
        return (prefixes >= self._prefix_start) & (prefixes < self._prefix_stop)

    def _gather_stats(
        self, stats: CollisionResolutionStats
    ) -> CollisionResolutionStats:
        """Gather statistics and artifact row counts after every rank closes writers."""
        if self._world_size == 1:
            return stats
        summaries: List[Any] = [None] * self._world_size
        dist.all_gather_object(summaries, (asdict(stats), self._bundle.rows))
        self._bundle.rows = {
            artifact: sum(rows[artifact] for _, rows in summaries)
            for artifact in self._bundle.rows
        }
        merged_stats = {}
        for name in asdict(stats):
            values = [summary[name] for summary, _ in summaries]
            if name == "max_final_items_per_sid":
                merged_stats[name] = max(values)
            else:
                merged_stats[name] = sum(values)
        return CollisionResolutionStats(**merged_stats)

    def _check_input_locations(self) -> None:
        """Reject a run whose inputs are not all present, before any pass runs."""
        wanted = [("--input_path", self._config.input_path)]
        if self._config.is_append:
            wanted += [
                (artifact, self._bundle.read_path(artifact))
                for artifact in (SID_TO_ITEMS_ARTIFACT, ITEM_TO_SID_ARTIFACT)
            ]
        missing = [
            f"{name} -> {path}"
            for name, path in wanted
            if not is_odps_path(path) and not glob.glob(path)
        ]
        if missing:
            raise ValueError(
                "these input locations hold no files:\n  "
                + "\n  ".join(missing)
                + "\nA directory that exists but holds no part files counts as "
                "missing, because the readers cannot tell the two apart."
            )

    def _check_output_locations(self) -> None:
        """Reject nonempty local generation directories before any rank writes.

        Rank 0 checks both output roots and broadcasts the result before a
        writer can create directories or overwrite part files.
        """
        nonempty_paths: List[str] = []
        if self._rank == 0:
            generation_paths = set()
            for artifact in (SID_TO_ITEMS_ARTIFACT, ITEM_TO_SID_ARTIFACT):
                location = self._bundle.write_path(artifact)
                if not is_odps_path(location):
                    generation_paths.add(os.path.dirname(location))
            nonempty_paths = [
                path
                for path in sorted(generation_paths)
                if os.path.exists(path) and os.listdir(path)
            ]
        if self._world_size > 1:
            check_result = [nonempty_paths]
            dist.broadcast_object_list(check_result, src=0)
            nonempty_paths = check_result[0]
        if nonempty_paths:
            raise ValueError(
                f"generation output is not empty: {nonempty_paths}. "
                "Use a new --generation or remove the incomplete generation "
                "before retrying."
            )

    def _make_reader(self, path: str, columns: List[str]) -> BaseReader:
        """Open a repository reader projecting ``columns``.

        The readers silently drop a projected column the source lacks, so a
        missing one is rejected here, before any batch is read.

        Args:
            path: Path to read.
            columns: Columns to project.

        Returns:
            A reader over ``path``.

        Raises:
            ValueError: If ``path`` lacks one of ``columns``.
        """
        reader = create_reader(
            input_path=path,
            batch_size=self._config.batch_size,
            selected_cols=columns,
            reader_type=self._config.reader_type,
            quota_name=self._config.odps_data_quota_name,
        )
        missing = [name for name in columns if name not in reader.schema.names]
        if missing:
            raise ValueError(f"columns {missing} are missing from {path!r}.")
        return reader

    def _iter_batches(
        self,
        reader: BaseReader,
        description: str,
        code_field: Optional[str] = None,
        keep_empty: bool = False,
    ) -> Iterator[Dict[str, pa.Array]]:
        """Stream batches, optionally selecting this rank's SID prefixes.

        ``keep_empty`` retains batches made empty by prefix selection so idle
        ranks can initialize input types. Empty source batches are skipped.
        """
        progress = ProgressLogger(
            description, start_n=0, miniters=self._config.progress_interval
        )
        scanned = 0
        for batch in reader.to_batches():
            batch_size = len(next(iter(batch.values())))
            if batch_size == 0:
                continue
            if code_field is not None and self._world_size > 1:
                keep = pa.array(
                    self._get_prefix_mask(
                        self._convert_codes_to_numpy(batch[code_field])
                    )
                )
                batch = {name: values.filter(keep) for name, values in batch.items()}
            if keep_empty or len(next(iter(batch.values()))):
                yield batch
            scanned += batch_size
            progress.log(scanned)

    @staticmethod
    def _convert_codes_to_numpy(values: pa.Array) -> np.ndarray:
        """Decode an SID column into an ``(N, n_layers)`` int32 matrix."""
        if _is_list_column(values):
            row_count = len(values)
            flat = pc.cast(values.flatten(), pa.int32()).to_numpy(zero_copy_only=False)
        elif pa.types.is_integer(values.type):
            return (
                pc.cast(values, pa.int32())
                .to_numpy(zero_copy_only=False)
                .reshape(-1, 1)
            )
        else:
            parts = pc.split_pattern(values, ",")
            row_count = len(parts)
            flat = pc.cast(parts.flatten(), pa.int32()).to_numpy(zero_copy_only=False)

        layer_count = flat.shape[0] // row_count
        if flat.shape[0] != row_count * layer_count:
            raise ValueError("ragged SID codes: all items must share n_layers.")
        return flat.reshape(row_count, layer_count)

    def _validate_item_ids(self, item_ids: pa.Array) -> None:
        """Validate one item ID batch and retain its Arrow type.

        Args:
            item_ids: Item IDs from one input batch.

        Raises:
            ValueError: If IDs contain nulls or their type changes between batches.
        """
        if item_ids.null_count:
            raise ValueError("item IDs must not contain null values.")
        if self._item_id_type is None:
            self._item_id_type = item_ids.type
        elif self._item_id_type != item_ids.type:
            raise ValueError(
                "item ID type changed between batches: "
                f"{self._item_id_type} vs {item_ids.type}."
            )

    def _load_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load item IDs and SIDs into arrays used by the NumPy core.

        Also resolves the default writer type from the reader class name before
        streaming, so downstream writes can default to the input format.
        """
        reader = self._make_reader(
            self._config.input_path,
            [self._config.item_id_field, self._config.code_field],
        )
        self._resolved_writer_type = self._config.item_to_sid_writer_type or (
            reader.__class__.__name__.replace("Reader", "Writer")
        )
        id_batches: List[np.ndarray] = []
        code_batches: List[np.ndarray] = []
        for batch in self._iter_batches(
            reader,
            "Reading SID input",
            self._config.code_field,
            keep_empty=True,
        ):
            item_ids = batch[self._config.item_id_field]
            self._validate_item_ids(item_ids)
            id_batches.append(item_ids.to_numpy(zero_copy_only=False))
            code_batches.append(
                self._convert_codes_to_numpy(batch[self._config.code_field])
                if len(item_ids)
                else np.empty((0, len(self._config.codebook)), dtype=np.int32)
            )

        if not id_batches:
            raise ValueError("SID input is empty.")
        numpy_codes = np.concatenate(code_batches, axis=0)
        del code_batches
        if numpy_codes.shape[1] != len(self._config.codebook):
            raise ValueError(
                f"codes have {numpy_codes.shape[1]} layers but --codebook has "
                f"{len(self._config.codebook)}."
            )
        item_id_array = np.concatenate(id_batches)
        del id_batches
        return item_id_array, numpy_codes

    def _candidate_last_matrix(self, values: pa.Array) -> np.ndarray:
        """Decode a flat candidate column into its per-candidate last codes.

        ``values`` is decoded like ``codes`` (list, integer, or comma string) into
        an ``(N, topk * n_layers)`` matrix, then split into ``topk`` groups of
        ``n_layers`` (from ``--codebook``), keeping each group's last-layer code.

        Args:
            values: One batch's flat ``candidate_codes`` column.

        Returns:
            The last-layer code of every candidate, shape ``(N, topk)``.

        Raises:
            ValueError: If the per-row width is not a multiple of ``n_layers``.
        """
        flat = self._convert_codes_to_numpy(values)
        per_row = flat.shape[1]
        n_layers = len(self._config.codebook)
        if per_row % n_layers != 0:
            raise ValueError(
                f"candidate_codes width {per_row} is not a multiple of n_layers "
                f"{n_layers}."
            )
        return flat[:, n_layers - 1 :: n_layers]

    def _load_candidate_last_codes(self, overflow_item_ids: np.ndarray) -> np.ndarray:
        """Load fixed-width candidates aligned to ``overflow_item_ids``."""
        columns = [self._config.item_id_field, self._config.candidate_codes_field]
        # NOTE: Duplicate IDs in other ranks' prefixes must not supply candidates.
        if self._world_size > 1:
            columns.append(self._config.code_field)
        reader = self._make_reader(self._config.input_path, columns)
        item_count = overflow_item_ids.shape[0]
        if item_count == 0:
            return np.empty((0, 0), dtype=np.int32)
        item_lookup = _ItemIdLookup(overflow_item_ids)
        candidates: Optional[np.ndarray] = None
        seen = np.zeros(item_count, dtype=bool)
        for batch in self._iter_batches(reader, "Scanning candidate input"):
            batch_ids = batch[self._config.item_id_field].to_numpy(zero_copy_only=False)
            matching_batch_indices, overflow_item_indices = item_lookup.match(batch_ids)
            if matching_batch_indices.size == 0:
                continue

            if self._world_size > 1:
                codes = self._convert_codes_to_numpy(
                    pc.take(
                        batch[self._config.code_field],
                        pa.array(matching_batch_indices, type=pa.int64()),
                    )
                )
                keep = self._get_prefix_mask(codes)
                matching_batch_indices = matching_batch_indices[keep]
                overflow_item_indices = overflow_item_indices[keep]
                if matching_batch_indices.size == 0:
                    continue

            batch_candidates = self._candidate_last_matrix(
                pc.take(
                    batch[self._config.candidate_codes_field],
                    pa.array(matching_batch_indices, type=pa.int64()),
                )
            )
            if candidates is None:
                candidates = np.empty(
                    (item_count, batch_candidates.shape[1]), dtype=np.int32
                )
            elif candidates.shape[1] != batch_candidates.shape[1]:
                raise ValueError(
                    "candidate topk changed between batches: "
                    f"{candidates.shape[1]} vs {batch_candidates.shape[1]}."
                )
            candidates[overflow_item_indices] = batch_candidates
            seen[overflow_item_indices] = True

        if candidates is None:
            raise ValueError(
                "the plan has overflow items but candidate_codes yielded no candidates."
            )
        item_lookup.broadcast_duplicate_targets(candidates)
        item_lookup.broadcast_duplicate_targets(seen)
        if not np.all(seen):
            missing = np.flatnonzero(~seen)
            preview = ",".join(str(value) for value in missing[:10])
            raise ValueError(
                f"candidate_codes missing for {missing.size} overflow items; "
                f"first overflow positions: {preview}."
            )
        return candidates

    def _offset_codes_column(
        self, codes: np.ndarray, writer_type: Optional[str]
    ) -> pa.Array:
        """Encode an SID matrix shifted into one contiguous vocabulary."""
        return self._codes_column(
            self._config.collision_config.offset_codes(codes), writer_type
        )

    def _align_codes_column(
        self, values: pa.Array, writer_type: Optional[str]
    ) -> pa.Array:
        """Return an SID column in the encoding the destination writer needs."""
        if writer_type == "CsvWriter":
            if pa.types.is_string(values.type) or pa.types.is_large_string(values.type):
                return values
        elif _is_list_column(values):
            codes_type = pa.list_(pa.int64())
            if values.type == codes_type:
                return values
        return self._codes_column(self._convert_codes_to_numpy(values), writer_type)

    def _check_prior_item_id_type(self, values: pa.Array) -> None:
        """Reject a published artifact whose item ID type differs from the input."""
        if values.type != self._item_id_type:
            raise ValueError(
                f"published item_to_sid item ID type {values.type} differs from the "
                f"input type {self._item_id_type}; the merged item_to_sid cannot mix "
                "them. Re-export the published generation or the prediction with "
                "one type."
            )

    def _load_existing_sid_state(
        self, new_item_ids: np.ndarray, new_codes: np.ndarray
    ) -> Tuple[ExistingSidState, int]:
        """Load the existing SID state, or empty state for a full resolve.

        Args:
            new_item_ids: IDs of the items this run places.
            new_codes: Their SID matrix, shape ``(N, n_layers)``; only its prefixes
                are used to select the existing SIDs those items can reach.

        Returns:
            Existing SID keys and item counts in the affected prefixes, and the
            number of existing items assigned to this rank.
        """
        if not self._config.is_append:
            logger.info("full-resolve mode: no published generation supplied")
            return ExistingSidState.empty(), 0

        key_batches: List[np.ndarray] = []
        count_batches: List[np.ndarray] = []
        overlap_preview: List[object] = []
        overlap_count = 0
        existing_item_count = 0
        previous_max_key = -1
        reader = self._make_reader(
            self._bundle.read_path(SID_TO_ITEMS_ARTIFACT), ["codebook", "itemids"]
        )
        wanted_prefixes = np.unique(self._config.collision_config.prefix_ids(new_codes))
        new_id_values = self._item_id_array(new_item_ids)
        for batch in self._iter_batches(reader, "Reading published sid_to_items"):
            codes = self._convert_codes_to_numpy(batch["codebook"])
            keys = self._config.collision_config.convert_codes_to_sid_keys(codes)
            if int(keys[0]) <= previous_max_key or np.any(keys[1:] <= keys[:-1]):
                raise ValueError(
                    "published sid_to_items must be ascending by SID key with one "
                    "row per SID; the merge would otherwise write duplicate "
                    "sids."
                )
            previous_max_key = int(keys[-1])

            item_counts = (
                pc.list_value_length(batch["itemids"])
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )
            existing_item_count += int(item_counts[self._get_prefix_mask(codes)].sum())
            flat_ids = pc.cast(batch["itemids"].flatten(), self._item_id_type)
            if len(flat_ids):
                overlap = pc.is_in(flat_ids, value_set=new_id_values)
                batch_overlap = overlap.true_count
                overlap_count += batch_overlap
                if batch_overlap and len(overlap_preview) < 10:
                    overlap_preview.extend(
                        flat_ids.filter(overlap).slice(0, 10).to_pylist()
                    )

            _, keep = find_in_sorted_array(
                wanted_prefixes, self._config.collision_config.prefix_ids(codes)
            )
            if np.any(keep):
                key_batches.append(keys[keep])
                count_batches.append(item_counts[keep])

        if overlap_count:
            preview = ",".join(str(value) for value in overlap_preview[:10])
            raise ValueError(
                f"{overlap_count} new item IDs are already published; an item "
                "cannot hold two SIDs. Remove them from --input_path. First "
                f"offending IDs: {preview}."
            )
        if key_batches:
            existing_sid_state = ExistingSidState(
                np.concatenate(key_batches), np.concatenate(count_batches)
            )
        else:
            existing_sid_state = ExistingSidState.empty()
        logger.info(
            "append mode: %d new items onto %d published items; %d published "
            "SIDs loaded from the prefixes this run touches, generation %s",
            new_item_ids.shape[0],
            existing_item_count,
            existing_sid_state.sid_keys.shape[0],
            self._config.from_generation,
        )
        return existing_sid_state, existing_item_count

    def _make_writer(
        self, output_path: str, writer_type: Optional[str] = None
    ) -> BaseWriter:
        """Create a repository writer, defaulting to the resolved writer type."""
        writer_type = writer_type or self._resolved_writer_type
        writer = create_writer(
            output_path,
            writer_type=writer_type,
            quota_name=self._config.odps_data_quota_name,
            world_size=self._world_size,
        )
        if self._world_size > 1 and not is_odps_path(output_path):
            # NOTE: Reader globs must visit rank files in SID prefix order.
            suffix = os.path.splitext(writer._output_path)[1]
            writer._output_path = os.path.join(
                output_path, f"part-{self._rank:05d}{suffix}"
            )
        return writer

    def _item_id_array(self, values: np.ndarray) -> pa.Array:
        """Encode item IDs with the input Arrow type preserved."""
        return pa.array(values, type=self._item_id_type)

    @staticmethod
    def _codes_column(codes: np.ndarray, writer_type: Optional[str]) -> pa.Array:
        """Encode an SID matrix for the actual output writer."""
        row_count, layer_count = codes.shape
        if codes.size > _INT32_MAX:
            raise ValueError(
                f"SID codes contain {codes.size} elements "
                f"({row_count} rows x {layer_count} layers), exceeding the "
                f"Arrow List int32 offset limit of {_INT32_MAX}."
            )
        if writer_type == "CsvWriter":
            values = pc.cast(pa.array(codes.reshape(-1)), pa.string())
            offsets = pa.array(np.arange(row_count + 1, dtype=np.int32) * layer_count)
            return pc.binary_join(pa.ListArray.from_arrays(offsets, values), ",")
        return pa.FixedSizeListArray.from_arrays(
            pa.array(codes.reshape(-1)), layer_count
        ).cast(pa.list_(pa.int64()))

    def _copy_prior_item_to_sid_rows(self, writer: BaseWriter) -> int:
        """Copy every published item_to_sid row into the merged output.

        The SID assignment is untouched; codes are re-encoded for this run's
        writer and ``offset_codebook`` is re-derived from ``codebook``.
        """
        if not self._config.is_append:
            return 0
        reader = self._make_reader(
            self._bundle.read_path(ITEM_TO_SID_ARTIFACT),
            ["item_id", "origin_codebook", "codebook", "index"],
        )
        writer_type = self._resolved_writer_type
        written = 0
        for batch in self._iter_batches(
            reader, "Copying published item_to_sid", "codebook"
        ):
            item_id_column = batch["item_id"]
            self._check_prior_item_id_type(item_id_column)
            writer.write(
                {
                    "item_id": item_id_column,
                    "origin_codebook": self._align_codes_column(
                        batch["origin_codebook"], writer_type
                    ),
                    "codebook": self._align_codes_column(
                        batch["codebook"], writer_type
                    ),
                    "offset_codebook": self._offset_codes_column(
                        self._convert_codes_to_numpy(batch["codebook"]), writer_type
                    ),
                    "index": pc.cast(batch["index"], pa.int64()),
                }
            )
            written += len(item_id_column)
        return written

    def _write_item_to_sid(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        result: CollisionResolutionResult,
        existing_item_count: int,
    ) -> None:
        """Write the corpus-complete item_to_sid.

        In append mode the published rows are streamed through first and this
        run's rows are appended, so the artifact stands alone and is exactly
        what the next append consumes.

        Args:
            item_ids: Input item IDs in original row order.
            origin_codes: Their SID matrix before relocation.
            result: Resolved last-layer codes and slots of those rows.
            existing_item_count: Number of existing items in this rank's prefix
                range, which the copied item_to_sid rows must match.

        Raises:
            ValueError: If the published item_to_sid and sid_to_items disagree.
        """
        writer = self._make_writer(self._bundle.write_path(ITEM_TO_SID_ARTIFACT))
        writers = [writer]
        if self._config.is_append:
            writers.append(
                self._make_writer(self._bundle.write_path(DELTA_ITEM_TO_SID_ARTIFACT))
            )
        copied = self._copy_prior_item_to_sid_rows(writer)
        if copied != existing_item_count:
            raise ValueError(
                f"published item_to_sid holds {copied} rows but the published "
                f"sid_to_items hold {existing_item_count} item IDs; one of the two "
                "artifacts is incomplete or they come from different generations "
                "of this tool. Appending onto them would drop published items or "
                "reuse their slots."
            )
        resolved_codes = origin_codes.copy()
        resolved_codes[:, -1] = result.resolved_last_codes
        writer_type = self._resolved_writer_type
        columns = {
            "item_id": self._item_id_array(item_ids),
            "origin_codebook": self._codes_column(origin_codes, writer_type),
            "codebook": self._codes_column(resolved_codes, writer_type),
            "offset_codebook": self._offset_codes_column(resolved_codes, writer_type),
            "index": pa.array(result.slot_indices, type=pa.int64()),
        }
        for item_writer in writers:
            item_writer.write(columns)
        del columns, resolved_codes
        for item_writer in writers:
            item_writer.close()
        self._bundle.rows[ITEM_TO_SID_ARTIFACT] = copied + item_ids.shape[0]
        if self._config.is_append:
            self._bundle.rows[DELTA_ITEM_TO_SID_ARTIFACT] = item_ids.shape[0]

    def _write_sid_to_items_outputs(
        self,
        item_ids: np.ndarray,
        plan: CollisionPlan,
        result: CollisionResolutionResult,
    ) -> None:
        """Group this run's rows by resolved SID and write them."""
        if plan.overflow_rows.size:
            sid_item_row_index = build_resolved_sid_item_row_index(plan, result)
        else:
            sid_item_row_index = build_origin_sid_item_row_index(plan)
        if self._config.is_append:
            self._write_merged_sid_to_items(
                item_ids, sid_item_row_index, plan.existing_sid_state
            )
        else:
            self._write_sid_to_items(item_ids, sid_item_row_index)

    def _write_sid_to_items_rows(
        self, writer: BaseWriter, sid_keys: np.ndarray, item_id_lists: pa.Array
    ) -> None:
        """Write one sid_to_items row per SID key, with its SID from the key."""
        codes = np.stack(
            np.unravel_index(sid_keys, self._config.codebook), axis=1
        ).astype(np.int32)
        writer.write(
            {
                "codebook": self._codes_column(codes, "ParquetWriter"),
                "offset_codebook": self._offset_codes_column(codes, "ParquetWriter"),
                "itemids": item_id_lists,
            }
        )

    def _write_merged_sid_to_items(
        self,
        item_ids: np.ndarray,
        sid_item_row_index: SidItemRowIndex,
        existing_sid_state: ExistingSidState,
    ) -> None:
        """Merge this run's SIDs into the published sid_to_items, writing both.

        Every new SID reaches the delta exactly once and whole: merged with
        the published items of its reader batch, or on its own.

        Args:
            item_ids: Input item IDs in original row order.
            sid_item_row_index: Input row indices ordered by resolved SID and slot.
            existing_sid_state: Published SIDs of the affected prefixes, including
                every published SID a new SID can share a key with.
        """
        writer = self._make_writer(
            self._bundle.write_path(SID_TO_ITEMS_ARTIFACT), "ParquetWriter"
        )
        delta_writer = self._make_writer(
            self._bundle.write_path(DELTA_SID_TO_ITEMS_ARTIFACT), "ParquetWriter"
        )
        input_sid_keys = sid_item_row_index.sid_keys
        input_item_id_lists = _item_lists(
            sid_item_row_index.offsets,
            self._item_id_array(item_ids[sid_item_row_index.item_row_indices]),
        )

        def write_new_sid_rows(low: int, high: int) -> None:
            """Write new SIDs ``[low, high)``, which no published SID shares."""
            if low == high:
                return
            item_id_lists = input_item_id_lists.slice(low, high - low)
            self._write_sid_to_items_rows(
                writer, input_sid_keys[low:high], item_id_lists
            )
            self._write_sid_to_items_rows(
                delta_writer, input_sid_keys[low:high], item_id_lists
            )

        reader = self._make_reader(
            self._bundle.read_path(SID_TO_ITEMS_ARTIFACT), ["codebook", "itemids"]
        )
        cursor = 0
        existing_sid_count = 0
        for batch in self._iter_batches(
            reader, "Writing merged sid_to_items", "codebook"
        ):
            codes = self._convert_codes_to_numpy(batch["codebook"])
            sid_keys = self._config.collision_config.convert_codes_to_sid_keys(codes)
            existing_sid_count += sid_keys.shape[0]
            low = int(np.searchsorted(input_sid_keys, sid_keys[0], side="left"))
            write_new_sid_rows(cursor, low)
            high = int(np.searchsorted(input_sid_keys, sid_keys[-1], side="right"))
            item_id_lists = batch["itemids"]
            if low == high:
                # No new SID in range: the merge would rebuild the batch.
                self._write_sid_to_items_rows(writer, sid_keys, item_id_lists)
            else:
                sid_keys, item_id_lists, updated_sid_indices = _merge_sid_to_items(
                    sid_keys,
                    item_id_lists,
                    input_sid_keys[low:high],
                    input_item_id_lists.slice(low, high - low),
                )
                self._write_sid_to_items_rows(writer, sid_keys, item_id_lists)
                self._write_sid_to_items_rows(
                    delta_writer,
                    sid_keys[updated_sid_indices],
                    item_id_lists.take(updated_sid_indices),
                )
            cursor = high
        write_new_sid_rows(cursor, input_sid_keys.shape[0])
        writer.close()
        delta_writer.close()

        _, shared = find_in_sorted_array(existing_sid_state.sid_keys, input_sid_keys)
        self._bundle.rows[SID_TO_ITEMS_ARTIFACT] = existing_sid_count + int(
            np.count_nonzero(~shared)
        )
        self._bundle.rows[DELTA_SID_TO_ITEMS_ARTIFACT] = input_sid_keys.shape[0]

    def _write_sid_to_items(
        self, item_ids: np.ndarray, sid_item_row_index: SidItemRowIndex
    ) -> None:
        """Write sid_to_items in SID and slot order.

        Args:
            item_ids: Input item IDs in original row order.
            sid_item_row_index: Input row indices ordered by SID and slot.
        """
        writer = self._make_writer(
            self._bundle.write_path(SID_TO_ITEMS_ARTIFACT), "ParquetWriter"
        )
        progress = ProgressLogger(
            "Writing resolved sid_to_items",
            start_n=0,
            miniters=self._config.progress_interval,
        )
        ordered_item_ids = self._item_id_array(
            item_ids[sid_item_row_index.item_row_indices]
        )
        offsets = sid_item_row_index.offsets
        item_id_lists = _item_lists(offsets, ordered_item_ids)
        self._write_sid_to_items_rows(
            writer, sid_item_row_index.sid_keys, item_id_lists
        )
        progress.log(item_ids.shape[0])
        writer.close()
        self._bundle.rows[SID_TO_ITEMS_ARTIFACT] = sid_item_row_index.sid_keys.shape[0]


def _item_lists(offsets: np.ndarray, ordered_item_ids: pa.Array) -> pa.ListArray:
    """Wrap flat item IDs into one ``itemids`` list per SID."""
    if offsets[-1] > _INT32_MAX:
        raise ValueError(
            f"SID item lists contain {offsets[-1]} items, exceeding the "
            f"Arrow List int32 offset limit of {_INT32_MAX}."
        )
    return pa.ListArray.from_arrays(
        pa.array(offsets, type=pa.int32()), ordered_item_ids
    )


def _merge_sid_to_items(
    sid_keys: np.ndarray,
    item_id_lists: pa.ListArray,
    input_sid_keys: np.ndarray,
    input_item_id_lists: pa.ListArray,
) -> Tuple[np.ndarray, pa.ListArray, np.ndarray]:
    """Merge key-sorted published SIDs with input SIDs in their range.

    The stable sort keeps each side in slot order and lists a shared SID's
    published items first.

    Returns:
        The merged SID keys, their item ID lists, and the
        positions of the merged SIDs that received input items.
    """
    item_sid_keys = np.concatenate(
        (
            np.repeat(sid_keys, pc.list_value_length(item_id_lists).to_numpy()),
            np.repeat(
                input_sid_keys, pc.list_value_length(input_item_id_lists).to_numpy()
            ),
        )
    )
    order = np.argsort(item_sid_keys, kind="stable")
    ordered_sid_keys = item_sid_keys[order]
    starts = get_same_value_start_indices(ordered_sid_keys)
    ordered_item_ids = (
        pa.chunked_array([item_id_lists.flatten(), input_item_id_lists.flatten()])
        .take(pa.array(order, type=pa.int64()))
        .combine_chunks()
    )
    merged_sid_keys = ordered_sid_keys[starts]
    return (
        merged_sid_keys,
        _item_lists(np.append(starts, item_sid_keys.shape[0]), ordered_item_ids),
        np.flatnonzero(find_in_sorted_array(input_sid_keys, merged_sid_keys)[1]),
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the SID collision-resolution command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve SID codebook collisions within each SID prefix on a best-effort "
            "basis; finite candidate sets or a full prefix may leave over-capacity "
            "sids."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--output_path",
        default=None,
        help=(
            "Bundle root. Artifacts land at <root>/<generation>/<artifact>, so "
            "one generation is one self-contained directory. Holds sid_to_items, "
            "delta_sid_to_items and the manifest, which are always files, so it "
            "can be a file or OSS path but never an odps:// table. Required "
            "unless --rate_only is set without --from_generation, since an "
            "append reads the published sid_to_items and manifest from it."
        ),
    )
    parser.add_argument(
        "--item_to_sid_root",
        default=None,
        help=(
            "Root for the item_to_sid family only, when it needs storage of its "
            "own -- an odps:// table, say. Defaults to --output_path."
        ),
    )
    parser.add_argument(
        "--generation",
        default=None,
        help="Generation being written, e.g. 'v2'. Required unless --rate_only.",
    )
    parser.add_argument(
        "--from_generation",
        default=None,
        help=(
            "Generation to append onto. Supplying it selects append mode: the new "
            "items in --input_path are placed without moving any published SID."
        ),
    )
    parser.add_argument(
        "--reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
        help=(
            "Format of --input_path, used only when the path carries no .csv / "
            ".parquet / odps:// marker. The published artifacts an append reads "
            "are opened in the format their own manifest records."
        ),
    )
    parser.add_argument(
        "--item_to_sid_writer_type",
        choices=["CsvWriter", "ParquetWriter", "OdpsWriter"],
        default=None,
        help=(
            "Writer for the item_to_sid family only; defaults to matching the "
            "input reader. sid_to_items and the manifest are always parquet "
            "and JSON."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help=(
            "Reader I/O batch size; does not limit the in-memory collision working "
            "set. In append mode it also governs the scans of the published "
            "item_to_sid and sid_to_items, so raise it for a large corpus."
        ),
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1_000_000,
        help="Minimum number of rows between progress logs.",
    )
    parser.add_argument("--item_id_field", default="item_id")
    parser.add_argument("--code_field", default="codes")
    parser.add_argument(
        "--codebook",
        required=True,
        help="Comma-separated per-layer sizes, e.g. '8192,8192,8192'.",
    )
    parser.add_argument("--candidate_codes_field", default="candidate_codes")
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=list(_RESOLVERS),
        default="candidate",
        help=(
            "Place overflow items greedily from model candidates (candidate), "
            "from model candidates by deterministic multi-round arbitration "
            "(iterative), or evenly over the least-loaded codes of their SID "
            "prefix without candidates (uniform)."
        ),
    )
    parser.add_argument(
        "--rate_only",
        action="store_true",
        help="Compute and log metrics without writing item_to_sid or sid_to_items.",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    return parser


def main() -> None:
    """Run SID collision resolution from command-line arguments."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="gloo")
    CollisionResolutionRunner(
        ResolveSidCollisionsConfig.from_namespace(build_parser().parse_args())
    ).run()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
