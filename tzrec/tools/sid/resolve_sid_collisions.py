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

Item IDs are expected to be unique in the upstream prediction output. To avoid
blocking a large job on rare upstream violations, this tool does not perform a
full-input uniqueness check or remove duplicate rows. Duplicate rows remain
independent items in collision accounting and outputs. When candidate-strategy
overflow rows share an item ID, they reuse one candidate list matched for that
ID. Duplicate data should still be fixed upstream.

The runner retains the first capacity items in each SID bucket, attempts to
relocate overflow items using fixed-width last-layer candidates, delegates
placement to the pure NumPy core, and writes item-level and grouped SID results
through TorchEasyRec readers and writers. CSV encodes each SID/candidate column
and each item-ID group as comma-separated values because Arrow's CSV writer
cannot serialize list columns, so an item ID containing a comma is rejected at
write time; ``candidate_codes`` is a flat ``topk * n_layers`` run split by the
``--codebook`` length.

The random strategy intentionally preserves the legacy deterministic baseline:
it draws with replacement from the full last-layer space. Placement skips an
item's origin, so an origin draw or a duplicate draw is not replaced.

Both the item map and the SID groups carry an ``offset_codebook`` column
alongside ``codebook``: the same SID with each layer shifted into one
contiguous vocabulary, so layer ``i`` is offset by ``sum(codebook[:i])``. With
``--codebook 64,64,64`` the SID ``[1, 2, 3]`` is written as ``[1, 66, 131]``.
It is derived from the emitted SID, so a relocated item's offset SID follows
where it landed, not what the model predicted.

It is a single-process tool -- launch it with ``python -m`` (no torchrun /
process group). The input is a Semantic-ID table from ``tzrec.predict`` (an
``item_id`` column, a ``codes`` ``list<int>`` column, and -- for the default
``--strategy candidate`` -- a flat ``candidate_codes`` ``list<int>`` column
(``topk * n_layers`` codes per item)).

Example::

    python -m tzrec.tools.sid.resolve_sid_collisions \
        --input_path 'sid_predict_output/*.parquet' \
        --codebook 256,256,256 --max_items_per_codebook 5 \
        --strategy candidate \
        --output_path sid_collision/map \
        --resolved_sid_groups_output_path sid_collision/resolved_groups

To also write the optional original-SID audit grouping, add
``--original_sid_groups_output_path sid_collision/original_groups``.

Append mode
-----------

Supplying ``--existing_sid_groups_path`` switches the tool to appending new
items onto an already-published corpus: ``--input_path`` then holds only the
new items, and **no published SID is ever moved**. Published items are never
rows in the plan, so they cannot be re-ranked; they are read only to be copied
into the merged outputs. A new item continues its bucket's slot numbering from
the published occupancy instead of restarting at one::

    python -m tzrec.tools.sid.resolve_sid_collisions \
        --input_path 'sid_predict_new_20260731/*.parquet' \
        --existing_sid_map_path    'sid_collision/map/*.parquet' \
        --existing_sid_groups_path 'sid_collision/resolved_groups/*.parquet' \
        --codebook 256,256,256 --max_items_per_codebook 5 \
        --strategy candidate \
        --output_path sid_collision/map_20260731 \
        --resolved_sid_groups_output_path sid_collision/resolved_groups_20260731 \
        --delta_map_output_path sid_collision/delta_20260731

Both outputs stay corpus-complete, so generation *n*'s outputs are exactly
generation *n+1*'s ``--existing_*`` inputs; writers truncate, so each
generation needs a fresh output directory.
"""

from __future__ import annotations

import argparse
import os
from contextlib import ExitStack, closing
from dataclasses import dataclass
from functools import cached_property
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Register the local reader/writer classes used through create_reader/writer.
import tzrec.datasets.parquet_dataset  # noqa: F401
from tzrec.datasets.csv_dataset import CsvWriter
from tzrec.datasets.dataset import (
    BaseReader,
    BaseWriter,
    create_reader,
    create_writer,
)
from tzrec.utils.logging_util import ProgressLogger, logger
from tzrec.utils.path_util import check_path_conflict
from tzrec.utils.sid.collision import (
    CodebookItemGrouping,
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionResult,
    CollisionResolutionStats,
    CollisionResolver,
    KnnCollisionResolver,
    PriorOccupancy,
    RandomCollisionResolver,
    build_original_item_grouping,
    build_resolved_item_grouping,
    concat_ranges,
    lookup_sorted,
    prepare_collision_plan,
    sid_band_ids,
    sid_bucket_keys,
    sid_offset_codes,
)

_MAP_WRITE_ROWS = 1_000_000
_GROUP_WRITE_ITEMS = 1_000_000
_ARROW_LIST_OFFSET_MAX = int(np.iinfo(np.int32).max)


def _is_list_column(values: pa.Array) -> bool:
    """Whether an Arrow column is one of the native list types."""
    return (
        pa.types.is_list(values.type)
        or pa.types.is_large_list(values.type)
        or pa.types.is_fixed_size_list(values.type)
    )


def _require_single_process() -> None:
    """Reject multi-process launches; the tool writes one complete map itself."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world_size != 1 or rank != 0:
        raise RuntimeError(
            "resolve_sid_collisions is single-process; launch it with `python -m` "
            f"(not torchrun): got WORLD_SIZE={world_size}, RANK={rank}. Under "
            "multiple ranks every rank reprocesses the full input and emits "
            "duplicate shards (local) or racing overwrite sessions (ODPS)."
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
        representative_sorted_rows = np.searchsorted(
            self._sorted_ids,
            self._sorted_ids[duplicate_sorted_rows],
            side="left",
        )
        self._representative_requested_rows = self._sorted_to_requested[
            representative_sorted_rows
        ]

    def match(self, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return matching source rows and requested-ID positions."""
        positions, found = lookup_sorted(self._sorted_ids, item_ids)
        source_rows = np.flatnonzero(found)
        return source_rows, self._sorted_to_requested[positions[source_rows]]

    def broadcast_duplicate_targets(self, values: np.ndarray) -> None:
        """Copy representative values to duplicate requested-ID positions."""
        values[self._duplicate_requested_rows] = values[
            self._representative_requested_rows
        ]


@dataclass(frozen=True)
class ResolveSidCollisionsConfig:
    """Validated configuration for SID collision-resolution orchestration."""

    input_path: str
    output_path: Optional[str]
    original_sid_groups_output_path: Optional[str]
    resolved_sid_groups_output_path: Optional[str]
    reader_type: Optional[str]
    writer_type: Optional[str]
    batch_size: int
    progress_interval: int
    item_id_field: str
    code_field: str
    candidate_codes_field: str
    layer_sizes: Tuple[int, ...]
    max_items_per_codebook: int
    strategy: str
    random_num_candidates: int
    rate_only: bool
    odps_data_quota_name: str
    existing_sid_map_path: Optional[str] = None
    existing_sid_groups_path: Optional[str] = None
    delta_map_output_path: Optional[str] = None
    delta_sid_groups_output_path: Optional[str] = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.progress_interval < 1:
            raise ValueError(
                f"progress_interval must be >= 1, got {self.progress_interval}."
            )
        if self.random_num_candidates < 1:
            raise ValueError(
                f"random_num_candidates must be >= 1, got {self.random_num_candidates}."
            )
        if self.strategy not in {"candidate", "random"}:
            raise ValueError(f"unsupported strategy: {self.strategy!r}.")
        if not self.rate_only and not self.output_path:
            raise ValueError("output_path is required unless rate_only is set.")
        if not self.rate_only and not self.resolved_sid_groups_output_path:
            raise ValueError(
                "resolved_sid_groups_output_path is required unless rate_only is set."
            )
        if self.is_append and not self.rate_only and not self.existing_sid_map_path:
            raise ValueError(
                "existing_sid_map_path is required in append mode unless rate_only "
                "is set; without it the merged map would hold only the new items."
            )

        candidates = [
            self.output_path,
            self.resolved_sid_groups_output_path,
            self.original_sid_groups_output_path,
        ]
        if self.is_append:
            candidates += [
                self.existing_sid_map_path,
                self.existing_sid_groups_path,
                self.delta_map_output_path,
                self.delta_sid_groups_output_path,
            ]
        paths = [self.input_path]
        paths.extend(path for path in candidates if path)
        has_conflict, conflict_message = check_path_conflict(paths)
        if has_conflict:
            raise ValueError(conflict_message)

        # Eagerly build the resolution config so its validation (layer_sizes,
        # capacity) fails fast here rather than lazily inside run().
        _ = self.resolution_config

    @property
    def is_append(self) -> bool:
        """Whether this run appends onto an already-published corpus.

        The previous round's resolved groups is the occupancy authority and is
        required by every append, so its presence selects the mode.
        """
        return self.existing_sid_groups_path is not None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ResolveSidCollisionsConfig":
        """Build a validated configuration from parsed CLI arguments."""
        try:
            layer_sizes = tuple(int(value) for value in args.codebook.split(","))
        except ValueError as err:
            raise ValueError(
                "--codebook must be 'int,int,...' with no empty fields, got "
                f"{args.codebook!r}."
            ) from err
        return cls(
            input_path=args.input_path,
            output_path=args.output_path,
            original_sid_groups_output_path=args.original_sid_groups_output_path,
            resolved_sid_groups_output_path=args.resolved_sid_groups_output_path,
            existing_sid_map_path=args.existing_sid_map_path,
            existing_sid_groups_path=args.existing_sid_groups_path,
            delta_map_output_path=args.delta_map_output_path,
            delta_sid_groups_output_path=args.delta_sid_groups_output_path,
            reader_type=args.reader_type,
            writer_type=args.writer_type,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            item_id_field=args.item_id_field,
            code_field=args.code_field,
            candidate_codes_field=args.candidate_codes_field,
            layer_sizes=layer_sizes,
            max_items_per_codebook=args.max_items_per_codebook,
            strategy=args.strategy,
            random_num_candidates=args.random_num_candidates,
            rate_only=args.rate_only,
            odps_data_quota_name=args.odps_data_quota_name,
        )

    @cached_property
    def resolution_config(self) -> CollisionResolutionConfig:
        """Return the pure-core configuration."""
        return CollisionResolutionConfig(
            layer_sizes=self.layer_sizes,
            capacity=self.max_items_per_codebook,
        )


class CollisionResolutionRunner:
    """Run best-effort SID collision resolution over repository I/O."""

    def __init__(
        self,
        config: ResolveSidCollisionsConfig,
    ) -> None:
        self._config = config
        self._resolver: CollisionResolver
        if self._config.strategy == "random":
            self._resolver = RandomCollisionResolver(
                self._config.random_num_candidates,
                progress_interval=self._config.progress_interval,
            )
        else:
            self._resolver = KnnCollisionResolver(
                progress_interval=self._config.progress_interval
            )
        self._default_writer_type: Optional[str] = None
        self._item_id_type: Optional[pa.DataType] = None

    def run(self) -> CollisionResolutionStats:
        """Read, resolve collisions, and write the resulting SID map."""
        _require_single_process()
        item_ids, codes = self._load_codes()
        if codes.shape[1] != len(self._config.layer_sizes):
            raise ValueError(
                f"codes have {codes.shape[1]} layers but --codebook has "
                f"{len(self._config.layer_sizes)}."
            )
        prior = self._load_prior_state(item_ids, codes)
        plan = prepare_collision_plan(
            item_ids, codes, self._config.resolution_config, prior=prior
        )
        collect_grouping = not self._config.rate_only and bool(plan.overflow_rows.size)
        candidate_last_codes = None
        if self._config.strategy != "random" and plan.overflow_rows.size:
            candidate_last_codes = self._load_candidate_last_codes(
                plan.overflow_item_ids
            )
        result = self._resolver.resolve(
            plan,
            candidate_last_codes,
            collect_grouping=collect_grouping,
        )
        del candidate_last_codes

        if self._config.rate_only:
            logger.info("rate_only: skipping map and SID group writes")
            del plan
        else:
            self._write_group_outputs(item_ids, codes, plan, result)
            del plan
            self._write_map(item_ids, codes, result)

        logger.info("SID collision resolution finished: %s", result.stats)
        return result.stats

    def _make_reader(
        self, selected_cols: List[str], input_path: Optional[str] = None
    ) -> BaseReader:
        """Open a repository reader projecting ``selected_cols``.

        Args:
            selected_cols: Columns to project.
            input_path: Path to read; defaults to the new-items input.

        Returns:
            A reader over the requested path.
        """
        return create_reader(
            input_path=input_path or self._config.input_path,
            batch_size=self._config.batch_size,
            selected_cols=selected_cols,
            reader_type=self._config.reader_type,
            quota_name=self._config.odps_data_quota_name,
        )

    def _iter_state_batches(
        self,
        path: str,
        fields: List[str],
        description: str,
    ) -> Iterator[Dict[str, pa.Array]]:
        """Stream one state artifact, checking columns and logging progress."""
        reader = self._make_reader(fields, input_path=path)
        missing = [name for name in fields if name not in reader.schema.names]
        if missing:
            raise ValueError(
                f"columns {missing} are missing from {path!r}; point the flag at "
                "an output directory written by this tool."
            )
        progress = ProgressLogger(description, start_n=0)
        scanned = 0
        last_progress_count = 0
        for batch in reader.to_batches():
            yield batch
            scanned += len(batch[fields[0]])
            if self._progress_interval_reached(scanned, last_progress_count):
                progress.log(scanned, suffix=f"{scanned} samples processed")
                last_progress_count = scanned

    def _progress_interval_reached(
        self, processed: int, last_progress_count: int
    ) -> bool:
        """Return whether enough samples passed since the last progress update."""
        return processed - last_progress_count >= self._config.progress_interval

    @staticmethod
    def _codes_matrix(values: pa.Array) -> np.ndarray:
        """Decode an SID column into an ``(N, n_layers)`` int64 matrix."""
        if _is_list_column(values):
            row_count = len(values)
            flat = (
                values.flatten()
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )
        elif pa.types.is_integer(values.type):
            array = values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            return array.reshape(-1, 1)
        else:
            parts = pc.split_pattern(values, ",")
            row_count = len(parts)
            flat = pc.cast(parts.flatten(), pa.int64()).to_numpy(zero_copy_only=False)

        if row_count == 0:
            return np.empty((0, 0), dtype=np.int64)
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
            [self._config.item_id_field, self._config.code_field]
        )
        self._default_writer_type = self._config.writer_type or (
            reader.__class__.__name__.replace("Reader", "Writer")
        )
        progress = ProgressLogger("Reading SID input", start_n=0)
        read_rows = 0
        last_progress_count = 0
        id_chunks: List[np.ndarray] = []
        code_chunks: List[np.ndarray] = []
        for batch in reader.to_batches():
            item_ids = batch[self._config.item_id_field]
            self._validate_item_ids(item_ids)
            id_chunks.append(item_ids.to_numpy(zero_copy_only=False))
            code_chunks.append(self._codes_matrix(batch[self._config.code_field]))
            batch_rows = len(item_ids)
            read_rows += batch_rows
            if self._progress_interval_reached(read_rows, last_progress_count):
                progress.log(read_rows, suffix=f"{read_rows} samples processed")
                last_progress_count = read_rows

        if not id_chunks:
            raise ValueError("SID input is empty.")
        item_id_array = np.concatenate(id_chunks)
        del id_chunks
        code_matrix = np.concatenate(code_chunks, axis=0)
        del code_chunks
        if code_matrix.shape[1] < 1:
            raise ValueError("SID codes must have at least one layer.")
        return item_id_array, code_matrix

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
        flat = self._codes_matrix(values)
        per_row = flat.shape[1]
        n_layers = len(self._config.layer_sizes)
        if per_row % n_layers != 0:
            raise ValueError(
                f"candidate_codes width {per_row} is not a multiple of n_layers "
                f"{n_layers}."
            )
        return flat[:, n_layers - 1 :: n_layers]

    def _load_candidate_last_codes(self, overflow_item_ids: np.ndarray) -> np.ndarray:
        """Load fixed-width candidates aligned to ``overflow_item_ids``."""
        item_count = overflow_item_ids.shape[0]
        item_id_lookup = _ItemIdLookup(overflow_item_ids)

        candidates: Optional[np.ndarray] = None
        seen = np.zeros(item_count, dtype=bool)
        field = self._config.candidate_codes_field
        reader = self._make_reader([self._config.item_id_field, field])
        scanned_items = 0
        last_progress_count = 0
        progress = ProgressLogger("Scanning candidate input", start_n=0)
        for batch in reader.to_batches():
            if field not in batch:
                raise ValueError(
                    f"candidate_codes field {field!r} is missing from an input batch."
                )
            batch_ids = batch[self._config.item_id_field].to_numpy(zero_copy_only=False)
            batch_items = batch_ids.shape[0]
            scanned_items += batch_items
            if self._progress_interval_reached(scanned_items, last_progress_count):
                progress.log(
                    scanned_items,
                    suffix=f"{scanned_items} samples processed",
                )
                last_progress_count = scanned_items
            source_rows, target_rows = item_id_lookup.match(batch_ids)
            if source_rows.size == 0:
                continue

            selected = pc.take(batch[field], pa.array(source_rows, type=pa.int64()))
            batch_candidates = self._candidate_last_matrix(selected)
            if candidates is None:
                candidates = np.empty(
                    (item_count, batch_candidates.shape[1]), dtype=np.int64
                )
            elif candidates.shape[1] != batch_candidates.shape[1]:
                raise ValueError(
                    "candidate topk changed between batches: "
                    f"{candidates.shape[1]} vs {batch_candidates.shape[1]}."
                )
            candidates[target_rows] = batch_candidates
            seen[target_rows] = True

        if candidates is None:
            raise ValueError(
                "map has overflow items but candidate_codes yielded no candidates."
            )
        item_id_lookup.broadcast_duplicate_targets(candidates)
        item_id_lookup.broadcast_duplicate_targets(seen)
        if not np.all(seen):
            missing = np.flatnonzero(~seen)
            preview = ",".join(str(value) for value in missing[:10])
            raise ValueError(
                f"candidate_codes missing for {missing.size} overflow items; "
                f"first overflow positions: {preview}."
            )
        return candidates

    def _load_prior_state(
        self, item_ids: np.ndarray, codes: np.ndarray
    ) -> PriorOccupancy:
        """Load and verify published state, or return empty for a full resolve."""
        if not self._config.is_append:
            logger.info("full-resolve mode: no existing SID state supplied")
            return PriorOccupancy.empty()

        layer_sizes = self._config.layer_sizes
        groups_path = self._config.existing_sid_groups_path
        assert groups_path is not None
        prior, published_items = self._load_prior_occupancy(
            groups_path, sid_band_ids(codes, layer_sizes), item_ids
        )
        self._check_existing_map_size(published_items)
        logger.info(
            "append mode: %d new items onto %d published items in %d touched "
            "buckets (groups=%s, map=%s)",
            item_ids.shape[0],
            published_items,
            prior.bucket_keys.shape[0],
            self._config.existing_sid_groups_path,
            self._config.existing_sid_map_path,
        )
        return prior

    def _decode_grouped_item_ids(self, values: pa.Array) -> Tuple[np.ndarray, pa.Array]:
        """Decode a grouped item-ID column into per-group counts and values."""
        if _is_list_column(values):
            lists = values
        else:
            lists = pc.split_pattern(pc.cast(values, pa.string()), ",")
        lengths = (
            pc.list_value_length(lists)
            .to_numpy(zero_copy_only=False)
            .astype(np.int64, copy=False)
        )
        flat = lists.flatten()
        if self._item_id_type is not None and flat.type != self._item_id_type:
            flat = pc.cast(flat, self._item_id_type)
        return lengths, flat

    def _offset_codes_column(self, codes: np.ndarray, is_csv: bool) -> pa.Array:
        """Encode an SID matrix shifted into one contiguous vocabulary."""
        return self._codes_column(
            sid_offset_codes(codes, self._config.layer_sizes), is_csv
        )

    def _align_codes_column(self, values: pa.Array, is_csv: bool) -> pa.Array:
        """Return an SID column in the encoding the destination writer needs."""
        source_is_text = pa.types.is_string(values.type) or pa.types.is_large_string(
            values.type
        )
        matches_writer = source_is_text if is_csv else _is_list_column(values)
        if matches_writer:
            return values
        return self._codes_column(self._codes_matrix(values), is_csv)

    def _check_state_item_id_type(self, values: pa.Array, source: str) -> None:
        """Reject an existing artifact whose item ID type differs from the input."""
        if len(values) == 0 or self._item_id_type is None:
            return
        if values.type != self._item_id_type:
            raise ValueError(
                f"{source} item ID type {values.type} differs from the new-items "
                f"input type {self._item_id_type}; the merged map cannot mix "
                "them. Re-export the state or the prediction with one type."
            )

    def _load_prior_occupancy(
        self, path: str, band_ids: np.ndarray, item_ids: np.ndarray
    ) -> Tuple[PriorOccupancy, int]:
        """Read published occupancy and total item count from the resolved groups."""
        layer_sizes = self._config.layer_sizes
        wanted_bands = np.unique(band_ids)
        new_item_ids = self._item_id_array(item_ids)

        key_chunks: List[np.ndarray] = []
        count_chunks: List[np.ndarray] = []
        overlapping: List[object] = []
        overlap_count = 0
        published_items = 0
        previous_max_key = -1
        for batch in self._iter_state_batches(
            path,
            ["codebook", "itemids"],
            "Reading published SID groups",
        ):
            keys = sid_bucket_keys(self._codes_matrix(batch["codebook"]), layer_sizes)
            if keys.size:
                if int(keys[0]) <= previous_max_key or np.any(keys[1:] <= keys[:-1]):
                    raise ValueError(
                        "existing SID groups must be ascending by SID key with one "
                        "row per bucket; the merge would otherwise emit duplicate "
                        "buckets."
                    )
                previous_max_key = int(keys[-1])

            lengths, flat_ids = self._decode_grouped_item_ids(batch["itemids"])
            published_items += int(lengths.sum())
            if len(flat_ids):
                published = pc.is_in(flat_ids, value_set=new_item_ids)
                batch_overlap = published.true_count
                overlap_count += batch_overlap
                if batch_overlap and len(overlapping) < 10:
                    overlapping.extend(
                        flat_ids.filter(published).slice(0, 10).to_pylist()
                    )

            _, keep = lookup_sorted(wanted_bands, keys // layer_sizes[-1])
            if np.any(keep):
                key_chunks.append(keys[keep])
                count_chunks.append(lengths[keep])

        if overlap_count:
            preview = ",".join(str(value) for value in overlapping[:10])
            raise ValueError(
                f"{overlap_count} new item IDs are already published; an item "
                "cannot hold two SIDs. Remove them from the append batch. First "
                f"offending IDs: {preview}."
            )
        if not key_chunks:
            return PriorOccupancy.empty(), published_items
        return (
            PriorOccupancy(np.concatenate(key_chunks), np.concatenate(count_chunks)),
            published_items,
        )

    def _check_existing_map_size(self, published_items: int) -> None:
        """Reject a published map whose size disagrees with the groups."""
        path = self._config.existing_sid_map_path
        if path is None:
            return
        map_rows = 0
        for batch in self._iter_state_batches(
            path, ["index"], "Counting published SID map"
        ):
            map_rows += len(batch["index"])
        if map_rows != published_items:
            raise ValueError(
                f"existing map holds {map_rows} rows but the existing groups hold "
                f"{published_items} item IDs; one of the two artifacts is "
                "incomplete or they come from different generations of this tool. "
                "Appending onto them would drop published items or reuse their "
                "slots."
            )

    def _make_writer(self, output_path: str) -> BaseWriter:
        """Create a repository writer for one independently resolved output."""
        if self._default_writer_type is None:
            raise RuntimeError("writer type is unavailable before reading input.")
        return create_writer(
            output_path,
            writer_type=self._default_writer_type,
            quota_name=self._config.odps_data_quota_name,
            world_size=1,
        )

    def _item_id_array(self, values: np.ndarray) -> pa.Array:
        """Encode item IDs with the input Arrow type preserved."""
        if self._item_id_type is None:
            raise RuntimeError("item ID type is unavailable before reading input.")
        return pa.array(values, type=self._item_id_type)

    @staticmethod
    def _grouped_item_ids_column(
        values: pa.Array, offsets: np.ndarray, is_csv: bool
    ) -> pa.Array:
        """Encode flat item IDs and offsets as one grouped output column."""
        arrow_offsets = pa.array(offsets, type=pa.int32())
        if not is_csv:
            return pa.ListArray.from_arrays(arrow_offsets, values)

        encoded_values = pc.cast(values, pa.string())
        if pc.any(pc.match_substring(encoded_values, ",")).as_py():
            raise ValueError(
                "CSV SID group output joins item IDs with commas, so an item ID "
                "cannot itself contain one; use parquet or ODPS for such IDs."
            )
        encoded_lists = pa.ListArray.from_arrays(arrow_offsets, encoded_values)
        return pc.binary_join(encoded_lists, ",")

    @staticmethod
    def _codes_column(codes: np.ndarray, is_csv: bool) -> pa.Array:
        """Encode an SID matrix for the actual output writer."""
        row_count, layer_count = codes.shape
        if is_csv:
            values = pc.cast(pa.array(codes.reshape(-1)), pa.string())
            offsets = pa.array(
                np.arange(
                    0,
                    (row_count + 1) * layer_count,
                    layer_count,
                    dtype=np.int64,
                )
            )
            return pc.binary_join(pa.LargeListArray.from_arrays(offsets, values), ",")
        if row_count > _ARROW_LIST_OFFSET_MAX // layer_count:
            raise ValueError("SID output exceeds Arrow list offset capacity.")
        values = pa.array(codes.reshape(-1))
        offsets = pa.array(
            np.arange(
                0,
                (row_count + 1) * layer_count,
                layer_count,
                dtype=np.int32,
            )
        )
        return pa.ListArray.from_arrays(offsets, values)

    def _write_new_map_rows(
        self,
        writer: BaseWriter,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        result: CollisionResolutionResult,
        progress: ProgressLogger,
        written: int,
    ) -> None:
        """Write this run's own item rows in chunks, without a full code copy."""
        is_csv = isinstance(writer, CsvWriter)
        output_count = item_ids.shape[0]
        last_progress_count = written
        write_chunk = _max_code_rows(is_csv, origin_codes.shape[1], _MAP_WRITE_ROWS)
        for start in range(0, output_count, write_chunk):
            end = min(start + write_chunk, output_count)
            selection = slice(start, end)
            origin_chunk = origin_codes[selection]
            final_chunk = origin_chunk.copy()
            final_chunk[:, -1] = result.resolved_last_codes[selection]
            writer.write(
                {
                    "item_id": self._item_id_array(item_ids[selection]),
                    "origin_codebook": self._codes_column(origin_chunk, is_csv),
                    "codebook": self._codes_column(final_chunk, is_csv),
                    "offset_codebook": self._offset_codes_column(final_chunk, is_csv),
                    "index": pa.array(result.slot_indices[selection], type=pa.int64()),
                }
            )
            written += end - start
            if self._progress_interval_reached(written, last_progress_count):
                progress.log(written, suffix=f"{written} samples processed")
                last_progress_count = written

    def _stream_existing_map_rows(self, writer: BaseWriter) -> int:
        """Copy every published map row into the merged output unchanged."""
        path = self._config.existing_sid_map_path
        if path is None or not self._config.is_append:
            return 0
        is_csv = isinstance(writer, CsvWriter)
        fields = ["item_id", "origin_codebook", "codebook", "index"]
        written = 0
        for batch in self._iter_state_batches(
            path, fields, "Copying published item map"
        ):
            item_id_column = batch["item_id"]
            self._check_state_item_id_type(item_id_column, "existing SID map")
            writer.write(
                {
                    "item_id": item_id_column,
                    "origin_codebook": self._align_codes_column(
                        batch["origin_codebook"], is_csv
                    ),
                    "codebook": self._align_codes_column(batch["codebook"], is_csv),
                    "offset_codebook": self._offset_codes_column(
                        self._codes_matrix(batch["codebook"]), is_csv
                    ),
                    "index": pc.cast(batch["index"], pa.int64()),
                }
            )
            written += len(item_id_column)
        return written

    def _write_map(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        result: CollisionResolutionResult,
    ) -> None:
        """Write the corpus-complete item map.

        In append mode the published rows are streamed through first and this
        run's rows are appended, so the artifact stands alone and is exactly
        what the next append consumes.
        """
        output_path = self._config.output_path
        if output_path is None:
            raise RuntimeError("map output path was not validated.")
        with closing(self._make_writer(output_path)) as writer:
            progress = ProgressLogger("Writing resolved item map", start_n=0)
            written = self._stream_existing_map_rows(writer)
            self._write_new_map_rows(
                writer, item_ids, origin_codes, result, progress, written
            )

        delta_path = self._config.delta_map_output_path
        if delta_path is None or not self._config.is_append:
            return
        with closing(self._make_writer(delta_path)) as writer:
            self._write_new_map_rows(
                writer,
                item_ids,
                origin_codes,
                result,
                ProgressLogger("Writing delta item map", start_n=0),
                0,
            )

    def _write_group_outputs(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        plan: CollisionPlan,
        result: CollisionResolutionResult,
    ) -> None:
        """Build and write original and resolved SID item groups sequentially.

        Args:
            item_ids: Input item IDs in original row order.
            origin_codes: Original full SID matrix.
            plan: Original SID aggregation and overflow plan.
            result: Completed collision-resolution result.
        """
        original_path = self._config.original_sid_groups_output_path
        resolved_path = self._config.resolved_sid_groups_output_path
        if resolved_path is None:
            raise RuntimeError("resolved SID group output path was not validated.")

        original_grouping: Optional[CodebookItemGrouping] = None
        if original_path is not None:
            original_grouping = build_original_item_grouping(plan)
            self._write_sid_groups(
                original_path,
                item_ids,
                origin_codes,
                original_grouping,
                resolved_last_codes=None,
                progress_description="Writing original SID item groups",
            )

        if plan.overflow_rows.size:
            del original_grouping
            grouping = build_resolved_item_grouping(plan, result)
            resolved_last_codes = result.resolved_last_codes
        else:
            grouping = original_grouping or build_original_item_grouping(plan)
            resolved_last_codes = None
        if self._config.is_append:
            existing_path = self._config.existing_sid_groups_path
            assert existing_path is not None
            self._write_merged_sid_groups(
                resolved_path,
                existing_path,
                item_ids,
                origin_codes,
                grouping,
                resolved_last_codes,
            )
            return
        self._write_sid_groups(
            resolved_path,
            item_ids,
            origin_codes,
            grouping,
            resolved_last_codes=resolved_last_codes,
            progress_description="Writing resolved SID item groups",
        )

    def _emit_group_rows(
        self,
        writer: BaseWriter,
        codes: np.ndarray,
        offsets: np.ndarray,
        values: pa.Array,
        rows: Optional[np.ndarray] = None,
    ) -> None:
        """Write one chunk of SID groups, optionally only ``rows`` of it."""
        is_csv = isinstance(writer, CsvWriter)
        if rows is not None:
            if rows.size == 0:
                return
            lengths = np.diff(offsets)[rows]
            values = values.take(
                pa.array(concat_ranges(offsets[:-1][rows], lengths), type=pa.int64())
            )
            codes = codes[rows]
            offsets = np.zeros(rows.shape[0] + 1, dtype=np.int64)
            np.cumsum(lengths, out=offsets[1:])
        writer.write(
            {
                "codebook": self._codes_column(codes, is_csv),
                "offset_codebook": self._offset_codes_column(codes, is_csv),
                "itemids": self._grouped_item_ids_column(values, offsets, is_csv),
            }
        )

    def _build_append_groups(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        grouping: CodebookItemGrouping,
        resolved_last_codes: Optional[np.ndarray],
    ) -> _AppendGroups:
        """Materialize this run's groups in the shape the merge consumes."""
        offsets = grouping.offsets
        representative_rows = grouping.row_order[offsets[:-1]]
        codes = origin_codes[representative_rows]
        if resolved_last_codes is not None:
            codes = codes.copy()
            codes[:, -1] = resolved_last_codes[representative_rows]
        return _AppendGroups(
            keys=grouping.sid_keys,
            counts=grouping.counts,
            offsets=offsets,
            codes=codes,
            item_ids=self._item_id_array(item_ids[grouping.row_order]),
        )

    def _write_merged_sid_groups(
        self,
        resolved_path: str,
        existing_path: str,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        grouping: CodebookItemGrouping,
        resolved_last_codes: Optional[np.ndarray],
    ) -> None:
        """Merge this run's groups into the published groups and write both."""
        layer_sizes = self._config.layer_sizes
        groups = self._build_append_groups(
            item_ids, origin_codes, grouping, resolved_last_codes
        )
        group_count = groups.keys.shape[0]

        with ExitStack() as stack:
            writer = stack.enter_context(closing(self._make_writer(resolved_path)))
            is_csv = isinstance(writer, CsvWriter)
            delta_writer: Optional[BaseWriter] = None
            if self._config.delta_sid_groups_output_path is not None:
                delta_writer = stack.enter_context(
                    closing(
                        self._make_writer(self._config.delta_sid_groups_output_path)
                    )
                )

            def emit(
                codes: np.ndarray,
                offsets: np.ndarray,
                values: pa.Array,
                delta_rows: Optional[np.ndarray],
            ) -> None:
                """Write one chunk to the corpus output and the delta output."""
                self._emit_group_rows(writer, codes, offsets, values)
                if delta_writer is not None:
                    self._emit_group_rows(
                        delta_writer, codes, offsets, values, delta_rows
                    )

            max_rows = _max_code_rows(is_csv, groups.codes.shape[1], group_count)

            def emit_new_only(low: int, high: int) -> None:
                """Write groups this run created in buckets nobody occupied."""
                for start, stop in _group_chunk_bounds(
                    groups.offsets, low, high, max_rows
                ):
                    child_low = int(groups.offsets[start])
                    child_high = int(groups.offsets[stop])
                    emit(
                        groups.codes[start:stop],
                        groups.offsets[start : stop + 1] - child_low,
                        groups.item_ids.slice(child_low, child_high - child_low),
                        None,
                    )

            cursor = 0
            for batch in self._iter_state_batches(
                existing_path,
                ["codebook", "itemids"],
                "Writing merged SID item groups",
            ):
                batch_codes = self._codes_matrix(batch["codebook"])
                batch_keys = sid_bucket_keys(batch_codes, layer_sizes)
                if batch_keys.size == 0:
                    continue
                low = int(np.searchsorted(groups.keys, batch_keys[0], side="left"))
                if cursor < low:
                    emit_new_only(cursor, low)
                high = int(np.searchsorted(groups.keys, batch_keys[-1], side="right"))
                batch_lengths, batch_values = self._decode_grouped_item_ids(
                    batch["itemids"]
                )
                emit(
                    *_merge_group_batch(
                        batch_keys,
                        batch_codes,
                        batch_lengths,
                        batch_values,
                        groups,
                        low,
                        high,
                    )
                )
                cursor = high
            emit_new_only(cursor, group_count)

    def _write_sid_groups(
        self,
        output_path: str,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        grouping: CodebookItemGrouping,
        resolved_last_codes: Optional[np.ndarray],
        progress_description: str,
    ) -> None:
        """Write codebook-to-item-ID groups in SID and slot order.

        Args:
            output_path: Destination passed to the selected repository writer.
            item_ids: Input item IDs in original row order.
            origin_codes: Original full SID matrix.
            grouping: Sorted SID groups and their flattened original row order.
            resolved_last_codes: Final last-layer values, or ``None`` when
                writing original SIDs.
            progress_description: Description identifying the group output.

        Raises:
            ValueError: If one group exceeds Arrow list offset capacity.
        """
        group_count = grouping.counts.shape[0]
        if np.any(grouping.counts > _ARROW_LIST_OFFSET_MAX):
            raise ValueError("one SID group exceeds Arrow list offset capacity.")

        offsets = grouping.offsets
        with closing(self._make_writer(output_path)) as writer:
            is_csv = isinstance(writer, CsvWriter)
            progress = ProgressLogger(progress_description, start_n=0)
            last_progress_count = 0
            max_codebook_rows = _max_code_rows(
                is_csv, origin_codes.shape[1], group_count
            )
            for group_start, group_end in _group_chunk_bounds(
                offsets, 0, group_count, max_codebook_rows
            ):
                child_start = int(offsets[group_start])
                child_end = int(offsets[group_end])
                rows = grouping.row_order[child_start:child_end]
                local_offsets = offsets[group_start : group_end + 1] - child_start
                representative_rows = grouping.row_order[offsets[group_start:group_end]]
                code_chunk = origin_codes[representative_rows]
                if resolved_last_codes is not None:
                    code_chunk[:, -1] = resolved_last_codes[representative_rows]
                self._emit_group_rows(
                    writer,
                    code_chunk,
                    local_offsets,
                    self._item_id_array(item_ids[rows]),
                )
                if self._progress_interval_reached(child_end, last_progress_count):
                    progress.log(
                        child_end,
                        suffix=f"{child_end} samples processed",
                    )
                    last_progress_count = child_end


def _max_code_rows(is_csv: bool, layer_count: int, row_cap: int) -> int:
    """Return how many SID rows one write can encode."""
    if is_csv:
        return row_cap
    return min(row_cap, _ARROW_LIST_OFFSET_MAX // layer_count)


def _group_chunk_bounds(
    offsets: np.ndarray, low: int, high: int, max_rows: int
) -> Iterator[Tuple[int, int]]:
    """Split ``[low, high)`` groups into writable ``(start, stop)`` chunks."""
    while low < high:
        child_limit = int(offsets[low]) + _GROUP_WRITE_ITEMS
        stop = int(np.searchsorted(offsets, child_limit, side="right") - 1)
        stop = min(max(stop, low + 1), low + max_rows, high)
        yield low, stop
        low = stop


@dataclass(frozen=True)
class _AppendGroups:
    """This run's SID groups, prepared for merging into the published groups."""

    keys: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    codes: np.ndarray
    item_ids: pa.Array


def _merge_group_batch(
    batch_keys: np.ndarray,
    batch_codes: np.ndarray,
    batch_lengths: np.ndarray,
    batch_values: pa.Array,
    groups: _AppendGroups,
    low: int,
    high: int,
) -> Tuple[np.ndarray, np.ndarray, pa.Array, np.ndarray]:
    """Merge one batch of published groups with the new groups it overlaps."""
    batch_rows = batch_keys.shape[0]
    candidate_keys = groups.keys[low:high]
    positions, matched = lookup_sorted(candidate_keys, batch_keys)

    new_for_batch = np.full(batch_rows, -1, dtype=np.int64)
    new_for_batch[matched] = low + positions[matched]
    consumed = np.zeros(candidate_keys.shape[0], dtype=bool)
    consumed[positions[matched]] = True
    unmatched = low + np.flatnonzero(~consumed)

    output_rows = batch_rows + unmatched.shape[0]
    batch_slots = np.arange(batch_rows) + np.searchsorted(
        groups.keys[unmatched], batch_keys, side="left"
    )
    unmatched_slots = np.arange(unmatched.shape[0]) + np.searchsorted(
        batch_keys, groups.keys[unmatched], side="left"
    )

    batch_of_output = np.full(output_rows, -1, dtype=np.int64)
    batch_of_output[batch_slots] = np.arange(batch_rows)
    new_of_output = np.full(output_rows, -1, dtype=np.int64)
    new_of_output[batch_slots] = new_for_batch
    new_of_output[unmatched_slots] = unmatched

    has_batch = batch_of_output >= 0
    has_new = new_of_output >= 0
    batch_index = np.maximum(batch_of_output, 0)
    new_index = np.maximum(new_of_output, 0)

    codes = np.empty((output_rows, batch_codes.shape[1]), dtype=np.int64)
    codes[batch_slots] = batch_codes
    codes[unmatched_slots] = groups.codes[unmatched]

    batch_part = np.where(has_batch, batch_lengths[batch_index], 0)
    new_part = np.where(has_new, groups.counts[new_index], 0)
    offsets = np.zeros(output_rows + 1, dtype=np.int64)
    np.cumsum(batch_part + new_part, out=offsets[1:])

    child_low = int(groups.offsets[low])
    child_high = int(groups.offsets[high])
    combined = pa.chunked_array(
        [batch_values, groups.item_ids.slice(child_low, child_high - child_low)]
    )
    batch_child_starts = np.zeros(batch_rows + 1, dtype=np.int64)
    np.cumsum(batch_lengths, out=batch_child_starts[1:])
    gather = np.empty(int(offsets[-1]), dtype=np.int64)
    gather[concat_ranges(offsets[:-1], batch_part)] = concat_ranges(
        batch_child_starts[batch_index], batch_part
    )
    gather[concat_ranges(offsets[:-1] + batch_part, new_part)] = concat_ranges(
        len(batch_values) + groups.offsets[new_index] - child_low, new_part
    )
    values = combined.take(pa.array(gather, type=pa.int64())).combine_chunks()
    return codes, offsets, values, np.flatnonzero(has_new)


def build_parser() -> argparse.ArgumentParser:
    """Build the SID collision-resolution command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve SID codebook collisions within each band on a best-effort "
            "basis; finite candidate sets may leave over-capacity buckets."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--output_path",
        default=None,
        help="Resolved item map output; required unless --rate_only is set.",
    )
    parser.add_argument(
        "--original_sid_groups_output_path",
        default=None,
        help=("Optional audit output grouping item IDs by their original SID."),
    )
    parser.add_argument(
        "--resolved_sid_groups_output_path",
        default=None,
        help=(
            "Output grouped item IDs keyed by their resolved SID; required "
            "unless --rate_only is set."
        ),
    )
    parser.add_argument(
        "--existing_sid_groups_path",
        default=None,
        help=(
            "Previous round's --resolved_sid_groups_output_path. Supplying it "
            "selects append mode: the new items in --input_path are placed "
            "without moving any already-published SID."
        ),
    )
    parser.add_argument(
        "--existing_sid_map_path",
        default=None,
        help=(
            "Previous round's --output_path. Required in append mode unless "
            "--rate_only, because only the map carries origin_codebook."
        ),
    )
    parser.add_argument(
        "--delta_map_output_path",
        default=None,
        help="Append mode only: map rows for the new items alone.",
    )
    parser.add_argument(
        "--delta_sid_groups_output_path",
        default=None,
        help=(
            "Append mode only: buckets this run changed, each with its complete "
            "post-append item list, for an idempotent index upsert."
        ),
    )
    parser.add_argument(
        "--reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter", "OdpsWriter"],
        default=None,
        help="Output writer; defaults to matching the input reader.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help=(
            "Reader I/O batch size; does not limit the in-memory collision working "
            "set. In append mode it also governs the scans of the published map "
            "and groups, so raise it for a large corpus."
        ),
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1_000_000,
        help="Number of processed samples between progress checks.",
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
        choices=["candidate", "random"],
        default="candidate",
        help="Use model candidates or deterministic legacy random draws.",
    )
    parser.add_argument(
        "--random_num_candidates",
        type=int,
        default=64,
        help="Full-space random draws per overflow item for random strategy.",
    )
    parser.add_argument(
        "--rate_only",
        action="store_true",
        help="Compute and log metrics without writing map or SID group outputs.",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    return parser


def main() -> None:
    """Run SID collision resolution from command-line arguments."""
    config = ResolveSidCollisionsConfig.from_namespace(build_parser().parse_args())
    CollisionResolutionRunner(config).run()


if __name__ == "__main__":
    main()
