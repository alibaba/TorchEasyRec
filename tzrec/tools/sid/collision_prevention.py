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

"""Offline SID collision prevention with TorchEasyRec-native I/O.

The runner caps each SID bucket, loads fixed-width last-layer candidates only
for overflow items, delegates collision resolution to the pure NumPy core, and
writes item-level and grouped SID results through TorchEasyRec readers and
writers. CSV uses compact SID strings and JSON item-ID arrays because Arrow's
CSV writer cannot serialize list columns.

The random strategy intentionally preserves the legacy deterministic baseline:
it draws with replacement from the full last-layer space. Placement skips an
item's origin, so an origin draw or a duplicate draw is not replaced.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Register the local reader/writer classes used through create_reader/writer.
import tzrec.datasets.csv_dataset  # noqa: F401
import tzrec.datasets.parquet_dataset  # noqa: F401
from tzrec.datasets.dataset import (
    BaseReader,
    BaseWriter,
    create_reader,
    create_writer,
)
from tzrec.datasets.odps_dataset import _parse_table_path
from tzrec.tools.sid.collision_resolution import (
    CodebookItemGrouping,
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionResult,
    CollisionResolutionStats,
    build_original_item_grouping,
    build_resolved_item_grouping,
    generate_random_candidate_last_codes,
    prepare_collision_plan,
    resolve_sid_collisions,
)
from tzrec.utils.logging_util import logger

_CODE_SEP = "|"
_CAND_SEP = ";"
_WRITE_CHUNK = 20_000_000
_CSV_GROUP_WRITE_CHUNK = 1_000_000
_ARROW_LIST_OFFSET_MAX = int(np.iinfo(np.int32).max)


def _output_path_identity(
    path: str,
) -> Union[str, Tuple[str, str, Optional[str], str, Optional[str]]]:
    """Return the destination identity used by the repository writer."""
    if not path.startswith("odps://"):
        return os.path.realpath(path)

    project, table_name, partitions, schema = _parse_table_path(path)
    partition_spec = partitions[0] if partitions and partitions[0] else None
    return "odps", project, schema, table_name, partition_spec


@dataclass(frozen=True)
class CollisionPreventionConfig:
    """Validated configuration for collision-prevention orchestration."""

    input_path: str
    output_path: str
    original_sid_groups_output_path: Optional[str]
    resolved_sid_groups_output_path: Optional[str]
    diagnostics_output_path: Optional[str]
    reader_type: Optional[str]
    writer_type: Optional[str]
    batch_size: int
    item_id_field: str
    code_field: str
    candidate_codes_field: str
    layer_sizes: Tuple[int, ...]
    max_items_per_codebook: int
    unassigned_policy: str
    strategy: str
    random_num_candidates: int
    rate_only: bool
    odps_data_quota_name: str

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.random_num_candidates < 1:
            raise ValueError(
                f"random_num_candidates must be >= 1, got {self.random_num_candidates}."
            )
        if self.strategy not in {"candidate", "random"}:
            raise ValueError(f"unsupported strategy: {self.strategy!r}.")
        has_original_groups = bool(self.original_sid_groups_output_path)
        has_resolved_groups = bool(self.resolved_sid_groups_output_path)
        if has_original_groups != has_resolved_groups:
            raise ValueError(
                "original_sid_groups_output_path and "
                "resolved_sid_groups_output_path must be supplied together."
            )
        if not self.rate_only and not has_original_groups:
            raise ValueError(
                "both SID group output paths are required unless rate_only is set."
            )

        named_paths = {
            "output_path": self.output_path,
            "original_sid_groups_output_path": self.original_sid_groups_output_path,
            "resolved_sid_groups_output_path": self.resolved_sid_groups_output_path,
            "diagnostics_output_path": self.diagnostics_output_path,
        }
        seen_paths: Dict[
            Union[str, Tuple[str, str, Optional[str], str, Optional[str]]], str
        ] = {}
        for name, path in named_paths.items():
            if not path:
                continue
            path_identity = _output_path_identity(path)
            previous_name = seen_paths.get(path_identity)
            if previous_name is not None:
                raise ValueError(f"{name} must differ from {previous_name}: {path!r}.")
            seen_paths[path_identity] = name
        _ = self.resolution_config

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CollisionPreventionConfig":
        """Build a validated configuration from parsed CLI arguments."""
        layer_sizes = tuple(
            int(value) for value in args.codebook.split(",") if value.strip()
        )
        return cls(
            input_path=args.input_path,
            output_path=args.output_path,
            original_sid_groups_output_path=getattr(
                args, "original_sid_groups_output_path", None
            ),
            resolved_sid_groups_output_path=getattr(
                args, "resolved_sid_groups_output_path", None
            ),
            diagnostics_output_path=args.diagnostics_output_path,
            reader_type=args.reader_type,
            writer_type=args.writer_type,
            batch_size=args.batch_size,
            item_id_field=args.item_id_field,
            code_field=args.code_field,
            candidate_codes_field=args.candidate_codes_field,
            layer_sizes=layer_sizes,
            max_items_per_codebook=args.max_items_per_codebook,
            unassigned_policy=args.unassigned_policy,
            strategy=args.strategy,
            random_num_candidates=args.random_num_candidates,
            rate_only=args.rate_only,
            odps_data_quota_name=args.odps_data_quota_name,
        )

    @property
    def resolution_config(self) -> CollisionResolutionConfig:
        """Return the pure-core configuration."""
        return CollisionResolutionConfig(
            layer_sizes=self.layer_sizes,
            capacity=self.max_items_per_codebook,
            fallback_policy=self.unassigned_policy,
        )


class CollisionRunner:
    """Run SID collision prevention over TorchEasyRec readers and writers."""

    def __init__(
        self,
        config: Union[CollisionPreventionConfig, argparse.Namespace],
    ) -> None:
        self._config = (
            config
            if isinstance(config, CollisionPreventionConfig)
            else CollisionPreventionConfig.from_namespace(config)
        )
        self._default_writer_type: Optional[str] = None
        self._item_id_type: Optional[pa.DataType] = None

    def run(self) -> CollisionResolutionStats:
        """Read, resolve collisions, and write the resulting SID map."""
        item_ids, codes = self._load_codes()
        plan = prepare_collision_plan(
            item_ids,
            codes,
            self._config.resolution_config,
        )
        candidate_last_codes = self._candidate_last_codes(plan)
        result = resolve_sid_collisions(
            plan,
            candidate_last_codes,
            collect_grouping=(
                not self._config.rate_only and plan.overflow_rows.size > 0
            ),
        )
        del candidate_last_codes

        if self._config.rate_only:
            logger.info("rate_only: skipping map and SID group writes")
            del plan
        else:
            self._write_group_outputs(item_ids, codes, plan, result)
            del plan
            self._write_map(item_ids, codes, result)
        if self._config.diagnostics_output_path:
            self._write_diagnostics(result.stats)

        logger.info("SID collision prevention finished: %s", result.stats)
        return result.stats

    def _make_reader(self, selected_cols: List[str]) -> BaseReader:
        """Open a repository reader projecting ``selected_cols``."""
        return create_reader(
            input_path=self._config.input_path,
            batch_size=self._config.batch_size,
            selected_cols=selected_cols,
            reader_type=self._config.reader_type,
            quota_name=self._config.odps_data_quota_name,
        )

    def _read_codes(self) -> Iterable[Dict[str, pa.Array]]:
        """Stream item IDs and SIDs while resolving the default writer type."""
        reader = self._make_reader(
            [self._config.item_id_field, self._config.code_field]
        )
        reader_name = reader.__class__.__name__
        self._default_writer_type = self._config.writer_type or reader_name.replace(
            "Reader", "Writer"
        )
        yield from reader.to_batches()

    @staticmethod
    def _is_list_type(data_type: pa.DataType) -> bool:
        """Return whether ``data_type`` is an Arrow list representation."""
        return (
            pa.types.is_list(data_type)
            or pa.types.is_large_list(data_type)
            or pa.types.is_fixed_size_list(data_type)
        )

    @classmethod
    def _codes_matrix(cls, values: pa.Array) -> np.ndarray:
        """Decode an SID column into an ``(N, n_layers)`` int64 matrix."""
        if cls._is_list_type(values.type):
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
            parts = pc.split_pattern(values, _CODE_SEP)
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
        """Load item IDs and SIDs into arrays used by the NumPy core."""
        id_chunks: List[np.ndarray] = []
        code_chunks: List[np.ndarray] = []
        for batch in self._read_codes():
            item_ids = batch[self._config.item_id_field]
            self._validate_item_ids(item_ids)
            id_chunks.append(item_ids.to_numpy(zero_copy_only=False))
            code_chunks.append(self._codes_matrix(batch[self._config.code_field]))

        if not id_chunks:
            raise ValueError("SID input is empty.")
        item_id_array = np.concatenate(id_chunks)
        code_matrix = np.concatenate(code_chunks, axis=0)
        if code_matrix.shape[1] < 1:
            raise ValueError("SID codes must have at least one layer.")
        return item_id_array, code_matrix

    def _candidate_last_codes(self, plan: CollisionPlan) -> np.ndarray:
        """Provide row-aligned candidate last codes for the overflow plan."""
        if plan.overflow_rows.size == 0:
            return np.empty((0, 0), dtype=np.int64)
        if self._config.strategy == "random":
            return generate_random_candidate_last_codes(
                plan.overflow_item_ids,
                self._config.random_num_candidates,
                self._config.layer_sizes[-1],
            )
        return self._load_candidate_last_codes(plan.overflow_item_ids)

    def _candidate_matrix(self, values: pa.Array) -> np.ndarray:
        """Decode candidate SIDs and retain only their last-layer codes."""
        if self._is_list_type(values.type):
            return self._candidate_list_matrix(values)
        return self._candidate_string_matrix(values)

    def _candidate_list_matrix(self, values: pa.Array) -> np.ndarray:
        """Decode a nested Arrow candidate column into an ``(N, K)`` matrix."""
        row_count = len(values)
        if row_count == 0:
            return np.empty((0, 0), dtype=np.int64)
        inner = values.flatten()
        candidate_count = len(inner) // row_count
        if len(inner) != row_count * candidate_count:
            raise ValueError("ragged candidate_codes: all items must share topk.")
        if candidate_count == 0:
            return np.empty((row_count, 0), dtype=np.int64)
        flat = (
            inner.flatten().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
        tuple_count = row_count * candidate_count
        layer_count = flat.shape[0] // tuple_count
        if layer_count < 1 or flat.shape[0] != tuple_count * layer_count:
            raise ValueError(
                "ragged candidate_codes: every candidate must share n_layers."
            )
        return flat.reshape(row_count, candidate_count, layer_count)[:, :, -1]

    @staticmethod
    def _candidate_string_matrix(values: pa.Array) -> np.ndarray:
        """Decode compact CSV candidate strings into an ``(N, K)`` matrix."""
        rows: List[List[int]] = []
        for value in values.to_pylist():
            last_codes: List[int] = []
            if value:
                for candidate in str(value).split(_CAND_SEP):
                    parts = [
                        int(part)
                        for part in candidate.strip().split(_CODE_SEP)
                        if part.strip()
                    ]
                    if parts:
                        last_codes.append(parts[-1])
            rows.append(last_codes)

        widths = {len(row) for row in rows}
        if len(widths) > 1:
            raise ValueError("ragged candidate_codes: all items must share topk.")
        width = widths.pop() if widths else 0
        if width == 0:
            return np.empty((len(rows), 0), dtype=np.int64)
        return np.asarray(rows, dtype=np.int64).reshape(len(rows), width)

    def _load_candidate_last_codes(self, overflow_item_ids: np.ndarray) -> np.ndarray:
        """Load fixed-width candidates aligned to ``overflow_item_ids``."""
        item_count = overflow_item_ids.shape[0]
        sorted_to_overflow = np.argsort(overflow_item_ids, kind="stable")
        sorted_ids = overflow_item_ids[sorted_to_overflow]
        if item_count > 1 and np.any(sorted_ids[1:] == sorted_ids[:-1]):
            raise ValueError("overflow item IDs must be unique.")

        candidates: Optional[np.ndarray] = None
        seen = np.zeros(item_count, dtype=bool)
        field = self._config.candidate_codes_field
        reader = self._make_reader([self._config.item_id_field, field])
        for batch in reader.to_batches():
            if field not in batch:
                break
            batch_ids = batch[self._config.item_id_field].to_numpy(zero_copy_only=False)
            positions = np.searchsorted(sorted_ids, batch_ids)
            in_bounds = positions < item_count
            source_rows = np.flatnonzero(in_bounds)
            if source_rows.size:
                source_rows = source_rows[
                    sorted_ids[positions[source_rows]] == batch_ids[source_rows]
                ]
            if source_rows.size == 0:
                continue

            target_rows = sorted_to_overflow[positions[source_rows]]
            if np.any(seen[target_rows]):
                raise ValueError("candidate input contains duplicate item IDs.")
            selected = pc.take(batch[field], pa.array(source_rows, type=pa.int64()))
            batch_candidates = self._candidate_matrix(selected)
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
        if not np.all(seen):
            missing = np.flatnonzero(~seen)
            preview = ",".join(str(value) for value in missing[:10])
            raise ValueError(
                f"candidate_codes missing for {missing.size} overflow items; "
                f"first overflow positions: {preview}."
            )
        return candidates

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

        if pa.types.is_integer(values.type):
            encoded_values = pc.cast(values, pa.string())
        else:
            try:
                encoded_values = pa.array(
                    [
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        for value in values.to_pylist()
                    ],
                    type=pa.string(),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "CSV SID group output cannot JSON-encode item ID type "
                    f"{values.type}."
                ) from error
        encoded_lists = pa.ListArray.from_arrays(arrow_offsets, encoded_values)
        joined_values = pc.binary_join(encoded_lists, ",")
        return pc.binary_join_element_wise("[", joined_values, "]", "")

    @staticmethod
    def _codes_column(codes: np.ndarray, is_csv: bool) -> pa.Array:
        """Encode an SID matrix for the actual output writer."""
        row_count, layer_count = codes.shape
        if layer_count < 1:
            raise ValueError("SID output must contain at least one layer.")
        if is_csv:
            columns = [
                pc.cast(pa.array(codes[:, layer]), pa.string())
                for layer in range(layer_count)
            ]
            return pc.binary_join_element_wise(*columns, _CODE_SEP)
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

    def _write_map(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        result: CollisionResolutionResult,
    ) -> None:
        """Write the resolved map in chunks without a full final-code copy."""
        writer = self._make_writer(self._config.output_path)
        is_csv = writer.__class__.__name__ == "CsvWriter"
        retained_rows = (
            np.flatnonzero(result.retained_mask)
            if result.retained_mask is not None
            else None
        )
        output_count = (
            retained_rows.shape[0] if retained_rows is not None else item_ids.shape[0]
        )
        try:
            write_chunk = _WRITE_CHUNK
            if not is_csv:
                write_chunk = min(
                    write_chunk,
                    _ARROW_LIST_OFFSET_MAX // origin_codes.shape[1],
                )
                if write_chunk < 1:
                    raise ValueError("one SID row exceeds Arrow list offset capacity.")
            for start in range(0, output_count, write_chunk):
                end = min(start + write_chunk, output_count)
                selection: Union[slice, np.ndarray]
                selection = (
                    retained_rows[start:end]
                    if retained_rows is not None
                    else slice(start, end)
                )
                origin_chunk = origin_codes[selection]
                final_chunk = origin_chunk.copy()
                final_chunk[:, -1] = result.resolved_last_codes[selection]
                writer.write(
                    {
                        "item_id": self._item_id_array(item_ids[selection]),
                        "origin_codebook": self._codes_column(origin_chunk, is_csv),
                        "codebook": self._codes_column(final_chunk, is_csv),
                        "index": pa.array(
                            result.slot_indices[selection], type=pa.int64()
                        ),
                    }
                )
        finally:
            writer.close()

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
        if original_path is None or resolved_path is None:
            raise RuntimeError("SID group output paths were not validated.")

        original_grouping = build_original_item_grouping(plan)
        self._write_sid_groups(
            original_path,
            item_ids,
            origin_codes,
            original_grouping,
            resolved_last_codes=None,
        )
        if plan.overflow_rows.size == 0:
            self._write_sid_groups(
                resolved_path,
                item_ids,
                origin_codes,
                original_grouping,
                resolved_last_codes=result.resolved_last_codes,
            )
            del original_grouping
        else:
            del original_grouping
            resolved_grouping = build_resolved_item_grouping(plan, result)
            self._write_sid_groups(
                resolved_path,
                item_ids,
                origin_codes,
                resolved_grouping,
                resolved_last_codes=result.resolved_last_codes,
            )
            del resolved_grouping

    def _write_sid_groups(
        self,
        output_path: str,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        grouping: CodebookItemGrouping,
        resolved_last_codes: Optional[np.ndarray],
    ) -> None:
        """Write codebook-to-item-ID groups in SID and slot order.

        Args:
            output_path: Destination passed to the selected repository writer.
            item_ids: Input item IDs in original row order.
            origin_codes: Original full SID matrix.
            grouping: Sorted SID groups and their flattened original row order.
            resolved_last_codes: Final last-layer values, or ``None`` when
                writing original SIDs.

        Raises:
            RuntimeError: If grouping dimensions or counts are inconsistent.
            ValueError: If one group exceeds Arrow list offset capacity.
        """
        group_count = grouping.counts.shape[0]
        if grouping.sid_keys.shape[0] != group_count:
            raise RuntimeError("SID group keys and counts must have the same length.")
        if np.any(grouping.counts <= 0):
            raise RuntimeError("SID group counts must be positive.")
        if np.any(grouping.counts > _ARROW_LIST_OFFSET_MAX):
            raise ValueError("one SID group exceeds Arrow list offset capacity.")

        offsets = grouping.offsets
        if int(offsets[-1]) != grouping.row_order.shape[0]:
            raise RuntimeError("SID group counts do not match grouped row count.")
        writer = self._make_writer(output_path)
        is_csv = writer.__class__.__name__ == "CsvWriter"
        child_chunk = (
            min(_WRITE_CHUNK, _CSV_GROUP_WRITE_CHUNK) if is_csv else _WRITE_CHUNK
        )
        try:
            max_codebook_rows = (
                group_count
                if is_csv
                else _ARROW_LIST_OFFSET_MAX // origin_codes.shape[1]
            )
            if not is_csv and max_codebook_rows < 1:
                raise ValueError("one SID row exceeds Arrow list offset capacity.")
            group_start = 0
            while group_start < group_count:
                child_limit = int(offsets[group_start]) + child_chunk
                group_end = int(np.searchsorted(offsets, child_limit, side="right") - 1)
                group_end = max(group_end, group_start + 1)
                group_end = min(group_end, group_start + max_codebook_rows)

                child_start = int(offsets[group_start])
                child_end = int(offsets[group_end])
                rows = grouping.row_order[child_start:child_end]
                local_offsets = (
                    offsets[group_start : group_end + 1] - child_start
                ).astype(np.int32, copy=False)
                representative_rows = grouping.row_order[offsets[group_start:group_end]]
                code_chunk = origin_codes[representative_rows]
                if resolved_last_codes is not None:
                    code_chunk[:, -1] = resolved_last_codes[representative_rows]
                flat_item_ids = self._item_id_array(item_ids[rows])
                writer.write(
                    {
                        "codebook": self._codes_column(code_chunk, is_csv),
                        "itemids": self._grouped_item_ids_column(
                            flat_item_ids, local_offsets, is_csv
                        ),
                    }
                )
                group_start = group_end
        finally:
            writer.close()

    def _write_diagnostics(self, stats: CollisionResolutionStats) -> None:
        """Write legacy-compatible diagnostic column names."""
        path = self._config.diagnostics_output_path
        if path is None:
            return
        writer = self._make_writer(path)
        try:
            writer.write(
                {
                    name: pa.array([value], type=pa.int64())
                    for name, value in stats.to_output_dict().items()
                }
            )
        finally:
            writer.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the collision-prevention command-line parser."""
    parser = argparse.ArgumentParser(
        description="Prevent SID codebook collisions (vectorized, within-band)."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--original_sid_groups_output_path",
        default=None,
        help=(
            "Output grouped item IDs keyed by their original SID; required unless "
            "--rate_only is set."
        ),
    )
    parser.add_argument(
        "--resolved_sid_groups_output_path",
        default=None,
        help=(
            "Output grouped retained item IDs keyed by their resolved SID; required "
            "unless --rate_only is set."
        ),
    )
    parser.add_argument("--diagnostics_output_path", default=None)
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
    parser.add_argument("--batch_size", type=int, default=100000)
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
        "--unassigned_policy",
        choices=["error", "drop", "keep_original"],
        default="error",
    )
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
    """Run collision prevention from command-line arguments."""
    config = CollisionPreventionConfig.from_namespace(build_parser().parse_args())
    CollisionRunner(config).run()


if __name__ == "__main__":
    main()
