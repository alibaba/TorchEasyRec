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

"""Offline SID codebook collision-prevention tool.

The default ('candidate') allocator is a deterministic post-process over predicted
SID rows plus explicit candidate SID rows. The opt-in 'random' strategy instead
generates random within-band last-layer candidates (no candidate input required) as
a baseline that ignores semantic nearest-neighbor proximity; it is still fully
reproducible given ``seed``.
"""

import argparse
import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pyarrow as pa

# Register the local reader/writer classes used through create_reader/writer.
import tzrec.datasets.csv_dataset  # noqa: F401
import tzrec.datasets.parquet_dataset  # noqa: F401
from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger


@dataclass(frozen=True)
class RawSidRow:
    """One raw item -> SID row."""

    item_id: Any
    item_key: str
    origin_codebook: str


@dataclass(frozen=True)
class CandidateSidRow:
    """One item -> candidate SID row."""

    item_key: str
    candidate_codebook: str
    priority: int
    score: float


@dataclass(frozen=True)
class AssignedSidRow:
    """One final item -> SID assignment row."""

    item_id: Any
    item_key: str
    origin_codebook: str
    codebook: str
    index: int


@dataclass(frozen=True)
class AssignmentStats:
    """Summary statistics for a collision-prevention run."""

    total_items: int
    raw_collision_buckets: int
    final_collision_buckets: int
    reassigned_count: int
    unassigned_count: int
    iteration_count: int
    max_final_bucket_size: int


class SidCollisionAssigner:
    """Deterministically assign overflow SID rows to explicit candidates."""

    def __init__(
        self,
        capacity: int,
        max_iters: int = 50,
        seed: int = 2026,
        score_order: str = "lower",
        unassigned_policy: str = "error",
        strategy: str = "candidate",
        code_delimiter: str = ",",
        random_last_layer_size: Optional[int] = None,
        random_num_candidates: int = 64,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {capacity}")
        if score_order not in ("lower", "higher"):
            raise ValueError("score_order must be 'lower' or 'higher'.")
        if unassigned_policy not in ("error", "drop", "keep_original"):
            raise ValueError(
                "unassigned_policy must be one of: error, drop, keep_original."
            )
        if strategy not in ("candidate", "random"):
            raise ValueError("strategy must be 'candidate' or 'random'.")
        if strategy == "random":
            if random_last_layer_size is None or random_last_layer_size < 2:
                raise ValueError(
                    "strategy='random' requires random_last_layer_size >= 2."
                )
            if random_num_candidates < 1:
                raise ValueError("random_num_candidates must be >= 1.")
        self.capacity = capacity
        self.max_iters = max_iters
        self.seed = seed
        self.score_order = score_order
        self.unassigned_policy = unassigned_policy
        self.strategy = strategy
        self.code_delimiter = code_delimiter
        self.random_last_layer_size = random_last_layer_size
        self.random_num_candidates = random_num_candidates
        self._candidate_sort_keys: Dict[
            CandidateSidRow, Tuple[int, float, int, str, str]
        ] = {}

    def assign(
        self,
        raw_rows: Sequence[RawSidRow],
        candidate_rows: Sequence[CandidateSidRow],
    ) -> Tuple[List[AssignedSidRow], AssignmentStats]:
        """Assign overflow SID rows to non-full candidate codebooks."""
        if len({row.item_key for row in raw_rows}) != len(raw_rows):
            raise ValueError("raw_rows contains duplicate item_id values.")

        raw_by_item = {row.item_key: row for row in raw_rows}
        by_origin: Dict[str, List[RawSidRow]] = defaultdict(list)
        for row in raw_rows:
            by_origin[row.origin_codebook].append(row)

        assigned: List[AssignedSidRow] = []
        assigned_items = set()
        code_counts: Dict[str, int] = defaultdict(int)

        for codebook, rows in by_origin.items():
            for index, row in enumerate(
                sorted(rows, key=self._assignment_sort_key)[: self.capacity],
                start=1,
            ):
                assigned.append(
                    AssignedSidRow(
                        item_id=row.item_id,
                        item_key=row.item_key,
                        origin_codebook=row.origin_codebook,
                        codebook=codebook,
                        index=index,
                    )
                )
                assigned_items.add(row.item_key)
                code_counts[codebook] += 1

        overflow_items = set(raw_by_item) - assigned_items
        if self.strategy == "random":
            candidate_rows = self._generate_random_candidates(
                overflow_items, raw_by_item
            )
        elif overflow_items and not candidate_rows:
            raise ValueError(
                "raw SID input has overflow rows, but no explicit candidate input was "
                "provided."
            )
        candidates_by_item = self._dedup_candidates(candidate_rows, raw_by_item)

        iteration_count = self._assign_candidates(
            raw_by_item,
            candidates_by_item,
            assigned,
            assigned_items,
            code_counts,
        )
        unassigned = self._handle_unassigned(
            raw_by_item,
            assigned,
            assigned_items,
            code_counts,
        )
        final_counts = self._final_counts(assigned)
        stats = AssignmentStats(
            total_items=len(raw_rows),
            raw_collision_buckets=self._raw_collision_buckets(raw_rows),
            final_collision_buckets=sum(
                1 for count in final_counts.values() if count > self.capacity
            ),
            reassigned_count=sum(
                1 for row in assigned if row.origin_codebook != row.codebook
            ),
            unassigned_count=(
                len(unassigned) if self.unassigned_policy != "keep_original" else 0
            ),
            iteration_count=iteration_count,
            max_final_bucket_size=max(final_counts.values()) if final_counts else 0,
        )
        return sorted(assigned, key=lambda r: (r.codebook, r.index, r.item_key)), stats

    @staticmethod
    def _stable_hash(*parts: Any) -> int:
        h = hashlib.blake2b(digest_size=8)
        for part in parts:
            h.update(str(part).encode("utf-8"))
            h.update(b"\x1f")
        return int.from_bytes(h.digest(), byteorder="big", signed=False)

    def _raw_collision_buckets(self, raw_rows: Sequence[RawSidRow]) -> int:
        counts: Dict[str, int] = defaultdict(int)
        for row in raw_rows:
            counts[row.origin_codebook] += 1
        return sum(1 for count in counts.values() if count > self.capacity)

    def _generate_random_candidates(
        self,
        overflow_items: Iterable[str],
        raw_by_item: Dict[str, RawSidRow],
    ) -> List[CandidateSidRow]:
        """Generate random within-band last-layer candidates for overflow items.

        Keeps every layer except the last and draws distinct random codes for the
        last layer in ``[0, random_last_layer_size)``, excluding the origin code, so
        each item stays in its own ``(prefix)`` band. Deterministic given ``seed``;
        unlike the 'candidate' strategy it ignores nearest-neighbor proximity.
        """
        assert self.random_last_layer_size is not None
        size = self.random_last_layer_size
        num_draws = min(self.random_num_candidates, size - 1)
        rows: List[CandidateSidRow] = []
        for item_key in overflow_items:
            raw = raw_by_item[item_key]
            parts = raw.origin_codebook.split(self.code_delimiter)
            prefix = parts[:-1]
            try:
                origin_last: Optional[int] = int(parts[-1])
            except ValueError:
                origin_last = None
            rng = random.Random(self._stable_hash(self.seed, item_key))
            drawn: set = set()
            while len(drawn) < num_draws:
                value = rng.randrange(size)
                if value in drawn or value == origin_last:
                    continue
                drawn.add(value)
                rows.append(
                    CandidateSidRow(
                        item_key=item_key,
                        candidate_codebook=self.code_delimiter.join(
                            [*prefix, str(value)]
                        ),
                        priority=1,
                        score=0.0,
                    )
                )
        return rows

    def _assignment_sort_key(self, row: RawSidRow) -> Tuple[int, str]:
        return (
            self._stable_hash(self.seed, row.origin_codebook, row.item_key),
            row.item_key,
        )

    def _candidate_sort_key(
        self,
        row: CandidateSidRow,
    ) -> Tuple[int, float, int, str, str]:
        # Memoized: this key's blake2b tie-breaker is pure but is compared inside
        # the up-to-max_iters assignment loop, so recomputing it would dominate
        # the phase. CandidateSidRow is frozen/hashable, so cache on the row.
        cached = self._candidate_sort_keys.get(row)
        if cached is not None:
            return cached
        score = row.score if self.score_order == "lower" else -row.score
        key = (
            row.priority,
            score,
            self._stable_hash(self.seed, row.item_key, row.candidate_codebook),
            row.item_key,
            row.candidate_codebook,
        )
        self._candidate_sort_keys[row] = key
        return key

    def _dedup_candidates(
        self,
        candidate_rows: Sequence[CandidateSidRow],
        raw_by_item: Dict[str, RawSidRow],
    ) -> Dict[str, List[CandidateSidRow]]:
        dedup_candidates: Dict[Tuple[str, str], CandidateSidRow] = {}
        for row in candidate_rows:
            if row.item_key not in raw_by_item:
                continue
            key = (row.item_key, row.candidate_codebook)
            current = dedup_candidates.get(key)
            if current is None or self._candidate_sort_key(
                row
            ) < self._candidate_sort_key(current):
                dedup_candidates[key] = row

        candidates_by_item: Dict[str, List[CandidateSidRow]] = defaultdict(list)
        for row in dedup_candidates.values():
            candidates_by_item[row.item_key].append(row)
        return candidates_by_item

    def _assign_candidates(
        self,
        raw_by_item: Dict[str, RawSidRow],
        candidates_by_item: Dict[str, List[CandidateSidRow]],
        assigned: List[AssignedSidRow],
        assigned_items: set,
        code_counts: Dict[str, int],
    ) -> int:
        iteration_count = 0
        for iteration in range(self.max_iters):
            unassigned = set(raw_by_item) - assigned_items
            if not unassigned:
                break

            available = self._available_candidates(
                unassigned,
                candidates_by_item,
                code_counts,
            )
            if not available:
                break

            accepted = self._select_candidates(available, code_counts)
            progress = 0
            for candidate in accepted:
                if candidate.item_key in assigned_items:
                    continue
                if code_counts[candidate.candidate_codebook] >= self.capacity:
                    continue
                raw = raw_by_item[candidate.item_key]
                code_counts[candidate.candidate_codebook] += 1
                assigned.append(
                    AssignedSidRow(
                        item_id=raw.item_id,
                        item_key=raw.item_key,
                        origin_codebook=raw.origin_codebook,
                        codebook=candidate.candidate_codebook,
                        index=code_counts[candidate.candidate_codebook],
                    )
                )
                assigned_items.add(candidate.item_key)
                progress += 1

            iteration_count = iteration + 1
            if progress == 0:
                break
        return iteration_count

    def _available_candidates(
        self,
        unassigned: Sequence[str],
        candidates_by_item: Dict[str, List[CandidateSidRow]],
        code_counts: Dict[str, int],
    ) -> List[CandidateSidRow]:
        available = []
        for item_key in unassigned:
            for candidate in candidates_by_item.get(item_key, []):
                if code_counts[candidate.candidate_codebook] < self.capacity:
                    available.append(candidate)
        return available

    def _select_candidates(
        self,
        available: Sequence[CandidateSidRow],
        code_counts: Dict[str, int],
    ) -> List[CandidateSidRow]:
        selected_by_codebook: List[CandidateSidRow] = []
        by_codebook: Dict[str, List[CandidateSidRow]] = defaultdict(list)
        for candidate in available:
            by_codebook[candidate.candidate_codebook].append(candidate)

        for codebook, rows in by_codebook.items():
            remaining = self.capacity - code_counts[codebook]
            if remaining <= 0:
                continue
            selected_by_codebook.extend(
                sorted(rows, key=self._candidate_sort_key)[:remaining]
            )

        best_by_item: Dict[str, CandidateSidRow] = {}
        for candidate in selected_by_codebook:
            current = best_by_item.get(candidate.item_key)
            if current is None or self._candidate_sort_key(
                candidate
            ) < self._candidate_sort_key(current):
                best_by_item[candidate.item_key] = candidate

        return sorted(
            best_by_item.values(),
            key=lambda r: (r.candidate_codebook, self._candidate_sort_key(r)),
        )

    def _handle_unassigned(
        self,
        raw_by_item: Dict[str, RawSidRow],
        assigned: List[AssignedSidRow],
        assigned_items: set,
        code_counts: Dict[str, int],
    ) -> List[str]:
        unassigned = sorted(set(raw_by_item) - assigned_items)
        if not unassigned:
            return unassigned

        if self.unassigned_policy == "error":
            preview = ",".join(unassigned[:10])
            raise RuntimeError(
                f"{len(unassigned)} items could not be assigned within capacity; "
                f"first unassigned item_ids: {preview}"
            )
        if self.unassigned_policy == "keep_original":
            for item_key in unassigned:
                raw = raw_by_item[item_key]
                code_counts[raw.origin_codebook] += 1
                assigned.append(
                    AssignedSidRow(
                        item_id=raw.item_id,
                        item_key=raw.item_key,
                        origin_codebook=raw.origin_codebook,
                        codebook=raw.origin_codebook,
                        index=code_counts[raw.origin_codebook],
                    )
                )
        return unassigned

    @staticmethod
    def _final_counts(rows: Sequence[AssignedSidRow]) -> Dict[str, int]:
        final_counts: Dict[str, int] = defaultdict(int)
        for row in rows:
            final_counts[row.codebook] += 1
        return final_counts


class CollisionRunner:
    """SID collision-prevention runner over the standard dataset reader/writer.

    The backend (CSV / Parquet / ODPS) is chosen by the reader/writer type, so
    the same path serves local files and MaxCompute tables.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Class name of the main-input reader, captured during the raw read; used
        # to derive the writer type when --writer_type is unset.
        self._input_reader_cls_name: Optional[str] = None

    def run(self) -> AssignmentStats:
        """Read inputs, assign, and write outputs via create_reader/create_writer."""
        raw_rows = self._load_raw_sid_rows()
        candidate_rows = self._load_candidate_rows()
        rows, stats = SidCollisionAssigner(
            capacity=self.args.max_items_per_codebook,
            max_iters=self.args.max_iters,
            seed=self.args.seed,
            score_order=self.args.score_order,
            unassigned_policy=self.args.unassigned_policy,
            strategy=self.args.strategy,
            code_delimiter=self.args.code_delimiter,
            random_last_layer_size=self.args.random_last_layer_size,
            random_num_candidates=self.args.random_num_candidates,
        ).assign(raw_rows, candidate_rows)
        self._write_assignments(rows)
        if self.args.diagnostics_output_path:
            self._write_diagnostics(stats)
        logger.info("SID collision prevention finished: %s", stats)
        return stats

    @staticmethod
    def _cell_to_code(value: Any, delimiter: str) -> str:
        if value is None:
            raise ValueError("SID code value cannot be null.")
        if isinstance(value, (list, tuple)):
            return delimiter.join(str(v) for v in value)
        return str(value)

    @classmethod
    def _split_compact_candidates(cls, value: Any, delimiter: str) -> List[str]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [cls._cell_to_code(v, ",") for v in value if v is not None]
        text = str(value).strip()
        if not text:
            return []
        return [part.strip() for part in text.split(delimiter) if part.strip()]

    @staticmethod
    def _array_to_pylist(batch: Dict[str, pa.Array], field: str) -> List[Any]:
        if field not in batch:
            raise ValueError(
                f"required field {field!r} not found; available fields: "
                f"{sorted(batch.keys())}"
            )
        return batch[field].to_pylist()

    def _read_batches(
        self,
        input_path: str,
        reader_type: Optional[str],
        selected_cols: Optional[List[str]] = None,
        capture_reader_cls: bool = False,
    ) -> Iterable[Dict[str, pa.Array]]:
        reader = create_reader(
            input_path=input_path,
            batch_size=self.args.batch_size,
            selected_cols=selected_cols,
            reader_type=reader_type,
            quota_name=self.args.odps_data_quota_name,
        )
        if capture_reader_cls:
            self._input_reader_cls_name = reader.__class__.__name__
        yield from reader.to_batches()

    def _load_raw_sid_rows(self) -> List[RawSidRow]:
        rows: List[RawSidRow] = []
        seen = set()
        code_fields = self._code_fields()
        code_field = None if code_fields else self.args.code_field
        if bool(code_field) == bool(code_fields):
            raise ValueError("Set exactly one of --code_field or --code_fields.")

        # Project only the required columns so wide / ODPS source tables don't
        # decode unused columns (create_reader forwards selected_cols to the reader).
        selected = [self.args.item_id_field, *(code_fields or [code_field])]
        for batch in self._read_batches(
            self.args.input_path,
            self.args.reader_type,
            selected_cols=selected,
            capture_reader_cls=True,
        ):
            item_ids = self._array_to_pylist(batch, self.args.item_id_field)
            if code_field:
                codes = self._array_to_pylist(batch, code_field)
                origin_codes = [
                    self._cell_to_code(v, self.args.code_delimiter) for v in codes
                ]
            else:
                assert code_fields is not None
                code_columns = [self._array_to_pylist(batch, f) for f in code_fields]
                origin_codes = [
                    self.args.code_delimiter.join(str(col[i]) for col in code_columns)
                    for i in range(len(item_ids))
                ]

            for item_id, origin_codebook in zip(item_ids, origin_codes):
                item_key = str(item_id)
                if item_key in seen:
                    raise ValueError(f"duplicate item_id in raw SID input: {item_key}")
                seen.add(item_key)
                rows.append(
                    RawSidRow(
                        item_id=item_id,
                        item_key=item_key,
                        origin_codebook=origin_codebook,
                    )
                )

        if not rows:
            raise ValueError("raw SID input is empty.")
        return rows

    def _load_candidate_rows(self) -> List[CandidateSidRow]:
        rows: List[CandidateSidRow] = []
        if not self.args.candidate_input_path:
            return rows

        candidate_codebook_field = (
            None
            if self.args.compact_candidate_field
            else self.args.candidate_codebook_field
        )
        if bool(candidate_codebook_field) == bool(self.args.compact_candidate_field):
            raise ValueError(
                "Set exactly one of --candidate_codebook_field or "
                "--compact_candidate_field."
            )

        reader_type = self.args.candidate_reader_type or self.args.reader_type
        item_id_field = self.args.candidate_item_id_field or self.args.item_id_field
        for batch in self._read_batches(self.args.candidate_input_path, reader_type):
            item_ids = self._array_to_pylist(batch, item_id_field)
            priorities = (
                self._array_to_pylist(batch, self.args.priority_field)
                if self.args.priority_field and self.args.priority_field in batch
                else None
            )
            scores = (
                self._array_to_pylist(batch, self.args.score_field)
                if self.args.score_field and self.args.score_field in batch
                else None
            )
            if candidate_codebook_field:
                candidates = self._array_to_pylist(batch, candidate_codebook_field)
                for i, (item_id, candidate) in enumerate(zip(item_ids, candidates)):
                    rows.append(
                        CandidateSidRow(
                            item_key=str(item_id),
                            candidate_codebook=self._cell_to_code(
                                candidate,
                                self.args.code_delimiter,
                            ),
                            priority=int(priorities[i])
                            if priorities is not None
                            else 1,
                            score=float(scores[i]) if scores is not None else 0.0,
                        )
                    )
            else:
                compact_values = self._array_to_pylist(
                    batch,
                    self.args.compact_candidate_field,
                )
                for item_id, compact_value in zip(item_ids, compact_values):
                    for priority, candidate in enumerate(
                        self._split_compact_candidates(
                            compact_value,
                            self.args.candidate_delimiter,
                        ),
                        start=1,
                    ):
                        rows.append(
                            CandidateSidRow(
                                item_key=str(item_id),
                                candidate_codebook=candidate,
                                priority=priority,
                                score=0.0,
                            )
                        )

        return rows

    def _code_fields(self) -> Optional[List[str]]:
        if not self.args.code_fields:
            return None
        return [
            field.strip() for field in self.args.code_fields.split(",") if field.strip()
        ]

    @staticmethod
    def _item_id_array(rows: Sequence[AssignedSidRow]) -> pa.Array:
        values = [row.item_id for row in rows]
        if all(isinstance(v, int) and not isinstance(v, bool) for v in values):
            return pa.array(values, type=pa.int64())
        return pa.array([str(v) for v in values], type=pa.string())

    def _writer_type(self) -> Optional[str]:
        # Derive the writer from the input reader (hitrate.py idiom) when unset,
        # so the output backend matches the input's. ODPS output still resolves
        # to OdpsWriter via create_writer's path detection regardless.
        if self.args.writer_type:
            return self.args.writer_type
        if self._input_reader_cls_name:
            return self._input_reader_cls_name.replace("Reader", "Writer")
        return None

    def _write_table(self, output_path: str, columns: Dict[str, pa.Array]) -> None:
        writer = create_writer(
            output_path,
            writer_type=self._writer_type(),
            quota_name=self.args.odps_data_quota_name,
            world_size=1,
        )
        writer.write(columns)
        writer.close()

    def _write_assignments(self, rows: Sequence[AssignedSidRow]) -> None:
        self._write_table(
            self.args.output_path,
            {
                "item_id": self._item_id_array(rows),
                "origin_codebook": pa.array(
                    [row.origin_codebook for row in rows], type=pa.string()
                ),
                "codebook": pa.array([row.codebook for row in rows], type=pa.string()),
                "index": pa.array([row.index for row in rows], type=pa.int64()),
            },
        )

    def _write_diagnostics(self, stats: AssignmentStats) -> None:
        self._write_table(
            self.args.diagnostics_output_path,
            {
                "total_items": pa.array([stats.total_items], type=pa.int64()),
                "raw_collision_buckets": pa.array(
                    [stats.raw_collision_buckets], type=pa.int64()
                ),
                "final_collision_buckets": pa.array(
                    [stats.final_collision_buckets], type=pa.int64()
                ),
                "reassigned_count": pa.array([stats.reassigned_count], type=pa.int64()),
                "unassigned_count": pa.array([stats.unassigned_count], type=pa.int64()),
                "iteration_count": pa.array([stats.iteration_count], type=pa.int64()),
                "max_final_bucket_size": pa.array(
                    [stats.max_final_bucket_size], type=pa.int64()
                ),
            },
        )


def assign_sid_collisions(
    raw_rows: Sequence[RawSidRow],
    candidate_rows: Sequence[CandidateSidRow],
    capacity: int,
    max_iters: int = 50,
    seed: int = 2026,
    score_order: str = "lower",
    unassigned_policy: str = "error",
    strategy: str = "candidate",
    code_delimiter: str = ",",
    random_last_layer_size: Optional[int] = None,
    random_num_candidates: int = 64,
) -> Tuple[List[AssignedSidRow], AssignmentStats]:
    """Assign overflow SID rows to non-full candidate codebooks."""
    return SidCollisionAssigner(
        capacity=capacity,
        max_iters=max_iters,
        seed=seed,
        score_order=score_order,
        unassigned_policy=unassigned_policy,
        strategy=strategy,
        code_delimiter=code_delimiter,
        random_last_layer_size=random_last_layer_size,
        random_num_candidates=random_num_candidates,
    ).assign(raw_rows, candidate_rows)


def run(args: argparse.Namespace) -> AssignmentStats:
    """Run collision prevention over the configured reader/writer backend."""
    return CollisionRunner(args).run()


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prevent SID codebook collisions with explicit candidates."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_input_path", default=None)
    parser.add_argument("--diagnostics_output_path", default=None)
    parser.add_argument(
        "--reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
    )
    parser.add_argument(
        "--candidate_reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter", "OdpsWriter"],
        default=None,
        help="Output writer; defaults to matching the input reader "
        "(CsvReader -> CsvWriter, etc.).",
    )
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--item_id_field", default="item_id")
    parser.add_argument("--candidate_item_id_field", default=None)
    parser.add_argument("--code_field", default="codes")
    parser.add_argument(
        "--code_fields",
        default=None,
        help="Comma-separated split code fields. Mutually exclusive with code_field.",
    )
    parser.add_argument("--code_delimiter", default=",")
    parser.add_argument("--candidate_codebook_field", default="candidate_codebook")
    parser.add_argument(
        "--compact_candidate_field",
        default=None,
        help="Compact string/list candidate field.",
    )
    parser.add_argument(
        "--candidate_delimiter",
        default="|",
        help="Delimiter for compact string candidate lists.",
    )
    parser.add_argument("--priority_field", default="priority")
    parser.add_argument("--score_field", default="score")
    parser.add_argument("--score_order", choices=["lower", "higher"], default="lower")
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--unassigned_policy",
        choices=["error", "drop", "keep_original"],
        default="error",
    )
    parser.add_argument(
        "--strategy",
        choices=["candidate", "random"],
        default="candidate",
        help="Reassignment strategy: 'candidate' uses explicit nearest-neighbor "
        "candidate rows; 'random' draws random within-band last-layer codes and "
        "needs no candidate input.",
    )
    parser.add_argument(
        "--random_last_layer_size",
        type=int,
        default=None,
        help="Code-space size of the last SID layer; required for --strategy random.",
    )
    parser.add_argument(
        "--random_num_candidates",
        type=int,
        default=64,
        help="Random last-layer codes drawn per overflow item for --strategy random.",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    return parser


def main() -> None:
    """Command line entrypoint."""
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
