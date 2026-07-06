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
from collections import OrderedDict, defaultdict
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


@dataclass(frozen=True)
class OdpsTableRef:
    """Parsed ODPS table URI."""

    project: str
    table: str
    partitions: Tuple[str, ...]
    schema: Optional[str] = None

    @classmethod
    def parse(cls, path: str) -> "OdpsTableRef":
        """Parse an ``odps://project/tables/table[/pt=value]`` path."""
        parts = path.split("/")
        if len(parts) < 5 or parts[0] != "odps:" or parts[3] != "tables":
            raise ValueError(
                f"invalid ODPS path {path!r}; expected "
                "odps://project/tables/table[/pt=value]"
            )
        table = parts[4]
        schema = None
        if "." in table:
            schema, table = table.split(".", 1)
        return cls(
            project=parts[2],
            table=table,
            partitions=tuple(p for p in parts[5:] if p),
            schema=schema,
        )

    @property
    def table_name(self) -> str:
        """Fully qualified ODPS table name."""
        table = f"{self.schema}.{self.table}" if self.schema else self.table
        return f"{self.project}.{table}"

    @property
    def partition_predicate(self) -> str:
        """SQL predicate suffix for this table's partition path."""
        if not self.partitions:
            return ""
        predicates = []
        for key, value in self._partition_pairs():
            predicates.append(f"{key}='{value}'")
        return " AND " + " AND ".join(predicates)

    @property
    def insert_target(self) -> str:
        """SQL INSERT target including partition spec."""
        if not self.partitions:
            return self.table_name
        specs = [f"{key}='{value}'" for key, value in self._partition_pairs()]
        return f"{self.table_name} PARTITION ({','.join(specs)})"

    @property
    def partition_schema(self) -> str:
        """CREATE TABLE partition schema suffix."""
        if not self.partitions:
            return ""
        fields = [f"{key} STRING" for key, _ in self._partition_pairs()]
        return f" PARTITIONED BY ({','.join(fields)})"

    def _partition_pairs(self) -> List[Tuple[str, str]]:
        pairs = []
        for part in self.partitions:
            if "=" not in part:
                raise ValueError(f"invalid ODPS partition segment: {part!r}")
            key, value = part.split("=", 1)
            pairs.append((key, value))
        return pairs


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
        score = row.score if self.score_order == "lower" else -row.score
        return (
            row.priority,
            score,
            self._stable_hash(self.seed, row.item_key, row.candidate_codebook),
            row.item_key,
            row.candidate_codebook,
        )

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


class LocalCollisionRunner:
    """CSV/Parquet SID collision-prevention runner."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> AssignmentStats:
        """Run local CSV/Parquet collision prevention."""
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
    ) -> Iterable[Dict[str, pa.Array]]:
        reader = create_reader(
            input_path=input_path,
            batch_size=self.args.batch_size,
            reader_type=reader_type,
            quota_name=self.args.odps_data_quota_name,
        )
        yield from reader.to_batches()

    def _load_raw_sid_rows(self) -> List[RawSidRow]:
        rows: List[RawSidRow] = []
        seen = set()
        code_fields = self._code_fields()
        code_field = None if code_fields else self.args.code_field
        if bool(code_field) == bool(code_fields):
            raise ValueError("Set exactly one of --code_field or --code_fields.")

        for batch in self._read_batches(self.args.input_path, self.args.reader_type):
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

    def _write_assignments(self, rows: Sequence[AssignedSidRow]) -> None:
        writer = create_writer(
            self.args.output_path,
            writer_type=self.args.writer_type,
            quota_name=self.args.odps_data_quota_name,
            world_size=1,
        )
        writer.write(
            OrderedDict(
                [
                    ("item_id", self._item_id_array(rows)),
                    (
                        "origin_codebook",
                        pa.array(
                            [row.origin_codebook for row in rows],
                            type=pa.string(),
                        ),
                    ),
                    (
                        "codebook",
                        pa.array([row.codebook for row in rows], type=pa.string()),
                    ),
                    ("index", pa.array([row.index for row in rows], type=pa.int64())),
                ]
            )
        )
        writer.close()

    def _write_diagnostics(self, stats: AssignmentStats) -> None:
        writer = create_writer(
            self.args.diagnostics_output_path,
            writer_type=self.args.writer_type,
            quota_name=self.args.odps_data_quota_name,
            world_size=1,
        )
        writer.write(
            OrderedDict(
                [
                    ("total_items", pa.array([stats.total_items], type=pa.int64())),
                    (
                        "raw_collision_buckets",
                        pa.array([stats.raw_collision_buckets], type=pa.int64()),
                    ),
                    (
                        "final_collision_buckets",
                        pa.array([stats.final_collision_buckets], type=pa.int64()),
                    ),
                    (
                        "reassigned_count",
                        pa.array([stats.reassigned_count], type=pa.int64()),
                    ),
                    (
                        "unassigned_count",
                        pa.array([stats.unassigned_count], type=pa.int64()),
                    ),
                    (
                        "iteration_count",
                        pa.array([stats.iteration_count], type=pa.int64()),
                    ),
                    (
                        "max_final_bucket_size",
                        pa.array([stats.max_final_bucket_size], type=pa.int64()),
                    ),
                ]
            )
        )
        writer.close()


class OdpsSqlGenerator:
    """Generate deterministic MaxCompute SQL for canonical candidate tables."""

    def __init__(self, args: argparse.Namespace) -> None:
        if not args.candidate_input_path:
            raise ValueError("--candidate_input_path is required for --backend odps.")
        if args.code_fields:
            raise ValueError(
                "--backend odps currently supports --code_field, not split code_fields."
            )
        if args.compact_candidate_field:
            raise ValueError(
                "--backend odps expects canonical candidate rows; compact candidates "
                "are supported in local CSV/Parquet mode."
            )

        self.args = args
        self.raw_ref = OdpsTableRef.parse(args.input_path)
        self.candidate_ref = OdpsTableRef.parse(args.candidate_input_path)
        self.output_ref = OdpsTableRef.parse(args.output_path)
        self.prefix = args.temp_prefix or "tmp_sid_collision"
        self.assigned = f"{self.prefix}_assigned"
        self.selected = f"{self.prefix}_selected"
        self.counts = f"{self.prefix}_counts"

    def generate(self) -> List[str]:
        """Generate the SQL statements for one ODPS collision-prevention run."""
        sqls = [
            "SET odps.sql.type.system.odps2=true",
            self._create_assigned_sql(),
            self._create_selected_sql(),
            self._create_counts_sql(),
            self._create_output_sql(),
            self._initial_assignment_sql(),
        ]
        for i in range(self.args.max_iters):
            sqls.extend(
                [
                    self._refresh_counts_sql(),
                    self._select_candidates_sql(),
                    self._insert_selected_sql(),
                    self._remaining_unassigned_sql(f"iteration {i + 1}"),
                ]
            )
        sqls.append(self._remaining_unassigned_sql("final"))
        if self.args.unassigned_policy == "keep_original":
            sqls.extend([self._refresh_counts_sql(), self._keep_original_sql()])
        sqls.append(self._final_insert_sql())
        return sqls

    @property
    def _raw_table(self) -> str:
        return self.raw_ref.table_name

    @property
    def _candidate_table(self) -> str:
        return self.candidate_ref.table_name

    @property
    def _raw_predicate(self) -> str:
        return self.raw_ref.partition_predicate

    @property
    def _candidate_predicate(self) -> str:
        return self.candidate_ref.partition_predicate

    @property
    def _score_expr(self) -> str:
        if self.args.score_field:
            return f"CAST({self.args.score_field} AS DOUBLE)"
        return "0.0"

    @property
    def _priority_expr(self) -> str:
        if self.args.priority_field:
            return f"CAST({self.args.priority_field} AS BIGINT)"
        return "1"

    @property
    def _score_order(self) -> str:
        return "score ASC" if self.args.score_order == "lower" else "score DESC"

    def _create_assigned_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.assigned} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, `index` BIGINT"
            f") LIFECYCLE {self.args.odps_lifecycle}"
        )

    def _create_selected_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.selected} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, "
            "priority BIGINT, score DOUBLE"
            f") LIFECYCLE {self.args.odps_lifecycle}"
        )

    def _create_counts_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.counts} ("
            "codebook STRING, cnt BIGINT"
            f") LIFECYCLE {self.args.odps_lifecycle}"
        )

    def _create_output_sql(self) -> str:
        return (
            f"CREATE TABLE IF NOT EXISTS {self.output_ref.table_name} ("
            "item_id STRING, origin_codebook STRING, codebook STRING, `index` BIGINT"
            f"){self.output_ref.partition_schema}"
        )

    def _initial_assignment_sql(self) -> str:
        return (
            f"INSERT OVERWRITE TABLE {self.assigned}\n"
            "SELECT item_id, origin_codebook, origin_codebook AS codebook,\n"
            "       rn AS `index`\n"
            "FROM (\n"
            f"  SELECT CAST({self.args.item_id_field} AS STRING) AS item_id,\n"
            f"         CAST({self.args.code_field} AS STRING) AS origin_codebook,\n"
            "         ROW_NUMBER() OVER (\n"
            f"           PARTITION BY CAST({self.args.code_field} AS STRING)\n"
            "           ORDER BY ABS(HASH(CONCAT("
            f"'{self.args.seed}', ':', CAST({self.args.item_id_field} AS STRING))))\n"
            "         ) AS rn\n"
            f"  FROM {self._raw_table}\n"
            f"  WHERE 1=1{self._raw_predicate}\n"
            ") t\n"
            f"WHERE rn <= {self.args.max_items_per_codebook}"
        )

    def _refresh_counts_sql(self) -> str:
        return (
            f"INSERT OVERWRITE TABLE {self.counts}\n"
            "SELECT codebook, COUNT(*) AS cnt\n"
            f"FROM {self.assigned}\n"
            "GROUP BY codebook"
        )

    def _select_candidates_sql(self) -> str:
        return (
            f"INSERT OVERWRITE TABLE {self.selected}\n"
            "SELECT item_id, origin_codebook, codebook, priority, score\n"
            "FROM (\n"
            "  SELECT c.item_id, c.origin_codebook, c.codebook, "
            "c.priority, c.score,\n"
            "         ROW_NUMBER() OVER (\n"
            "           PARTITION BY c.codebook\n"
            f"           ORDER BY c.priority ASC, c.{self._score_order}, "
            "ABS(HASH(CONCAT("
            f"'{self.args.seed}', ':', c.item_id, ':', c.codebook)))\n"
            "         ) AS rn,\n"
            "         COALESCE(cnt.cnt, 0) AS current_cnt\n"
            "  FROM (\n"
            "    SELECT CAST("
            f"{self.args.candidate_item_id_field or self.args.item_id_field}"
            " AS STRING) AS item_id,\n"
            "           CAST("
            f"{self.args.candidate_origin_codebook_field} AS STRING"
            ") AS origin_codebook,\n"
            "           CAST("
            f"{self.args.candidate_codebook_field}"
            " AS STRING) AS codebook,\n"
            f"           {self._priority_expr} AS priority,\n"
            f"           {self._score_expr} AS score\n"
            f"    FROM {self._candidate_table}\n"
            f"    WHERE 1=1{self._candidate_predicate}\n"
            "  ) c\n"
            "  INNER JOIN (\n"
            f"    SELECT CAST({self.args.item_id_field} AS STRING) AS item_id\n"
            f"    FROM {self._raw_table}\n"
            f"    WHERE 1=1{self._raw_predicate}\n"
            "  ) r ON c.item_id = r.item_id\n"
            f"  LEFT OUTER JOIN {self.assigned} a ON c.item_id = a.item_id\n"
            f"  LEFT OUTER JOIN {self.counts} cnt ON c.codebook = cnt.codebook\n"
            "  WHERE a.item_id IS NULL\n"
            f"    AND COALESCE(cnt.cnt, 0) < {self.args.max_items_per_codebook}\n"
            ") ranked\n"
            f"WHERE rn <= {self.args.max_items_per_codebook} - current_cnt"
        )

    def _insert_selected_sql(self) -> str:
        # ``codebook_rn`` must rank ONLY the surviving (``item_rn = 1``) rows so
        # the emitted ``index`` stays dense and contiguous with the rows already
        # in ``assigned`` (counted by ``current_cnt``). Ranking before the
        # ``item_rn = 1`` filter leaves a gap whenever a codebook's top-ranked
        # candidate wins a different codebook (its row here is dropped): that gap
        # both wastes capacity and is re-issued next iteration as a duplicate
        # ``(codebook, index)`` pair, since ``current_cnt`` (a COUNT of inserted
        # rows) then undercounts the real slot high-water mark. Hence the nested
        # shape: pick each item's best codebook first, THEN densely number the
        # survivors per codebook.
        return (
            f"INSERT INTO TABLE {self.assigned}\n"
            "SELECT item_id, origin_codebook, codebook,\n"
            "       current_cnt + codebook_rn AS `index`\n"
            "FROM (\n"
            "  SELECT item_id, origin_codebook, codebook, current_cnt,\n"
            "         ROW_NUMBER() OVER (\n"
            "           PARTITION BY codebook\n"
            f"           ORDER BY priority ASC, {self._score_order}, "
            "ABS(HASH(CONCAT("
            f"'{self.args.seed}', ':', item_id, ':', codebook)))\n"
            "         ) AS codebook_rn\n"
            "  FROM (\n"
            "    SELECT s.item_id, s.origin_codebook, s.codebook, "
            "s.priority, s.score,\n"
            "           ROW_NUMBER() OVER (\n"
            "             PARTITION BY s.item_id\n"
            f"             ORDER BY s.priority ASC, s.{self._score_order}, "
            "ABS(HASH(CONCAT("
            f"'{self.args.seed}', ':', s.item_id, ':', s.codebook)))\n"
            "           ) AS item_rn,\n"
            "           COALESCE(cnt.cnt, 0) AS current_cnt\n"
            f"    FROM {self.selected} s\n"
            f"    LEFT OUTER JOIN {self.counts} cnt "
            "ON s.codebook = cnt.codebook\n"
            "  ) x\n"
            "  WHERE item_rn = 1\n"
            ") y\n"
            f"WHERE current_cnt + codebook_rn <= {self.args.max_items_per_codebook}"
        )

    def _remaining_unassigned_sql(self, label: str) -> str:
        return (
            f"-- {label}: remaining_unassigned\n"
            "SELECT COUNT(*) AS remaining_unassigned\n"
            f"FROM {self._raw_table} r\n"
            f"LEFT OUTER JOIN {self.assigned} a\n"
            f"ON CAST(r.{self.args.item_id_field} AS STRING) = a.item_id\n"
            f"WHERE a.item_id IS NULL{self._raw_predicate}"
        )

    def _keep_original_sql(self) -> str:
        return (
            f"INSERT INTO TABLE {self.assigned}\n"
            "SELECT u.item_id, u.origin_codebook, "
            "u.origin_codebook AS codebook,\n"
            "       COALESCE(cnt.cnt, 0) + u.rn AS `index`\n"
            "FROM (\n"
            f"  SELECT CAST(r.{self.args.item_id_field} AS STRING) AS item_id,\n"
            f"         CAST(r.{self.args.code_field} AS STRING) AS "
            "origin_codebook,\n"
            "         ROW_NUMBER() OVER (\n"
            f"           PARTITION BY CAST(r.{self.args.code_field} AS STRING)\n"
            "           ORDER BY ABS(HASH(CONCAT("
            f"'{self.args.seed}', ':', CAST(r.{self.args.item_id_field} AS STRING))))\n"
            "         ) AS rn\n"
            f"  FROM {self._raw_table} r\n"
            f"  LEFT OUTER JOIN {self.assigned} a\n"
            f"  ON CAST(r.{self.args.item_id_field} AS STRING) = a.item_id\n"
            f"  WHERE a.item_id IS NULL{self._raw_predicate}\n"
            ") u\n"
            f"LEFT OUTER JOIN {self.counts} cnt\n"
            "ON u.origin_codebook = cnt.codebook"
        )

    def _final_insert_sql(self) -> str:
        return (
            f"INSERT OVERWRITE TABLE {self.output_ref.insert_target}\n"
            "SELECT item_id, origin_codebook, codebook, `index`\n"
            f"FROM {self.assigned}"
        )


class OdpsCollisionRunner:
    """ODPS SID collision-prevention runner."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> None:
        """Run MaxCompute SQL collision prevention."""
        sqls = OdpsSqlGenerator(self.args).generate()
        if self.args.dry_run_sql:
            print(";\n\n".join(sqls) + ";")
            return

        from odps import ODPS

        from tzrec.datasets.odps_dataset import _create_odps_account

        output_ref = OdpsTableRef.parse(self.args.output_path)
        account, endpoint = _create_odps_account()
        odps = ODPS(account=account, project=output_ref.project, endpoint=endpoint)
        for sql in sqls:
            logger.info("Executing ODPS SQL:\n%s", sql)
            instance = odps.execute_sql(sql)
            instance.wait_for_success()
            if not self._is_remaining_unassigned_sql(sql):
                continue

            remaining_unassigned = self._read_scalar(instance)
            logger.info(
                "ODPS remaining unassigned item count: %s",
                remaining_unassigned,
            )
            if not self._is_final_remaining_unassigned_sql(sql):
                continue
            self._handle_final_unassigned(remaining_unassigned)

    @staticmethod
    def _is_remaining_unassigned_sql(sql: str) -> bool:
        return " AS remaining_unassigned" in sql

    @staticmethod
    def _is_final_remaining_unassigned_sql(sql: str) -> bool:
        return sql.lstrip().startswith("-- final:")

    @staticmethod
    def _read_scalar(instance: Any) -> int:
        with instance.open_reader() as reader:
            for record in reader:
                return int(record[0])
        return 0

    def _handle_final_unassigned(self, remaining_unassigned: int) -> None:
        if remaining_unassigned == 0:
            return
        if self.args.unassigned_policy == "error":
            raise RuntimeError(
                f"{remaining_unassigned} items could not be assigned within "
                "capacity in ODPS collision-prevention run."
            )
        if self.args.unassigned_policy == "drop":
            logger.warning(
                "Dropping %s unassigned items because --unassigned_policy=drop.",
                remaining_unassigned,
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


def run_local(args: argparse.Namespace) -> AssignmentStats:
    """Run local CSV/Parquet collision prevention."""
    return LocalCollisionRunner(args).run()


def generate_odps_sql(args: argparse.Namespace) -> List[str]:
    """Generate deterministic MaxCompute SQL for canonical candidate tables."""
    return OdpsSqlGenerator(args).generate()


def run_odps(args: argparse.Namespace) -> None:
    """Run MaxCompute SQL collision prevention."""
    OdpsCollisionRunner(args).run()


def build_parser() -> argparse.ArgumentParser:
    """Build the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Prevent SID codebook collisions with explicit candidates."
    )
    parser.add_argument("--backend", choices=["local", "odps"], default="local")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--candidate_input_path", default=None)
    parser.add_argument("--diagnostics_output_path", default=None)
    parser.add_argument(
        "--reader_type", choices=["CsvReader", "ParquetReader"], default=None
    )
    parser.add_argument(
        "--candidate_reader_type",
        choices=["CsvReader", "ParquetReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter"],
        default="ParquetWriter",
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
    parser.add_argument("--candidate_origin_codebook_field", default="origin_codebook")
    parser.add_argument(
        "--compact_candidate_field",
        default=None,
        help="Compact string/list candidate field for local mode.",
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
        "needs no candidate input (local backend only).",
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
    parser.add_argument("--temp_prefix", default=None)
    parser.add_argument("--odps_lifecycle", type=int, default=7)
    parser.add_argument("--dry_run_sql", action="store_true", default=False)
    return parser


def main() -> None:
    """Command line entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    if args.backend == "local":
        run_local(args)
    else:
        if args.strategy == "random":
            raise NotImplementedError(
                "strategy='random' is only supported for --backend local."
            )
        run_odps(args)


if __name__ == "__main__":
    main()
