# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import itertools
import random
import unittest
from typing import List, Tuple

import numpy as np
import pyarrow as pa
from parameterized import parameterized

from tzrec.datasets.utils import (
    _normalize_type_str,
    build_sampler_input,
    calc_remaining_intervals,
    calc_slice_intervals,
    combine_negs_to_candidate_sequence,
    get_input_fields_proto,
    plan_rank_worker_intervals,
)
from tzrec.protos import data_pb2
from tzrec.protos.data_pb2 import FieldType
from tzrec.utils.test_util import parameterized_name_func


class DatasetUtilsTest(unittest.TestCase):
    @staticmethod
    def _rank_batches(
        rows: List[int],
        world_size: int,
        num_workers: int,
        batch_size: int,
        equalize: bool,
        min_batch_size: int,
    ) -> Tuple[List[List[int]], int, List[int]]:
        """Simulate every worker's buffered stream.

        Returns per-rank batch sizes, rows read, and per-worker row totals.
        """
        per_rank = []
        worker_totals = []
        num_read = 0
        seen = [set() for _ in rows]
        for rank in range(world_size):
            batches = []
            for worker in range(num_workers):
                intervals = plan_rank_worker_intervals(
                    rows,
                    rank,
                    world_size,
                    worker,
                    num_workers,
                    batch_size,
                    equalize,
                    min_batch_size,
                )
                total = 0
                for t, source_intervals in enumerate(intervals):
                    # one contiguous chunk per worker and source
                    assert len(source_intervals) <= 1, source_intervals
                    for start, end in source_intervals:
                        assert 0 <= start < end <= rows[t]
                        assert seen[t].isdisjoint(range(start, end))
                        seen[t].update(range(start, end))
                        total += end - start
                num_read += total
                worker_totals.append(total)
                assert not 0 < total % batch_size < min_batch_size, total
                # a worker's stream is whole batches plus at most one final tail
                batches.extend([batch_size] * (total // batch_size))
                if total % batch_size >= max(min_batch_size, 1):
                    batches.append(total % batch_size)
            per_rank.append(batches)
        return per_rank, num_read, worker_totals

    def test_plan_rank_worker_intervals_invariants(self):
        rng = random.Random(0)
        for batch_size, world_size, num_workers, equalize in itertools.product(
            (1, 2, 3, 4, 8), (1, 2, 3, 4), (1, 2, 3, 4), (False, True)
        ):
            for min_batch_size in sorted({0, 1, min(2, batch_size), batch_size}):
                cases = [
                    [rng.randrange(3 * batch_size * world_size + 5) for _ in range(n)]
                    for n in (1, 2, 3, 5)
                    for _ in range(15)
                ]
                cases += [[r] for r in range(4 * batch_size * world_size + 3)]
                for rows in cases:
                    per_rank, num_read, _ = self._rank_batches(
                        rows,
                        world_size,
                        num_workers,
                        batch_size,
                        equalize,
                        min_batch_size,
                    )
                    msg = (
                        f"rows={rows} world={world_size} workers={num_workers} "
                        f"bs={batch_size} min_bs={min_batch_size}"
                    )
                    for batches in per_rank:
                        # at most one partial batch per rank and pass
                        self.assertLessEqual(
                            sum(1 for b in batches if b != batch_size), 1, msg
                        )
                    if equalize:
                        self.assertEqual(len({len(b) for b in per_rank}), 1, msg)
                    if not equalize and min_batch_size == 0:
                        self.assertEqual(num_read, sum(rows), msg)
                    else:
                        # the extra rows of the pass, plus a short tail per rank
                        max_drop = world_size - 1
                        max_drop += max(min_batch_size - 1, 0) * world_size
                        self.assertLessEqual(sum(rows) - num_read, max_drop, msg)

    def test_plan_rank_worker_intervals_uniform_sources_balance(self):
        # dealing to the least loaded worker keeps the skew independent of the
        # number of sources
        for num_workers, num_sources in ((2, 2), (2, 40), (4, 4), (4, 40), (8, 40)):
            _, _, worker_totals = self._rank_batches(
                [13] * num_sources, 1, num_workers, 4, True, 0
            )
            self.assertLessEqual(max(worker_totals) - min(worker_totals), 2 * 4)

    @parameterized.expand(
        [
            # the failing job: 8200 rows, 8 workers -> one 8-row tail, no drop
            [[8200], 1, 8, 1024, 0, 9, [8], 8200],
            [[8200], 1, 8, 1024, 2, 9, [8], 8200],
            [[8201], 1, 8, 1024, 2, 9, [9], 8201],
            # extra row would buy a whole step on one rank -> drop it
            [[33], 2, 4, 4, 0, 4, [], 32],
            # residue-1 tails are kept without min_batch_size
            [[34], 2, 4, 4, 0, 5, [1], 34],
            [[35], 2, 4, 4, 0, 5, [2], 35],
            # ... and dropped together with the extra row when min_batch_size=2
            [[35], 2, 4, 4, 2, 4, [], 32],
            [[36], 2, 4, 4, 2, 5, [2], 36],
            # too few rows for one 2-row batch per rank -> empty pass
            [[3], 2, 4, 4, 2, 0, [], 0],
            # tails of two sources carry across the boundary: 2049 = 2 * 1024 + 1
            [[1000, 1049], 1, 1, 1024, 0, 3, [1], 2049],
            [[1000, 1049], 1, 1, 1024, 2, 2, [], 2048],
            # sources are one stream: 300 rows are exactly three batches
            [[150, 150], 1, 4, 100, 0, 3, [], 300],
            [[6, 7], 1, 3, 4, 0, 4, [1], 13],
            [[13, 13], 1, 2, 4, 0, 7, [2], 26],
            # ranks split the stream too: 80 rows over 8 ranks lose nothing
            [[33, 47], 8, 1, 10, 0, 1, [], 80],
            [[33, 47], 8, 2, 10, 2, 1, [], 80],
            # drop_remainder: no partial batch at all
            [[8200], 1, 8, 1024, 1024, 8, [], 8192],
        ],
        name_func=parameterized_name_func,
    )
    def test_plan_rank_worker_intervals(
        self,
        rows,
        world_size,
        num_workers,
        batch_size,
        min_batch_size,
        steps,
        rank0_tails,
        num_read,
    ):
        per_rank, actual_read, _ = self._rank_batches(
            rows, world_size, num_workers, batch_size, True, min_batch_size
        )
        self.assertEqual([len(b) for b in per_rank], [steps] * world_size)
        self.assertEqual([b for b in per_rank[0] if b != batch_size], rank0_tails)
        self.assertEqual(actual_read, num_read)

    def test_plan_rank_worker_intervals_predict_keeps_rows(self):
        per_rank, num_read, _ = self._rank_batches([35], 2, 4, 4, False, 0)
        self.assertEqual([len(b) for b in per_rank], [5, 5])
        self.assertEqual(num_read, 35)

    def test_calc_slice_intervals_two_ranks_two_workers(self):
        # 35 rows: rank 0 owns [0, 18) with the extra row, rank 1 owns [18, 35);
        # inside each rank the 4 full batches are dealt two per worker and the
        # tail opens on worker 0; chunks are laid out in worker order
        expected = {0: [(0, 10)], 1: [(10, 18)], 2: [(18, 27)], 3: [(27, 35)]}
        for worker_id, intervals in expected.items():
            result = calc_slice_intervals(
                [("/data/test.parquet", 35)],
                worker_id=worker_id,
                num_workers=4,
                batch_size=4,
                equalize_rank_steps=True,
                world_size=2,
            )
            self.assertEqual(result, [intervals])

    def test_calc_slice_intervals_two_ranks_resume(self):
        # remaining [(100, 500), (600, 1000)] = 800 rows, 400 per rank; rank 1's
        # share [400, 800) spans the gap between the two remaining intervals
        checkpoint_state = {
            "/data/test.parquet:0": 99,
            "/data/test.parquet:500": 599,
        }
        result = calc_slice_intervals(
            [("/data/test.parquet", 1000)],
            worker_id=2,
            num_workers=4,
            batch_size=128,
            equalize_rank_steps=True,
            checkpoint_state=checkpoint_state,
            world_size=2,
        )
        # rank 1, worker 0 gets two full batches -> logical [400, 656)
        self.assertEqual(result, [[(600, 856)]])

    def test_calc_remaining_intervals_no_checkpoint(self):
        """Test remaining intervals when no checkpoint exists."""
        result = calc_remaining_intervals(
            checkpoint_state=None,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        self.assertEqual(result, [(0, 1000)])

    def test_calc_remaining_intervals_empty_checkpoint(self):
        """Test remaining intervals with empty checkpoint state."""
        result = calc_remaining_intervals(
            checkpoint_state={},
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        self.assertEqual(result, [(0, 1000)])

    def test_calc_remaining_intervals_single_worker(self):
        """Test remaining intervals with single worker checkpoint."""
        # Worker consumed rows 0-499 (checkpoint at row 499)
        checkpoint_state = {"/data/test.parquet:0": 499}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Remaining: [500, 1000)
        self.assertEqual(result, [(500, 1000)])

    def test_calc_remaining_intervals_multiple_workers(self):
        """Test remaining intervals with multiple workers' checkpoints."""
        # 2 workers: worker0 [0, 500), worker1 [500, 1000)
        # Worker0 consumed up to row 299
        # Worker1 consumed up to row 799
        checkpoint_state = {
            "/data/test.parquet:0": 299,
            "/data/test.parquet:500": 799,
        }
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Remaining: [300, 500), [800, 1000)
        self.assertEqual(result, [(300, 500), (800, 1000)])

    def test_calc_remaining_intervals_fully_consumed(self):
        """Test remaining intervals when all data is consumed."""
        # Worker consumed all rows up to 999 (last row)
        checkpoint_state = {"/data/test.parquet:0": 999}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # No remaining intervals
        self.assertEqual(result, [])

    def test_calc_remaining_intervals_unstarted_leading_range(self):
        """A range with no key yet was never read, even before the first key."""
        checkpoint_state = {"/data/test.parquet:4": 7}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=8,
        )
        self.assertEqual(result, [(0, 4)])

    def test_calc_remaining_intervals_unrelated_path(self):
        """Test remaining intervals when checkpoint is for different path."""
        checkpoint_state = {"/data/other.parquet:0": 499}
        result = calc_remaining_intervals(
            checkpoint_state=checkpoint_state,
            input_path="/data/test.parquet",
            total_rows=1000,
        )
        # Unrelated checkpoint, return full range
        self.assertEqual(result, [(0, 1000)])

    def test_calc_slice_intervals_single_worker(self):
        """Test calc_slice_intervals with single worker."""
        # Simulate intervals [(100, 500), (600, 1000)] via checkpoint_state
        # checkpoint at row 99 means rows 0-99 consumed, remaining starts at 100
        # checkpoint at row 599 means rows 500-599 consumed, remaining starts at 600
        checkpoint_state = {
            "/data/test.parquet:0": 99,
            "/data/test.parquet:500": 599,
        }
        result = calc_slice_intervals(
            [("/data/test.parquet", 1000)],
            worker_id=0,
            num_workers=1,
            batch_size=1,
            checkpoint_state=checkpoint_state,
        )[0]
        self.assertEqual(result, [(100, 500), (600, 1000)])

    def _assert_tiles(self, slices, remaining):
        """Slices are pairwise disjoint and their union is the remaining rows."""
        rows = [r for s in slices for start, end in s for r in range(start, end)]
        self.assertEqual(len(rows), len(set(rows)))
        self.assertEqual(
            sorted(rows), [r for start, end in remaining for r in range(start, end)]
        )

    def test_calc_slice_intervals_two_workers(self):
        """Two even shares tile the remaining intervals without overlap."""
        checkpoint_state = {
            "/data/test.parquet:0": 99,
            "/data/test.parquet:500": 599,
        }
        slices = [
            calc_slice_intervals(
                [("/data/test.parquet", 1000)],
                worker_id=worker_id,
                num_workers=2,
                batch_size=1,
                checkpoint_state=checkpoint_state,
            )[0]
            for worker_id in range(2)
        ]
        self._assert_tiles(slices, [(100, 500), (600, 1000)])

    def test_calc_slice_intervals_two_sources_resume(self):
        """Even shares over two sources, the first partly consumed."""
        sources = [("a", 1000), ("b", 500)]
        slices = [
            calc_slice_intervals(
                sources,
                worker_id=worker_id,
                num_workers=2,
                batch_size=128,
                checkpoint_state={"a:0": 399},
            )
            for worker_id in range(2)
        ]
        self._assert_tiles([s[0] for s in slices], [(400, 1000)])
        self._assert_tiles([s[1] for s in slices], [(0, 500)])

    def test_calc_slice_intervals_worker_ahead_in_next_source(self):
        """Rows of a source chunk nobody has started yet survive a resume.

        Sources [12, 8], 2 workers, batch 4: worker 1 has one batch in source 0
        and reaches source 1 first, so the checkpoint keys source1:4 but not
        source1:0.
        """
        sources = [("source0", 12), ("source1", 8)]
        checkpoint_state = {"source0:0": 7, "source0:8": 11, "source1:4": 7}
        slices = [
            calc_slice_intervals(
                sources,
                worker_id=worker_id,
                num_workers=2,
                batch_size=4,
                equalize_rank_steps=True,
                checkpoint_state=checkpoint_state,
                world_size=1,
            )
            for worker_id in range(2)
        ]
        self._assert_tiles([s[0] for s in slices], [])
        self._assert_tiles([s[1] for s in slices], [(0, 4)])

    def test_calc_slice_intervals_empty_intervals(self):
        """Test calc_slice_intervals with empty intervals (fully consumed)."""
        # All data consumed: checkpoint at row 999 (last row)
        checkpoint_state = {"/data/test.parquet:0": 999}
        result = calc_slice_intervals(
            [("/data/test.parquet", 1000)],
            worker_id=0,
            num_workers=2,
            batch_size=1,
            checkpoint_state=checkpoint_state,
        )[0]
        self.assertEqual(result, [])

    def test_calc_slice_intervals_topology_change(self):
        """Resuming with 3 workers tiles what 2 workers left behind."""
        checkpoint_state = {
            "/data/test.parquet:0": 299,
            "/data/test.parquet:500": 799,
        }
        slices = [
            calc_slice_intervals(
                [("/data/test.parquet", 1000)],
                worker_id=worker_id,
                num_workers=3,
                batch_size=1,
                checkpoint_state=checkpoint_state,
            )[0]
            for worker_id in range(3)
        ]
        self._assert_tiles(slices, [(300, 500), (800, 1000)])

    @parameterized.expand(
        [
            # (name, input_data, item_id_field, user_id_field,
            #  seq_delim, expected_output)
            (
                # NegativeSampler-style: no user_id_field; item_id is
                # delimited string; gets flattened.
                "string_item_id_no_user_id",
                {"item_id": pa.array(["1;2", "3"]), "label": pa.array([1, 0])},
                "item_id",
                None,
                ";",
                {"item_id": ["1", "2", "3"], "label": [1, 0]},
            ),
            (
                # NegativeSamplerV2 / HardNeg style: item_id arrives as
                # list<int64>; user_id is expanded by per-row pos count.
                "list_item_id_expands_user_id",
                {
                    "item_id": pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                    "user_id": pa.array(["u0", "u1"]),
                },
                "item_id",
                "user_id",
                ";",
                {"item_id": [1, 2, 3], "user_id": ["u0", "u0", "u1"]},
            ),
            (
                # DSSM-style: item_id is scalar int64. No flatten, no
                # user_id expand.
                "scalar_item_id_passthrough",
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "user_id": pa.array(["u0", "u1", "u2"]),
                },
                "item_id",
                "user_id",
                ";",
                {"item_id": [1, 2, 3], "user_id": ["u0", "u1", "u2"]},
            ),
            (
                # item_id_field is a top-level scalar -> seq_delim="" -> pass through.
                "empty_seq_delim_passthrough",
                {"item_id": pa.array(["1", "2"])},
                "item_id",
                None,
                "",
                {"item_id": ["1", "2"]},
            ),
            (
                # Sampler config without item_id_field at all -> still
                # shallow-copied, no transformation.
                "no_item_id_field",
                {"a": pa.array([1, 2])},
                None,
                None,
                "",
                {"a": [1, 2]},
            ),
        ]
    )
    def test_build_sampler_input(
        self,
        _name,
        input_data,
        item_id_field,
        user_id_field,
        seq_delim,
        expected_output,
    ):
        # Snapshot input_data so we can verify the function didn't mutate it.
        input_snapshot = {k: v.to_pylist() for k, v in input_data.items()}

        out = build_sampler_input(
            input_data,
            item_id_field=item_id_field,
            user_id_field=user_id_field,
            seq_delim=seq_delim,
        )

        # Contract 1: output equals expected (per-column pylist compare).
        self.assertEqual({k: v.to_pylist() for k, v in out.items()}, expected_output)
        # Contract 2: input_data is not mutated.
        self.assertEqual(
            {k: v.to_pylist() for k, v in input_data.items()}, input_snapshot
        )
        # Contract 3: returned dict is a different object (shallow copy).
        self.assertIsNot(out, input_data)

    # Every case verifies output rows, pos_lengths, and output type.
    @parameterized.expand(
        [
            # name, pos_data, negs, expected_rows, expected_pos_lengths, expected_type
            (
                "single_pos_string",
                pa.array(["1", "2", "3"]),
                pa.array(["10", "20", "30"]),
                ["1", "2", "3;10;20;30"],
                [1, 1, 1],
                pa.string(),
            ),
            (
                "multivalue_pos_string",
                pa.array(["1;2", "3;4;5"]),
                pa.array(["10", "20"]),
                ["1;2", "3;4;5;10;20"],
                [2, 3],
                pa.string(),
            ),
            (
                # String pos + list<T> negs: still uses string path; combine
                # flattens the list-wrapped negs before joining.
                "string_pos_list_negs",
                pa.array(["1", "2"]),
                pa.array([[10], [20], [30]], type=pa.list_(pa.int64())),
                ["1", "2;10;20;30"],
                [1, 1],
                pa.string(),
            ),
            (
                "empty_negs_string",
                pa.array(["1", "2"]),
                pa.array([], type=pa.string()),
                ["1", "2"],
                [1, 1],
                pa.string(),
            ),
            (
                # list<T> pos + flat T negs -> list<T> out (no string round-trip).
                "list_pos_flat_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([10, 20], type=pa.int64()),
                [[1, 2], [3, 10, 20]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                # The sampler emits list<T> of 1-element lists when the
                # attr's field schema is list-typed (see _to_arrow_array in
                # sampler.py:168); combine flattens that shape.
                "list_pos_list_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([[10], [20], [30]], type=pa.list_(pa.int64())),
                [[1, 2], [3, 10, 20, 30]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                "list_pos_empty_negs",
                pa.array([[1, 2], [3]], type=pa.list_(pa.int64())),
                pa.array([], type=pa.int64()),
                [[1, 2], [3]],
                [2, 1],
                pa.list_(pa.int64()),
            ),
            (
                # large_list<T> pos -> large_list<T> out.
                "large_list_pos",
                pa.array([["a", "b"], ["c"]], type=pa.large_list(pa.string())),
                pa.array(["d", "e"], type=pa.string()),
                [["a", "b"], ["c", "d", "e"]],
                [2, 1],
                pa.large_list(pa.string()),
            ),
            (
                # Empty pos batch + non-empty negs -> empty output.
                # negs are dropped because there's no last row to land
                # them in (consistent with the list-path empty-batch
                # behavior).
                "empty_pos_batch_string",
                pa.array([], type=pa.string()),
                pa.array(["10"]),
                [],
                [],
                pa.string(),
            ),
            (
                # Single-row batch: row 0 is also row B-1, gets the
                # negs.
                "single_row_string",
                pa.array(["1;2"]),
                pa.array(["10"]),
                ["1;2;10"],
                [2],
                pa.string(),
            ),
        ]
    )
    def test_combine_negs_to_candidate_sequence(
        self,
        _name,
        pos_data,
        negs,
        expected_rows,
        expected_pos_lengths,
        expected_type,
    ):
        result, pos_lengths = combine_negs_to_candidate_sequence(
            pos_data=pos_data, negs=negs, seq_delim=";"
        )
        self.assertEqual(result.to_pylist(), expected_rows)
        np.testing.assert_array_equal(
            pos_lengths, np.array(expected_pos_lengths, dtype=np.int32)
        )
        self.assertTrue(result.type.equals(expected_type))

    def test_normalize_type_str_basic_types(self):
        """Test normalizing basic types."""
        self.assertEqual(_normalize_type_str("int32"), "INT32")
        self.assertEqual(_normalize_type_str("INT64"), "INT64")
        self.assertEqual(_normalize_type_str("string"), "STRING")
        self.assertEqual(_normalize_type_str("float"), "FLOAT")
        self.assertEqual(_normalize_type_str("double"), "DOUBLE")

    def test_normalize_type_str_aliases(self):
        """Test ODPS aliases: BIGINT -> INT64, INT -> INT32."""
        self.assertEqual(_normalize_type_str("BIGINT"), "INT64")
        self.assertEqual(_normalize_type_str("bigint"), "INT64")
        self.assertEqual(_normalize_type_str("INT"), "INT32")
        self.assertEqual(_normalize_type_str("int"), "INT32")

    def test_normalize_type_str_array_types(self):
        """Test array types with aliases."""
        self.assertEqual(_normalize_type_str("ARRAY<BIGINT>"), "ARRAY<INT64>")
        self.assertEqual(_normalize_type_str("ARRAY<INT>"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("ARRAY<INT64>"), "ARRAY<INT64>")
        self.assertEqual(_normalize_type_str("ARRAY<INT32>"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("ARRAY<STRING>"), "ARRAY<STRING>")
        self.assertEqual(_normalize_type_str("array<float>"), "ARRAY<FLOAT>")

    def test_normalize_type_str_nested_array_types(self):
        """Test nested array types."""
        self.assertEqual(
            _normalize_type_str("ARRAY<ARRAY<BIGINT>>"), "ARRAY<ARRAY<INT64>>"
        )
        self.assertEqual(
            _normalize_type_str("ARRAY<ARRAY<INT>>"), "ARRAY<ARRAY<INT32>>"
        )

    def test_normalize_type_str_map_types(self):
        """Test map types with aliases."""
        self.assertEqual(_normalize_type_str("MAP<STRING,BIGINT>"), "MAP<STRING,INT64>")
        self.assertEqual(_normalize_type_str("MAP<STRING,INT>"), "MAP<STRING,INT32>")
        self.assertEqual(_normalize_type_str("MAP<BIGINT,STRING>"), "MAP<INT64,STRING>")
        self.assertEqual(_normalize_type_str("MAP<INT,STRING>"), "MAP<INT32,STRING>")
        self.assertEqual(_normalize_type_str("MAP<BIGINT,BIGINT>"), "MAP<INT64,INT64>")

    def test_normalize_type_str_spaces(self):
        """Test handling of spaces around commas and angle brackets."""
        self.assertEqual(_normalize_type_str("MAP<STRING, INT>"), "MAP<STRING,INT32>")
        self.assertEqual(
            _normalize_type_str("MAP< STRING , BIGINT >"), "MAP<STRING,INT64>"
        )
        self.assertEqual(_normalize_type_str("ARRAY< INT >"), "ARRAY<INT32>")
        self.assertEqual(_normalize_type_str("  BIGINT  "), "INT64")

    def test_get_input_fields_proto_basic_types(self):
        """Test parsing basic types from input_fields_str."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "user_id:BIGINT;item_id:INT64;label:FLOAT"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 3)
        self.assertEqual(fields[0].input_name, "user_id")
        self.assertEqual(fields[0].input_type, FieldType.INT64)
        self.assertEqual(fields[1].input_name, "item_id")
        self.assertEqual(fields[1].input_type, FieldType.INT64)
        self.assertEqual(fields[2].input_name, "label")
        self.assertEqual(fields[2].input_type, FieldType.FLOAT)

    def test_get_input_fields_proto_array_types(self):
        """Test parsing array types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "ids:ARRAY<BIGINT>;values:ARRAY<FLOAT>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0].input_name, "ids")
        self.assertEqual(fields[0].input_type, FieldType.ARRAY_INT64)
        self.assertEqual(fields[1].input_name, "values")
        self.assertEqual(fields[1].input_type, FieldType.ARRAY_FLOAT)

    def test_get_input_fields_proto_map_types(self):
        """Test parsing map types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "feat_map:MAP<STRING, BIGINT>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "feat_map")
        self.assertEqual(fields[0].input_type, FieldType.MAP_STRING_INT64)

    def test_get_input_fields_proto_nested_array(self):
        """Test parsing nested array types."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "nested:ARRAY<ARRAY<INT>>"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "nested")
        self.assertEqual(fields[0].input_type, FieldType.ARRAY_ARRAY_INT32)

    def test_get_input_fields_proto_empty_string(self):
        """Test empty input_fields_str returns empty list."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = ""

        fields = get_input_fields_proto(data_config)
        self.assertEqual(len(fields), 0)

    def test_get_input_fields_proto_trailing_semicolon(self):
        """Test handling of trailing semicolon."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "user_id:BIGINT;item_id:INT64;"

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 2)
        self.assertEqual(fields[0].input_name, "user_id")
        self.assertEqual(fields[1].input_name, "item_id")

    def test_get_input_fields_proto_fallback_to_input_fields(self):
        """Test fallback to input_fields when input_fields_str is not set."""
        data_config = data_pb2.DataConfig()
        field1 = data_config.input_fields.add()
        field1.input_name = "test_field"
        field1.input_type = FieldType.STRING

        fields = get_input_fields_proto(data_config)

        self.assertEqual(len(fields), 1)
        self.assertEqual(fields[0].input_name, "test_field")
        self.assertEqual(fields[0].input_type, FieldType.STRING)

    def test_get_input_fields_proto_invalid_format(self):
        """Test error handling for invalid format."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "invalid_field"

        with self.assertRaises(ValueError) as context:
            get_input_fields_proto(data_config)
        self.assertIn("Invalid input_fields_str format", str(context.exception))

    def test_get_input_fields_proto_unknown_type(self):
        """Test error handling for unknown type."""
        data_config = data_pb2.DataConfig()
        data_config.input_fields_str = "field1:UNKNOWN_TYPE"

        with self.assertRaises(ValueError) as context:
            get_input_fields_proto(data_config)
        self.assertIn("Unknown field type", str(context.exception))


if __name__ == "__main__":
    unittest.main()
