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

import os
import shutil
import tempfile
import unittest
from collections import Counter, defaultdict

import pyarrow as pa
from pyarrow import csv, parquet

from tzrec.tools.sid.collision_prevention import (
    CandidateSidRow,
    RawSidRow,
    assign_sid_collisions,
    build_parser,
    run,
)


class SidCollisionPreventionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_assign_sid_collisions_respects_capacity(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
            RawSidRow("item_3", "item_3", "B"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "C", 1, 0.1),
            CandidateSidRow("item_1", "C", 1, 0.1),
            CandidateSidRow("item_2", "C", 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=2,
            seed=7,
        )

        self.assertEqual(len(assigned), 4)
        self.assertEqual(len({row.item_key for row in assigned}), 4)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)
        self.assertEqual(stats.raw_collision_buckets, 1)
        self.assertEqual(stats.final_collision_buckets, 0)
        self.assertEqual(stats.reassigned_count, 1)
        self.assertEqual(stats.unassigned_count, 0)

    def test_random_strategy_reassigns_within_band_without_candidates(self) -> None:
        # Three items collide on SID (lv1,lv2,lv3) = (1,2,3); capacity 1.
        raw_rows = [
            RawSidRow("item_0", "item_0", "1,2,3"),
            RawSidRow("item_1", "item_1", "1,2,3"),
            RawSidRow("item_2", "item_2", "1,2,3"),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            [],  # random needs no candidate input
            capacity=1,
            strategy="random",
            random_last_layer_size=16,
            seed=7,
        )

        self.assertEqual(len(assigned), 3)
        codebooks = {row.item_key: row.codebook for row in assigned}
        # every item keeps its (lv1,lv2) band, only the last layer varies ...
        for codebook in codebooks.values():
            self.assertTrue(codebook.startswith("1,2,"))
        # ... capacity 1 keeps exactly one item at the origin SID and reassigns the
        # other two to distinct random last-layer codes -> near-injective.
        self.assertEqual(len(set(codebooks.values())), 3)
        self.assertEqual(sum(1 for cb in codebooks.values() if cb == "1,2,3"), 1)
        self.assertEqual(stats.final_collision_buckets, 0)
        self.assertEqual(stats.reassigned_count, 2)
        self.assertEqual(stats.unassigned_count, 0)

    def test_random_strategy_is_deterministic_given_seed(self) -> None:
        raw_rows = [RawSidRow(f"item_{i}", f"item_{i}", "0,0,0") for i in range(4)]
        kwargs = dict(capacity=1, strategy="random", random_last_layer_size=64, seed=11)
        first, _ = assign_sid_collisions(raw_rows, [], **kwargs)
        second, _ = assign_sid_collisions(raw_rows, [], **kwargs)
        self.assertEqual(
            {r.item_key: r.codebook for r in first},
            {r.item_key: r.codebook for r in second},
        )

    def test_random_strategy_requires_last_layer_size(self) -> None:
        raw_rows = [RawSidRow("item_0", "item_0", "1,2,3")]
        with self.assertRaisesRegex(ValueError, "random_last_layer_size"):
            assign_sid_collisions(raw_rows, [], capacity=1, strategy="random")

    def test_missing_candidates_errors_on_overflow(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
        ]
        with self.assertRaisesRegex(ValueError, "no explicit candidate input"):
            assign_sid_collisions(raw_rows, [], capacity=1)

    def test_duplicate_candidates_do_not_consume_capacity_twice(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "C", 1, 0.1),
            CandidateSidRow("item_0", "C", 2, 0.2),
            CandidateSidRow("item_1", "C", 1, 0.1),
            CandidateSidRow("item_2", "C", 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=2,
            seed=7,
        )

        self.assertEqual(len(assigned), 3)
        self.assertEqual(stats.unassigned_count, 0)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)

    def test_local_csv_outputs_codebooks_as_strings(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.csv")
        cand_path = os.path.join(self.test_dir, "cand.csv")
        out_dir = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "codes": ["A", "A", "A"],
                }
            ),
            raw_path,
        )
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "candidate_codebook": ["C", "C", "C"],
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertEqual(result.schema.field("origin_codebook").type, pa.string())
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_local_parquet_accepts_list_codes(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.parquet")
        cand_path = os.path.join(self.test_dir, "cand.parquet")
        out_dir = os.path.join(self.test_dir, "out_parquet")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "codes": pa.array([[1, 2], [1, 2], [1, 2]]),
                }
            ),
            raw_path,
        )
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "candidate_codebook": ["1,3", "1,3", "1,3"],
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--writer_type",
                "ParquetWriter",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = parquet.read_table(os.path.join(out_dir, "part-0.parquet"))
        self.assertIn("1,2", set(result["origin_codebook"].to_pylist()))
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_local_csv_accepts_split_codes_and_compact_candidates(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw_split.csv")
        cand_path = os.path.join(self.test_dir, "cand_compact.csv")
        out_dir = os.path.join(self.test_dir, "out_split")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "code_0": ["A", "A", "A"],
                    "code_1": ["B", "B", "B"],
                }
            ),
            raw_path,
        )
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "sorted_index": ["A|C", "A|C", "A|C"],
                }
            ),
            cand_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--candidate_input_path",
                cand_path,
                "--output_path",
                out_dir,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--code_field",
                "",
                "--code_fields",
                "code_0,code_1",
                "--compact_candidate_field",
                "sorted_index",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertIn("A,B", set(result["origin_codebook"].to_pylist()))
        self.assertIn("A", set(result["codebook"].to_pylist()))
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_local_reassignment_indices_are_unique_and_dense(self) -> None:
        # After reassignment every (codebook, index) pair must be unique and each
        # bucket's indices must form a contiguous 1..N run. (Which two items
        # overflow A is decided by a seeded hash, so every item carries the same
        # B fallback.)
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
            RawSidRow("item_3", "item_3", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "B", 1, 0.1),
            CandidateSidRow("item_1", "B", 1, 0.2),
            CandidateSidRow("item_2", "B", 1, 0.3),
            CandidateSidRow("item_3", "B", 1, 0.4),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=2,
            seed=7,
        )

        self.assertEqual(len(assigned), 4)
        self.assertEqual(stats.unassigned_count, 0)
        pairs = [(row.codebook, row.index) for row in assigned]
        self.assertEqual(len(pairs), len(set(pairs)))
        by_codebook = defaultdict(list)
        for codebook, index in pairs:
            by_codebook[codebook].append(index)
        for indices in by_codebook.values():
            self.assertEqual(sorted(indices), list(range(1, len(indices) + 1)))

    def test_local_drop_policy_omits_unplaceable_items(self) -> None:
        # A (capacity 1) keeps one item; the other two overflow and both want B,
        # which fits only one -> one item stays unplaceable and reaches
        # _handle_unassigned's drop branch.
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "B", 1, 0.1),
            CandidateSidRow("item_1", "B", 1, 0.2),
            CandidateSidRow("item_2", "B", 1, 0.3),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=1,
            seed=7,
            unassigned_policy="drop",
        )

        self.assertEqual(stats.unassigned_count, 1)
        # The unplaceable item is dropped: A keeps 1, B keeps 1, nothing else.
        self.assertEqual(len(assigned), 2)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 1)

    def test_local_keep_original_readds_over_capacity(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "B", 1, 0.1),
            CandidateSidRow("item_1", "B", 1, 0.2),
            CandidateSidRow("item_2", "B", 1, 0.3),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows,
            candidate_rows,
            capacity=1,
            seed=7,
            unassigned_policy="keep_original",
        )

        self.assertEqual(stats.unassigned_count, 0)
        self.assertEqual(len(assigned), 3)
        # The unplaceable item is re-added at its origin: keep_original is the
        # only policy allowed to exceed capacity (A now holds 2 > capacity 1).
        self.assertEqual(Counter(row.codebook for row in assigned)["A"], 2)
        self.assertEqual(stats.max_final_bucket_size, 2)

    def test_local_error_policy_raises_on_unplaceable(self) -> None:
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
            RawSidRow("item_2", "item_2", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "B", 1, 0.1),
            CandidateSidRow("item_1", "B", 1, 0.2),
            CandidateSidRow("item_2", "B", 1, 0.3),
        ]

        # Distinct from the "no explicit candidate input" ValueError: candidates
        # are provided but cannot place every overflow item.
        with self.assertRaisesRegex(RuntimeError, "could not be assigned"):
            assign_sid_collisions(
                raw_rows,
                candidate_rows,
                capacity=1,
                seed=7,
                unassigned_policy="error",
            )

    def test_local_score_order_higher_prefers_high_score(self) -> None:
        # One item overflows A (capacity 1); it can go to B (score 0.1) or
        # C (score 0.9). score_order flips which score wins. Both items carry both
        # candidates so the choice is independent of which item the seeded hash
        # picks to overflow.
        raw_rows = [
            RawSidRow("item_0", "item_0", "A"),
            RawSidRow("item_1", "item_1", "A"),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", "B", 1, 0.1),
            CandidateSidRow("item_0", "C", 1, 0.9),
            CandidateSidRow("item_1", "B", 1, 0.1),
            CandidateSidRow("item_1", "C", 1, 0.9),
        ]

        lower, _ = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=1, seed=7, score_order="lower"
        )
        higher, _ = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=1, seed=7, score_order="higher"
        )

        reassigned_lower = next(
            row for row in lower if row.origin_codebook != row.codebook
        )
        reassigned_higher = next(
            row for row in higher if row.origin_codebook != row.codebook
        )
        self.assertEqual(reassigned_lower.codebook, "B")
        self.assertEqual(reassigned_higher.codebook, "C")


if __name__ == "__main__":
    unittest.main()
