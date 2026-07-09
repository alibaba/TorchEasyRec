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

    # ---- assigner (codebook is a tuple[int]) ----

    def test_assign_sid_collisions_respects_capacity(self) -> None:
        raw_rows = [
            RawSidRow("item_0", (1,)),
            RawSidRow("item_1", (1,)),
            RawSidRow("item_2", (1,)),
            RawSidRow("item_3", (2,)),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", (3,), 1, 0.1),
            CandidateSidRow("item_1", (3,), 1, 0.1),
            CandidateSidRow("item_2", (3,), 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=2, seed=7
        )

        self.assertEqual(len(assigned), 4)
        self.assertEqual(len({row.item_key for row in assigned}), 4)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)
        self.assertEqual(stats.raw_collision_buckets, 1)
        self.assertEqual(stats.final_collision_buckets, 0)
        self.assertEqual(stats.reassigned_count, 1)
        self.assertEqual(stats.unassigned_count, 0)

    def test_random_strategy_reassigns_within_band_without_candidates(self) -> None:
        # Three items collide on SID (1,2,3); capacity 1.
        raw_rows = [
            RawSidRow("item_0", (1, 2, 3)),
            RawSidRow("item_1", (1, 2, 3)),
            RawSidRow("item_2", (1, 2, 3)),
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
        # every item keeps its (1,2) band; only the last layer varies ...
        for codebook in codebooks.values():
            self.assertEqual(codebook[:2], (1, 2))
        # ... capacity 1 keeps one item at the origin SID and reassigns the other
        # two to distinct random last-layer codes -> near-injective.
        self.assertEqual(len(set(codebooks.values())), 3)
        self.assertEqual(sum(1 for cb in codebooks.values() if cb == (1, 2, 3)), 1)
        self.assertEqual(stats.final_collision_buckets, 0)
        self.assertEqual(stats.reassigned_count, 2)
        self.assertEqual(stats.unassigned_count, 0)

    def test_random_strategy_is_deterministic_given_seed(self) -> None:
        raw_rows = [RawSidRow(f"item_{i}", (0, 0, 0)) for i in range(4)]
        kwargs = dict(capacity=1, strategy="random", random_last_layer_size=64, seed=11)
        first, _ = assign_sid_collisions(raw_rows, [], **kwargs)
        second, _ = assign_sid_collisions(raw_rows, [], **kwargs)
        self.assertEqual(
            {r.item_key: r.codebook for r in first},
            {r.item_key: r.codebook for r in second},
        )

    def test_random_strategy_requires_last_layer_size(self) -> None:
        raw_rows = [RawSidRow("item_0", (1, 2, 3))]
        with self.assertRaisesRegex(ValueError, "random_last_layer_size"):
            assign_sid_collisions(raw_rows, [], capacity=1, strategy="random")

    def test_missing_candidates_errors_on_overflow(self) -> None:
        raw_rows = [RawSidRow("item_0", (1,)), RawSidRow("item_1", (1,))]
        with self.assertRaisesRegex(ValueError, "no explicit candidate input"):
            assign_sid_collisions(raw_rows, [], capacity=1)

    def test_duplicate_candidates_do_not_consume_capacity_twice(self) -> None:
        raw_rows = [
            RawSidRow("item_0", (1,)),
            RawSidRow("item_1", (1,)),
            RawSidRow("item_2", (1,)),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", (3,), 1, 0.1),
            CandidateSidRow("item_0", (3,), 2, 0.2),
            CandidateSidRow("item_1", (3,), 1, 0.1),
            CandidateSidRow("item_2", (3,), 1, 0.1),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=2, seed=7
        )

        self.assertEqual(len(assigned), 3)
        self.assertEqual(stats.unassigned_count, 0)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 2)

    def test_local_reassignment_indices_are_unique_and_dense(self) -> None:
        # After reassignment every (codebook, index) pair must be unique and each
        # bucket's indices must form a contiguous 1..N run.
        raw_rows = [RawSidRow(f"item_{i}", (1,)) for i in range(4)]
        candidate_rows = [
            CandidateSidRow("item_0", (2,), 1, 0.1),
            CandidateSidRow("item_1", (2,), 1, 0.2),
            CandidateSidRow("item_2", (2,), 1, 0.3),
            CandidateSidRow("item_3", (2,), 1, 0.4),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=2, seed=7
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
        # (1,) (capacity 1) keeps one item; the other two overflow and both want
        # (2,), which fits only one -> one item is unplaceable (drop branch).
        raw_rows = [
            RawSidRow("item_0", (1,)),
            RawSidRow("item_1", (1,)),
            RawSidRow("item_2", (1,)),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", (2,), 1, 0.1),
            CandidateSidRow("item_1", (2,), 1, 0.2),
            CandidateSidRow("item_2", (2,), 1, 0.3),
        ]

        assigned, stats = assign_sid_collisions(
            raw_rows, candidate_rows, capacity=1, seed=7, unassigned_policy="drop"
        )

        self.assertEqual(stats.unassigned_count, 1)
        self.assertEqual(len(assigned), 2)
        self.assertLessEqual(max(Counter(row.codebook for row in assigned).values()), 1)

    def test_local_keep_original_readds_over_capacity(self) -> None:
        raw_rows = [
            RawSidRow("item_0", (1,)),
            RawSidRow("item_1", (1,)),
            RawSidRow("item_2", (1,)),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", (2,), 1, 0.1),
            CandidateSidRow("item_1", (2,), 1, 0.2),
            CandidateSidRow("item_2", (2,), 1, 0.3),
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
        # keep_original is the only policy allowed to exceed capacity.
        self.assertEqual(Counter(row.codebook for row in assigned)[(1,)], 2)
        self.assertEqual(stats.max_final_bucket_size, 2)

    def test_local_error_policy_raises_on_unplaceable(self) -> None:
        raw_rows = [
            RawSidRow("item_0", (1,)),
            RawSidRow("item_1", (1,)),
            RawSidRow("item_2", (1,)),
        ]
        candidate_rows = [
            CandidateSidRow("item_0", (2,), 1, 0.1),
            CandidateSidRow("item_1", (2,), 1, 0.2),
            CandidateSidRow("item_2", (2,), 1, 0.3),
        ]

        with self.assertRaisesRegex(RuntimeError, "could not be assigned"):
            assign_sid_collisions(
                raw_rows,
                candidate_rows,
                capacity=1,
                seed=7,
                unassigned_policy="error",
            )

    def test_local_score_order_higher_prefers_high_score(self) -> None:
        # One item overflows (1,) (capacity 1); it can go to (2,) (score 0.1) or
        # (3,) (score 0.9). score_order flips which score wins.
        raw_rows = [RawSidRow("item_0", (1,)), RawSidRow("item_1", (1,))]
        candidate_rows = [
            CandidateSidRow("item_0", (2,), 1, 0.1),
            CandidateSidRow("item_0", (3,), 1, 0.9),
            CandidateSidRow("item_1", (2,), 1, 0.1),
            CandidateSidRow("item_1", (3,), 1, 0.9),
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
        self.assertEqual(reassigned_lower.codebook, (2,))
        self.assertEqual(reassigned_higher.codebook, (3,))

    # ---- file backends: Parquet/ODPS list<int64>, CSV string fallback ----

    def test_local_parquet_writes_list_codebooks(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.parquet")
        out_dir = os.path.join(self.test_dir, "out_parquet")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2, 3], type=pa.int64()),
                    "codes": pa.array(
                        [[1, 2], [1, 2], [1, 2]], type=pa.list_(pa.int64())
                    ),
                    "candidate_codebook": pa.array(
                        [[1, 3], [1, 3], [1, 3]], type=pa.list_(pa.int64())
                    ),
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            raw_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
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
        # Parquet/ODPS keep codebooks as list<int64>.
        self.assertEqual(result.schema.field("codebook").type, pa.list_(pa.int64()))
        origins = {tuple(x) for x in result["origin_codebook"].to_pylist()}
        self.assertIn((1, 2), origins)
        codebooks = [tuple(x) for x in result["codebook"].to_pylist()]
        self.assertLessEqual(max(Counter(codebooks).values()), 2)

    def test_local_csv_writes_codebooks_as_strings(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw.csv")
        out_dir = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    # multi-layer so the codebook string keeps a comma and CSV
                    # read-back doesn't re-infer it as int.
                    "item_id": ["1", "2", "3"],
                    "codes": ["1,2", "1,2", "1,2"],
                    "candidate_codebook": ["3,4", "3,4", "3,4"],
                    "priority": [1, 1, 1],
                    "score": [0.1, 0.1, 0.1],
                }
            ),
            raw_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
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
        # CSV fallback encodes codebooks as delimited strings.
        self.assertEqual(result.schema.field("origin_codebook").type, pa.string())
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_writer_type_defaults_to_matching_the_reader(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw_derive.csv")
        out_dir = os.path.join(self.test_dir, "out_derive")
        csv.write_csv(
            pa.table({"item_id": ["1", "2", "3"], "codes": ["1", "1", "1"]}),
            raw_path,
        )
        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--output_path",
                out_dir,
                "--max_items_per_codebook",
                "3",
            ]
        )
        run(args)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertEqual(result.num_rows, 3)

    def test_local_csv_accepts_compact_candidates(self) -> None:
        raw_path = os.path.join(self.test_dir, "raw_compact.csv")
        out_dir = os.path.join(self.test_dir, "out_compact")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["1", "2", "3"],
                    "codes": ["1,2", "1,2", "1,2"],
                    "candidates": ["1|3", "1|3", "1|3"],
                }
            ),
            raw_path,
        )

        args = build_parser().parse_args(
            [
                "--input_path",
                raw_path,
                "--output_path",
                out_dir,
                "--reader_type",
                "CsvReader",
                "--writer_type",
                "CsvWriter",
                "--compact_candidate_field",
                "candidates",
                "--max_items_per_codebook",
                "2",
            ]
        )
        stats = run(args)

        self.assertEqual(stats.reassigned_count, 1)
        result = csv.read_csv(os.path.join(out_dir, "part-0.csv"))
        self.assertIn("1,2", set(result["origin_codebook"].to_pylist()))
        self.assertTrue({"1", "3"} & set(result["codebook"].to_pylist()))
        self.assertLessEqual(max(Counter(result["codebook"].to_pylist()).values()), 2)

    def test_csv_and_parquet_yield_identical_decisions(self) -> None:
        # The same logical data as CSV (delimited strings) and Parquet
        # (list<int64>) must normalize to the same tuple codebooks, so the
        # assignment decisions are identical across backends.
        item_ids = list(range(20))
        origin = [[i % 3, i % 2] for i in item_ids]  # heavy collision on few buckets
        cand = [[9, i % 5] for i in item_ids]

        def decisions(mode: str) -> dict:
            if mode == "parquet":
                path = os.path.join(self.test_dir, "eq.parquet")
                parquet.write_table(
                    pa.table(
                        {
                            "item_id": pa.array(item_ids, type=pa.int64()),
                            "codes": pa.array(origin, type=pa.list_(pa.int64())),
                            "candidate_codebook": pa.array(
                                cand, type=pa.list_(pa.int64())
                            ),
                        }
                    ),
                    path,
                )
                reader, writer = "ParquetReader", "ParquetWriter"
            else:
                path = os.path.join(self.test_dir, "eq.csv")
                csv.write_csv(
                    pa.table(
                        {
                            "item_id": [str(i) for i in item_ids],
                            "codes": [",".join(map(str, c)) for c in origin],
                            "candidate_codebook": [",".join(map(str, c)) for c in cand],
                        }
                    ),
                    path,
                )
                reader, writer = "CsvReader", "CsvWriter"
            out = os.path.join(self.test_dir, f"eq_out_{mode}")
            run(
                build_parser().parse_args(
                    [
                        "--input_path",
                        path,
                        "--output_path",
                        out,
                        "--reader_type",
                        reader,
                        "--writer_type",
                        writer,
                        "--max_items_per_codebook",
                        "2",
                        "--unassigned_policy",
                        "keep_original",
                        "--seed",
                        "5",
                    ]
                )
            )
            if mode == "parquet":
                d = parquet.read_table(os.path.join(out, "part-0.parquet")).to_pydict()
                return {
                    int(d["item_id"][i]): tuple(d["codebook"][i])
                    for i in range(len(d["item_id"]))
                }
            d = csv.read_csv(os.path.join(out, "part-0.csv")).to_pydict()
            return {
                int(d["item_id"][i]): tuple(
                    int(x) for x in str(d["codebook"][i]).split(",")
                )
                for i in range(len(d["item_id"]))
            }

        self.assertEqual(decisions("parquet"), decisions("csv"))


if __name__ == "__main__":
    unittest.main()
