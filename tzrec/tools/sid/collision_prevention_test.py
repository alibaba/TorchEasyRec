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

import json
import os
import random
import shutil
import tempfile
import unittest
from argparse import Namespace
from collections import Counter
from unittest import mock

import numpy as np
import pyarrow as pa
from pyarrow import csv, parquet

from tzrec.datasets.odps_dataset import _type_pa_to_table
from tzrec.tools.sid import collision_prevention
from tzrec.tools.sid.collision_prevention import CollisionRunner

# Defaults mirror the __main__ argparse; tests override only what they exercise.
_DEFAULTS = dict(
    input_path=None,
    output_path=None,
    original_sid_groups_output_path=None,
    resolved_sid_groups_output_path=None,
    reader_type=None,
    writer_type=None,
    batch_size=100000,
    item_id_field="item_id",
    code_field="codes",
    candidate_codes_field="candidate_codes",
    max_items_per_codebook=2,
    strategy="candidate",
    codebook="8,8",
    random_num_candidates=64,
    rate_only=False,
    odps_data_quota_name="pay-as-you-go",
)


def _parquet(path, item_ids, codes, candidate_codes=None):
    cols = {
        "item_id": pa.array(item_ids, type=pa.int64()),
        "codes": pa.array(codes, type=pa.list_(pa.int64())),
    }
    if candidate_codes is not None:
        # flatten each row's [[c0, ..], ..] (topk x n_layers) to a flat list<int>
        flat = [[code for cand in row for code in cand] for row in candidate_codes]
        cols["candidate_codes"] = pa.array(flat, type=pa.list_(pa.int64()))
    parquet.write_table(pa.table(cols), path)


class SidCollisionPreventionTest(unittest.TestCase):
    def setUp(self) -> None:
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _run(self, inp, out, **kw):
        args = dict(_DEFAULTS)
        original_groups, resolved_groups = self._group_paths(out)
        args.update(
            input_path=inp,
            output_path=out,
            original_sid_groups_output_path=original_groups,
            resolved_sid_groups_output_path=resolved_groups,
        )
        args.update(kw)
        return CollisionRunner(Namespace(**args)).run()

    @staticmethod
    def _group_paths(out):
        return f"{out}_original_groups", f"{out}_resolved_groups"

    def _read_parquet(self, out_dir):
        return parquet.read_table(os.path.join(out_dir, "part-0.parquet")).to_pydict()

    # ---- candidate strategy ----

    def test_candidate_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # 5 items in bucket (0,0); candidates vary only the last layer within band 0.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        stats = self._run(inp, out, max_items_per_codebook=2)
        self.assertEqual(stats.total_items, 5)
        self.assertEqual(stats.relocated_count, 3)
        self.assertEqual(stats.unresolved_count, 0)
        self.assertEqual(stats.max_final_bucket_size, 2)
        d = self._read_parquet(out)
        # every final SID keeps prefix 0 (stays in band) ...
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        # ... and no bucket exceeds the cap.
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)

    def test_three_layer_candidate_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # 3-layer SIDs: band = (code_0, code_1). Three items collide in bucket
        # (0, 0, 0); candidates keep the (0, 0) band and vary the last layer, so
        # the flat topk*3 candidate column must be split at stride 3.
        _parquet(
            inp,
            [0, 1, 2],
            [[0, 0, 0]] * 3,
            [[[0, 0, 1], [0, 0, 2], [0, 0, 3]]] * 3,
        )
        stats = self._run(inp, out, codebook="2,3,4", max_items_per_codebook=2)
        self.assertEqual(stats.total_items, 3)
        self.assertEqual(stats.relocated_count, 1)
        self.assertEqual(stats.unresolved_count, 0)
        self.assertEqual(stats.max_final_bucket_size, 2)
        d = self._read_parquet(out)
        # The relocated item took the first free candidate last code (1); every
        # SID keeps its (0, 0) band prefix.
        self.assertEqual(
            sorted(tuple(cb) for cb in d["codebook"]),
            [(0, 0, 0), (0, 0, 0), (0, 0, 1)],
        )

    def test_candidate_last_matrix_splits_at_n_layers(self) -> None:
        # A flat topk*n_layers candidate column splits into topk groups of
        # n_layers, keeping each group's LAST code (stride n_layers), not a
        # contiguous tail. codebook 2,3,4 -> n_layers=3, so the flat run
        # [0,0,1, 0,0,2, 0,0,3] decodes to last codes [1, 2, 3].
        runner = CollisionRunner(
            Namespace(
                **{
                    **_DEFAULTS,
                    "codebook": "2,3,4",
                    "rate_only": True,
                    "input_path": "x",
                    "output_path": "y",
                }
            )
        )
        flat = pa.array([[0, 0, 1, 0, 0, 2, 0, 0, 3]], type=pa.list_(pa.int64()))
        self.assertEqual(runner._candidate_last_matrix(flat).tolist(), [[1, 2, 3]])

    def test_output_is_list_int64(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)
        self._run(inp, out, max_items_per_codebook=2)
        schema = parquet.read_table(os.path.join(out, "part-0.parquet")).schema
        self.assertEqual(schema.field("codebook").type, pa.list_(pa.int64()))
        self.assertEqual(schema.field("origin_codebook").type, pa.list_(pa.int64()))

    def test_keep_original_when_only_origin_candidate(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # candidate == origin last -> skipped -> nothing to place; kept over cap.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 0]]] * 5)
        stats = self._run(inp, out, max_items_per_codebook=2)
        self.assertEqual(stats.relocated_count, 0)
        self.assertEqual(stats.unresolved_count, 3)
        d = self._read_parquet(out)
        self.assertEqual(len(d["item_id"]), 5)  # every item preserved
        self.assertTrue(all(tuple(cb) == (0, 0) for cb in d["codebook"]))
        _, resolved_path = self._group_paths(out)
        resolved = self._read_parquet(resolved_path)
        self.assertEqual(len(resolved["itemids"][0]), 5)
        for item_id, index in zip(d["item_id"], d["index"]):
            self.assertEqual(resolved["itemids"][0][index - 1], item_id)

    def test_raises_when_overflow_but_no_candidates(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)  # no candidate_codes column
        with self.assertRaises(ValueError):
            self._run(inp, out, max_items_per_codebook=2)

    def test_no_overflow_does_not_require_candidates(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0], [0, 1], [0, 2]])
        stats = self._run(inp, out, max_items_per_codebook=1)
        self.assertEqual(stats.relocated_count, 0)
        self.assertEqual(len(self._read_parquet(out)["item_id"]), 3)

    def test_candidates_align_across_batches(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        candidates = [[[0, code]] for code in [1, 2, 5, 3, 6]]
        _parquet(inp, list(range(5)), [[0, 0]] * 5, candidates)
        self._run(inp, out, batch_size=2, max_items_per_codebook=2)
        result = self._read_parquet(out)
        by_id = dict(zip(result["item_id"], result["codebook"]))
        self.assertEqual(by_id[0], [0, 1])
        self.assertEqual(by_id[1], [0, 2])
        self.assertEqual(by_id[3], [0, 3])

    def test_group_outputs_follow_sid_and_slot_order(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        item_ids = list(range(10))
        # 4 items in bucket (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3
        candidates = [[[0, 0]] for _ in item_ids]
        candidates[1] = [[0, 0], [0, 1], [0, 2]]
        candidates[3] = [[0, 1], [0, 2], [0, 3]]
        candidates[8] = [[1, 0], [1, 1], [1, 2]]
        _parquet(inp, item_ids, codes, candidates)

        self._run(inp, out, max_items_per_codebook=2)

        original_path, resolved_path = self._group_paths(out)
        original = parquet.read_table(os.path.join(original_path, "part-0.parquet"))
        resolved = parquet.read_table(os.path.join(resolved_path, "part-0.parquet"))
        self.assertEqual(original.column_names, ["codebook", "itemids"])
        self.assertEqual(resolved.column_names, ["codebook", "itemids"])
        self.assertEqual(
            original.to_pydict(),
            {
                "codebook": [[0, 0], [0, 1], [0, 2], [1, 0]],
                "itemids": [[2, 0, 1, 3], [4], [6, 5], [9, 7, 8]],
            },
        )
        self.assertEqual(
            resolved.to_pydict(),
            {
                "codebook": [[0, 0], [0, 1], [0, 2], [0, 3], [1, 0], [1, 1]],
                "itemids": [[2, 0], [4, 1], [6, 5], [3], [9, 7], [8]],
            },
        )

        resolved_items = {
            tuple(codebook): item_group
            for codebook, item_group in zip(
                resolved["codebook"].to_pylist(),
                resolved["itemids"].to_pylist(),
            )
        }
        item_map = self._read_parquet(out)
        for item_id, codebook, index in zip(
            item_map["item_id"], item_map["codebook"], item_map["index"]
        ):
            self.assertEqual(resolved_items[tuple(codebook)][index - 1], item_id)

    def test_no_overflow_reuses_groups_with_gapped_prefixes(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(
            inp,
            list(range(6)),
            [[7, 10], [7, 2], [12, 1], [7, 2], [12, 1], [7, 10]],
        )

        with mock.patch.object(
            collision_prevention,
            "resolve_sid_collisions",
            wraps=collision_prevention.resolve_sid_collisions,
        ) as resolve:
            self._run(inp, out, codebook="16,16", max_items_per_codebook=2)
        self.assertFalse(resolve.call_args.kwargs["collect_grouping"])

        original_path, resolved_path = self._group_paths(out)
        original = parquet.read_table(
            os.path.join(original_path, "part-0.parquet")
        ).to_pydict()
        resolved = parquet.read_table(
            os.path.join(resolved_path, "part-0.parquet")
        ).to_pydict()
        self.assertEqual(original, resolved)
        self.assertEqual(original["codebook"], [[7, 2], [7, 10], [12, 1]])

    # ---- random strategy ----

    def test_random_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)
        stats = self._run(
            inp,
            out,
            max_items_per_codebook=2,
            strategy="random",
        )
        self.assertEqual(stats.relocated_count, 3)
        d = self._read_parquet(out)
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)
        _, resolved_path = self._group_paths(out)
        resolved = self._read_parquet(resolved_path)
        resolved_items = {
            tuple(codebook): item_group
            for codebook, item_group in zip(resolved["codebook"], resolved["itemids"])
        }
        for item_id, codebook, index in zip(d["item_id"], d["codebook"], d["index"]):
            self.assertEqual(resolved_items[tuple(codebook)][index - 1], item_id)

    def test_random_requires_last_codebook_ge_2(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)
        with self.assertRaises(ValueError):
            self._run(
                inp, out, max_items_per_codebook=2, strategy="random", codebook="8,1"
            )

    def test_single_layer_sid(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(4)), [[0]] * 4)
        stats = self._run(
            inp,
            out,
            max_items_per_codebook=1,
            strategy="random",
            codebook="16",
        )
        self.assertEqual(stats.relocated_count, 3)
        d = self._read_parquet(out)
        self.assertEqual(len({tuple(cb) for cb in d["codebook"]}), 4)

    # ---- determinism ----

    def test_deterministic_and_order_independent(self) -> None:
        rng = random.Random(0)
        n = 200
        item_ids = list(range(n))
        codes = [[rng.randrange(3), rng.randrange(2)] for _ in item_ids]
        cands = [[[c[0], j] for j in range(8)] for c in codes]

        def decisions(order):
            inp = os.path.join(self.test_dir, f"in_{order[0]}_{len(order)}.parquet")
            out = os.path.join(self.test_dir, f"out_{order[0]}_{len(order)}")
            _parquet(
                inp,
                [item_ids[i] for i in order],
                [codes[i] for i in order],
                [cands[i] for i in order],
            )
            self._run(
                inp,
                out,
                max_items_per_codebook=2,
            )
            d = self._read_parquet(out)
            original_path, resolved_path = self._group_paths(out)
            return (
                {
                    d["item_id"][i]: tuple(d["codebook"][i])
                    for i in range(len(d["item_id"]))
                },
                self._read_parquet(original_path),
                self._read_parquet(resolved_path),
            )

        base = decisions(list(range(n)))
        self.assertEqual(base, decisions(list(range(n))))  # run-twice
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        self.assertEqual(base, decisions(shuffled))  # order-independent

    # ---- CSV backend ----

    def test_csv_code_encoder_joins_all_layers(self) -> None:
        codes = np.asarray([[1, -2, 3], [4, 5, 6]], dtype=np.int64)

        encoded = CollisionRunner._codes_column(codes, is_csv=True)

        self.assertEqual(encoded.to_pylist(), ["1,-2,3", "4,5,6"])

    def test_csv_backend_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.csv")
        out = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["0", "1", "2", "3", "4"],
                    "codes": ["0,0"] * 5,
                    "candidate_codes": ["0,1,0,2,0,3"] * 5,
                }
            ),
            inp,
        )
        stats = self._run(
            inp,
            out,
            reader_type="CsvReader",
            writer_type="CsvWriter",
            max_items_per_codebook=2,
        )
        self.assertEqual(stats.relocated_count, 3)
        result = csv.read_csv(os.path.join(out, "part-0.csv"))
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertTrue(all(s.startswith("0,") for s in result["codebook"].to_pylist()))

        original_path, resolved_path = self._group_paths(out)
        for path in (original_path, resolved_path):
            groups = csv.read_csv(os.path.join(path, "part-0.csv"))
            self.assertEqual(groups.column_names, ["codebook", "itemids"])
            self.assertTrue(
                all(
                    isinstance(json.loads(itemids), list)
                    for itemids in groups["itemids"].to_pylist()
                )
            )

    def test_csv_group_itemids_use_json_escaping(self) -> None:
        inp = os.path.join(self.test_dir, "in.csv")
        out = os.path.join(self.test_dir, "out")
        item_ids = ["a;b", "x|y", "with,comma", 'with"quote', "back\\slash"]
        csv.write_csv(
            pa.table(
                {
                    "item_id": item_ids,
                    "codes": ["0,0"] * len(item_ids),
                }
            ),
            inp,
        )

        self._run(
            inp,
            out,
            reader_type="CsvReader",
            writer_type="CsvWriter",
            max_items_per_codebook=len(item_ids),
        )

        original_path, _ = self._group_paths(out)
        groups = csv.read_csv(os.path.join(original_path, "part-0.csv"))
        decoded_ids = json.loads(groups["itemids"][0].as_py())
        self.assertCountEqual(decoded_ids, item_ids)

    def test_parquet_input_with_csv_output(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)
        self._run(
            inp,
            out,
            writer_type="CsvWriter",
            max_items_per_codebook=2,
        )
        result = csv.read_csv(os.path.join(out, "part-0.csv"))
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertTrue(all("," in sid for sid in result["codebook"].to_pylist()))
        original_path, resolved_path = self._group_paths(out)
        for path in (original_path, resolved_path):
            schema = csv.read_csv(os.path.join(path, "part-0.csv")).schema
            self.assertEqual(schema.names, ["codebook", "itemids"])
            self.assertEqual(schema.field("itemids").type, pa.string())

    def test_csv_input_with_parquet_output(self) -> None:
        inp = os.path.join(self.test_dir, "in.csv")
        out = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["a", "b", "c"],
                    "codes": ["0,0"] * 3,
                    "candidate_codes": ["0,1"] * 3,
                }
            ),
            inp,
        )
        self._run(
            inp,
            out,
            reader_type="CsvReader",
            writer_type="ParquetWriter",
            max_items_per_codebook=2,
        )
        schema = parquet.read_table(os.path.join(out, "part-0.parquet")).schema
        self.assertEqual(schema.field("codebook").type, pa.list_(pa.int64()))
        original_path, resolved_path = self._group_paths(out)
        for path in (original_path, resolved_path):
            group_schema = parquet.read_table(
                os.path.join(path, "part-0.parquet")
            ).schema
            self.assertEqual(group_schema.names, ["codebook", "itemids"])
            self.assertEqual(group_schema.field("itemids").type, pa.list_(pa.string()))

    def test_preserves_integer_item_id_type(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([0, 1, 2], type=pa.int32()),
                    "codes": pa.array(
                        [[0, 0], [0, 1], [0, 2]],
                        type=pa.list_(pa.int64()),
                    ),
                }
            ),
            inp,
        )
        self._run(inp, out, max_items_per_codebook=1)
        schema = parquet.read_table(os.path.join(out, "part-0.parquet")).schema
        self.assertEqual(schema.field("item_id").type, pa.int32())
        original_path, resolved_path = self._group_paths(out)
        for path in (original_path, resolved_path):
            group_schema = parquet.read_table(
                os.path.join(path, "part-0.parquet")
            ).schema
            self.assertEqual(group_schema.field("itemids").type, pa.list_(pa.int32()))

    def test_writer_defaults_to_reader(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)
        self._run(inp, out, max_items_per_codebook=3)
        self.assertTrue(os.path.exists(os.path.join(out, "part-0.parquet")))

    # ---- misc ----

    def test_group_output_paths_are_required_for_normal_runs(self) -> None:
        args = dict(_DEFAULTS)
        args.update(input_path="input", output_path="map")
        with self.assertRaisesRegex(ValueError, "both SID group output paths"):
            CollisionRunner(Namespace(**args))

    def test_group_output_paths_must_be_supplied_together(self) -> None:
        args = dict(_DEFAULTS)
        args.update(
            input_path="input",
            output_path="map",
            original_sid_groups_output_path="original",
        )
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            CollisionRunner(Namespace(**args))

    def test_output_paths_must_be_distinct(self) -> None:
        args = dict(_DEFAULTS)
        args.update(
            input_path="input",
            output_path="map",
            original_sid_groups_output_path="map/../map",
            resolved_sid_groups_output_path="resolved",
        )
        with self.assertRaisesRegex(ValueError, "must differ from output_path"):
            CollisionRunner(Namespace(**args))

    def test_odps_output_paths_reject_trailing_slash_alias(self) -> None:
        args = dict(_DEFAULTS)
        args.update(
            input_path="input",
            output_path="odps://project/tables/map",
            original_sid_groups_output_path="odps://project/tables/map/",
            resolved_sid_groups_output_path="odps://project/tables/resolved",
        )
        with self.assertRaisesRegex(ValueError, "must differ from output_path"):
            CollisionRunner(Namespace(**args))

    def test_odps_output_paths_reject_ignored_partition_alias(self) -> None:
        args = dict(_DEFAULTS)
        args.update(
            input_path="input",
            output_path="odps://project/tables/schema.map/dt=20260713",
            original_sid_groups_output_path=(
                "odps://project/tables/schema.map/dt=20260713&dt=20260712"
            ),
            resolved_sid_groups_output_path="odps://project/tables/schema.resolved",
        )
        with self.assertRaisesRegex(ValueError, "must differ from output_path"):
            CollisionRunner(Namespace(**args))

    def test_odps_group_writes_use_native_array_columns(self) -> None:
        class OdpsWriter:
            def __init__(self) -> None:
                self.writes = []
                self.closed = False

            def write(self, output_dict) -> None:
                self.writes.append(output_dict)

            def close(self) -> None:
                self.closed = True

        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "map")
        parquet.write_table(
            pa.table(
                {
                    "item_id": pa.array([1, 2], type=pa.int32()),
                    "codes": pa.array([[0, 0], [0, 1]], type=pa.list_(pa.int64())),
                }
            ),
            inp,
        )
        created_writers = []

        def make_writer(output_path):
            writer = OdpsWriter()
            created_writers.append((output_path, writer))
            return writer

        with mock.patch.object(
            CollisionRunner, "_make_writer", side_effect=make_writer
        ):
            self._run(
                inp,
                out,
                writer_type="OdpsWriter",
                max_items_per_codebook=1,
            )

        original_path, resolved_path = self._group_paths(out)
        self.assertEqual(
            [path for path, _ in created_writers],
            [original_path, resolved_path, out],
        )
        self.assertTrue(all(writer.closed for _, writer in created_writers))
        for _, writer in created_writers[:2]:
            columns = writer.writes[0]
            self.assertEqual(list(columns), ["codebook", "itemids"])
            self.assertEqual(columns["codebook"].type, pa.list_(pa.int64()))
            self.assertEqual(columns["itemids"].type, pa.list_(pa.int32()))
            self.assertEqual(
                _type_pa_to_table(columns["codebook"].type), "ARRAY<BIGINT>"
            )
            self.assertEqual(_type_pa_to_table(columns["itemids"].type), "ARRAY<INT>")
        self.assertEqual(_type_pa_to_table(pa.list_(pa.string())), "ARRAY<STRING>")

    def test_writer_closes_when_write_fails(self) -> None:
        class FailingWriter:
            def __init__(self) -> None:
                self.closed = False

            def write(self, output_dict) -> None:
                raise RuntimeError("write failed")

            def close(self) -> None:
                self.closed = True

        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0], [[0, 0]])
        writer = FailingWriter()

        with (
            mock.patch.object(CollisionRunner, "_make_writer", return_value=writer),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            self._run(inp, out, max_items_per_codebook=1)

        self.assertTrue(writer.closed)

    def test_group_writer_chunks_by_item_count_without_splitting_groups(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(4)), [[0, 0], [0, 0], [0, 0], [0, 1]])

        with mock.patch("tzrec.tools.sid.collision_prevention._GROUP_WRITE_ITEMS", 2):
            self._run(inp, out, max_items_per_codebook=3)

        original_path, _ = self._group_paths(out)
        output_file = os.path.join(original_path, "part-0.parquet")
        self.assertEqual(parquet.ParquetFile(output_file).metadata.num_row_groups, 2)
        groups = parquet.read_table(output_file).to_pydict()
        self.assertEqual([len(group) for group in groups["itemids"]], [3, 1])

    def test_map_writer_chunks_by_row_count(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, code] for code in range(5)])

        with mock.patch("tzrec.tools.sid.collision_prevention._MAP_WRITE_ROWS", 2):
            self._run(inp, out, max_items_per_codebook=1)

        output_file = os.path.join(out, "part-0.parquet")
        self.assertEqual(parquet.ParquetFile(output_file).metadata.num_row_groups, 3)

    def test_writers_bound_codebook_list_offsets(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0], [0, 1], [0, 2]])

        with (
            mock.patch(
                "tzrec.tools.sid.collision_prevention._ARROW_LIST_OFFSET_MAX", 4
            ),
            mock.patch("tzrec.tools.sid.collision_prevention._MAP_WRITE_ROWS", 100),
            mock.patch("tzrec.tools.sid.collision_prevention._GROUP_WRITE_ITEMS", 100),
        ):
            self._run(inp, out, max_items_per_codebook=1)

        original_path, resolved_path = self._group_paths(out)
        for path in (out, original_path, resolved_path):
            output_file = os.path.join(path, "part-0.parquet")
            self.assertEqual(
                parquet.ParquetFile(output_file).metadata.num_row_groups, 2
            )

    def test_rate_only_skips_write(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        with mock.patch.object(
            collision_prevention,
            "resolve_sid_collisions",
            wraps=collision_prevention.resolve_sid_collisions,
        ) as resolve:
            stats = self._run(
                inp,
                out,
                max_items_per_codebook=2,
                rate_only=True,
            )
        self.assertFalse(resolve.call_args.kwargs["collect_grouping"])
        self.assertEqual(stats.relocated_count, 3)
        self.assertFalse(os.path.exists(os.path.join(out, "part-0.parquet")))
        original_path, resolved_path = self._group_paths(out)
        self.assertFalse(os.path.exists(original_path))
        self.assertFalse(os.path.exists(resolved_path))

    def test_rate_only_allows_omitted_group_paths(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1], [[0, 0], [0, 0]])

        stats = self._run(
            inp,
            out,
            original_sid_groups_output_path=None,
            resolved_sid_groups_output_path=None,
            max_items_per_codebook=2,
            rate_only=True,
        )

        self.assertEqual(stats.total_items, 2)
        self.assertFalse(os.path.exists(out))

    def test_empty_input_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [], [])
        with self.assertRaises(ValueError):
            self._run(inp, out, max_items_per_codebook=2)

    def test_duplicate_item_id_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 1], [[0, 0], [0, 1], [0, 2]])
        with self.assertRaisesRegex(ValueError, "item IDs must be unique"):
            self._run(inp, out, max_items_per_codebook=2)

    def test_empty_codebook_token_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1], [[0, 0], [0, 1]])
        with self.assertRaisesRegex(ValueError, "codebook"):
            self._run(inp, out, codebook="8,,8")


if __name__ == "__main__":
    unittest.main()
