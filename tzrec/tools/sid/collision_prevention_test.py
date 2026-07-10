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
import random
import shutil
import tempfile
import unittest
from argparse import Namespace
from collections import Counter

import pyarrow as pa
from pyarrow import csv, parquet

from tzrec.tools.sid.collision_prevention import CollisionRunner

# Defaults mirror the __main__ argparse; tests override only what they exercise.
_DEFAULTS = dict(
    input_path=None,
    output_path=None,
    diagnostics_output_path=None,
    reader_type=None,
    writer_type=None,
    batch_size=100000,
    item_id_field="item_id",
    code_field="codes",
    candidate_codes_field="candidate_codes",
    candidate_depth=None,
    max_items_per_codebook=2,
    unassigned_policy="error",
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
        cols["candidate_codes"] = pa.array(
            candidate_codes, type=pa.list_(pa.list_(pa.int64()))
        )
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
        args.update(input_path=inp, output_path=out)
        args.update(kw)
        return CollisionRunner(Namespace(**args)).run()

    def _read_parquet(self, out_dir):
        return parquet.read_table(os.path.join(out_dir, "part-0.parquet")).to_pydict()

    # ---- candidate strategy ----

    def test_candidate_reassigns_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # 5 items in bucket (0,0); candidates vary only the last layer within band 0.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        stats = self._run(
            inp, out, max_items_per_codebook=2, unassigned_policy="keep_original"
        )
        self.assertEqual(stats.total_items, 5)
        self.assertEqual(stats.reassigned_count, 3)
        self.assertEqual(stats.unassigned_count, 0)
        self.assertEqual(stats.max_final_bucket_size, 2)
        d = self._read_parquet(out)
        # every final SID keeps prefix 0 (stays in band) ...
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        # ... and no bucket exceeds the cap.
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)

    def test_output_is_list_int64(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)
        self._run(inp, out, max_items_per_codebook=2, unassigned_policy="keep_original")
        schema = parquet.read_table(os.path.join(out, "part-0.parquet")).schema
        self.assertEqual(schema.field("codebook").type, pa.list_(pa.int64()))
        self.assertEqual(schema.field("origin_codebook").type, pa.list_(pa.int64()))

    def test_keep_original_when_only_origin_candidate(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        # candidate == origin last -> skipped -> nothing to place.
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 0]]] * 5)
        stats = self._run(
            inp, out, max_items_per_codebook=2, unassigned_policy="keep_original"
        )
        self.assertEqual(stats.reassigned_count, 0)
        self.assertEqual(stats.unassigned_count, 3)
        d = self._read_parquet(out)
        self.assertTrue(all(tuple(cb) == (0, 0) for cb in d["codebook"]))

    def test_error_policy_raises_on_overflow(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 0]]] * 5)
        with self.assertRaises(RuntimeError):
            self._run(inp, out, max_items_per_codebook=2)

    def test_drop_policy_excludes_unassigned(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 0]]] * 5)
        stats = self._run(inp, out, max_items_per_codebook=2, unassigned_policy="drop")
        self.assertEqual(stats.unassigned_count, 3)
        d = self._read_parquet(out)
        self.assertEqual(len(d["item_id"]), 2)

    def test_raises_when_overflow_but_no_candidates(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5)  # no candidate_codes column
        with self.assertRaises(ValueError):
            self._run(inp, out, max_items_per_codebook=2)

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
            unassigned_policy="keep_original",
        )
        self.assertEqual(stats.reassigned_count, 3)
        d = self._read_parquet(out)
        self.assertTrue(all(cb[0] == 0 for cb in d["codebook"]))
        self.assertLessEqual(max(Counter(map(tuple, d["codebook"])).values()), 2)

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
            unassigned_policy="keep_original",
        )
        self.assertEqual(stats.reassigned_count, 3)
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
                unassigned_policy="keep_original",
            )
            d = self._read_parquet(out)
            return {
                d["item_id"][i]: tuple(d["codebook"][i])
                for i in range(len(d["item_id"]))
            }

        base = decisions(list(range(n)))
        self.assertEqual(base, decisions(list(range(n))))  # run-twice
        shuffled = list(range(n))
        rng.shuffle(shuffled)
        self.assertEqual(base, decisions(shuffled))  # order-independent

    # ---- CSV backend ----

    def test_csv_backend_within_band(self) -> None:
        inp = os.path.join(self.test_dir, "in.csv")
        out = os.path.join(self.test_dir, "out")
        csv.write_csv(
            pa.table(
                {
                    "item_id": ["0", "1", "2", "3", "4"],
                    "codes": ["0|0"] * 5,
                    "candidate_codes": ["0|1;0|2;0|3"] * 5,
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
            unassigned_policy="keep_original",
        )
        self.assertEqual(stats.reassigned_count, 3)
        result = csv.read_csv(os.path.join(out, "part-0.csv"))
        self.assertEqual(result.schema.field("codebook").type, pa.string())
        self.assertTrue(all(s.startswith("0|") for s in result["codebook"].to_pylist()))

    def test_writer_defaults_to_reader(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [0, 1, 2], [[0, 0]] * 3, [[[0, 1]]] * 3)
        self._run(inp, out, max_items_per_codebook=3)
        self.assertTrue(os.path.exists(os.path.join(out, "part-0.parquet")))

    # ---- misc ----

    def test_rate_only_skips_write(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        stats = self._run(
            inp,
            out,
            max_items_per_codebook=2,
            unassigned_policy="keep_original",
            rate_only=True,
        )
        self.assertEqual(stats.reassigned_count, 3)
        self.assertFalse(os.path.exists(os.path.join(out, "part-0.parquet")))

    def test_diagnostics_output(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        diag = os.path.join(self.test_dir, "diag")
        _parquet(inp, list(range(5)), [[0, 0]] * 5, [[[0, 1], [0, 2], [0, 3]]] * 5)
        self._run(
            inp,
            out,
            max_items_per_codebook=2,
            unassigned_policy="keep_original",
            diagnostics_output_path=diag,
        )
        d = parquet.read_table(os.path.join(diag, "part-0.parquet")).to_pydict()
        self.assertEqual(d["total_items"][0], 5)
        self.assertEqual(d["reassigned_count"][0], 3)

    def test_empty_input_raises(self) -> None:
        inp = os.path.join(self.test_dir, "in.parquet")
        out = os.path.join(self.test_dir, "out")
        _parquet(inp, [], [])
        with self.assertRaises(ValueError):
            self._run(inp, out, max_items_per_codebook=2)


if __name__ == "__main__":
    unittest.main()
