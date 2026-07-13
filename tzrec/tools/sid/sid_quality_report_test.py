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

import argparse
import math
import os
import shutil
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tzrec.tools.sid import sid_quality_report as sqr
from tzrec.utils import misc_util
from tzrec.utils.test_util import make_test_dir


class SidQualityReportTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        self.test_dir = None  # only the subprocess test needs a filesystem

    def tearDown(self):
        if self.success and self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_compute_gini(self):
        # a perfectly uniform frequency distribution has Gini 0.
        self.assertAlmostEqual(sqr.compute_gini([2, 2, 2, 2]), 0.0)
        # 3 items on one SID, 2 on another, 3 singletons.
        self.assertAlmostEqual(sqr.compute_gini([3, 2, 1, 1, 1]), 0.25)
        self.assertEqual(sqr.compute_gini([]), 0.0)

    def test_compute_entropy(self):
        self.assertEqual(sqr.compute_entropy([]), 0.0)
        # uniform over k categories -> entropy ln(k).
        self.assertAlmostEqual(sqr.compute_entropy([5, 5, 5, 5]), float(np.log(4)))
        self.assertAlmostEqual(
            sqr.compute_entropy([3, 2, 1, 1, 1]), 1.4941751382893085, places=10
        )
        # zero-count entries (e.g. unused codebook slots) are ignored, not NaN.
        self.assertAlmostEqual(
            sqr.compute_entropy([3, 0, 2, 0, 1, 1, 1]),
            sqr.compute_entropy([3, 2, 1, 1, 1]),
            places=10,
        )

    def test_parse_codes_list_and_string_agree(self):
        list_col = pa.array([[1, 2, 3], [4, 5, 6]], type=pa.list_(pa.int64()))
        str_col = pa.array(["1,2,3", "4,5,6"], type=pa.string())
        self.assertEqual(sqr.parse_codes(list_col), [[1, 2, 3], [4, 5, 6]])
        self.assertEqual(sqr.parse_codes(str_col), [[1, 2, 3], [4, 5, 6]])
        # null rows become empty lists.
        self.assertEqual(
            sqr.parse_codes(pa.array([[1, 2], None], type=pa.list_(pa.int64()))),
            [[1, 2], []],
        )
        # a non-integer / whitespace token yields [] (malformed) -- never crashes.
        self.assertEqual(
            sqr.parse_codes(pa.array(["1,2,3", "1,x,3", "1, ,3"])),
            [[1, 2, 3], [], []],
        )
        # a null element is preserved so the caller can drop that row.
        self.assertEqual(
            sqr.parse_codes(pa.array([[1, None, 3]], type=pa.list_(pa.int64()))),
            [[1, None, 3]],
        )

    def test_build_arr_drops_malformed(self):
        # clean list<int> batch: fast Arrow path, nothing dropped.
        arr, dropped = sqr.build_arr(
            pa.array([[1, 2, 3], [4, 5, 6]], type=pa.list_(pa.int64())), 3
        )
        self.assertEqual(dropped, 0)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3], [4, 5, 6]])))
        # a null row, a wrong-width row, and a null element are each dropped;
        # the valid rows survive (no whole-batch drop, no crash).
        arr, dropped = sqr.build_arr(
            pa.array(
                [[1, 2, 3], None, [4, 5], [7, None, 9], [10, 11, 12]],
                type=pa.list_(pa.int64()),
            ),
            3,
        )
        self.assertEqual(dropped, 3)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3], [10, 11, 12]])))
        # a null ELEMENT with no null row must be caught (else flatten -> float NaN
        # -> bincount crash); it takes the per-row path via flat.null_count.
        arr, dropped = sqr.build_arr(
            pa.array([[1, 2, 3], [4, None, 6]], type=pa.list_(pa.int64())), 3
        )
        self.assertEqual(dropped, 1)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3]])))
        # string path: a non-integer row is dropped, not crashed on.
        arr, dropped = sqr.build_arr(pa.array(["1,2,3", "1,x,3"]), 3)
        self.assertEqual(dropped, 1)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3]])))

    def test_update_layer_hist(self):
        arr = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int64)
        hist = sqr.update_layer_hist(None, arr, [2, 2])
        self.assertTrue(np.array_equal(hist[0], np.array([2, 1])))
        self.assertTrue(np.array_equal(hist[1], np.array([1, 2])))
        # a second batch accumulates into the existing histograms.
        hist = sqr.update_layer_hist(hist, np.array([[1, 1]], dtype=np.int64), [2, 2])
        self.assertTrue(np.array_equal(hist[0], np.array([2, 2])))
        self.assertTrue(np.array_equal(hist[1], np.array([1, 3])))

    def test_parse_codebook(self):
        self.assertEqual(sqr._parse_codebook("256, 256, 256"), [256, 256, 256])
        self.assertEqual(sqr._parse_codebook(" 16 "), [16])
        with self.assertRaises(argparse.ArgumentTypeError):
            sqr._parse_codebook("16,abc,16")
        with self.assertRaises(argparse.ArgumentTypeError):
            sqr._parse_codebook("0,16")

    def test_end_to_end(self):
        self.test_dir = make_test_dir()
        # 8 items: (1,2,3)x2, (4,5,6), (7,8,9), (10,11,12), (0,0,0)x3.
        codes = [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [10, 11, 12],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        input_dir = os.path.join(self.test_dir, "predict_output")
        os.makedirs(input_dir)
        pq.write_table(
            pa.table({"codes": pa.array(codes, type=pa.list_(pa.int64()))}),
            os.path.join(input_dir, "part-0.parquet"),
        )

        summary_out = os.path.join(self.test_dir, "summary")
        layer_out = os.path.join(self.test_dir, "layer")
        cmd_str = (
            "PYTHONPATH=. python -m tzrec.tools.sid.sid_quality_report "
            f"--input_path {input_dir}/*.parquet --codes_field codes "
            "--codebook 16,16,16 "
            f"--summary_output {summary_out} --layer_stats_output {layer_out}"
        )
        ran = misc_util.run_cmd(
            cmd_str,
            os.path.join(self.test_dir, "log_sid_quality_report.txt"),
            timeout=600,
        )
        self.assertTrue(ran)

        summary = pq.read_table(
            os.path.join(summary_out, "part-0.parquet")
        ).to_pylist()[0]
        self.assertEqual(summary["total"], 8)
        self.assertEqual(summary["unique_sid"], 5)
        self.assertAlmostEqual(summary["no_collision_rate"], 0.625)
        self.assertAlmostEqual(summary["collision_free_item_rate"], 0.375)
        self.assertEqual(summary["max_collision"], 3)
        self.assertAlmostEqual(summary["gini"], 0.25)
        self.assertAlmostEqual(summary["entropy"], 1.4941751382893085)
        self.assertAlmostEqual(summary["max_entropy"], math.log(16**3))
        self.assertAlmostEqual(
            summary["entropy_ratio"], 1.4941751382893085 / math.log(16**3)
        )

        layer = pq.read_table(os.path.join(layer_out, "part-0.parquet")).to_pylist()
        self.assertEqual(len(layer), 3)
        for row in layer:
            self.assertEqual(row["codebook_size"], 16)
            self.assertAlmostEqual(row["coverage"], 0.3125)
            self.assertEqual(row["dead_codes"], 11)
            self.assertAlmostEqual(row["perplexity"], math.exp(1.4941751382893085))

        self.success = True

    def test_end_to_end_skips_malformed_and_out_of_range(self):
        self.test_dir = make_test_dir()
        # valid in-range: (1,2,3)x2, (9,9,9). dropped: null row, width-2 row,
        # null-element row, and an out-of-range row (20 >= codebook 16).
        codes = [
            [1, 2, 3],
            [1, 2, 3],
            None,
            [4, 5],
            [1, None, 3],
            [9, 9, 9],
            [20, 1, 1],
        ]
        input_dir = os.path.join(self.test_dir, "predict_output")
        os.makedirs(input_dir)
        pq.write_table(
            pa.table({"codes": pa.array(codes, type=pa.list_(pa.int64()))}),
            os.path.join(input_dir, "part-0.parquet"),
        )
        summary_out = os.path.join(self.test_dir, "summary")
        cmd_str = (
            "PYTHONPATH=. python -m tzrec.tools.sid.sid_quality_report "
            f"--input_path {input_dir}/*.parquet --codes_field codes "
            f"--codebook 16,16,16 --summary_output {summary_out}"
        )
        ran = misc_util.run_cmd(
            cmd_str, os.path.join(self.test_dir, "log_malformed.txt"), timeout=600
        )
        self.assertTrue(ran)  # must not crash on null/ragged/out-of-range rows
        summary = pq.read_table(
            os.path.join(summary_out, "part-0.parquet")
        ).to_pylist()[0]
        self.assertEqual(summary["total"], 3)  # only the valid, in-range rows
        self.assertEqual(summary["unique_sid"], 2)  # (1,2,3) and (9,9,9)
        self.assertEqual(summary["max_collision"], 2)  # (1,2,3) appears twice
        self.success = True


if __name__ == "__main__":
    unittest.main()
