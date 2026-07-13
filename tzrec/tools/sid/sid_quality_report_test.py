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

# Asymmetric codebook + distinct per-layer usage: exercises mixed-radix ordering
# and per-layer indexing, which a symmetric codebook with identical layers cannot.
_ASYM_CODEBOOK = [4, 8, 16]
_ASYM_ROWS = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
    [1, 2, 2],
    [2, 3, 3],
    [3, 7, 15],
]
# Expectations for that fixture, derived independently of the tool:
_ASYM_SID_COUNTS = [3, 1, 1, 1, 1]
_ASYM_LAYER_USAGE = [
    [3, 2, 1, 1],
    [3, 1, 1, 1, 1],
    [3, 1, 1, 1, 1],
]
_ASYM_LAYER_COVERAGE = [1.0, 0.625, 0.3125]
_ASYM_LAYER_DEAD = [0, 3, 11]
_ASYM_TOP_SIDS = [
    ("0,0,0", 3),
    ("1,1,1", 1),
    ("1,2,2", 1),
    ("2,3,3", 1),
    ("3,7,15", 1),
]


def _run(rows, codebook, **kwargs):
    """Run the report over a single ``codes`` batch built from ``rows``."""
    batch = {"codes": pa.array(rows, type=pa.list_(pa.int64()))}
    return sqr.run_report([batch], "codes", codebook, **kwargs)


class SidQualityReportTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        self.test_dir = None  # only the subprocess test needs a filesystem

    def tearDown(self):
        if self.success and self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _assert_asym_summary(self, summary):
        """Assert the summary computed for the _ASYM fixture (dict-like input)."""
        self.assertEqual(summary["total"], 7)
        self.assertEqual(summary["unique_sid"], 5)
        self.assertAlmostEqual(summary["no_collision_rate"], 5 / 7)
        self.assertAlmostEqual(summary["uniquely_identified_item_rate"], 4 / 7)
        self.assertEqual(summary["max_collision"], 3)
        self.assertAlmostEqual(summary["gini"], sqr.compute_gini(_ASYM_SID_COUNTS))
        entropy = sqr.compute_entropy(_ASYM_SID_COUNTS)
        max_entropy = math.log(math.prod(_ASYM_CODEBOOK))
        self.assertAlmostEqual(summary["entropy"], entropy)
        self.assertAlmostEqual(summary["max_entropy"], max_entropy)
        self.assertAlmostEqual(summary["entropy_ratio"], entropy / max_entropy)

    def _assert_asym_layers(self, cols):
        """Assert the per-layer stats for the _ASYM fixture (columnar dict input)."""
        self.assertEqual(list(cols["codebook_size"]), _ASYM_CODEBOOK)
        self.assertEqual(list(cols["coverage"]), _ASYM_LAYER_COVERAGE)
        self.assertEqual(list(cols["dead_codes"]), _ASYM_LAYER_DEAD)
        for got, usage in zip(cols["perplexity"], _ASYM_LAYER_USAGE):
            self.assertAlmostEqual(got, math.exp(sqr.compute_entropy(usage)))

    def test_compute_gini(self):
        self.assertAlmostEqual(sqr.compute_gini([2, 2, 2, 2]), 0.0)
        self.assertAlmostEqual(sqr.compute_gini([3, 2, 1, 1, 1]), 0.25)
        self.assertEqual(sqr.compute_gini([]), 0.0)

    def test_compute_entropy(self):
        self.assertEqual(sqr.compute_entropy([]), 0.0)
        self.assertAlmostEqual(sqr.compute_entropy([5, 5, 5, 5]), float(np.log(4)))
        self.assertAlmostEqual(
            sqr.compute_entropy([3, 2, 1, 1, 1]), 1.4941751382893085, places=10
        )
        # zero-count entries (unused codebook slots) are ignored, not NaN.
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
        self.assertEqual(
            sqr.parse_codes(pa.array([[1, 2], None], type=pa.list_(pa.int64()))),
            [[1, 2], []],
        )
        # non-integer / whitespace token -> [] (malformed), never a crash.
        self.assertEqual(
            sqr.parse_codes(pa.array(["1,2,3", "1,x,3", "1, ,3"])),
            [[1, 2, 3], [], []],
        )
        # a blank field is malformed, not compacted: "1,2,,3" must not become [1,2,3].
        self.assertEqual(
            sqr.parse_codes(pa.array(["1,2,3", "1,2,,3"])),
            [[1, 2, 3], []],
        )
        # an int64-overflowing token is dropped, not an OverflowError.
        self.assertEqual(
            sqr.parse_codes(pa.array(["1,2,3", "1,99999999999999999999,3"])),
            [[1, 2, 3], []],
        )
        # a null element is preserved so the caller can drop the row.
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
        # null row, wrong-width row, and null element are each dropped, not the batch.
        arr, dropped = sqr.build_arr(
            pa.array(
                [[1, 2, 3], None, [4, 5], [7, None, 9], [10, 11, 12]],
                type=pa.list_(pa.int64()),
            ),
            3,
        )
        self.assertEqual(dropped, 3)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3], [10, 11, 12]])))
        # null element w/o null row: caught via flat.null_count
        # (else flatten -> NaN -> bincount crash).
        arr, dropped = sqr.build_arr(
            pa.array([[1, 2, 3], [4, None, 6]], type=pa.list_(pa.int64())), 3
        )
        self.assertEqual(dropped, 1)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3]])))
        # string path: a non-integer row is dropped, not crashed on.
        arr, dropped = sqr.build_arr(pa.array(["1,2,3", "1,x,3"]), 3)
        self.assertEqual(dropped, 1)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3]])))
        # string path: an int64-overflowing token drops the row.
        arr, dropped = sqr.build_arr(pa.array(["1,2,3", "1,99999999999999999999,3"]), 3)
        self.assertEqual(dropped, 1)
        self.assertTrue(np.array_equal(arr, np.array([[1, 2, 3]])))

    def test_update_layer_hist(self):
        arr = np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int64)
        hist = sqr.update_layer_hist(None, arr, [2, 2])
        self.assertTrue(np.array_equal(hist[0], np.array([2, 1])))
        self.assertTrue(np.array_equal(hist[1], np.array([1, 2])))
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

    def test_run_report_asymmetric(self):
        # asymmetric codebook: validates radix ordering, per-layer indexing, decode.
        summary, layer_stats, top_sids = _run(
            _ASYM_ROWS, _ASYM_CODEBOOK, log_top_sids=5
        )
        self._assert_asym_summary(summary)
        self._assert_asym_layers(layer_stats)
        self.assertEqual(top_sids, _ASYM_TOP_SIDS)

    def test_run_report_mixed_radix_is_bijective(self):
        # distinct tuples a degenerate radix (plain sum) would merge -- (0,0,1),
        # (0,1,0), (1,0,0) all sum to 1; a correct mixed-radix keeps them distinct.
        rows = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [3, 7, 15],
        ]
        summary, _, _ = _run(rows, _ASYM_CODEBOOK)
        self.assertEqual(summary["total"], 8)
        self.assertEqual(summary["unique_sid"], 8)
        self.assertEqual(summary["max_collision"], 1)
        self.assertEqual(summary["no_collision_rate"], 1.0)

    def test_run_report_skips_malformed_and_out_of_range(self):
        # malformed (null / wrong-width / null-element) and out-of-range (20>=16)
        # rows dropped; only (1,2,3)x2 and (9,9,9) survive.
        rows = [
            [1, 2, 3],
            [1, 2, 3],
            None,
            [4, 5],
            [1, None, 3],
            [9, 9, 9],
            [20, 1, 1],
        ]
        summary, _, _ = _run(rows, [16, 16, 16])
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["unique_sid"], 2)
        self.assertEqual(summary["max_collision"], 2)

    def test_run_report_capacity_overflow_raises(self):
        # pass a valid row so ONLY the capacity guard (not "no valid rows") can raise.
        with self.assertRaisesRegex(ValueError, "exceeds int64"):
            _run([[0, 0]], [3037000500, 3037000500])

    def test_run_report_no_valid_rows_raises(self):
        # match the message so the explicit guard, not numpy's concat error, is pinned.
        with self.assertRaisesRegex(ValueError, "no valid rows"):
            _run([None, [1, 2]], [16, 16, 16])

    def test_run_report_entropy_ratio_nan_on_unit_capacity(self):
        # codebook (1,): max_entropy=log(1)=0 -> entropy_ratio is nan, not a crash.
        summary, layer_stats, _ = _run([[0], [0]], [1])
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["unique_sid"], 1)
        self.assertTrue(math.isnan(summary["entropy_ratio"]))
        self.assertEqual(layer_stats["coverage"], [1.0])

    def test_end_to_end(self):
        # subprocess smoke test of the CLI + writers (asymmetric codebook end to end).
        self.test_dir = make_test_dir()
        input_dir = os.path.join(self.test_dir, "predict_output")
        os.makedirs(input_dir)
        pq.write_table(
            pa.table({"codes": pa.array(_ASYM_ROWS, type=pa.list_(pa.int64()))}),
            os.path.join(input_dir, "part-0.parquet"),
        )

        summary_out = os.path.join(self.test_dir, "summary")
        layer_out = os.path.join(self.test_dir, "layer")
        cmd_str = (
            "PYTHONPATH=. python -m tzrec.tools.sid.sid_quality_report "
            f"--input_path {input_dir}/*.parquet --codes_field codes "
            "--codebook 4,8,16 "
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
        self._assert_asym_summary(summary)

        layer_rows = pq.read_table(
            os.path.join(layer_out, "part-0.parquet")
        ).to_pylist()
        self._assert_asym_layers({k: [r[k] for r in layer_rows] for k in layer_rows[0]})

        self.success = True


if __name__ == "__main__":
    unittest.main()
