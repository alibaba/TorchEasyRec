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

import math
import unittest
from dataclasses import fields
from unittest import mock

import numpy as np

from tzrec.utils.sid import quality
from tzrec.utils.sid.quality import (
    SidQualityAccumulator,
    compare_sid_quality,
    compute_entropy,
    compute_gini,
    valid_code_rows,
)

_CODEBOOK = [4, 8, 16]
_ROWS = np.asarray(
    [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
        [1, 2, 2],
        [2, 3, 3],
        [3, 7, 15],
    ],
    dtype=np.int64,
)
_SID_COUNTS = [3, 1, 1, 1, 1]
_LAYER_USAGE = [
    [3, 2, 1, 1],
    [3, 1, 1, 1, 1],
    [3, 1, 1, 1, 1],
]


class SidQualityMathTest(unittest.TestCase):
    def test_compute_gini(self) -> None:
        self.assertAlmostEqual(compute_gini([2, 2, 2, 2]), 0.0)
        self.assertAlmostEqual(compute_gini([3, 2, 1, 1, 1]), 0.25)
        self.assertEqual(compute_gini([]), 0.0)
        self.assertEqual(compute_gini([0, 0]), 0.0)

    def test_compute_entropy(self) -> None:
        self.assertEqual(compute_entropy([]), 0.0)
        self.assertAlmostEqual(compute_entropy([5, 5, 5, 5]), float(np.log(4)))
        self.assertAlmostEqual(
            compute_entropy([3, 2, 1, 1, 1]), 1.4941751382893085, places=10
        )
        self.assertAlmostEqual(
            compute_entropy([3, 0, 2, 0, 1, 1, 1]),
            compute_entropy([3, 2, 1, 1, 1]),
            places=10,
        )

    def test_valid_code_rows(self) -> None:
        codes = np.asarray([[0, 0], [1, 2], [-1, 0], [0, 3]], dtype=np.int64)
        np.testing.assert_array_equal(
            valid_code_rows(codes, [2, 3]), [True, True, False, False]
        )

    def test_valid_code_rows_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            valid_code_rows(np.empty((0, 0), dtype=np.int64), [])
        with self.assertRaisesRegex(ValueError, "positive"):
            valid_code_rows(np.empty((0, 2), dtype=np.int64), [2, 0])
        with self.assertRaisesRegex(ValueError, "integers"):
            valid_code_rows(np.empty((0, 2), dtype=np.int64), [2.0, 3.0])
        with self.assertRaisesRegex(ValueError, "shape"):
            valid_code_rows(np.asarray([1, 2]), [2, 3])
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            valid_code_rows(np.asarray([[0.0, 1.0]]), [2, 3])
        with self.assertRaisesRegex(ValueError, "fit int64"):
            valid_code_rows(
                np.asarray([[np.iinfo(np.uint64).max]], dtype=np.uint64), [2]
            )


class SidQualityAccumulatorTest(unittest.TestCase):
    def test_accumulator_matches_existing_metric_golden(self) -> None:
        accumulator = SidQualityAccumulator(_CODEBOOK, top_sids=5)
        accumulator.update(_ROWS[:3])
        accumulator.update(_ROWS[3:])
        self.assertEqual(accumulator.total, 7)

        result = accumulator.finalize()
        metrics = result.metrics
        self.assertEqual(metrics.total, 7)
        self.assertEqual(metrics.unique_sid, 5)
        self.assertAlmostEqual(metrics.no_collision_rate, 5 / 7)
        self.assertAlmostEqual(metrics.uniquely_identified_item_rate, 4 / 7)
        self.assertEqual(metrics.max_collision, 3)
        self.assertAlmostEqual(metrics.gini, compute_gini(_SID_COUNTS))
        entropy = compute_entropy(_SID_COUNTS)
        max_entropy = math.log(math.prod(_CODEBOOK))
        self.assertAlmostEqual(metrics.entropy, entropy)
        self.assertAlmostEqual(metrics.max_entropy, max_entropy)
        self.assertAlmostEqual(metrics.entropy_ratio, entropy / max_entropy)

        self.assertEqual(
            [layer.codebook_size for layer in result.layer_metrics], _CODEBOOK
        )
        self.assertEqual(
            [layer.coverage for layer in result.layer_metrics],
            [1.0, 0.625, 0.3125],
        )
        self.assertEqual(
            [layer.dead_codes for layer in result.layer_metrics], [0, 3, 11]
        )
        for layer, usage in zip(result.layer_metrics, _LAYER_USAGE):
            self.assertAlmostEqual(layer.perplexity, math.exp(compute_entropy(usage)))
        self.assertEqual(
            result.top_sids,
            (
                ("0,0,0", 3),
                ("1,1,1", 1),
                ("1,2,2", 1),
                ("2,3,3", 1),
                ("3,7,15", 1),
            ),
        )

    def test_mixed_radix_encoding_is_bijective(self) -> None:
        rows = np.asarray(
            [
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 0],
                [0, 1, 1],
                [1, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [3, 7, 15],
            ],
            dtype=np.int64,
        )
        accumulator = SidQualityAccumulator(_CODEBOOK)
        accumulator.update(rows)
        result = accumulator.finalize()
        self.assertEqual(result.metrics.unique_sid, 8)
        self.assertEqual(result.metrics.max_collision, 1)
        self.assertEqual(result.metrics.no_collision_rate, 1.0)

    def test_accumulator_accepts_empty_batches(self) -> None:
        accumulator = SidQualityAccumulator([2, 2])
        accumulator.update(np.empty((0, 2), dtype=np.int64))
        accumulator.update(np.asarray([[1, 1]], dtype=np.int64))
        self.assertEqual(accumulator.finalize().metrics.total, 1)

    def test_accumulator_can_skip_a_completed_range_scan(self) -> None:
        accumulator = SidQualityAccumulator([2, 2])
        codes = np.asarray([[0, 1], [1, 0]], dtype=np.int64)

        with mock.patch.object(quality, "_code_rows_in_range") as in_range:
            accumulator.update(codes, assume_in_range=True)

        in_range.assert_not_called()
        self.assertEqual(accumulator.finalize().metrics.total, 2)

    def test_accumulator_rejects_invalid_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            SidQualityAccumulator([])
        with self.assertRaisesRegex(ValueError, "positive"):
            SidQualityAccumulator([2, 0])
        with self.assertRaisesRegex(ValueError, "exceeds int64"):
            SidQualityAccumulator([3037000500, 3037000500])
        with self.assertRaisesRegex(ValueError, "top_sids must be positive"):
            SidQualityAccumulator([2], top_sids=0)

    def test_accumulator_rejects_invalid_updates(self) -> None:
        accumulator = SidQualityAccumulator([2, 3])
        with self.assertRaisesRegex(ValueError, "shape"):
            accumulator.update(np.asarray([0, 1], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "integer dtype"):
            accumulator.update(np.asarray([[0.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, "outside"):
            accumulator.update(np.asarray([[0, 3]], dtype=np.int64))

    def test_accumulator_finalization_is_one_shot(self) -> None:
        empty_accumulator = SidQualityAccumulator([2])
        with self.assertRaisesRegex(ValueError, "no valid"):
            empty_accumulator.finalize()

        accumulator = SidQualityAccumulator([2])
        accumulator.update(np.asarray([[0]], dtype=np.int64))
        accumulator.finalize()
        with self.assertRaisesRegex(ValueError, "already"):
            accumulator.finalize()
        with self.assertRaisesRegex(ValueError, "finalized"):
            accumulator.update(np.asarray([[1]], dtype=np.int64))

    def test_unit_capacity_entropy_ratio_is_nan(self) -> None:
        accumulator = SidQualityAccumulator([1])
        accumulator.update(np.asarray([[0], [0]], dtype=np.int64))
        result = accumulator.finalize()
        self.assertTrue(math.isnan(result.metrics.entropy_ratio))
        self.assertEqual(result.layer_metrics[0].coverage, 1.0)


class CompareSidQualityTest(unittest.TestCase):
    def test_compare_subtracts_every_metric_and_preserves_layer_metadata(
        self,
    ) -> None:
        before_codes = np.asarray(
            [[0, 0, 0], [0, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=np.int64
        )
        after_codes = np.asarray(
            [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]], dtype=np.int64
        )
        before_accumulator = SidQualityAccumulator([2, 2, 2])
        after_accumulator = SidQualityAccumulator([2, 2, 2])
        before_accumulator.update(before_codes)
        after_accumulator.update(after_codes)
        before = before_accumulator.finalize()
        after = after_accumulator.finalize()

        delta = compare_sid_quality(before, after)
        for metric_field in fields(delta.metrics):
            before_value = getattr(before.metrics, metric_field.name)
            after_value = getattr(after.metrics, metric_field.name)
            delta_value = getattr(delta.metrics, metric_field.name)
            self.assertAlmostEqual(delta_value, after_value - before_value)
        self.assertIsNone(delta.top_sids)
        for layer_index, layer_delta in enumerate(delta.layer_metrics):
            self.assertEqual(layer_delta.layer, layer_index)
            self.assertEqual(layer_delta.codebook_size, 2)
        self.assertEqual(delta.layer_metrics[0].coverage, 0.0)
        self.assertEqual(delta.layer_metrics[0].dead_codes, 0)
        self.assertEqual(delta.layer_metrics[0].perplexity, 0.0)
        self.assertEqual(delta.layer_metrics[1].coverage, 0.0)
        self.assertEqual(delta.layer_metrics[1].dead_codes, 0)
        self.assertEqual(delta.layer_metrics[1].perplexity, 0.0)

    def test_compare_rejects_different_cohort_sizes(self) -> None:
        before_accumulator = SidQualityAccumulator([2])
        after_accumulator = SidQualityAccumulator([2])
        before_accumulator.update(np.asarray([[0]], dtype=np.int64))
        after_accumulator.update(np.asarray([[0], [1]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "same item cohort"):
            compare_sid_quality(
                before_accumulator.finalize(), after_accumulator.finalize()
            )

    def test_compare_rejects_different_codebooks(self) -> None:
        before_accumulator = SidQualityAccumulator([2])
        after_accumulator = SidQualityAccumulator([3])
        before_accumulator.update(np.asarray([[0]], dtype=np.int64))
        after_accumulator.update(np.asarray([[0]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "different codebooks"):
            compare_sid_quality(
                before_accumulator.finalize(), after_accumulator.finalize()
            )

    def test_compare_rejects_different_layer_counts(self) -> None:
        before_accumulator = SidQualityAccumulator([2])
        after_accumulator = SidQualityAccumulator([2, 2])
        before_accumulator.update(np.asarray([[0]], dtype=np.int64))
        after_accumulator.update(np.asarray([[0, 0]], dtype=np.int64))
        with self.assertRaisesRegex(ValueError, "different layer counts"):
            compare_sid_quality(
                before_accumulator.finalize(), after_accumulator.finalize()
            )


if __name__ == "__main__":
    unittest.main()
