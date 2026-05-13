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


import unittest

import torch
from parameterized import parameterized

from tzrec.metrics.normalized_entropy import NormalizedEntropy


class NormalizedEntropyTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "constant_pred",
                [0.5] * 8,
                [0, 1, 0, 1, 1, 0, 1, 0],
                1.0,
            ),
            (
                "perfect_classifier",
                [0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [0, 1, 0, 1, 1, 0, 1, 0],
                0.0,
            ),
            (
                "balanced_symmetric",
                [0.1, 0.4, 0.6, 0.9],
                [0, 0, 1, 1],
                0.44448435,
            ),
            (
                "imbalanced_informed",
                [0.2, 0.5, 0.8],
                [0, 1, 1],
                0.59670544,
            ),
            (
                "two_sample",
                [0.3, 0.7],
                [0, 1],
                0.51457322,
            ),
        ]
    )
    def test_ne(self, _name, preds, target, expected):
        p = torch.tensor(preds, dtype=torch.float32)
        t = torch.tensor(target, dtype=torch.long)
        expected_t = torch.tensor(expected, dtype=torch.float32)

        single = NormalizedEntropy()
        single.update(p, t)
        torch.testing.assert_close(single.compute(), expected_t, atol=1e-6, rtol=0)

        # Two-shot update covers accumulation invariance.
        split = NormalizedEntropy()
        mid = len(p) // 2
        split.update(p[:mid], t[:mid])
        split.update(p[mid:], t[mid:])
        torch.testing.assert_close(split.compute(), expected_t, atol=1e-6, rtol=0)

    @parameterized.expand([("all_zero", 0), ("all_one", 1)])
    def test_degenerate_label_is_finite(self, _name, label_value):
        target = torch.full((128,), label_value, dtype=torch.long)
        preds = torch.full((128,), 0.3, dtype=torch.float32)
        metric = NormalizedEntropy()
        metric.update(preds, target)
        self.assertTrue(torch.isfinite(metric.compute()).item())

    def test_boundary_predictions_are_finite(self):
        target = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        preds = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        metric = NormalizedEntropy()
        metric.update(preds, target)
        self.assertTrue(torch.isfinite(metric.compute()).item())


if __name__ == "__main__":
    unittest.main()
