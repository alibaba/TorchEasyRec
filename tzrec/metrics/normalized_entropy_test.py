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

import numpy as np
import torch
from parameterized import parameterized

from tzrec.metrics.normalized_entropy import NormalizedEntropy


class NormalizedEntropyTest(unittest.TestCase):
    def test_constant_predictor_equals_one(self):
        target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
        p_avg = target.to(torch.float64).mean().item()
        preds = torch.full_like(target, fill_value=0.0, dtype=torch.float32)
        preds[:] = p_avg
        metric = NormalizedEntropy()
        metric.update(preds, target)
        torch.testing.assert_close(
            metric.compute(), torch.tensor(1.0), atol=1e-5, rtol=0
        )

    def test_perfect_classifier_is_small(self):
        target = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
        preds = target.to(torch.float32)
        metric = NormalizedEntropy()
        metric.update(preds, target)
        self.assertLess(metric.compute().item(), 1e-3)

    @parameterized.expand(
        [
            ("balanced_symmetric", [0.1, 0.4, 0.6, 0.9], [0, 0, 1, 1], 0.44448435),
            ("imbalanced_informed", [0.2, 0.5, 0.8], [0, 1, 1], 0.59670544),
            ("two_sample", [0.3, 0.7], [0, 1], 0.51457322),
        ]
    )
    def test_known_value(self, _name, preds, target, expected):
        metric = NormalizedEntropy()
        metric.update(
            torch.tensor(preds, dtype=torch.float32),
            torch.tensor(target, dtype=torch.long),
        )
        torch.testing.assert_close(
            metric.compute(),
            torch.tensor(expected, dtype=torch.float32),
            atol=1e-6,
            rtol=0,
        )

    @parameterized.expand([("all_zero", 0), ("all_one", 1)])
    def test_degenerate_label_is_finite(self, _name, label_value):
        target = torch.full((128,), label_value, dtype=torch.long)
        preds = torch.full((128,), 0.3, dtype=torch.float32)
        metric = NormalizedEntropy()
        metric.update(preds, target)
        value = metric.compute()
        self.assertTrue(torch.isfinite(value).item())

    def test_boundary_predictions_are_finite(self):
        target = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        preds = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=torch.float32)
        metric = NormalizedEntropy()
        metric.update(preds, target)
        self.assertTrue(torch.isfinite(metric.compute()).item())

    def test_multi_batch_accumulation_matches_single_shot(self):
        rng = np.random.default_rng(7)
        n = 2048
        target = torch.from_numpy(rng.integers(0, 2, size=n).astype(np.int64))
        preds = torch.from_numpy(rng.uniform(0.01, 0.99, size=n).astype(np.float32))

        full = NormalizedEntropy()
        full.update(preds, target)

        split = NormalizedEntropy()
        split.update(preds[: n // 2], target[: n // 2])
        split.update(preds[n // 2 :], target[n // 2 :])

        torch.testing.assert_close(split.compute(), full.compute(), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
