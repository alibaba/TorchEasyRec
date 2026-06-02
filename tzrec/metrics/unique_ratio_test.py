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

from tzrec.metrics.unique_ratio import UniqueRatio


class UniqueRatioTest(unittest.TestCase):
    def test_single_batch_ratio(self) -> None:
        metric = UniqueRatio()
        # 3 distinct rows out of 4 -> 0.75.
        metric.update(torch.tensor([[1, 2], [1, 2], [3, 4], [5, 6]]))
        self.assertAlmostEqual(metric.compute().item(), 0.75, places=6)

    def test_mean_over_batches(self) -> None:
        metric = UniqueRatio()
        metric.update(torch.tensor([[1, 1], [1, 1]]))  # 1/2 = 0.5
        metric.update(torch.tensor([[1, 1], [2, 2]]))  # 2/2 = 1.0
        # Per-batch mean = 0.75 (a global distinct/total would give 0.5).
        self.assertAlmostEqual(metric.compute().item(), 0.75, places=6)

    def test_empty_batch_skipped(self) -> None:
        metric = UniqueRatio()
        metric.update(torch.empty(0, 3, dtype=torch.long))
        self.assertEqual(metric.count.item(), 0.0)
        self.assertTrue(torch.isnan(metric.compute()))


if __name__ == "__main__":
    unittest.main()
