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

from tzrec.metrics.relative_l1 import RelativeL1


class RelativeL1Test(unittest.TestCase):
    def test_zero_on_identity(self) -> None:
        metric = RelativeL1()
        x = torch.randn(8, 4)
        metric.update(x, x.clone())
        self.assertAlmostEqual(metric.compute().item(), 0.0, places=6)

    def test_matches_formula(self) -> None:
        metric = RelativeL1(epsilon=1e-4)
        p = torch.tensor([[1.0, 0.0]])
        t = torch.tensor([[0.0, 2.0]])
        # |t-p|/(max(|t|,|p|)+eps): [1/(1+eps), 2/(2+eps)], mean of the two.
        expected = (1.0 / (1.0 + 1e-4) + 2.0 / (2.0 + 1e-4)) / 2
        metric.update(p, t)
        self.assertAlmostEqual(metric.compute().item(), expected, places=5)

    def test_count_weighted_across_updates(self) -> None:
        """Aggregation is element-wise, not a mean of per-batch means."""
        metric = RelativeL1()
        metric.update(torch.zeros(1, 4), torch.ones(1, 4))  # 4 elems, rel ~1
        metric.update(torch.ones(3, 4), torch.ones(3, 4))  # 12 elems, rel 0
        # Element-weighted: 4 nonzero over 16 elems -> ~0.25, NOT (1+0)/2 = 0.5.
        per = 1.0 / (1.0 + 1e-4)  # rel of a 0-vs-1 element (with epsilon)
        self.assertAlmostEqual(metric.compute().item(), 4 * per / 16, places=6)

    def test_nan_before_update(self) -> None:
        self.assertTrue(torch.isnan(RelativeL1().compute()))


if __name__ == "__main__":
    unittest.main()
