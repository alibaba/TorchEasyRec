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

from tzrec.loss.sid_recon_loss import SidReconLoss


class SidReconLossTest(unittest.TestCase):
    """Tests for the per-row reconstruction-distance module."""

    def test_l2_is_per_row_mse(self) -> None:
        d = SidReconLoss("l2")(torch.ones(3, 4), torch.zeros(3, 4))
        self.assertEqual(d.shape, (3,))
        torch.testing.assert_close(d, torch.ones(3))  # mean of 1^2 over dim -1

    def test_l1_is_per_row_mae(self) -> None:
        d = SidReconLoss("l1")(torch.ones(2, 5), torch.zeros(2, 5))
        torch.testing.assert_close(d, torch.ones(2))

    def test_cos_is_one_minus_cosine(self) -> None:
        x = torch.tensor([[1.0, 0.0]])
        # identical vectors -> cosine 1 -> distance 0
        d = SidReconLoss("cos")(x, x.clone())
        torch.testing.assert_close(d, torch.zeros(1), atol=1e-6, rtol=0)

    @parameterized.expand([("l2",), ("l1",), ("cos",)])
    def test_each_type_finite_and_backprops(self, recon_type) -> None:
        x_hat = torch.randn(4, 6, requires_grad=True)
        loss = SidReconLoss(recon_type)(x_hat, torch.randn(4, 6)).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # grad must flow back to the (decoder) input
        self.assertIsNotNone(x_hat.grad)

    def test_unknown_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "recon_type"):
            SidReconLoss("nope")


if __name__ == "__main__":
    unittest.main()
