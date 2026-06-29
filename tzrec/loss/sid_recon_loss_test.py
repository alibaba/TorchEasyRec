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
    """Tests for the reconstruction-loss module (per-row distance + reduction)."""

    def test_l2_is_per_row_mse(self) -> None:
        d = SidReconLoss("l2")._per_row(torch.ones(3, 4), torch.zeros(3, 4))
        self.assertEqual(d.shape, (3,))
        torch.testing.assert_close(d, torch.ones(3))  # mean of 1^2 over dim -1

    def test_l1_is_per_row_mae(self) -> None:
        d = SidReconLoss("l1")._per_row(torch.ones(2, 5), torch.zeros(2, 5))
        torch.testing.assert_close(d, torch.ones(2))

    def test_cos_is_one_minus_cosine(self) -> None:
        x = torch.tensor([[1.0, 0.0]])
        # identical vectors -> cosine 1 -> distance 0
        d = SidReconLoss("cos")._per_row(x, x.clone())
        torch.testing.assert_close(d, torch.zeros(1), atol=1e-6, rtol=0)

    @parameterized.expand([("l2",), ("l1",), ("cos",)])
    def test_each_type_scalar_and_backprops(self, recon_type) -> None:
        x_hat = torch.randn(4, 6, requires_grad=True)
        loss = SidReconLoss(recon_type)(x_hat, torch.randn(4, 6))
        self.assertEqual(loss.shape, ())  # forward reduces to a scalar
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # grad must flow back to the (decoder) input
        self.assertIsNotNone(x_hat.grad)

    def test_unknown_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "recon_type"):
            SidReconLoss("nope")

    # --- masked-mean reduction (forward's mask handling) ---

    def test_no_mask_is_plain_mean(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(SidReconLoss._masked_mean(x), x.mean())

    def test_mask_averages_over_valid_rows_only(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        torch.testing.assert_close(
            SidReconLoss._masked_mean(x, mask), torch.tensor(2.0)
        )  # (1+3)/2

    def test_empty_mask_is_zero_not_nan(self) -> None:
        out = SidReconLoss._masked_mean(
            torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3, dtype=torch.bool)
        )
        self.assertEqual(out.item(), 0.0)

    def test_forward_applies_mask(self) -> None:
        # l1 per-row of [[1,1],[2,2],[3,3],[4,4]] vs 0 is [1,2,3,4]; mask keeps 0,2.
        x_hat = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
        mask = torch.tensor([True, False, True, False])
        loss = SidReconLoss("l1")(x_hat, torch.zeros(4, 2), mask)
        torch.testing.assert_close(loss, torch.tensor(2.0))  # (1+3)/2


if __name__ == "__main__":
    unittest.main()
