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

from tzrec.models.sid_model import _masked_mean, recon_loss


class ReconLossTest(unittest.TestCase):
    """Tests for the shared ``BaseSidModel`` reconstruction-distance factory."""

    def test_l2_is_per_row_mse(self) -> None:
        d = recon_loss("l2")(torch.ones(3, 4), torch.zeros(3, 4))
        self.assertEqual(d.shape, (3,))
        torch.testing.assert_close(d, torch.ones(3))  # mean of 1^2 over dim -1

    def test_l1_is_per_row_mae(self) -> None:
        d = recon_loss("l1")(torch.ones(2, 5), torch.zeros(2, 5))
        torch.testing.assert_close(d, torch.ones(2))

    def test_cos_is_one_minus_cosine(self) -> None:
        x = torch.tensor([[1.0, 0.0]])
        # identical vectors -> cosine 1 -> distance 0
        d = recon_loss("cos")(x, x.clone())
        torch.testing.assert_close(d, torch.zeros(1), atol=1e-6, rtol=0)

    @parameterized.expand([("l2",), ("l1",), ("cos",)])
    def test_each_type_finite_and_backprops(self, recon_type) -> None:
        x_hat = torch.randn(4, 6, requires_grad=True)
        loss = recon_loss(recon_type)(x_hat, torch.randn(4, 6)).mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()  # grad must flow back to the (decoder) input
        self.assertIsNotNone(x_hat.grad)

    def test_unknown_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "recon_type"):
            recon_loss("nope")


class MaskedMeanTest(unittest.TestCase):
    """Tests for the shared ``BaseSidModel`` masked-mean reduction."""

    def test_no_mask_is_plain_mean(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        torch.testing.assert_close(_masked_mean(x), x.mean())

    def test_mask_averages_over_valid_rows_only(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([True, False, True, False])
        torch.testing.assert_close(_masked_mean(x, mask), torch.tensor(2.0))  # (1+3)/2

    def test_empty_mask_is_zero_not_nan(self) -> None:
        out = _masked_mean(
            torch.tensor([1.0, 2.0, 3.0]), torch.zeros(3, dtype=torch.bool)
        )
        self.assertEqual(out.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
