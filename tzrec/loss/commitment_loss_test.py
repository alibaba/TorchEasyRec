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

from tzrec.loss.commitment_loss import CommitmentLoss


class CommitmentLossTest(unittest.TestCase):
    """Tests for the standalone CommitmentLoss module."""

    @parameterized.expand([("l2",), ("l1",), ("cos",)])
    def test_branch_runs_and_backprops(self, commitment_type) -> None:
        """Each commitment_type runs end-to-end; grad reaches both operands."""
        torch.manual_seed(0)
        loss_fn = CommitmentLoss(
            latent_weight=(1.0, 0.5), commitment_type=commitment_type
        )
        B, L, D = 4, 3, 8
        encoder_out = torch.randn(B, D, requires_grad=True)
        latents = torch.randn(B, L, D, requires_grad=True)
        out = loss_fn(encoder_out, latents)
        self.assertEqual(out.shape, ())
        self.assertTrue(torch.isfinite(out))
        out.backward()
        # loss1 (encoder-toward-quant) feeds encoder_out; loss2 feeds latents.
        self.assertIsNotNone(encoder_out.grad)
        self.assertIsNotNone(latents.grad)
        self.assertTrue(torch.isfinite(encoder_out.grad).all())

    def test_latent_weight_wrong_length_raises(self) -> None:
        """latent_weight must be exactly [w1, w2]."""
        for bad in ([1.0], [1.0, 0.5, 0.25]):
            with self.assertRaisesRegex(ValueError, "latent_weight"):
                CommitmentLoss(latent_weight=bad)

    def test_invalid_commitment_type_raises(self) -> None:
        """An unknown commitment_type is rejected."""
        with self.assertRaisesRegex(AssertionError, "commitment_type"):
            CommitmentLoss(commitment_type="bogus")

    def test_weights_scale_the_two_directions(self) -> None:
        """w1/w2 weight the encoder-toward-quant / quant-toward-encoder terms."""
        torch.manual_seed(0)
        encoder_out = torch.randn(4, 8)
        latents = torch.randn(4, 3, 8)
        base = CommitmentLoss(latent_weight=(1.0, 0.5), commitment_type="l2")
        zero = CommitmentLoss(latent_weight=(0.0, 0.0), commitment_type="l2")
        self.assertGreater(base(encoder_out, latents).item(), 0.0)
        self.assertEqual(zero(encoder_out, latents).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
