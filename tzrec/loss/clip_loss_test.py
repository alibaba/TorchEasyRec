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

from tzrec.loss.clip_loss import MaskedCLIPLoss


class AllGatherWithGradTest(unittest.TestCase):
    def test_single_process_identity(self) -> None:
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        out = MaskedCLIPLoss._all_gather_with_grad([a, b])
        self.assertIs(out[0], a)
        self.assertIs(out[1], b)


class MaskedCLIPLossTest(unittest.TestCase):
    """Single-process tests for the masked CLIP loss."""

    def _features(self, B: int, D: int) -> dict:
        torch.manual_seed(0)
        scale = torch.tensor(np.log(1 / 0.07)).exp()
        return {
            "image_embed": torch.randn(B, D, requires_grad=True),
            "text_embed": torch.randn(B, D, requires_grad=True),
            "image_embed_ori": torch.randn(B, D),
            "text_embed_ori": torch.randn(B, D),
            "logit_scale_self": scale,
            "logit_scale_cl": scale,
            "logit_scale": scale,
        }

    def test_forward_all_clip_finite(self) -> None:
        loss_fn = MaskedCLIPLoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        out = loss_fn(feats, mask)
        self.assertIn("clip_loss", out)
        self.assertTrue(torch.isfinite(out["clip_loss"]))
        self.assertGreater(out["clip_loss"].item(), 0.0)

    def test_all_recon_mask_zero_loss(self) -> None:
        loss_fn = MaskedCLIPLoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)  # no clip rows
        out = loss_fn(feats, mask)
        # No clip rows -> masked average is exactly zero (and finite).
        self.assertTrue(torch.isfinite(out["clip_loss"]))
        self.assertAlmostEqual(out["clip_loss"].item(), 0.0, places=6)

    def test_all_recon_mask_finite_gradient(self) -> None:
        # Regression: with float("-inf") column fill an all-recon batch produced
        # a NaN gradient (0 * NaN) that survived the row mask. The finite fill
        # must keep the backward finite (and zero, since no clip row contributes).
        loss_fn = MaskedCLIPLoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)
        loss_fn(feats, mask)["clip_loss"].backward()
        grad = feats["image_embed"].grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertAlmostEqual(grad.abs().sum().item(), 0.0, places=6)

    def test_backward_flows_to_embeddings(self) -> None:
        loss_fn = MaskedCLIPLoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        loss_fn(feats, mask)["clip_loss"].backward()
        self.assertIsNotNone(feats["image_embed"].grad)
        self.assertTrue(torch.isfinite(feats["image_embed"].grad).all())

    def test_mask_holds_under_large_scale(self) -> None:
        # The column fill is finfo.min (below any real logit) rather than a
        # hardcoded -1e4, so masking holds even when logit_scale is large and
        # the *_ori operands are un-normalized (real logits can dwarf 1e4).
        # Loss/grad must stay finite and acc valid; eval exercises the argmax.
        loss_fn = MaskedCLIPLoss()
        loss_fn.eval()
        feats = self._features(6, 8)
        big = torch.tensor(3000.0)
        feats["logit_scale"] = big
        feats["logit_scale_self"] = big
        feats["logit_scale_cl"] = big
        feats["image_embed_ori"] = feats["image_embed_ori"] * 50
        feats["text_embed_ori"] = feats["text_embed_ori"] * 50
        mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        out = loss_fn(feats, mask)
        self.assertTrue(torch.isfinite(out["clip_loss"]))
        loss_fn.train()
        feats["image_embed"].grad = None
        loss_fn(feats, mask)["clip_loss"].backward()
        self.assertTrue(torch.isfinite(feats["image_embed"].grad).all())


if __name__ == "__main__":
    unittest.main()
