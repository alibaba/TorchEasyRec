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

from tzrec.loss.clip_loss import (
    MaskedCLIPLoss,
    _all_gather_with_grad,
)


class AllGatherWithGradTest(unittest.TestCase):
    def test_single_process_identity(self) -> None:
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        out = _all_gather_with_grad([a, b])
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

    def test_backward_flows_to_embeddings(self) -> None:
        loss_fn = MaskedCLIPLoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        loss_fn(feats, mask)["clip_loss"].backward()
        self.assertIsNotNone(feats["image_embed"].grad)
        self.assertTrue(torch.isfinite(feats["image_embed"].grad).all())


if __name__ == "__main__":
    unittest.main()
