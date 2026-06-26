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

from tzrec.modules.sid.types import QuantizeForwardMode
from tzrec.modules.sid.vector_quantize import (
    VectorQuantizeLayer,
)


class VectorQuantizeTest(unittest.TestCase):
    """Tests for a single VectorQuantizeLayer layer."""

    def test_l2_compute_distances(self) -> None:
        layer = VectorQuantizeLayer(embed_dim=2, n_embed=2, distance_type="l2")
        # Pin the codebook to (0,0) and (0,1) so distances are exact.
        layer.embedding.weight.data.copy_(torch.tensor([[0.0, 0.0], [0.0, 1.0]]))
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        d = layer._compute_distances(x)
        self.assertEqual(d.shape, (2, 2))
        # row0: dist² to (0,0)=0, to (0,1)=1; row1: to (0,0)=1, to (0,1)=2
        torch.testing.assert_close(d, torch.tensor([[0.0, 1.0], [1.0, 2.0]]))

    @parameterized.expand(
        [
            ("ste_l2", QuantizeForwardMode.STE, "l2", True),
            ("ste_cosine", QuantizeForwardMode.STE, "cosine", True),
            ("ste_no_sinkhorn", QuantizeForwardMode.STE, "l2", False),
            # Gumbel must run without Sinkhorn (the combo is asserted against).
            ("gumbel_l2", QuantizeForwardMode.GUMBEL_SOFTMAX, "l2", False),
        ]
    )
    def test_train_forward(self, _name, mode, distance_type, use_sinkhorn) -> None:
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(
            embed_dim=8,
            n_embed=16,
            forward_mode=mode,
            distance_type=distance_type,
            use_sinkhorn=use_sinkhorn,
        )
        vq.train()
        x = torch.randn(5, 8, requires_grad=True)
        out = vq.quantize(x)
        self.assertEqual(out.embeddings.shape, (5, 8))
        self.assertEqual(out.ids.shape, (5,))
        self.assertTrue((out.ids >= 0).all() and (out.ids < 16).all())
        self.assertTrue(torch.isfinite(out.embeddings).all())

    def test_sinkhorn_balances_assignment(self) -> None:
        """Sinkhorn spreads clustered points across codes; argmin collapses them.

        Functional check (not just shape/finiteness): feed points clustered at
        one anchor — argmin sends all to that code, while Sinkhorn's uniform
        assignment must use more than one code.
        """
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(
            embed_dim=2, n_embed=4, use_sinkhorn=True, sinkhorn_iters=10
        )
        vq.train()
        with torch.no_grad():
            vq.embedding.weight.copy_(
                torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
            )
        x = torch.randn(16, 2) * 0.1  # all clustered at anchor 0
        sinkhorn_ids = vq.quantize(x).ids
        vq.use_sinkhorn = False
        argmin_ids = vq.quantize(x).ids
        self.assertEqual(argmin_ids.unique().numel(), 1)
        self.assertGreater(sinkhorn_ids.unique().numel(), 1)

    def test_sinkhorn_gumbel_combo_rejected(self) -> None:
        """Sinkhorn + Gumbel would desync `ids` and `emb`; constructor rejects it."""
        with self.assertRaisesRegex(AssertionError, "GUMBEL_SOFTMAX"):
            VectorQuantizeLayer(
                embed_dim=8,
                n_embed=16,
                forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
                use_sinkhorn=True,
            )

    def test_sinkhorn_epsilon_must_be_positive(self) -> None:
        """Reject a non-positive sinkhorn_epsilon (it overflows exp(-cost*eps))."""
        with self.assertRaisesRegex(ValueError, "sinkhorn_epsilon"):
            VectorQuantizeLayer(
                embed_dim=8, n_embed=16, use_sinkhorn=True, sinkhorn_epsilon=0.0
            )

    def test_train_forward_backward_reaches_codebook(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(embed_dim=8, n_embed=16, use_sinkhorn=False)
        vq.train()
        x = torch.randn(5, 8, requires_grad=True)
        out = vq.quantize(x)
        out.embeddings.sum().backward()
        # The layer returns the raw codebook vector, so gradient reaches the
        # codebook (the encoder STE is applied on the aggregate by the RVQ).
        self.assertIsNotNone(vq.embedding.weight.grad)
        self.assertTrue(torch.isfinite(vq.embedding.weight.grad).all())

    def test_eval_forward_is_plain_lookup(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(embed_dim=4, n_embed=8)
        vq.eval()
        x = torch.randn(3, 4)
        out = vq.quantize(x)
        # In eval, emb == embedding(ids) exactly.
        torch.testing.assert_close(out.embeddings, vq.embedding(out.ids))

    def test_gumbel_train_ids_match_embedding(self) -> None:
        # In gumbel training the saved code must index the codebook vector
        # actually used (the hard sample), so emb forward == embedding(ids).
        # (Under the old code ids came from argmin and could disagree with the
        # gumbel-sampled embedding.)
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(
            embed_dim=8,
            n_embed=16,
            forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
            use_sinkhorn=False,
        )
        vq.train()
        out = vq.quantize(torch.randn(5, 8))
        torch.testing.assert_close(out.embeddings, vq.embedding(out.ids))

    def test_gumbel_train_distances_are_differentiable(self) -> None:
        # Gumbel needs the assignment differentiable: grad must reach the input.
        torch.manual_seed(0)
        vq = VectorQuantizeLayer(
            embed_dim=8,
            n_embed=16,
            forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
            use_sinkhorn=False,
        )
        vq.train()
        x = torch.randn(5, 8, requires_grad=True)
        vq.quantize(x).embeddings.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
