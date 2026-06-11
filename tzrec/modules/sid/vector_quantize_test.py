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
    VectorQuantize,
    _squared_euclidean_distance,
)


class SquaredEuclideanDistanceTest(unittest.TestCase):
    """Tests for the squared-L2 distance helper used by VectorQuantize."""

    def test_squared_euclidean_distance(self) -> None:
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        y = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        d = _squared_euclidean_distance(x, y)
        self.assertEqual(d.shape, (2, 2))
        # row0: dist to (0,0)=0, to (0,1)=1; row1: to (0,0)=1, to (0,1)=2
        torch.testing.assert_close(d, torch.tensor([[0.0, 1.0], [1.0, 2.0]]))


class VectorQuantizeTest(unittest.TestCase):
    """Tests for a single VectorQuantize layer."""

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
        vq = VectorQuantize(
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

    def test_sinkhorn_gumbel_combo_rejected(self) -> None:
        """Sinkhorn + Gumbel would desync `ids` and `emb`; constructor rejects it."""
        with self.assertRaisesRegex(AssertionError, "GUMBEL_SOFTMAX"):
            VectorQuantize(
                embed_dim=8,
                n_embed=16,
                forward_mode=QuantizeForwardMode.GUMBEL_SOFTMAX,
                use_sinkhorn=True,
            )

    def test_train_forward_backward_reaches_input(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantize(embed_dim=8, n_embed=16, use_sinkhorn=False)
        vq.train()
        x = torch.randn(5, 8, requires_grad=True)
        out = vq.quantize(x)
        out.embeddings.sum().backward()
        # STE routes gradient back through x.
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_eval_forward_is_plain_lookup(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantize(embed_dim=4, n_embed=8)
        vq.eval()
        x = torch.randn(3, 4)
        out = vq.quantize(x)
        # In eval, emb == embedding(ids) exactly.
        torch.testing.assert_close(out.embeddings, vq.embedding(out.ids))


if __name__ == "__main__":
    unittest.main()
