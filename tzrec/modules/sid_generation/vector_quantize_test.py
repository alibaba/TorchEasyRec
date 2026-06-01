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

from tzrec.modules.sid_generation.types import QuantizeForwardMode
from tzrec.modules.sid_generation.vector_quantize import VectorQuantize


class VectorQuantizeTest(unittest.TestCase):
    """Tests for a single VectorQuantize layer."""

    @parameterized.expand(
        [
            ("ste_l2", QuantizeForwardMode.STE, "l2", True),
            ("ste_cosine", QuantizeForwardMode.STE, "cosine", True),
            ("ste_no_sinkhorn", QuantizeForwardMode.STE, "l2", False),
            ("gumbel_l2", QuantizeForwardMode.GUMBEL_SOFTMAX, "l2", True),
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
        out = vq(x)
        self.assertEqual(out.embeddings.shape, (5, 8))
        self.assertEqual(out.ids.shape, (5,))
        self.assertTrue((out.ids >= 0).all() and (out.ids < 16).all())
        self.assertTrue(torch.isfinite(out.embeddings).all())

    def test_train_forward_backward_reaches_input(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantize(embed_dim=8, n_embed=16, use_sinkhorn=False)
        vq.train()
        x = torch.randn(5, 8, requires_grad=True)
        out = vq(x)
        out.embeddings.sum().backward()
        # STE routes gradient back through x.
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_eval_forward_is_plain_lookup(self) -> None:
        torch.manual_seed(0)
        vq = VectorQuantize(embed_dim=4, n_embed=8)
        vq.eval()
        x = torch.randn(3, 4)
        out = vq(x)
        # In eval, emb == embedding(ids) exactly.
        torch.testing.assert_close(out.embeddings, vq.embedding(out.ids))


if __name__ == "__main__":
    unittest.main()
