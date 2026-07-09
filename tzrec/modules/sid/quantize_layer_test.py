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

from tzrec.modules.sid.quantize_layer import QuantizeLayer
from tzrec.modules.sid.types import QuantizeOutput


class _StubQuantizeLayer(QuantizeLayer):
    """Minimal concrete subclass: a fixed codebook, nearest-row assignment.

    Exercises the base class's concrete ``__init__`` / ``lookup`` without
    pulling in a backend (FAISS / nn.Embedding).
    """

    def __init__(self, n_embed: int, embed_dim: int) -> None:
        super().__init__(n_embed, embed_dim)
        # A deterministic codebook so lookup/quantize are checkable by hand.
        self._codebook = torch.arange(n_embed * embed_dim, dtype=torch.float32).reshape(
            n_embed, embed_dim
        )

    def quantize(self, x: torch.Tensor, topk: int = 1) -> QuantizeOutput:
        dist = torch.cdist(x, self._codebook)
        if not self.training:
            return self._topk_output(dist, topk)
        ids = dist.argmin(dim=-1)
        return QuantizeOutput(embeddings=self.lookup(ids), ids=ids)

    def get_codebook_embeddings(self) -> torch.Tensor:
        return self._codebook


class QuantizeLayerTest(unittest.TestCase):
    """Tests for the shared QuantizeLayer base class."""

    def setUp(self) -> None:
        self.layer = _StubQuantizeLayer(n_embed=4, embed_dim=3)

    def test_init_stores_codebook_shape(self) -> None:
        self.assertEqual(self.layer.n_embed, 4)
        self.assertEqual(self.layer.embed_dim, 3)

    def test_lookup_gathers_codebook_rows(self) -> None:
        ids = torch.tensor([0, 2, 3, 1])
        out = self.layer.lookup(ids)
        torch.testing.assert_close(out, self.layer.get_codebook_embeddings()[ids])
        self.assertEqual(out.shape, (4, 3))

    def test_quantize_assigns_exact_codebook_rows(self) -> None:
        # Feeding codebook rows back in must recover their own indices.
        x = self.layer.get_codebook_embeddings().clone()
        out = self.layer.quantize(x)
        torch.testing.assert_close(out.ids, torch.arange(4))
        torch.testing.assert_close(out.embeddings, x)

    def test_quantize_returns_topk_neighbors(self) -> None:
        layer = _StubQuantizeLayer(n_embed=4, embed_dim=1)
        layer.eval()
        x = torch.tensor([[1.1], [2.9]])
        out = layer.quantize(x, topk=2)
        torch.testing.assert_close(out.ids, torch.tensor([1, 3]))
        torch.testing.assert_close(out.topk_ids, torch.tensor([[1, 2], [3, 2]]))
        torch.testing.assert_close(
            out.topk_scores,
            torch.tensor([[0.1, 0.9], [0.1, 0.9]]),
        )

    def test_abstract_methods_unoverridden_raise(self) -> None:
        # The abstract methods are documented to raise if a subclass forgets
        # to implement them; QuantizeLayer relies on nn.Module (no ABCMeta),
        # so this guards that the bodies still fail loudly rather than no-op.
        class _Incomplete(QuantizeLayer):
            pass

        layer = _Incomplete(n_embed=2, embed_dim=2)
        with self.assertRaises(NotImplementedError):
            layer.get_codebook_embeddings()
        with self.assertRaises(NotImplementedError):
            layer.quantize(torch.zeros(1, 2))


if __name__ == "__main__":
    unittest.main()
