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
from torch import nn

from tzrec.modules.sid.residual_kmeans_quantizer import (
    ResidualKMeansQuantizer,
)
from tzrec.modules.sid.residual_quantizer import (
    ResidualQuantizer,
    normalize_n_embed,
)
from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
from tzrec.modules.sid.types import ResidualQuantizerOutput


class NormalizeNEmbedTest(unittest.TestCase):
    def test_scalar_broadcasts(self) -> None:
        self.assertEqual(normalize_n_embed(256, 3), [256, 256, 256])

    def test_list_passes_through(self) -> None:
        self.assertEqual(normalize_n_embed([8, 4, 2], 3), [8, 4, 2])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(AssertionError):
            normalize_n_embed([8, 4], 3)


class ResidualQuantizerBaseTest(unittest.TestCase):
    """The abstract base owns shared state but not the backend primitives."""

    def test_shared_state_and_output_dim(self) -> None:
        rq = ResidualQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertEqual(rq.output_dim(), 4)
        self.assertEqual(rq.n_embed_list, [8, 8])
        self.assertEqual(len(rq.layers), 0)  # subclasses populate this

    def test_abstract_primitives_raise(self) -> None:
        rq = ResidualQuantizer(embed_dim=4, n_layers=2)
        x = torch.randn(3, 4)
        with self.assertRaises(NotImplementedError):
            rq.forward(x)
        with self.assertRaises(NotImplementedError):
            rq.get_codes(x)
        with self.assertRaises(NotImplementedError):
            rq.get_codebook_embeddings(0)
        # decode_codes is concrete but delegates to the abstract _lookup_code.
        with self.assertRaises(NotImplementedError):
            rq.decode_codes(torch.zeros(3, 2, dtype=torch.long))


class _FakeQuantizer(ResidualQuantizer):
    """Minimal concrete subclass to exercise the base residual walk.

    Implements only the two per-layer primitives over a learnable codebook,
    so the base's _residual_pass / get_codes / decode_codes can be tested
    without pulling in the K-Means or VQ backends.
    """

    def __init__(self, embed_dim, n_layers, n_embed=5, normalize_residuals=False):
        super().__init__(embed_dim, n_layers, n_embed, normalize_residuals)
        self.books = nn.ParameterList(
            [
                nn.Parameter(torch.randn(self.n_embed_list[i], embed_dim))
                for i in range(n_layers)
            ]
        )

    def _quantize_layer(self, layer_idx, residual, temperature=1.0):
        codes = (residual.detach() @ self.books[layer_idx].t()).argmax(dim=-1)
        return codes, self.books[layer_idx][codes]

    def _lookup_code(self, layer_idx, code_idx):
        return self.books[layer_idx][code_idx]

    def forward(self, input):
        return self._residual_pass(input)

    def get_codebook_embeddings(self, layer_idx):
        return self.books[layer_idx]


class ResidualQuantizerWalkTest(unittest.TestCase):
    """Exercise the concrete residual walk the base owns (via a fake backend)."""

    def test_residual_pass_shapes_and_aggregate(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=3, n_embed=5)
        x = torch.randn(6, 4)
        ids, agg, cum = fq._residual_pass(x)
        self.assertEqual(ids.shape, (6, 3))
        self.assertEqual(fq.get_codes(x).shape, (6, 3))
        manual = sum(fq._lookup_code(i, ids[:, i]) for i in range(3))
        torch.testing.assert_close(agg, manual)  # aggregated == Σ quantized_i
        self.assertTrue(torch.equal(cum[-1], agg))

    def test_detach_invariant(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=5)
        x = torch.randn(5, 4, requires_grad=True)
        _, agg, _ = fq._residual_pass(x)
        # Codebook grad flows, but the residual chain is detached, so the
        # input receives no gradient.
        self.assertTrue(agg.requires_grad)
        agg.sum().backward()
        self.assertIsNotNone(fq.books[0].grad)
        self.assertIsNone(x.grad)

    def test_normalize_residuals_changes_assignment(self) -> None:
        # Same input and same codebook (re-seeded before each build), so the
        # only difference is the normalize_residuals branch — it must change
        # the residual a later layer sees and hence the codes it assigns.
        x = torch.randn(8, 4)
        torch.manual_seed(1)
        fq_off = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=6)
        torch.manual_seed(1)
        fq_on = _FakeQuantizer(
            embed_dim=4, n_layers=2, n_embed=6, normalize_residuals=True
        )
        ids_off, _, _ = fq_off._residual_pass(x)
        ids_on, _, _ = fq_on._residual_pass(x)
        self.assertEqual(ids_on.shape, (8, 2))
        self.assertFalse(torch.equal(ids_off, ids_on))

    def test_decode_codes_sum_and_dtype(self) -> None:
        torch.manual_seed(0)
        fq = _FakeQuantizer(embed_dim=4, n_layers=3, n_embed=5)
        codes = torch.randint(0, 5, (6, 3))
        recon = fq.decode_codes(codes)
        self.assertEqual(recon.shape, (6, 4))
        manual = sum(fq.books[i][codes[:, i]] for i in range(3))
        torch.testing.assert_close(recon, manual)
        # device/dtype follow the codebook (regression for the fp32-pin fix).
        fq16 = _FakeQuantizer(embed_dim=4, n_layers=2, n_embed=5).to(torch.bfloat16)
        recon16 = fq16.decode_codes(torch.randint(0, 5, (3, 2)))
        self.assertEqual(recon16.dtype, torch.bfloat16)


class ResidualVectorQuantizerTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.rvq = ResidualVectorQuantizer(
            embed_dim=8, n_layers=3, n_embed=16, kmeans_init=False
        )

    def test_is_subclass(self) -> None:
        self.assertIsInstance(self.rvq, ResidualQuantizer)

    def test_forward_output(self) -> None:
        self.rvq.train()
        out = self.rvq(torch.randn(5, 8))
        self.assertIsInstance(out, ResidualQuantizerOutput)
        self.assertEqual(out.cluster_ids.shape, (5, 3))
        self.assertEqual(out.quantized_embeddings.shape, (5, 8))
        self.assertTrue(torch.isfinite(out.quantization_loss).all())

    def test_decode_codes_shared_base(self) -> None:
        codes = torch.randint(0, 16, (5, 3))
        recon = self.rvq.decode_codes(codes)
        self.assertEqual(recon.shape, (5, 8))

    def test_get_codes_no_grad(self) -> None:
        codes = self.rvq.get_codes(torch.randn(4, 8))
        self.assertEqual(codes.shape, (4, 3))

    def test_forward_get_codes_consistent_eval(self) -> None:
        """get_codes (shared base walk) matches forward's ids in eval."""
        self.rvq.eval()
        x = torch.randn(6, 8)
        fwd_ids = self.rvq(x).cluster_ids
        gc_ids = self.rvq.get_codes(x)
        self.assertFalse(gc_ids.requires_grad)
        torch.testing.assert_close(gc_ids, fwd_ids)

    def test_faiss_kmeans_init_seeds_codebook(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8, n_layers=2, n_embed=16, kmeans_init=True
        )
        self.assertFalse(bool(rvq.initted.item()))
        rvq.train()
        # First training forward triggers the FAISS warm-start.
        rvq(torch.randn(512, 8))
        self.assertTrue(bool(rvq.initted.item()))
        for layer in rvq.layers:
            self.assertTrue(torch.isfinite(layer.embedding.weight).all())
            self.assertGreater(layer.embedding.weight.abs().sum().item(), 0.0)


class ResidualKMeansQuantizerTest(unittest.TestCase):
    def test_is_subclass(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertIsInstance(rkq, ResidualQuantizer)

    def test_non_uniform_codebook_supported(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=3, n_embed=[8, 4, 16])
        self.assertEqual(rkq.n_embed_list, [8, 4, 16])
        self.assertEqual([layer.centroids.shape[0] for layer in rkq.layers], [8, 4, 16])

    def test_forward_returns_zeros_before_fit(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertFalse(all(layer.is_initialized for layer in rkq.layers))
        codes, quantized = rkq(torch.randn(5, 4))
        self.assertEqual(codes.shape, (5, 2))
        self.assertEqual(quantized.shape, (5, 4))

    def test_forward_is_fx_traceable(self) -> None:
        """Predict forward must FX-trace.

        torchrec's inference pipeline symbolically traces the model, so the
        per-batch distance path must be free of data-dependent control flow.
        """
        import torch.fx as fx

        torch.manual_seed(0)
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        for layer in rkq.layers:  # populate centroids -> is_initialized=True
            layer.load_centroids_(torch.randn(8, 4))
        traced = fx.symbolic_trace(rkq)
        x = torch.randn(5, 4)
        c_eager, q_eager = rkq(x)
        c_traced, q_traced = traced(x)
        torch.testing.assert_close(c_traced, c_eager)
        torch.testing.assert_close(q_traced, q_eager)

    def test_train_offline_non_uniform(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        n_embed = [8, 4, 16]
        rkq = ResidualKMeansQuantizer(
            embed_dim=4, n_layers=3, n_embed=n_embed, faiss_kmeans_kwargs={"niter": 5}
        )
        rkq.train_offline(torch.randn(512, 4), verbose=False)
        self.assertTrue(all(layer.is_initialized for layer in rkq.layers))
        # Each layer fit its own K centroids; codes stay in per-layer range.
        codes, _ = rkq(torch.randn(7, 4))
        self.assertEqual(codes.shape, (7, 3))
        for i, k in enumerate(n_embed):
            self.assertTrue((codes[:, i] >= 0).all() and (codes[:, i] < k).all())

    def test_train_offline_then_decode(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        rkq = ResidualKMeansQuantizer(
            embed_dim=4, n_layers=2, n_embed=8, faiss_kmeans_kwargs={"niter": 5}
        )
        rkq.train_offline(torch.randn(256, 4), verbose=False)
        self.assertTrue(all(layer.is_initialized for layer in rkq.layers))

        codes, _ = rkq(torch.randn(5, 4))
        self.assertTrue((codes >= 0).all() and (codes < 8).all())
        recon = rkq.decode_codes(codes)  # inherited from the base
        self.assertEqual(recon.shape, (5, 4))

    def test_forward_get_codes_consistent(self) -> None:
        """Forward ids and get_codes both route through the shared walk."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        rkq = ResidualKMeansQuantizer(
            embed_dim=4, n_layers=3, n_embed=8, faiss_kmeans_kwargs={"niter": 5}
        )
        rkq.train_offline(torch.randn(256, 4), verbose=False)
        x = torch.randn(9, 4)
        fwd_ids, fwd_quant = rkq(x)
        torch.testing.assert_close(rkq.get_codes(x), fwd_ids)
        # forward's residual-sum equals the centroid-sum reconstruction.
        torch.testing.assert_close(fwd_quant, rkq.decode_codes(fwd_ids))


if __name__ == "__main__":
    unittest.main()
