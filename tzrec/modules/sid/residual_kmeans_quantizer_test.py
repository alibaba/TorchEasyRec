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

from tzrec.modules.sid.residual_kmeans_quantizer import (
    ResidualKMeansQuantizer,
)
from tzrec.modules.sid.residual_quantizer import (
    ResidualQuantizer,
)


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
