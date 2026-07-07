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
from tzrec.modules.sid.types import ResidualQuantizerOutput


class ResidualKMeansQuantizerTest(unittest.TestCase):
    def test_is_subclass(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertIsInstance(rkq, ResidualQuantizer)

    def test_non_uniform_codebook_supported(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=3, n_embed=[8, 4, 16])
        self.assertEqual(rkq.n_embed_list, [8, 4, 16])
        self.assertEqual([layer.centroids.shape[0] for layer in rkq.layers], [8, 4, 16])

    def test_train_offline_raises_on_too_few_points(self) -> None:
        """N < largest K fails fast (clear message before faiss's own throw)."""
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=1, n_embed=8)
        with self.assertRaisesRegex(RuntimeError, "largest layer K"):
            rkq.train_offline(torch.randn(4, 4), verbose=False)

    def test_train_offline_raises_on_wrong_dim(self) -> None:
        """An input whose width != embed_dim fails fast."""
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=1, n_embed=8)
        with self.assertRaisesRegex(RuntimeError, "inputs must be"):
            rkq.train_offline(torch.randn(16, 8), verbose=False)

    def test_forward_returns_zeros_before_fit(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertFalse(all(layer.is_initialized for layer in rkq.layers))
        out = rkq(torch.randn(5, 4))
        self.assertIsInstance(out, ResidualQuantizerOutput)
        self.assertEqual(out.cluster_ids.shape, (5, 2))
        self.assertEqual(out.quantized_embeddings.shape, (5, 4))
        self.assertEqual(out.latents.shape, (5, 2, 4))
        self.assertIsNone(out.candidate_codes)
        self.assertIsNone(out.candidate_scores)

    def test_candidate_output_before_fit_degrades_gracefully(self) -> None:
        # Candidate output enabled + inference mode, but the codebook was never
        # fit: the pre-fit KMeans layer emits no top-k metadata, so the model
        # must stay callable (like the greedy path) and emit no candidates
        # rather than crash in _build_code_candidates.
        rkq = ResidualKMeansQuantizer(
            embed_dim=4,
            n_layers=2,
            n_embed=8,
            candidate_output_config={
                "enabled": True,
                "topk": 3,
                "strategy": "last_layer_knn",
            },
        )
        rkq.eval()
        rkq.set_is_inference(True)
        self.assertFalse(rkq.is_fitted)
        out = rkq(torch.randn(5, 4))  # must not raise
        self.assertEqual(out.cluster_ids.shape, (5, 2))
        self.assertIsNone(out.candidate_codes)
        self.assertIsNone(out.candidate_scores)

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
        eager = rkq(x)
        traced_out = traced(x)
        torch.testing.assert_close(traced_out.cluster_ids, eager.cluster_ids)
        torch.testing.assert_close(
            traced_out.quantized_embeddings, eager.quantized_embeddings
        )

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
        out = rkq(torch.randn(7, 4))
        self.assertEqual(out.cluster_ids.shape, (7, 3))
        for i, k in enumerate(n_embed):
            self.assertTrue(
                (out.cluster_ids[:, i] >= 0).all() and (out.cluster_ids[:, i] < k).all()
            )

    def test_candidate_output_last_layer_knn(self) -> None:
        """Candidate SIDs keep the greedy prefix and vary only the last layer."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        rkq = ResidualKMeansQuantizer(
            embed_dim=1,
            n_layers=2,
            n_embed=[2, 4],
            candidate_output_config={
                "enabled": True,
                "topk": 3,
                "strategy": "last_layer_knn",
            },
        )
        rkq.eval()
        rkq.set_is_inference(True)
        # Deterministic centroids (set directly; no fit needed).
        rkq.layers[0].load_centroids_(torch.tensor([[0.0], [10.0]]))
        rkq.layers[1].load_centroids_(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))

        x = torch.tensor([[2.2], [0.9]])
        out = rkq(x)

        self.assertEqual(out.candidate_codes.shape, (2, 3, 2))  # (B, topk, n_layers)
        self.assertEqual(out.candidate_scores.shape, (2, 3))
        # The first candidate is the greedy SID (nearest == get_codes order).
        torch.testing.assert_close(out.candidate_codes[:, 0, :], rkq.get_codes(x))
        # The first-layer greedy prefix is unchanged for every candidate.
        self.assertTrue(
            torch.equal(
                out.candidate_codes[:, :, 0],
                out.candidate_codes[:, :1, 0].expand(-1, 3),
            )
        )
        # For 2.2, last-layer origin is 2; nearest alternatives are 3 then 1.
        self.assertEqual(out.candidate_codes[0, :, 1].tolist(), [2, 3, 1])

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
        out = rkq(x)
        torch.testing.assert_close(rkq.get_codes(x), out.cluster_ids)
        # forward's residual-sum equals the centroid-sum reconstruction.
        torch.testing.assert_close(
            out.quantized_embeddings,
            rkq.decode_codes(out.cluster_ids),
        )


if __name__ == "__main__":
    unittest.main()
