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

from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
    faiss_residual_kmeans,
)


class GumbelResidualVQTest(unittest.TestCase):
    """Gumbel-Softmax forward_mode: grad reaches encoder + codebook (not STE)."""

    def test_default_gumbel_config_disables_sinkhorn(self) -> None:
        # use_sinkhorn defaults True; gumbel must auto-disable it (not crash).
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=3,
            n_embed=16,
            forward_mode="gumbel_softmax",
            use_sinkhorn=True,
            kmeans_init=False,
        )
        self.assertTrue(all(not layer.use_sinkhorn for layer in rvq.layers))

    def test_gumbel_grad_flows_via_soft_assignment(self) -> None:
        # The fix: the gradient from the reconstruction path (no commitment
        # loss) must reach BOTH the encoder input and the codebook through the
        # soft gumbel embedding. Under the old code it reached neither (the soft
        # embedding was discarded), so gumbel silently trained like STE.
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=3,
            n_embed=16,
            forward_mode="gumbel_softmax",
            use_sinkhorn=False,
            kmeans_init=False,
        )
        rvq.train()
        z = torch.randn(32, 8, requires_grad=True)
        rvq(z).quantized_embeddings.sum().backward()
        self.assertIsNotNone(z.grad)
        self.assertGreater(z.grad.abs().sum().item(), 0.0)
        cb_grad = rvq.layers[0].embedding.weight.grad
        self.assertIsNotNone(cb_grad)
        self.assertGreater(cb_grad.abs().sum().item(), 0.0)

    def test_ste_codebook_grad_is_detached_on_recon_path(self) -> None:
        # Contrast: STE detaches the aggregate, so the recon path gives the
        # codebook no gradient (it trains via the commitment loss instead).
        torch.manual_seed(0)
        rvq = ResidualVectorQuantizer(
            embed_dim=8,
            n_layers=2,
            n_embed=16,
            forward_mode="ste",
            use_sinkhorn=False,
            kmeans_init=False,
        )
        rvq.train()
        z = torch.randn(16, 8, requires_grad=True)
        rvq(z).quantized_embeddings.sum().backward()
        cb_grad = rvq.layers[0].embedding.weight.grad
        self.assertTrue(cb_grad is None or cb_grad.abs().sum().item() == 0.0)


class FaissResidualKmeansTest(unittest.TestCase):
    """Tests for the FAISS residual K-Means warm-start helper."""

    def test_faiss_residual_kmeans_per_layer_centers(self) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")
        torch.manual_seed(0)
        samples = torch.randn(512, 6)
        centers = faiss_residual_kmeans(
            samples, [8, 4], {"niter": 5, "verbose": False, "seed": 1}
        )
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0].shape, (8, 6))
        self.assertEqual(centers[1].shape, (4, 6))
        self.assertTrue(torch.isfinite(centers[0]).all())
        # Centroids come back on the input device (CPU fit, device-preserving).
        self.assertEqual(centers[0].device, samples.device)


if __name__ == "__main__":
    unittest.main()
