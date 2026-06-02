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

from tzrec.modules.sid_generation.residual_kmeans_quantizer import (
    ResidualKMeansQuantizer,
)
from tzrec.modules.sid_generation.residual_quantizer import (
    ResidualQuantizer,
    normalize_n_embed,
)
from tzrec.modules.sid_generation.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
from tzrec.modules.sid_generation.types import ResidualQuantizerOutput


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
        self.assertEqual(
            [layer.centroids.shape[0] for layer in rkq.layers], [8, 4, 16]
        )

    def test_forward_returns_zeros_before_fit(self) -> None:
        rkq = ResidualKMeansQuantizer(embed_dim=4, n_layers=2, n_embed=8)
        self.assertFalse(all(layer.is_initialized for layer in rkq.layers))
        codes, quantized = rkq(torch.randn(5, 4))
        self.assertEqual(codes.shape, (5, 2))
        self.assertEqual(quantized.shape, (5, 4))

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


if __name__ == "__main__":
    unittest.main()
