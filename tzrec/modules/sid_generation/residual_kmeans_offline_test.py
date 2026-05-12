# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for offline FAISS K-Means backend in residual_kmeans.

Skipped automatically if ``faiss`` is not importable.
"""

import io
import unittest

import torch

try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from tzrec.modules.sid_generation.kmeans import MiniBatchKMeans
from tzrec.modules.sid_generation.residual_kmeans import (
    RQKMeans,
    ResidualKMeans,
)


def _seed(s: int = 42) -> None:
    import random

    import numpy as np

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@unittest.skipUnless(FAISS_AVAILABLE, "faiss not installed")
class ResidualKMeansOfflineTest(unittest.TestCase):
    """End-to-end behaviour of ``train_mode='offline_faiss'``."""

    def setUp(self) -> None:
        _seed(42)
        self.dim = 16
        self.n_layers = 2
        self.n_embed = 32
        self.x = torch.randn(512, self.dim)

    def _build(self, train_mode: str = "offline_faiss") -> RQKMeans:
        return RQKMeans(
            embed_dim=self.dim,
            n_layers=self.n_layers,
            n_embed=self.n_embed,
            train_mode=train_mode,
            faiss_kmeans_kwargs={
                "niter": 5,
                "verbose": False,
                "spherical": False,
                "seed": 1234,
            },
        )

    def test_offline_train_shapes_and_codes(self) -> None:
        model = self._build()
        model.train_offline(self.x, verbose=False)
        codes = model.get_codes(self.x)
        self.assertEqual(codes.shape, (self.x.shape[0], self.n_layers))
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < self.n_embed).all())

    def test_offline_then_inference_consistency(self) -> None:
        """``forward(eval)`` and ``get_codes`` agree after offline train."""
        model = self._build()
        model.train_offline(self.x, verbose=False)
        model.eval()
        out = model(self.x)
        codes_forward = out["codes"]
        codes_inference = model.get_codes(self.x)
        self.assertTrue(torch.equal(codes_forward, codes_inference))

    def test_state_dict_roundtrip_to_online_mode(self) -> None:
        """Offline ckpt loads cleanly into a freshly built online model."""
        offline_model = self._build()
        offline_model.train_offline(self.x, verbose=False)
        sd = offline_model.state_dict()

        online_model = self._build(train_mode="online")
        online_model.load_state_dict(sd)
        online_model.eval()
        codes_a = offline_model.get_codes(self.x)
        codes_b = online_model.get_codes(self.x)
        self.assertTrue(torch.equal(codes_a, codes_b))

    def test_train_offline_in_online_mode_raises(self) -> None:
        model = self._build(train_mode="online")
        with self.assertRaises(AssertionError):
            model.train_offline(self.x, verbose=False)

    def test_offline_locked_blocks_train_step(self) -> None:
        """After offline init, layer.train_step must raise."""
        model = self._build()
        model.train_offline(self.x, verbose=False)
        layer = model.quantizer.layers[0]
        self.assertTrue(layer.offline_locked)
        with self.assertRaises(RuntimeError) as ctx:
            layer.train_step(self.x[:4])
        self.assertIn("offline", str(ctx.exception).lower())

    def test_offline_locked_persists_after_state_dict_roundtrip(self) -> None:
        """Lock survives load_state_dict into a fresh online-mode model."""
        offline_model = self._build()
        offline_model.train_offline(self.x, verbose=False)
        online_model = self._build(train_mode="online")
        online_model.load_state_dict(offline_model.state_dict())
        for layer in online_model.quantizer.layers:
            self.assertTrue(layer.offline_locked)
            with self.assertRaises(RuntimeError):
                layer.train_step(self.x[:4])

    def test_unlock_for_online_finetune(self) -> None:
        """Explicit unlock allows train_step again."""
        model = self._build()
        model.train_offline(self.x, verbose=False)
        layer = model.quantizer.layers[0]
        layer.unlock_for_online_finetune_()
        self.assertFalse(layer.offline_locked)
        # train_step now runs without raising
        codes, emb = layer.train_step(self.x[:8])
        self.assertEqual(codes.shape, (8,))
        self.assertEqual(emb.shape, (8, self.dim))

    def test_normalize_residuals_offline(self) -> None:
        """``normalize_residuals=True`` path runs and converges."""
        model = RQKMeans(
            embed_dim=self.dim,
            n_layers=self.n_layers,
            n_embed=self.n_embed,
            normalize_residuals=True,
            train_mode="offline_faiss",
            faiss_kmeans_kwargs={"niter": 5, "verbose": False, "seed": 1234},
        )
        model.train_offline(self.x, verbose=False)
        codes = model.get_codes(self.x)
        self.assertEqual(codes.shape, (self.x.shape[0], self.n_layers))


class MiniBatchKMeansOfflineLockTest(unittest.TestCase):
    """Direct tests for the offline-lock mechanism on MiniBatchKMeans."""

    def test_load_centroids_marks_initialized_and_locked(self) -> None:
        layer = MiniBatchKMeans(n_clusters=4, n_features=3)
        self.assertFalse(layer.is_initialized)
        self.assertFalse(layer.offline_locked)

        centroids = torch.randn(4, 3)
        layer.load_centroids_(centroids)

        self.assertTrue(layer.is_initialized)
        self.assertTrue(layer.offline_locked)
        self.assertTrue(torch.allclose(layer.centroids, centroids))
        self.assertTrue(torch.equal(
            layer.cluster_counts, torch.zeros_like(layer.cluster_counts)
        ))

    def test_train_step_raises_after_load_centroids(self) -> None:
        layer = MiniBatchKMeans(n_clusters=4, n_features=3)
        layer.load_centroids_(torch.randn(4, 3))
        with self.assertRaises(RuntimeError):
            layer.train_step(torch.randn(8, 3))

    def test_unlock_then_train_step(self) -> None:
        layer = MiniBatchKMeans(n_clusters=4, n_features=3)
        layer.load_centroids_(torch.randn(4, 3))
        layer.unlock_for_online_finetune_()
        codes, emb = layer.train_step(torch.randn(8, 3))
        self.assertEqual(codes.shape, (8,))
        self.assertEqual(emb.shape, (8, 3))


if __name__ == "__main__":
    # Silence FAISS C++ stdout if needed: not redirected here.
    unittest.main()
