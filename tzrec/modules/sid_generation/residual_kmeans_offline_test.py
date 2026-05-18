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

import unittest

import torch

try:
    import faiss  # noqa: F401

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from tzrec.modules.sid_generation.kmeans import KMeansLayer
from tzrec.modules.sid_generation.residual_kmeans import RQKMeans


def _seed(s: int = 42) -> None:
    import random

    import numpy as np

    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


@unittest.skipUnless(FAISS_AVAILABLE, "faiss not installed")
class ResidualKMeansOfflineTest(unittest.TestCase):
    """End-to-end behaviour of the FAISS-only training path."""

    def setUp(self) -> None:
        _seed(42)
        self.dim = 16
        self.n_layers = 2
        self.n_embed = 32
        self.x = torch.randn(512, self.dim)

    def _build(self) -> RQKMeans:
        return RQKMeans(
            embed_dim=self.dim,
            n_layers=self.n_layers,
            n_embed=self.n_embed,
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

    def test_state_dict_roundtrip(self) -> None:
        """Offline ckpt loads cleanly into a freshly built model."""
        a = self._build()
        a.train_offline(self.x, verbose=False)
        sd = a.state_dict()

        b = self._build()
        b.load_state_dict(sd)
        b.eval()
        codes_a = a.get_codes(self.x)
        codes_b = b.get_codes(self.x)
        self.assertTrue(torch.equal(codes_a, codes_b))

    def test_normalize_residuals_offline(self) -> None:
        """``normalize_residuals=True`` path runs and converges."""
        model = RQKMeans(
            embed_dim=self.dim,
            n_layers=self.n_layers,
            n_embed=self.n_embed,
            normalize_residuals=True,
            faiss_kmeans_kwargs={"niter": 5, "verbose": False, "seed": 1234},
        )
        model.train_offline(self.x, verbose=False)
        codes = model.get_codes(self.x)
        self.assertEqual(codes.shape, (self.x.shape[0], self.n_layers))


class KMeansLayerLoadTest(unittest.TestCase):
    """Direct tests for the centroid-injection API on KMeansLayer."""

    def test_load_centroids_marks_initialized(self) -> None:
        layer = KMeansLayer(n_clusters=4, n_features=3)
        self.assertFalse(layer.is_initialized)

        centroids = torch.randn(4, 3)
        layer.load_centroids_(centroids)

        self.assertTrue(layer.is_initialized)
        self.assertTrue(torch.allclose(layer.centroids, centroids))

    def test_predict_after_load(self) -> None:
        layer = KMeansLayer(n_clusters=4, n_features=3)
        layer.load_centroids_(torch.randn(4, 3))
        codes = layer.predict(torch.randn(8, 3))
        self.assertEqual(codes.shape, (8,))
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 4).all())


if __name__ == "__main__":
    unittest.main()
