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

from tzrec.modules.sid_generation.kmeans import (
    KMeansLayer,
    _kmeans,
    _residual_kmeans,
    _squared_euclidean_distance,
    recon_diagnostics,
)


class KmeansHelpersTest(unittest.TestCase):
    """Tests for the pure-torch K-Means helpers."""

    def test_recon_diagnostics_zero_on_identity(self) -> None:
        x = torch.randn(8, 4)
        mse, rel = recon_diagnostics(x, x.clone())
        self.assertAlmostEqual(mse.item(), 0.0, places=6)
        self.assertAlmostEqual(rel.item(), 0.0, places=6)

    def test_squared_euclidean_distance(self) -> None:
        x = torch.tensor([[0.0, 0.0], [1.0, 0.0]])
        y = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
        d = _squared_euclidean_distance(x, y)
        self.assertEqual(d.shape, (2, 2))
        # row0: dist to (0,0)=0, to (0,1)=1; row1: to (0,0)=1, to (0,1)=2
        torch.testing.assert_close(d, torch.tensor([[0.0, 1.0], [1.0, 2.0]]))

    def test_squared_euclidean_distance_chunked_matches(self) -> None:
        x = torch.randn(120, 5)
        y = torch.randn(7, 5)
        full = _squared_euclidean_distance(x, y, chunk_size=1000)
        chunked = _squared_euclidean_distance(x, y, chunk_size=16)
        torch.testing.assert_close(full, chunked)

    def test_kmeans_shapes_and_assignment_range(self) -> None:
        torch.manual_seed(0)
        samples = torch.randn(200, 6)
        centroids, assignments = _kmeans(samples, n_clusters=8, n_iters=5)
        self.assertEqual(centroids.shape, (8, 6))
        self.assertEqual(assignments.shape, (200,))
        self.assertTrue((assignments >= 0).all() and (assignments < 8).all())

    def test_residual_kmeans_per_layer_centers(self) -> None:
        torch.manual_seed(0)
        samples = torch.randn(200, 6)
        centers = _residual_kmeans(samples, [8, 4], n_iters=5)
        self.assertEqual(len(centers), 2)
        self.assertEqual(centers[0].shape, (8, 6))
        self.assertEqual(centers[1].shape, (4, 6))


class KMeansLayerTest(unittest.TestCase):
    """Tests for the single KMeansLayer."""

    def test_uninitialized_by_default(self) -> None:
        layer = KMeansLayer(n_clusters=4, n_features=3)
        self.assertFalse(layer.is_initialized)
        self.assertEqual(layer.centroids.abs().sum().item(), 0.0)

    def test_load_centroids_and_predict(self) -> None:
        layer = KMeansLayer(n_clusters=2, n_features=2)
        centroids = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        layer.load_centroids_(centroids)
        self.assertTrue(layer.is_initialized)

        batch = torch.tensor([[0.1, 0.0], [9.0, 11.0]])
        codes = layer.predict(batch)
        torch.testing.assert_close(codes, torch.tensor([0, 1]))

    def test_load_centroids_shape_mismatch_raises(self) -> None:
        layer = KMeansLayer(n_clusters=2, n_features=2)
        with self.assertRaises(AssertionError):
            layer.load_centroids_(torch.zeros(3, 2))

    def test_mid_fit_checkpoint_rejected(self) -> None:
        layer = KMeansLayer(n_clusters=2, n_features=2)
        sd = layer.state_dict()
        # Simulate a mid-fit checkpoint: flag True but centroids still zero.
        sd["_is_initialized"] = torch.tensor(True)
        fresh = KMeansLayer(n_clusters=2, n_features=2)
        with self.assertRaisesRegex(RuntimeError, "mid-FAISS-fit"):
            fresh.load_state_dict(sd)

    def test_post_fit_checkpoint_round_trips(self) -> None:
        layer = KMeansLayer(n_clusters=2, n_features=2)
        layer.load_centroids_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        fresh = KMeansLayer(n_clusters=2, n_features=2)
        fresh.load_state_dict(layer.state_dict())
        self.assertTrue(fresh.is_initialized)
        torch.testing.assert_close(fresh.centroids, layer.centroids)


if __name__ == "__main__":
    unittest.main()
