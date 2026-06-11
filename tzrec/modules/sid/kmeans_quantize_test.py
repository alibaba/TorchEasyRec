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

from tzrec.modules.sid.kmeans_quantize import (
    KMeansQuantizeLayer,
    ReservoirSampler,
    faiss_residual_kmeans,
)


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


class KMeansQuantizeLayerTest(unittest.TestCase):
    """Tests for the single KMeansQuantizeLayer."""

    def test_uninitialized_by_default(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=3)
        self.assertFalse(layer.is_initialized)
        self.assertEqual(layer.centroids.abs().sum().item(), 0.0)

    def test_load_centroids_and_quantize(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        centroids = torch.tensor([[0.0, 0.0], [10.0, 10.0]])
        layer.load_centroids_(centroids)
        self.assertTrue(layer.is_initialized)

        batch = torch.tensor([[0.1, 0.0], [9.0, 11.0]])
        out = layer.quantize(batch)
        torch.testing.assert_close(out.ids, torch.tensor([0, 1]))
        # embeddings are the gathered centroids; lookup matches.
        torch.testing.assert_close(out.embeddings, centroids[out.ids])
        torch.testing.assert_close(layer.lookup(out.ids), out.embeddings)

    def test_quantize_uninitialized_returns_zeros(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=3)
        out = layer.quantize(torch.randn(5, 3))
        self.assertEqual(out.ids.shape, (5,))
        self.assertEqual(int(out.ids.abs().sum()), 0)
        torch.testing.assert_close(out.embeddings, torch.zeros(5, 3))

    def test_load_centroids_shape_mismatch_raises(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        with self.assertRaises(RuntimeError):
            layer.load_centroids_(torch.zeros(3, 2))

    def test_mid_fit_checkpoint_rejected(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        sd = layer.state_dict()
        # Simulate a mid-fit checkpoint: flag True but centroids still zero.
        sd["_is_initialized"] = torch.tensor(True)
        fresh = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        with self.assertRaisesRegex(RuntimeError, "mid-FAISS-fit"):
            fresh.load_state_dict(sd)

    def test_post_fit_checkpoint_round_trips(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        layer.load_centroids_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        fresh = KMeansQuantizeLayer(n_embed=2, embed_dim=2)
        fresh.load_state_dict(layer.state_dict())
        self.assertTrue(fresh.is_initialized)
        torch.testing.assert_close(fresh.centroids, layer.centroids)


class ReservoirSamplerTest(unittest.TestCase):
    """Tests for the bounded reservoir sampler (Vitter Algorithm R)."""

    def test_empty_sample(self) -> None:
        """sample() before any add returns an empty (0, dim) tensor."""
        r = ReservoirSampler(capacity=10, dim=4)
        self.assertEqual(r.sample().shape, (0, 4))
        self.assertEqual(r.n_seen, 0)
        self.assertEqual(r.n_filled, 0)

    def test_caps_memory(self) -> None:
        """The buffer is bounded at capacity regardless of stream length."""
        cap, dim, B = 10, 8, 16
        r = ReservoirSampler(capacity=cap, dim=dim)
        for _ in range(20):  # 320 rows >> cap
            r.add(torch.randn(B, dim))
        self.assertEqual(r.n_seen, 20 * B)
        self.assertEqual(r.n_filled, cap)
        self.assertEqual(r.sample().shape, (cap, dim))

    def test_phase2_replacement(self) -> None:
        """Phase-2 replacement keeps a valid sample of real, in-range rows.

        Feeds identifiable rows (each row's value == its global stream index),
        then asserts every slot still holds an intact fed row, all indices are
        in range, and replacement past the initial fill actually happened —
        exercising the accept-prob / slot-write logic that the count/shape-only
        ``test_caps_memory`` cannot.
        """
        torch.manual_seed(0)
        dim, cap, B, n_batches = 4, 8, 4, 50
        r = ReservoirSampler(capacity=cap, dim=dim)

        gidx = 0
        for _ in range(n_batches):
            rows = (
                torch.arange(gidx, gidx + B, dtype=torch.float32)
                .unsqueeze(1)
                .expand(B, dim)
                .contiguous()
            )
            gidx += B
            r.add(rows)

        total = B * n_batches
        self.assertEqual(r.n_seen, total)
        self.assertEqual(r.n_filled, cap)

        res = r.sample()
        idx = res[:, 0].round().long()
        # Each stored row is an intact fed row (all columns equal its index).
        self.assertTrue(
            torch.equal(res, idx.unsqueeze(1).float().expand_as(res)),
            "reservoir holds corrupted (non-fed) rows",
        )
        # All indices are valid stream positions.
        self.assertTrue((idx >= 0).all() and (idx < total).all())
        # Phase-2 replacement happened: at least one slot holds a row added
        # after the reservoir filled (index >= cap).
        self.assertTrue((idx >= cap).any(), "no Phase-2 replacement occurred")

    def test_reset(self) -> None:
        """reset() drops the buffer and counters."""
        r = ReservoirSampler(capacity=10, dim=4)
        r.add(torch.randn(5, 4))
        self.assertEqual(r.n_filled, 5)
        r.reset()
        self.assertEqual(r.n_seen, 0)
        self.assertEqual(r.n_filled, 0)
        self.assertEqual(r.sample().shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
