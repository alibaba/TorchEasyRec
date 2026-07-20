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
    faiss_kmeans_fit,
)
from tzrec.utils.test_util import faiss_unavailable


@unittest.skipIf(*faiss_unavailable)
class FaissKmeansFitTest(unittest.TestCase):
    """Tests for the shared one-layer FAISS fit primitive."""

    def test_fit_returns_trained_kmeans(self) -> None:
        torch.manual_seed(0)
        # numpy input (the RQ-VAE call path; no faiss torch-utils needed).
        x = torch.randn(200, 6).numpy()
        km = faiss_kmeans_fit(x, 6, 8, {"niter": 5, "seed": 1, "verbose": False})
        self.assertEqual(tuple(km.centroids.shape), (8, 6))

    def test_raises_on_too_few_points(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "need >= 8 points"):
            faiss_kmeans_fit(torch.randn(4, 6).numpy(), 6, 8)


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

    def test_quantize_returns_topk_centroids(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=1)
        centroids = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        layer.load_centroids_(centroids)
        layer.eval()

        out = layer.quantize(torch.tensor([[1.2], [2.8]]), topk=3)

        torch.testing.assert_close(out.ids, torch.tensor([1, 3]))
        torch.testing.assert_close(out.topk_ids, torch.tensor([[1, 2, 0], [3, 2, 1]]))
        torch.testing.assert_close(
            out.topk_scores,
            torch.tensor([[0.04, 0.64, 1.44], [0.04, 0.64, 3.24]]),
        )

    def test_quantize_uninitialized_returns_zeros(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=3)
        out = layer.quantize(torch.randn(5, 3))
        self.assertEqual(out.ids.shape, (5,))
        self.assertEqual(int(out.ids.abs().sum()), 0)
        torch.testing.assert_close(out.embeddings, torch.zeros(5, 3))
        self.assertIsNone(out.topk_ids)
        self.assertIsNone(out.topk_scores)

        # Eval before the fit is also a no-op: no topk/candidate metadata either.
        layer.eval()
        out = layer.quantize(torch.randn(5, 3), topk=2)
        self.assertIsNone(out.topk_ids)
        self.assertIsNone(out.topk_scores)

    def test_training_quantize_skips_topk_metadata(self) -> None:
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=1)
        layer.load_centroids_(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        layer.train()

        out = layer.quantize(torch.tensor([[1.2], [2.8]]), topk=3)

        torch.testing.assert_close(out.ids, torch.tensor([1, 3]))
        self.assertIsNone(out.topk_ids)
        self.assertIsNone(out.topk_scores)

    def test_quantize_topk_out_of_range_raises(self) -> None:
        # nearest_neighbors (shared QuantizeLayer guard) rejects topk<1 and
        # topk>n_embed on the eval quantize path.
        layer = KMeansQuantizeLayer(n_embed=4, embed_dim=1)
        layer.load_centroids_(torch.tensor([[0.0], [1.0], [2.0], [3.0]]))
        layer.eval()
        with self.assertRaisesRegex(ValueError, r"topk must be in \[1, 4\]"):
            layer.quantize(torch.tensor([[1.0]]), topk=0)
        with self.assertRaisesRegex(ValueError, r"topk must be in \[1, 4\]"):
            layer.quantize(torch.tensor([[1.0]]), topk=5)

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
        # Phase-2 replacement dominates the final sample: with a correct accept
        # probability the expected post-fill survivor count is
        # cap*(total-cap)/total ~= cap, so require well over half. A near-empty
        # phase-2 count means the accept rate is broken (``.any()`` would only
        # catch replacement being disabled outright).
        n_phase2 = (idx >= cap).sum().item()
        self.assertGreater(n_phase2, cap // 2, f"too few Phase-2 rows: {n_phase2}")

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
