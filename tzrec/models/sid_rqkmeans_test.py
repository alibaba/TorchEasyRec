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

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.models.sid_rqkmeans import SidRqkmeans
from tzrec.protos import model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils import misc_util
from tzrec.utils.state_dict_util import init_parameters

WORLD_SIZE = 2


def _make_batch(batch_size: int, input_dim: int, device: str = "cpu") -> Batch:
    """Create a minimal Batch with dense embedding features."""
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["item_emb"],
        tensors=[torch.randn(batch_size, input_dim, device=device)],
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={},
        labels={},
    )


def _build_model(input_dim=32, n_layers=2, niter=5, codebook=None) -> SidRqkmeans:
    """Build a SidRqkmeans configured for offline FAISS fit.

    Module-level (not a method) so the spawned DDP workers below can build
    the same model; callers move it to a device / init params as needed.
    SID models read the item-embedding dense feature directly from the batch
    and do not consume feature_groups, so none is set.
    """
    from google.protobuf.struct_pb2 import Struct

    n_embed_list = codebook if codebook is not None else [16] * n_layers
    faiss_kwargs = Struct()
    faiss_kwargs.update({"niter": niter, "verbose": False, "seed": 1234})
    cfg = sid_model_pb2.SidRqkmeans(
        input_dim=input_dim,
        codebook=n_embed_list,
        normalize_residuals=False,
        faiss_kmeans_kwargs=faiss_kwargs,
        embedding_feature_name="item_emb",
    )
    return SidRqkmeans(
        model_config=model_pb2.ModelConfig(sid_rqkmeans=cfg),
        features=[],
        labels=[],
    )


class SidRqkmeansOfflineTest(unittest.TestCase):
    """Single-process tests for SidRqkmeans (FAISS-only)."""

    def _create_model(self, input_dim=32, n_layers=2, niter=5, codebook=None):
        """Create a SidRqkmeans on CPU with params initialized."""
        model = _build_model(input_dim, n_layers, niter, codebook)
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_proto_parse(self) -> None:
        """Verify faiss_kmeans_kwargs are parsed correctly."""
        model = self._create_model()
        self.assertEqual(model._faiss_kwargs.get("niter"), 5)
        self.assertEqual(model._faiss_kwargs.get("seed"), 1234)
        self.assertFalse(model._faiss_kwargs.get("verbose"))
        self.assertEqual(model._n_seen, 0)
        self.assertIsNone(model._reservoir)

    def test_predict_collects_buffer(self) -> None:
        """In train mode, predict reservoir-samples; never fits."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim)
        model.train()

        for _ in range(4):
            batch = _make_batch(B, input_dim)
            preds = model.predict(batch)
            self.assertIn("codes", preds)

        # Reservoir holds all 4*B samples (well under the cap) and tracks
        # the running count.
        self.assertEqual(model._n_seen, 4 * B)
        self.assertEqual(model._n_filled, 4 * B)
        # FAISS not yet triggered: layers should be uninitialized
        for layer in model._quantizer.layers:
            self.assertFalse(layer.is_initialized)

    def test_reservoir_caps_memory(self) -> None:
        """Reservoir bounds the buffer at _sample_cap regardless of corpus."""
        B, input_dim = 16, 8
        model = self._create_model(input_dim=input_dim)
        model._sample_cap = 10  # force a tiny cap
        model._reset_reservoir()
        model.train()
        for _ in range(20):  # 320 rows >> cap
            model.predict(_make_batch(B, input_dim))
        self.assertEqual(model._n_seen, 20 * B)
        self.assertEqual(model._n_filled, 10)
        self.assertEqual(model._reservoir.shape, (10, input_dim))

    def test_on_train_end_runs_faiss(self) -> None:
        """on_train_end triggers FAISS fit and clears buffer."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        model = self._create_model(input_dim=input_dim)
        model.train()

        # Accumulate enough samples (FAISS K-Means needs at least K points)
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        self.assertGreater(model._n_seen, 0)

        # Trigger one-shot FAISS fit
        model.on_train_end()

        # Reservoir should be released after the fit
        self.assertEqual(model._n_seen, 0)
        self.assertIsNone(model._reservoir)
        # All layers should be initialized + centroids non-zero
        for layer in model._quantizer.layers:
            self.assertTrue(bool(layer._is_initialized.item()))
            self.assertGreater(layer.centroids.abs().sum().item(), 0.0)

        # After fit, predict on eval should produce valid codes
        model.eval()
        preds = model.predict(_make_batch(B, input_dim))
        codes = preds["codes"]
        self.assertEqual(codes.shape, (B, 2))
        self.assertTrue((codes >= 0).all() and (codes < 16).all())

    def test_non_uniform_codebook_end_to_end(self) -> None:
        """Non-uniform codebook [8, 4, 16]: fit then emit per-layer codes."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        codebook = [8, 4, 16]
        model = self._create_model(input_dim=input_dim, codebook=codebook)
        # Reservoir cap derives from the LARGEST K (16), not the first (8).
        self.assertEqual(
            model._sample_cap,
            16 * int(model._faiss_kwargs.get("max_points_per_centroid", 256)),
        )

        model.train()
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        model.on_train_end()

        for k, layer in zip(codebook, model._quantizer.layers):
            self.assertTrue(bool(layer._is_initialized.item()))
            self.assertEqual(layer.centroids.shape[0], k)

        model.eval()
        codes = model.predict(_make_batch(B, input_dim))["codes"]
        self.assertEqual(codes.shape, (B, 3))
        for i, k in enumerate(codebook):
            self.assertTrue((codes[:, i] >= 0).all() and (codes[:, i] < k).all())

    def test_on_train_end_noop_on_empty_buffer(self) -> None:
        """on_train_end on an empty buffer is a warned no-op."""
        model = self._create_model()
        model.on_train_end()  # should not raise

    def test_post_fit_checkpoint_round_trips(self) -> None:
        """Fit → save state_dict → load into fresh instance → predict.

        After loading, ``predict`` must return real (non-zero) codes —
        the centroids and the ``_is_initialized`` flag both need to come
        through the state_dict.
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        src = self._create_model(input_dim=input_dim)
        src.train()
        for _ in range(8):
            src.predict(_make_batch(B, input_dim))
        src.on_train_end()
        sd = src.state_dict()

        dst = self._create_model(input_dim=input_dim)
        dst.load_state_dict(sd)
        dst.eval()
        codes = dst.predict(_make_batch(B, input_dim))["codes"]
        self.assertGreater(
            codes.abs().sum().item(),
            0,
            "post-fit checkpoint resume produced all-zero codes",
        )

    def test_mid_fit_checkpoint_rejected_on_load(self) -> None:
        """Tampered state (_is_initialized=True + zero centroids) raises."""
        model = self._create_model()
        sd = model.state_dict()
        # Simulate a checkpoint that captured the flag mid-fit (before
        # load_centroids_ ran): True flag, zero centroids.
        layer0_prefix = next(
            k.rsplit("._is_initialized", 1)[0]
            for k in sd
            if k.endswith("._is_initialized")
        )
        sd[f"{layer0_prefix}._is_initialized"] = torch.tensor(True)

        fresh = self._create_model()
        with self.assertRaisesRegex(RuntimeError, "mid-FAISS-fit"):
            fresh.load_state_dict(sd)


# --------------------------------------------------------------------------
# Distributed (multi-process) test for the DDP on_train_end path: the
# cross-rank gather_object -> FAISS fit -> broadcast sequence the in-process
# tests above cannot reach. NCCL on GPU when >=2 devices, else gloo/CPU.
# --------------------------------------------------------------------------
def _init_dist(rank: int, world_size: int, port: int) -> torch.device:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    if use_cuda:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return torch.device("cpu")


def _on_train_end_worker(rank: int, world_size: int, port: int) -> None:
    device = _init_dist(rank, world_size, port)
    input_dim, n_layers, k = 16, 2, 16
    model = _build_model(input_dim, n_layers, codebook=[k] * n_layers).to(device)
    model.train()

    torch.manual_seed(100 + rank)
    for _ in range(6):
        model.predict(_make_batch(32, input_dim, device))
    assert model._n_seen == 6 * 32, f"rank{rank}: reservoir not filled"

    # gather_object -> rank0 FAISS fit -> broadcast centroids + fill flag.
    model.on_train_end()

    for layer in model._quantizer.layers:
        assert bool(layer._is_initialized.item()), f"rank{rank}: layer uninit"
        assert layer.centroids.abs().sum().item() > 0.0, f"rank{rank}: zero centroids"
    # Centroids were broadcast from rank0 -> must be bit-identical across ranks.
    for layer in model._quantizer.layers:
        cmin, cmax = layer.centroids.clone(), layer.centroids.clone()
        dist.all_reduce(cmin, op=dist.ReduceOp.MIN)
        dist.all_reduce(cmax, op=dist.ReduceOp.MAX)
        assert torch.allclose(cmin, cmax), f"rank{rank}: centroids differ across ranks"

    model.eval()
    codes = model.predict(_make_batch(8, input_dim, device))["codes"]
    assert codes.shape == (8, n_layers), f"rank{rank}: bad codes shape {codes.shape}"
    assert (codes >= 0).all() and (codes < k).all(), f"rank{rank}: codes out of range"
    dist.destroy_process_group()


class SidRqkmeansDistTest(unittest.TestCase):
    """2-rank test for SidRqkmeans.on_train_end (gather -> fit -> broadcast)."""

    def test_on_train_end_ddp(self) -> None:
        port = misc_util.get_free_port()
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(WORLD_SIZE):
            p = ctx.Process(target=_on_train_end_worker, args=(rank, WORLD_SIZE, port))
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed (exitcode={p.exitcode}).")


if __name__ == "__main__":
    unittest.main()
