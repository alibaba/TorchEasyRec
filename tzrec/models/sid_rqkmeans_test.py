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


def _batch_from_rows(rows: torch.Tensor) -> Batch:
    """Wrap explicit ``item_emb`` rows in a minimal Batch."""
    dense_feature = KeyedTensor.from_tensor_list(keys=["item_emb"], tensors=[rows])
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={},
        labels={},
    )


def _make_batch(batch_size: int, input_dim: int, device: str = "cpu") -> Batch:
    """Create a minimal Batch with random dense embedding features."""
    return _batch_from_rows(torch.randn(batch_size, input_dim, device=device))


def _build_model(
    input_dim=32,
    n_layers=2,
    niter=5,
    codebook=None,
    normalize_residuals=False,
    train_sample_size=0,
) -> SidRqkmeans:
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
        normalize_residuals=normalize_residuals,
        faiss_kmeans_kwargs=faiss_kwargs,
        embedding_feature_name="item_emb",
        train_sample_size=train_sample_size,
    )
    return SidRqkmeans(
        model_config=model_pb2.ModelConfig(sid_rqkmeans=cfg),
        features=[],
        labels=[],
    )


class SidRqkmeansOfflineTest(unittest.TestCase):
    """Single-process tests for SidRqkmeans (FAISS-only)."""

    def _create_model(
        self,
        input_dim=32,
        n_layers=2,
        niter=5,
        codebook=None,
        normalize_residuals=False,
        train_sample_size=0,
    ):
        """Create a SidRqkmeans on CPU with params initialized."""
        model = _build_model(
            input_dim,
            n_layers,
            niter,
            codebook,
            normalize_residuals,
            train_sample_size,
        )
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

    def test_sample_cap_from_train_sample_size(self) -> None:
        """Explicit train_sample_size drives the per-rank cap (ceil-div)."""
        from unittest import mock

        # Single process (world_size=1): cap == train_sample_size.
        model = self._create_model(train_sample_size=900)
        self.assertEqual(model._sample_cap, 900)

        # Per-rank ceil-div across world_size (patch dist + recompute the cap).
        for world_size, expected in [(4, 225), (7, 129), (1000, 1)]:
            with (
                mock.patch.object(dist, "is_initialized", return_value=True),
                mock.patch.object(dist, "get_world_size", return_value=world_size),
            ):
                model._init_reservoir()
            self.assertEqual(model._sample_cap, expected)

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

    def test_reservoir_phase2_replacement(self) -> None:
        """Phase-2 replacement keeps a valid reservoir of real, in-range rows.

        Feeds identifiable rows (each row's value == its global stream index),
        then asserts every reservoir slot still holds an intact fed row, all
        indices are in range, and replacement past the initial fill actually
        happened — exercising the accept-prob / slot-write logic that the
        count/shape-only ``test_reservoir_caps_memory`` cannot.
        """
        torch.manual_seed(0)
        input_dim, cap, B, n_batches = 4, 8, 4, 50
        model = self._create_model(input_dim=input_dim)
        model._sample_cap = cap
        model._reset_reservoir()
        model.train()

        gidx = 0
        for _ in range(n_batches):
            rows = (
                torch.arange(gidx, gidx + B, dtype=torch.float32)
                .unsqueeze(1)
                .expand(B, input_dim)
                .contiguous()
            )
            gidx += B
            model.predict(_batch_from_rows(rows))

        total = B * n_batches
        self.assertEqual(model._n_seen, total)
        self.assertEqual(model._n_filled, cap)

        res = model._reservoir
        idx = res[:, 0].round().long()
        # Each stored row is an intact fed row (all columns equal its index),
        # never zeros/garbage.
        self.assertTrue(
            torch.equal(res, idx.unsqueeze(1).float().expand_as(res)),
            "reservoir holds corrupted (non-fed) rows",
        )
        # All indices are valid stream positions.
        self.assertTrue((idx >= 0).all() and (idx < total).all())
        # Phase-2 replacement happened: at least one slot holds a row added
        # after the reservoir filled (index >= cap).
        self.assertTrue((idx >= cap).any(), "no Phase-2 replacement occurred")

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

        # Trigger one-shot FAISS fit; a real fit must request a tail checkpoint
        self.assertTrue(model.on_train_end())

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

    def test_normalize_residuals_end_to_end(self) -> None:
        """train_offline with normalize_residuals=True fits + predicts.

        Exercises the ``F.normalize`` site inside ``train_offline`` (a second
        normalize independent of ``_residual_pass``), which the other tests —
        all built with normalize_residuals=False — never reach.
        """
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        model = self._create_model(input_dim=input_dim, normalize_residuals=True)
        self.assertTrue(model._quantizer.normalize_residuals)

        model.train()
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        self.assertTrue(model.on_train_end())

        for layer in model._quantizer.layers:
            self.assertTrue(layer.is_initialized)

        model.eval()
        codes = model.predict(_make_batch(B, input_dim))["codes"]
        self.assertEqual(codes.shape, (B, 2))
        self.assertTrue((codes >= 0).all() and (codes < 16).all())

    def test_eval_and_inference_predict_contract(self) -> None:
        """Eval exposes quantized/input_embedding; inference is codes-only."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        model.on_train_end()

        # Eval mode: reconstruction outputs are present for update_metric.
        model.eval()
        eval_preds = model.predict(_make_batch(B, input_dim))
        self.assertIn("quantized", eval_preds)
        self.assertIn("input_embedding", eval_preds)

        # Inference (serving) mode: codes-only contract.
        model.set_is_inference(True)
        inf_preds = model.predict(_make_batch(B, input_dim))
        self.assertEqual(set(inf_preds.keys()), {"codes"})

    def test_eval_metric_path(self) -> None:
        """init_metric/update_metric report finite mse + rel_loss in eval."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        model.on_train_end()

        model.init_metric()
        model.eval()
        preds = model.predict(_make_batch(B, input_dim))
        model.update_metric(preds, _make_batch(B, input_dim))
        metrics = model.compute_metric()
        for key in ("mse", "rel_loss", "unique_sid_ratio"):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(torch.as_tensor(metrics[key])).all())

    def test_on_train_end_noop_on_empty_buffer(self) -> None:
        """on_train_end on an empty buffer is a warned no-op."""
        model = self._create_model()
        # No fit happened, so no tail checkpoint is requested.
        self.assertFalse(model.on_train_end())  # should not raise

    def test_post_fit_checkpoint_round_trips(self) -> None:
        """Fit → save state_dict → load into fresh instance → predict.

        The reloaded model must produce the *same* codes as the source on the
        same batch — verifying the centroids round-trip exactly, not merely
        that they came through as non-zero.
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

        # Same batch through both → identical codes (exact round-trip).
        batch = _make_batch(B, input_dim)
        src.eval()
        dst.eval()
        src_codes = src.predict(batch)["codes"]
        dst_codes = dst.predict(batch)["codes"]
        self.assertGreater(
            dst_codes.abs().sum().item(),
            0,
            "post-fit checkpoint resume produced all-zero codes",
        )
        torch.testing.assert_close(dst_codes, src_codes)

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
    # Every rank fitted/received the codebook, so each requests a tail ckpt.
    assert model.on_train_end(), f"rank{rank}: on_train_end should request ckpt"

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


def _on_train_end_fail_worker(rank: int, world_size: int, port: int) -> None:
    """Worker that forces rank0's FAISS fit to fail.

    Every rank must then raise the coordinated ``RuntimeError`` (driven by the
    fit-status broadcast) instead of deadlocking on the centroid broadcast. A
    worker returns 0 only if it caught that expected error.
    """
    device = _init_dist(rank, world_size, port)
    input_dim, n_layers, k = 16, 2, 16
    model = _build_model(input_dim, n_layers, codebook=[k] * n_layers).to(device)
    model.train()
    for _ in range(6):
        model.predict(_make_batch(32, input_dim, device))

    # Force the rank0-only fit to raise (no faiss needed: only rank0 fits, and
    # we replace its fit). The status flag must turn this into an all-ranks
    # raise, not a hang.
    if rank == 0:

        def _boom(*args, **kwargs):
            raise RuntimeError("forced rank0 fit failure")

        model._quantizer.train_offline = _boom

    try:
        model.on_train_end()
    except RuntimeError:
        dist.destroy_process_group()
        return  # expected: coordinated failure reached this rank
    dist.destroy_process_group()
    raise AssertionError(
        f"rank{rank}: on_train_end did not raise on a rank0 fit failure"
    )


def _run_dist_workers(worker, world_size: int, timeout: int = 120) -> None:
    """Spawn ``world_size`` procs running ``worker(rank, world_size, port)``.

    Joins with a timeout so a deadlock (e.g. a dropped barrier / reordered
    broadcast) fails the test instead of hanging CI, and raises on a hung or
    nonzero-exit worker.
    """
    port = misc_util.get_free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(world_size):
        p = ctx.Process(target=worker, args=(rank, world_size, port))
        p.start()
        procs.append(p)
    for i, p in enumerate(procs):
        p.join(timeout=timeout)
        if p.is_alive():
            p.terminate()
            raise RuntimeError(f"worker-{i} deadlocked (timed out after {timeout}s).")
        if p.exitcode != 0:
            raise RuntimeError(f"worker-{i} failed (exitcode={p.exitcode}).")


class SidRqkmeansDistTest(unittest.TestCase):
    """2-rank test for SidRqkmeans.on_train_end (gather -> fit -> broadcast)."""

    def test_on_train_end_ddp(self) -> None:
        _run_dist_workers(_on_train_end_worker, WORLD_SIZE)

    def test_on_train_end_ddp_rank0_failure(self) -> None:
        """A rank0-only fit failure raises on every rank — never deadlocks.

        Guards the status-flag-before-centroid-broadcast ordering: a regression
        that reordered/dropped it would hang, which the join timeout turns into
        a CI failure instead of a hung job.
        """
        _run_dist_workers(_on_train_end_fail_worker, WORLD_SIZE)


if __name__ == "__main__":
    unittest.main()
