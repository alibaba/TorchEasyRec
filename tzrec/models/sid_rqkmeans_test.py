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
from unittest import mock

import torch
import torch.distributed as dist
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.models.sid_rqkmeans import SidRqkmeans
from tzrec.protos import model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils.state_dict_util import init_parameters


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


class SidRqkmeansOfflineTest(unittest.TestCase):
    """Single-process tests for SidRqkmeans (FAISS-only)."""

    def setUp(self) -> None:
        # SidRqkmeans is CPU-only and refuses to init when CUDA is visible. The
        # GPU CI runners have CUDA, so simulate a CPU-only host for every
        # construction-based test. (test_init_raises_on_gpu overrides this.)
        patcher = mock.patch.object(torch.cuda, "is_available", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _create_model(
        self,
        input_dim=32,
        n_layers=2,
        niter=5,
        codebook=None,
        normalize_residuals=False,
        train_sample_size=0,
    ):
        """Build a SidRqkmeans on CPU with params initialized.

        SID models read the item-embedding dense feature directly from the
        batch and do not consume feature_groups, so none is set.
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
        model = SidRqkmeans(
            model_config=model_pb2.ModelConfig(sid_rqkmeans=cfg),
            features=[],
            labels=[],
        )
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_proto_parse(self) -> None:
        """Verify faiss_kmeans_kwargs are parsed correctly."""
        model = self._create_model()
        self.assertEqual(model._faiss_kwargs.get("niter"), 5)
        self.assertEqual(model._faiss_kwargs.get("seed"), 1234)
        self.assertFalse(model._faiss_kwargs.get("verbose"))
        self.assertEqual(model._reservoir.n_seen, 0)
        self.assertEqual(model._reservoir.n_filled, 0)

    def test_sample_cap_from_train_sample_size(self) -> None:
        """train_sample_size (when set) drives the reservoir cap directly."""
        # Explicit train_sample_size: cap == train_sample_size.
        model = self._create_model(train_sample_size=900)
        self.assertEqual(model._reservoir.capacity, 900)

        # Default (train_sample_size=0): cap == the FAISS fit's subsample size.
        model = self._create_model()
        self.assertEqual(
            model._reservoir.capacity, model._quantizer.default_fit_sample_size()
        )

    def test_init_raises_on_too_small_train_sample_size(self) -> None:
        """train_sample_size below the largest codebook fails fast at init."""
        with self.assertRaisesRegex(RuntimeError, "largest codebook"):
            self._create_model(codebook=[16, 16], train_sample_size=8)

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
        self.assertEqual(model._reservoir.n_seen, 4 * B)
        self.assertEqual(model._reservoir.n_filled, 4 * B)
        # FAISS not yet triggered: layers should be uninitialized
        for layer in model._quantizer.layers:
            self.assertFalse(layer.is_initialized)

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
        self.assertGreater(model._reservoir.n_seen, 0)

        # Trigger one-shot FAISS fit; a real fit must request a tail checkpoint
        self.assertTrue(model.on_train_end())

        # Reservoir should be released after the fit
        self.assertEqual(model._reservoir.n_seen, 0)
        self.assertEqual(model._reservoir.n_filled, 0)
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
            model._reservoir.capacity,
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
        """Eval exposes codes + quantized only; inference is codes-only."""
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

        # Eval mode: the centroid-sum reconstruction is exposed for
        # update_metric; the input embedding is NOT threaded through
        # predictions (it is re-extracted from the batch in update_metric).
        model.eval()
        eval_preds = model.predict(_make_batch(B, input_dim))
        self.assertEqual(set(eval_preds.keys()), {"codes", "quantized"})

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
        # Same batch through predict + update_metric: the reconstruction target
        # is re-extracted from this batch, so it must match the predicted one.
        batch = _make_batch(B, input_dim)
        preds = model.predict(batch)
        model.update_metric(preds, batch)
        metrics = model.compute_metric()
        for key in ("mse", "rel_loss", "unique_sid_ratio"):
            self.assertIn(key, metrics)
            self.assertTrue(torch.isfinite(torch.as_tensor(metrics[key])).all())

    def test_update_metric_skipped_before_fit(self) -> None:
        """Pre-fit eval (unfitted codebook) does not pollute metric state."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim)
        model.init_metric()
        model.eval()
        # Codebook not fitted yet: predict emits zeros; update_metric must skip.
        batch = _make_batch(B, input_dim)
        model.update_metric(model.predict(batch), batch)
        self.assertEqual(model._metric_modules["unique_sid_ratio"].count.item(), 0.0)

    def test_on_train_end_noop_on_empty_buffer(self) -> None:
        """on_train_end on an empty buffer is a warned no-op."""
        model = self._create_model()
        # No fit happened, so no tail checkpoint is requested.
        self.assertFalse(model.on_train_end())  # should not raise

    def test_init_raises_under_ddp(self) -> None:
        """SidRqkmeans is single-process only: world_size>1 fails fast in init."""
        with (
            mock.patch.object(dist, "is_available", return_value=True),
            mock.patch.object(dist, "is_initialized", return_value=True),
            mock.patch.object(dist, "get_world_size", return_value=2),
            self.assertRaisesRegex(RuntimeError, "single-process"),
        ):
            self._create_model()

    def test_init_raises_on_gpu(self) -> None:
        """SidRqkmeans is CPU-only: a visible CUDA device fails fast in init."""
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=True),
            self.assertRaisesRegex(RuntimeError, "CPU-only"),
        ):
            self._create_model()

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


if __name__ == "__main__":
    unittest.main()
