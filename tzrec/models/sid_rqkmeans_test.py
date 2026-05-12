# Copyright (c) 2024, Alibaba Group;
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
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.models.sid_rqkmeans import SidRqkmeans
from tzrec.protos import model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils.state_dict_util import init_parameters


def _make_batch(batch_size: int, input_dim: int) -> Batch:
    """Create a minimal Batch with dense embedding features."""
    dense_feature = KeyedTensor.from_tensor_list(
        keys=["item_emb"], tensors=[torch.randn(batch_size, input_dim)]
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: dense_feature},
        sparse_features={},
        labels={},
    )


class SidRqkmeansTest(unittest.TestCase):
    """Tests for SidRqkmeans model."""

    def _create_model(self, input_dim=32, n_layers=2):
        """Helper to create a SidRqkmeans model with minimal config."""
        n_embed_str = ",".join(["16"] * n_layers)
        sid_rqkmeans_cfg = sid_model_pb2.SidRqkmeans(
            input_dim=input_dim,
            codebook=n_embed_str,
            normalize_residuals=True,
            init_buffer_size=64,
            embedding_feature_name="item_emb",
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqkmeans=sid_rqkmeans_cfg,
        )
        model = SidRqkmeans(
            model_config=model_config, features=[], labels=[]
        )
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_kmeans_train_mode(self) -> None:
        """Test SidRqkmeans in train mode: predict -> loss -> metric."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        model.train()
        model.init_loss()
        model.init_metric()

        batch = _make_batch(B, input_dim)
        predictions = model.predict(batch)

        # Train mode should return codes + quantized + input_embedding
        self.assertIn("codes", predictions)
        self.assertIn("quantized", predictions)
        self.assertIn("input_embedding", predictions)
        self.assertEqual(predictions["codes"].shape[0], B)
        self.assertEqual(predictions["quantized"].shape, (B, input_dim))

        # Loss should return dummy_loss == 0
        losses = model.loss(predictions, batch)
        self.assertIn("dummy_loss", losses)
        self.assertEqual(losses["dummy_loss"].item(), 0.0)

        # Backward should not raise
        total_loss = losses["dummy_loss"]
        total_loss.backward()

        # Metric update should not raise
        model.update_metric(predictions, batch, losses)
        metrics = model.compute_metric()
        self.assertIn("mse", metrics)
        self.assertIn("unique_sid_ratio", metrics)

    def test_kmeans_eval_mode(self) -> None:
        """Test SidRqkmeans in eval mode."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        # Run one train step first to initialize centroids
        model.train()
        batch = _make_batch(B, input_dim)
        _ = model.predict(batch)

        # Switch to eval
        model.eval()
        predictions = model.predict(batch)

        self.assertIn("codes", predictions)
        self.assertIn("quantized", predictions)
        self.assertIn("input_embedding", predictions)

    def test_kmeans_inference_mode(self) -> None:
        """Test SidRqkmeans in inference mode: only codes returned."""
        B, input_dim = 4, 32
        model = self._create_model(input_dim=input_dim)
        # Run one train step first to initialize centroids
        model.train()
        batch = _make_batch(B, input_dim)
        _ = model.predict(batch)

        # Switch to inference
        model.eval()
        model.set_is_inference(True)
        predictions = model.predict(batch)

        self.assertIn("codes", predictions)
        self.assertNotIn("input_embedding", predictions)

    def test_kmeans_multi_step(self) -> None:
        """Test multiple train steps update centroids."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim)
        model.train()

        # Run multiple training steps
        for _ in range(3):
            batch = _make_batch(B, input_dim)
            predictions = model.predict(batch)

        # After training, codes should be valid indices
        codes = predictions["codes"]
        self.assertEqual(codes.shape, (B, 2))  # n_layers=2
        self.assertTrue((codes >= 0).all())
        self.assertTrue((codes < 16).all())  # n_embed=16


class SidRqkmeansOfflineTest(unittest.TestCase):
    """Tests for SidRqkmeans offline_faiss mode."""

    def _create_offline_model(self, input_dim=32, n_layers=2, niter=5):
        """Create a SidRqkmeans configured for offline_faiss mode."""
        from google.protobuf.struct_pb2 import Struct
        n_embed_str = ",".join(["16"] * n_layers)

        faiss_kwargs = Struct()
        faiss_kwargs.update({"niter": niter, "verbose": False, "seed": 1234})

        sid_rqkmeans_cfg = sid_model_pb2.SidRqkmeans(
            input_dim=input_dim,
            codebook=n_embed_str,
            normalize_residuals=False,
            init_buffer_size=64,
            train_mode="offline_faiss",
            faiss_kmeans_kwargs=faiss_kwargs,
            embedding_feature_name="item_emb",
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqkmeans=sid_rqkmeans_cfg,
        )
        model = SidRqkmeans(
            model_config=model_config, features=[], labels=[]
        )
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_offline_proto_parse(self) -> None:
        """Verify train_mode + faiss_kmeans_kwargs are parsed correctly."""
        model = self._create_offline_model()
        self.assertEqual(model._train_mode, "offline_faiss")
        self.assertEqual(model._faiss_kwargs.get("niter"), 5)
        self.assertEqual(model._faiss_kwargs.get("seed"), 1234)
        self.assertFalse(model._faiss_kwargs.get("verbose"))
        # Buffer should be initialized as empty list
        self.assertEqual(model._offline_buffer, [])

    def test_offline_default_train_mode_is_online(self) -> None:
        """Backward-compat: not setting train_mode should fallback to 'online'."""
        sid_rqkmeans_cfg = sid_model_pb2.SidRqkmeans(
            input_dim=32,
            codebook="16,16",
            normalize_residuals=True,
            init_buffer_size=64,
            embedding_feature_name="item_emb",
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqkmeans=sid_rqkmeans_cfg,
        )
        model = SidRqkmeans(
            model_config=model_config, features=[], labels=[]
        )
        self.assertEqual(model._train_mode, "online")
        # No offline_buffer attribute in online mode
        self.assertFalse(hasattr(model, "_offline_buffer"))

    def test_offline_predict_collects_buffer(self) -> None:
        """In offline+train mode, predict should append to buffer; never fit."""
        B, input_dim = 8, 32
        model = self._create_offline_model(input_dim=input_dim)
        model.train()

        for _ in range(4):
            batch = _make_batch(B, input_dim)
            preds = model.predict(batch)
            self.assertIn("codes", preds)

        # Buffer accumulates 4 batches of B samples each
        self.assertEqual(len(model._offline_buffer), 4)
        total = sum(t.shape[0] for t in model._offline_buffer)
        self.assertEqual(total, 4 * B)
        # FAISS not yet triggered: layer should be unlocked
        for layer in model._rqkmeans.quantizer.layers:
            self.assertFalse(bool(layer._offline_locked.item()))

    def test_offline_flush_runs_faiss_and_locks(self) -> None:
        """flush_offline_fit triggers FAISS fit, locks layers, clears buffer."""
        try:
            import faiss  # noqa: F401
        except ImportError:
            self.skipTest("faiss not installed")

        B, input_dim = 64, 32
        model = self._create_offline_model(input_dim=input_dim)
        model.train()

        # Accumulate enough samples (FAISS K-Means needs at least K points)
        for _ in range(8):
            model.predict(_make_batch(B, input_dim))
        self.assertGreater(len(model._offline_buffer), 0)

        # Trigger one-shot FAISS fit
        model.flush_offline_fit()

        # Buffer should be cleared
        self.assertEqual(model._offline_buffer, [])
        # All layers should be locked + initialized + centroids non-zero
        for layer in model._rqkmeans.quantizer.layers:
            self.assertTrue(bool(layer._offline_locked.item()))
            self.assertTrue(bool(layer._is_initialized.item()))
            self.assertGreater(layer.centroids.abs().sum().item(), 0.0)

        # After fit, predict on eval should produce valid codes
        model.eval()
        preds = model.predict(_make_batch(B, input_dim))
        codes = preds["codes"]
        self.assertEqual(codes.shape, (B, 2))
        self.assertTrue((codes >= 0).all() and (codes < 16).all())

    def test_offline_flush_noop_in_online_mode(self) -> None:
        """flush_offline_fit should be a no-op for online mode."""
        sid_rqkmeans_cfg = sid_model_pb2.SidRqkmeans(
            input_dim=32, codebook="16,16",
            embedding_feature_name="item_emb",
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["item_emb"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            sid_rqkmeans=sid_rqkmeans_cfg,
        )
        model = SidRqkmeans(
            model_config=model_config, features=[], labels=[]
        )
        # Should not raise
        model.flush_offline_fit()


if __name__ == "__main__":
    unittest.main()
