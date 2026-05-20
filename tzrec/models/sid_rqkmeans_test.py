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


class SidRqkmeansOfflineTest(unittest.TestCase):
    """Tests for SidRqkmeans (FAISS-only)."""

    def _create_model(self, input_dim=32, n_layers=2, niter=5):
        """Create a SidRqkmeans configured for offline FAISS fit."""
        from google.protobuf.struct_pb2 import Struct

        n_embed_str = ",".join(["16"] * n_layers)

        faiss_kwargs = Struct()
        faiss_kwargs.update({"niter": niter, "verbose": False, "seed": 1234})

        sid_rqkmeans_cfg = sid_model_pb2.SidRqkmeans(
            input_dim=input_dim,
            codebook=n_embed_str,
            normalize_residuals=False,
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
        model = SidRqkmeans(model_config=model_config, features=[], labels=[])
        init_parameters(model, device=torch.device("cpu"))
        return model

    def test_proto_parse(self) -> None:
        """Verify faiss_kmeans_kwargs are parsed correctly."""
        model = self._create_model()
        self.assertEqual(model._faiss_kwargs.get("niter"), 5)
        self.assertEqual(model._faiss_kwargs.get("seed"), 1234)
        self.assertFalse(model._faiss_kwargs.get("verbose"))
        self.assertEqual(model._offline_buffer, [])

    def test_predict_collects_buffer(self) -> None:
        """In train mode, predict should append to buffer; never fit."""
        B, input_dim = 8, 32
        model = self._create_model(input_dim=input_dim)
        model.train()

        for _ in range(4):
            batch = _make_batch(B, input_dim)
            preds = model.predict(batch)
            self.assertIn("codes", preds)

        # Buffer accumulates 4 batches of B samples each
        self.assertEqual(len(model._offline_buffer), 4)
        total = sum(t.shape[0] for t in model._offline_buffer)
        self.assertEqual(total, 4 * B)
        # FAISS not yet triggered: layers should be uninitialized
        for layer in model._rqkmeans.quantizer.layers:
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
        self.assertGreater(len(model._offline_buffer), 0)

        # Trigger one-shot FAISS fit
        model.on_train_end()

        # Buffer should be cleared
        self.assertEqual(model._offline_buffer, [])
        # All layers should be initialized + centroids non-zero
        for layer in model._rqkmeans.quantizer.layers:
            self.assertTrue(bool(layer._is_initialized.item()))
            self.assertGreater(layer.centroids.abs().sum().item(), 0.0)

        # After fit, predict on eval should produce valid codes
        model.eval()
        preds = model.predict(_make_batch(B, input_dim))
        codes = preds["codes"]
        self.assertEqual(codes.shape, (B, 2))
        self.assertTrue((codes >= 0).all() and (codes < 16).all())

    def test_on_train_end_noop_on_empty_buffer(self) -> None:
        """on_train_end on an empty buffer is a warned no-op."""
        model = self._create_model()
        model.on_train_end()  # should not raise


if __name__ == "__main__":
    unittest.main()
