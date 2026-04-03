# Copyright (c) 2024-2025, Alibaba Group;
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
from torchrec import KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.hstu import HSTUMatch
from tzrec.models.model import TrainWrapper
from tzrec.modules.utils import Kernel
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    metric_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model, gpu_unavailable


def _build_model_config():
    """Build HSTUMatch model config for tests."""
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="uih",
            feature_names=["historical_ids"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate",
            feature_names=["item_id"],
            group_type=model_pb2.FeatureGroupType.DEEP,
        ),
    ]
    return model_pb2.ModelConfig(
        feature_groups=feature_groups,
        hstu_match=match_model_pb2.HSTUMatch(
            hstu_tower=tower_pb2.HSTUMatchTower(
                input="uih",
                hstu=module_pb2.HSTU(
                    stu=module_pb2.STU(
                        embedding_dim=48,
                        num_heads=1,
                        hidden_dim=48,
                        attention_dim=48,
                        output_dropout_ratio=0.2,
                    ),
                    attn_num_layers=2,
                    positional_encoder=module_pb2.GRPositionalEncoder(
                        num_position_buckets=512,
                    ),
                    input_preprocessor=module_pb2.GRInputPreprocessor(
                        sequence_preprocessor=module_pb2.GRSequencePreprocessor(),
                    ),
                    output_postprocessor=module_pb2.GROutputPostprocessor(
                        l2norm_postprocessor=module_pb2.GRL2NormPostprocessor(),
                    ),
                ),
                max_seq_len=210,
            ),
            temperature=0.05,
        ),
        losses=[
            loss_pb2.LossConfig(softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy())
        ],
        metrics=[metric_pb2.MetricConfig(recall_at_k=metric_pb2.RecallAtK(top_k=1))],
    )


def _build_features():
    """Build features for HSTUMatch tests."""
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="historical_ids",
                sequence_length=210,
                embedding_dim=48,
                num_buckets=3953,
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="item_id",
                embedding_dim=48,
                num_buckets=1000,
            )
        ),
    ]
    return create_features(feature_cfgs, neg_fields=["item_id"])


def _build_model(device):
    """Build HSTUMatch model on device."""
    model_config = _build_model_config()
    features = _build_features()
    hstu = HSTUMatch(
        model_config=model_config,
        features=features,
        labels=["label"],
        sampler_type="negative_sampler",
    )
    init_parameters(hstu, device=device)
    hstu.to(device)
    hstu.set_kernel(Kernel.PYTORCH)
    return hstu


def _build_batch(device):
    """Build test batch with 2 users.

    UIH: user1 has 3 items, user2 has 4 items.
    Candidates: 2 pos (1 per user) + 2 neg items.
    """
    # BASE: UIH sequences + positive items
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["historical_ids"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        lengths=torch.tensor([3, 4]),
    )
    # NEG: positive items (first batch_size) + negative items
    neg_sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["item_id"],
        values=torch.tensor([10, 11, 20, 21]),
        lengths=torch.tensor([1, 1, 1, 1]),  # 2 pos + 2 neg, each 1 item
    )
    return Batch(
        sparse_features={
            BASE_DATA_GROUP: sparse_feature,
            NEG_DATA_GROUP: neg_sparse_feature,
        },
        labels={"label": torch.tensor([1, 1])},
    ).to(device)


class HSTUMatchTest(unittest.TestCase):
    """Tests for HSTUMatch model with STU and jagged sequences."""

    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_match_train(self) -> None:
        """Test HSTUMatch training: forward + loss + backward."""
        device = torch.device("cuda")
        hstu = _build_model(device)
        batch = _build_batch(device)

        train_model = TrainWrapper(hstu, device=device).to(device)
        total_loss, (losses, predictions, batch) = train_model(batch)

        self.assertIn("similarity", predictions)
        self.assertIn("softmax_cross_entropy", losses)
        self.assertTrue(total_loss.requires_grad)
        self.assertFalse(torch.isnan(total_loss))

    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_match_eval(self) -> None:
        """Test HSTUMatch evaluation: forward + metrics."""
        device = torch.device("cuda")
        hstu = _build_model(device)
        batch = _build_batch(device)

        train_model = TrainWrapper(hstu, device=device).to(device)
        _, (_, predictions, batch) = train_model(batch)

        hstu.update_metric(predictions, batch)
        metric_result = hstu.compute_metric()
        self.assertIn("recall@1", metric_result)

    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_match_export(self) -> None:
        """Test HSTUMatch export: FX trace for serving."""
        device = torch.device("cuda")
        hstu = _build_model(device)
        batch = _build_batch(device)

        hstu.eval()
        hstu = create_test_model(hstu, TestGraphType.FX_TRACE)
        predictions = hstu(batch)

        self.assertIn("similarity", predictions)
        sim = predictions["similarity"]
        self.assertEqual(sim.dim(), 2)
        self.assertEqual(sim.size(0), 2)

    @unittest.skipIf(*gpu_unavailable)
    def test_hstu_match_predict(self) -> None:
        """Test HSTUMatch predict: inference mode forward pass."""
        device = torch.device("cuda")
        hstu = _build_model(device)
        batch = _build_batch(device)

        hstu.eval()
        with torch.no_grad():
            predictions = hstu.predict(batch)

        self.assertIn("similarity", predictions)
        sim = predictions["similarity"]
        self.assertEqual(sim.dim(), 2)
        self.assertEqual(sim.size(0), 2)
        self.assertFalse(torch.isnan(sim).any())


if __name__ == "__main__":
    unittest.main()
