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
from parameterized import parameterized
from torchrec import KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.hstu import HSTUMatch
from tzrec.modules.utils import Kernel
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model, gpu_unavailable


class HSTUMatchTest(unittest.TestCase):
    """Tests for the refactored HSTUMatch model with STU and jagged sequences."""

    @unittest.skipIf(*gpu_unavailable)
    @parameterized.expand([[TestGraphType.NORMAL], [TestGraphType.FX_TRACE]])
    def test_hstu_match(self, graph_type) -> None:
        """Test HSTUMatch with separate uih/candidate JAGGED_SEQUENCE groups."""
        device = torch.device("cuda")

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
                sequence_id_feature=feature_pb2.IdFeature(
                    feature_name="candidate_ids",
                    sequence_length=10,
                    embedding_dim=48,
                    num_buckets=3953,
                    embedding_name="historical_ids",
                )
            ),
        ]
        features = create_features(feature_cfgs)

        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="uih",
                feature_names=["historical_ids"],
                group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="candidate",
                feature_names=["candidate_ids"],
                group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
            ),
        ]

        model_config = model_pb2.ModelConfig(
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
                            sequence_preprocessor=(module_pb2.GRSequencePreprocessor()),
                        ),
                        output_postprocessor=module_pb2.GROutputPostprocessor(
                            l2norm_postprocessor=(module_pb2.GRL2NormPostprocessor()),
                        ),
                    ),
                    max_seq_len=210,
                ),
                temperature=0.05,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )

        hstu = HSTUMatch(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(hstu, device=device)
        hstu.to(device)
        hstu.set_kernel(Kernel.PYTORCH)
        hstu.eval()
        hstu = create_test_model(hstu, graph_type)

        # Build test batch: 2 users
        # UIH: user1 has 3 history items, user2 has 4
        # Candidates: user1 has 2 candidates (1 pos + 1 neg),
        #             user2 has 2 candidates
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["historical_ids", "candidate_ids"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]),
            lengths=torch.tensor(
                [3, 4, 2, 2]  # uih: [3,4], candidate: [2,2]
            ),
        )

        batch = Batch(
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={"label": torch.tensor([1, 1])},
        ).to(device)

        predictions = hstu(batch)
        self.assertIn("similarity", predictions)
        sim = predictions["similarity"]
        self.assertEqual(sim.dim(), 2)
        self.assertEqual(sim.size(0), 2)  # batch_size


if __name__ == "__main__":
    unittest.main()
