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
from parameterized import parameterized
from torchrec import KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.hstu import HSTUMatch
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    seq_encoder_pb2,
    tower_pb2,
)
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class HSTUTest(unittest.TestCase):
    @parameterized.expand([[TestGraphType.NORMAL], [TestGraphType.FX_TRACE]])
    def test_hstu(self, graph_type) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
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
                    embedding_name="item_id",
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="sequence",
                feature_names=["historical_ids"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            hstu_match=match_model_pb2.HSTUMatch(
                hstu_tower=tower_pb2.HSTUMatchTower(
                    input="sequence",
                    hstu_encoder=seq_encoder_pb2.HSTUEncoder(
                        sequence_dim=48,
                        attn_dim=48,
                        linear_dim=48,
                        input="sequence",
                        max_seq_length=210,
                        num_blocks=2,
                        num_heads=1,
                        linear_activation="silu",
                        linear_config="uvqk",
                        max_output_len=0,
                    ),
                ),
                temperature=0.05,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        hstu = HSTUMatch(model_config=model_config, features=features, labels=["label"])
        init_parameters(hstu, device=torch.device("cpu"))
        hstu = create_test_model(hstu, graph_type)

        # Create test sequence data
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["historical_ids"],
            values=torch.tensor([1, 2, 3, 4, 5, 2, 7, 8, 9, 4, 11, 5, 13, 14, 15]),
            # sequence length is
            # 2, 3, 2 (neg_seq, first is pos), 2 (neg_seq, first is pos)...
            lengths=torch.tensor([2, 3, 2, 2, 2, 2, 2]),
        )

        batch = Batch(
            sparse_features={
                NEG_DATA_GROUP: sparse_feature,
                BASE_DATA_GROUP: sparse_feature,
            },
            labels={"label": torch.tensor([1, 1])},
        )

        predictions = hstu(batch)
        self.assertEqual(predictions["similarity"].size(), (5, 2))


if __name__ == "__main__":
    unittest.main()
