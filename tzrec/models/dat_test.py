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
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.dat import DAT
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class DATTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dat(self, graph_type) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_u", embedding_dim=4, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_i", embedding_dim=4, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_u")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_i")
            ),
        ]
        features = create_features(feature_cfgs, neg_fields=["cat_i", "int_i"])
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="user",
                feature_names=["cat_u", "int_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item",
                feature_names=["cat_i", "int_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="user_augment",
                feature_names=["cat_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item_augment",
                feature_names=["cat_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            dat=match_model_pb2.DAT(
                user_tower=tower_pb2.DATTower(
                    input="user",
                    augment_input="user_augment",
                    mlp=module_pb2.MLP(hidden_units=[12, 6]),
                ),
                item_tower=tower_pb2.DATTower(
                    input="item",
                    augment_input="item_augment",
                    mlp=module_pb2.MLP(hidden_units=[12, 6]),
                ),
                output_dim=4,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        dat = DAT(model_config=model_config, features=features, labels=["label"])
        init_parameters(dat, device=torch.device("cpu"))
        dat = create_test_model(dat, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_u"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_u"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sparse_neg_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_i"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_neg_feature = KeyedTensor.from_tensor_list(
            keys=["int_i"], tensors=[torch.tensor([[0.2], [0.3], [0.4], [0.5]])]
        )

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: dense_feature,
                NEG_DATA_GROUP: dense_neg_feature,
            },
            sparse_features={
                BASE_DATA_GROUP: sparse_feature,
                NEG_DATA_GROUP: sparse_neg_feature,
            },
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = dat(batch.to_dict())
        else:
            predictions = dat(batch)
        self.assertEqual(predictions["similarity"].size(), (2, 3))


if __name__ == "__main__":
    unittest.main()
    print(1)
