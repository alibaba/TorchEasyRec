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

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.wukong import WuKong
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class WuKongTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_wukong(self, graph_type=TestGraphType.NORMAL) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b", embedding_dim=16, num_buckets=1000
                )
            ),
        ]
        features = create_features(feature_cfgs)

        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            wukong=rank_model_pb2.WuKong(
                layers=[
                    module_pb2.WuKongLayer(
                        rank_feature_num=2,
                        feature_num_mlp=module_pb2.MLP(hidden_units=[4]),
                    ),
                    module_pb2.WuKongLayer(
                        rank_feature_num=2,
                        feature_num_mlp=module_pb2.MLP(hidden_units=[4]),
                    ),
                    module_pb2.WuKongLayer(
                        rank_feature_num=2,
                        feature_num_mlp=module_pb2.MLP(hidden_units=[4]),
                    ),
                ],
                final=module_pb2.MLP(hidden_units=[4, 2]),
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        wukong = WuKong(model_config=model_config, features=features, labels=["label"])
        init_parameters(wukong, device=torch.device("cpu"))
        wukong = create_test_model(wukong, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
            ],
            values=torch.tensor(list(range(6))),
            lengths=torch.tensor([1, 2, 1, 2]),
        )

        batch = Batch(
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = wukong(batch.to_dict())
        else:
            predictions = wukong(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
