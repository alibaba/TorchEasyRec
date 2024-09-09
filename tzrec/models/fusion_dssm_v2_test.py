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

# Copyright (c) Alibaba, Inc. and its affiliates.
import unittest

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.fusion_dssm_v2 import FusionDSSMV2
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class FusionDSSMV2Test(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False],
            [TestGraphType.NORMAL, True, False],
            [TestGraphType.NORMAL, False, True],
            [TestGraphType.FX_TRACE, False, False],
            [TestGraphType.FX_TRACE, True, False],
            [TestGraphType.FX_TRACE, False, True],
        ]
    )
    def test_fusion_dssm_v2(self, graph_type, use_senet=False, perm_feat=False) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_u", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_i", embedding_dim=8, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_u")
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_i")
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="rec",
                feature_names=["cat_u", "int_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="search",
                feature_names=["cat_u", "int_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item",
                feature_names=["cat_i", "int_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            fusion_dssm_v2=match_model_pb2.FusionDSSMV2(
                user_tower=[
                    tower_pb2.Tower(
                        input="rec", mlp=module_pb2.MLP(hidden_units=[12, 6])
                    ),
                    tower_pb2.Tower(
                        input="search", mlp=module_pb2.MLP(hidden_units=[12, 6])
                    ),
                ],
                item_tower=tower_pb2.Tower(
                    input="item", mlp=module_pb2.MLP(hidden_units=[12, 6])
                ),
                output_dim=4,
                use_senet=use_senet,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        if perm_feat:
            model_config.fusion_dssm_v2.user_tower_perm_feat.append(
                match_model_pb2.UserTowerPermFeat(
                    tower_name="rec", features=["cat_u"], nflods=2
                )
            )
        dssm = FusionDSSMV2(
            model_config=model_config, features=features, labels=["label", "channel"]
        )
        init_parameters(dssm, device=torch.device("cpu"))
        dssm = create_test_model(dssm, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_u", "cat_i"],
            values=torch.tensor([1, 2, 3, 1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 0, 0, 1, 2, 1, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_u", "int_i"],
            tensors=[
                torch.tensor([[0.2], [0.3], [0.0], [0.0]]),
                torch.tensor([[0.2], [0.3], [0.4], [0.5]]),
            ],
        )

        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: dense_feature,
            },
            sparse_features={
                BASE_DATA_GROUP: sparse_feature,
            },
            labels={"channel": torch.tensor([0, 1])},
        )
        predictions = dssm(batch)
        if perm_feat:
            self.assertEqual(predictions["similarity_rec"].size(), (1, 5))
        else:
            self.assertEqual(predictions["similarity_rec"].size(), (1, 3))
        self.assertEqual(predictions["similarity_search"].size(), (1, 3))


if __name__ == "__main__":
    unittest.main()
