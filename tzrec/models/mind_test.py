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

from tzrec.datasets.utils import (
    BASE_DATA_GROUP,
    HARD_NEG_INDICES,
    NEG_DATA_GROUP,
    Batch,
)
from tzrec.features.feature import create_features
from tzrec.models.mind import MIND
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class MINDTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mind(self, graph_type) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_u", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_i", embedding_dim=16, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_50_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_i", embedding_dim=16, num_buckets=1000
                            )
                        )
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs, neg_fields=["cat_i"])
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="user",
                feature_names=["cat_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item",
                feature_names=["cat_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="hist",
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
                feature_names=["click_50_seq__cat_i"],
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            mind=match_model_pb2.MIND(
                user_tower=tower_pb2.MINDUserTower(
                    input="user",
                    history_input="hist",
                    user_mlp=module_pb2.MLP(hidden_units=[12, 6]),
                    hist_seq_mlp=module_pb2.MLP(hidden_units=[12, 16]),
                    capsule_config=module_pb2.B2ICapsule(
                        max_k=2,
                        max_seq_len=20,
                        high_dim=4,
                        num_iters=3,
                        routing_logits_scale=1.0,
                        routing_logits_stddev=0.1,
                        squash_pow=2.0,
                    ),
                    concat_mlp=module_pb2.MLP(hidden_units=[16, 8]),
                ),
                item_tower=tower_pb2.Tower(
                    input="item", mlp=module_pb2.MLP(hidden_units=[16, 8])
                ),
                simi_pow=10,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        mind = MIND(
            model_config=model_config,
            features=features,
            labels=["label"],
            sampler_type="negative_sampler",
        )
        init_parameters(mind, device=torch.device("cpu"))
        mind = create_test_model(mind, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_u", "click_50_seq__cat_i"],
            values=torch.tensor([1, 3, 4, 5]),
            lengths=torch.tensor([1, 3]),
        )

        sparse_neg_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_i"],
            values=torch.tensor([1, 2, 3]),
            lengths=torch.tensor([1, 1, 1]),
        )

        batch = Batch(
            sparse_features={
                BASE_DATA_GROUP: sparse_feature,
                NEG_DATA_GROUP: sparse_neg_feature,
            },
            labels={},
        )

        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = mind(batch.to_dict())
        else:
            predictions = mind(batch)
        self.assertEqual(predictions["similarity"].size(), (1, 3))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mind_hard_neg(self, graph_type) -> None:
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
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_50_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_i", embedding_dim=16, num_buckets=1000
                            )
                        )
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs, neg_fields=["cat_i"])
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="user",
                feature_names=["cat_u"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="item",
                feature_names=["cat_i"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="hist",
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
                feature_names=["click_50_seq__cat_i"],
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            mind=match_model_pb2.MIND(
                user_tower=tower_pb2.MINDUserTower(
                    input="user",
                    history_input="hist",
                    user_mlp=module_pb2.MLP(hidden_units=[12, 6]),
                    hist_seq_mlp=module_pb2.MLP(hidden_units=[12, 16]),
                    capsule_config=module_pb2.B2ICapsule(
                        max_k=2,
                        max_seq_len=20,
                        high_dim=4,
                        num_iters=3,
                        routing_logits_scale=1.0,
                        routing_logits_stddev=0.1,
                        squash_pow=2.0,
                    ),
                    concat_mlp=module_pb2.MLP(hidden_units=[16, 8]),
                ),
                item_tower=tower_pb2.Tower(
                    input="item", mlp=module_pb2.MLP(hidden_units=[16, 8])
                ),
                simi_pow=10,
            ),
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
        )
        mind = MIND(
            model_config=model_config,
            features=features,
            labels=["label"],
            sampler_type="hard_negative_sampler",
        )
        init_parameters(mind, device=torch.device("cpu"))
        mind = create_test_model(mind, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_u", "click_50_seq__cat_i"],
            values=torch.tensor([1, 3, 4, 5]),
            lengths=torch.tensor([1, 3]),
        )

        sparse_neg_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_i"],
            values=torch.tensor([1, 2, 3, 4, 5]),
            lengths=torch.tensor([1, 1, 1, 1, 1]),
        )

        hard_neg_indices = torch.tensor([[0, 0], [0, 1]])

        batch = Batch(
            sparse_features={
                BASE_DATA_GROUP: sparse_feature,
                NEG_DATA_GROUP: sparse_neg_feature,
            },
            labels={},
            additional_infos={HARD_NEG_INDICES: hard_neg_indices},
        )

        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = mind(batch.to_dict())
        else:
            predictions = mind(batch)
        self.assertEqual(predictions["similarity"].size(), (1, 5))


if __name__ == "__main__":
    unittest.main()
