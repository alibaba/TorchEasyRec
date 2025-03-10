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

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.rocket_launching import RocketLaunching
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
)
from tzrec.protos.models import general_rank_model_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class RocketLaunchingTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
        ]
    )
    def test_rocket_launching(self, graph_type, is_training=True) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b", embedding_dim=8, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_a",
                                expression="item:cat_a",
                                embedding_dim=16,
                                num_buckets=100,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_b",
                                expression="item:cat_b",
                                embedding_dim=8,
                                num_buckets=1000,
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="int_a",
                                expression="item:int_a",
                            )
                        ),
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="t1",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="click_seq",
                        feature_names=[
                            "cat_a",
                            "cat_b",
                            "int_a",
                            "click_seq__cat_a",
                            "click_seq__cat_b",
                            "click_seq__int_a",
                        ],
                    )
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                ],
            )
        ]

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            rocket_launching=general_rank_model_pb2.RocketLaunching(
                share_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                booster_mlp=module_pb2.MLP(hidden_units=[32, 16, 8]),
                light_mlp=module_pb2.MLP(hidden_units=[16, 8]),
                feature_based_distillation=True,
            ),
        )
        rocket_launching = RocketLaunching(
            model_config=model_config,
            features=features,
            labels=["label"],
        )
        init_parameters(rocket_launching, device=torch.device("cpu"))
        if not is_training:
            rocket_launching.eval()
        rocket_launching = create_test_model(rocket_launching, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
            ],
            values=torch.tensor(list(range(14))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 2, 3, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([3, 2]),
        ).to_dict()

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features=sequence_dense_feature,
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = rocket_launching(batch.to_dict())
        else:
            predictions = rocket_launching(batch)
        self.assertEqual(predictions["logits_light"].size(), (2,))
        self.assertEqual(predictions["probs_light"].size(), (2,))
        if not is_training:
            self.assertTrue("logits_booster" not in predictions)
            self.assertTrue("probs_booster" not in predictions)
        else:
            self.assertEqual(predictions["logits_booster"].size(), (2,))
            self.assertEqual(predictions["probs_booster"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
