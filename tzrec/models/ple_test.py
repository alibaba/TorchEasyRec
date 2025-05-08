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
from tzrec.models.ple import PLE
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
    tower_pb2,
)
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model


class PLETest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_ple(self, graph_type) -> None:
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
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="t1",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        ]

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            ple=multi_task_rank_pb2.PLE(
                extraction_networks=[
                    module_pb2.ExtractionNetwork(
                        network_name="layer1",
                        expert_num_per_task=3,
                        share_num=4,
                        task_expert_net=module_pb2.MLP(hidden_units=[12, 8, 4]),
                        share_expert_net=module_pb2.MLP(hidden_units=[12, 8, 6, 4]),
                    ),
                    module_pb2.ExtractionNetwork(
                        network_name="layer2",
                        expert_num_per_task=3,
                        share_num=3,
                        task_expert_net=module_pb2.MLP(hidden_units=[8, 12, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[8, 12, 8]),
                    ),
                    module_pb2.ExtractionNetwork(
                        network_name="layer3",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[12, 6]),
                        share_expert_net=module_pb2.MLP(hidden_units=[12, 6]),
                    ),
                ],
                task_towers=[
                    tower_pb2.TaskTower(
                        tower_name="is_click",
                        label_name="label1",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                    tower_pb2.TaskTower(
                        tower_name="is_buy",
                        label_name="label2",
                        # mlp=module_pb2.MLP(hidden_units=[12, 6]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                    tower_pb2.TaskTower(
                        tower_name="cost_price",
                        label_name="label3",
                        mlp=module_pb2.MLP(hidden_units=[12, 6]),
                        losses=[loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())],
                    ),
                ],
            ),
        )
        ple = PLE(
            model_config=model_config,
            features=features,
            labels=["label1", "label2", "label3"],
        )
        init_parameters(ple, device=torch.device("cpu"))
        ple = create_test_model(ple, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = ple(batch.to_dict())
        else:
            predictions = ple(batch)
        self.assertEqual(predictions["logits_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["logits_is_buy"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        self.assertEqual(predictions["y_cost_price"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_ple_has_sequences(self, graph_type) -> None:
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
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="buy_seq",
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
                    ),
                    model_pb2.SeqGroupConfig(
                        group_name="buy_seq",
                        feature_names=[
                            "cat_a",
                            "int_a",
                            "buy_seq__cat_a",
                            "buy_seq__int_a",
                        ],
                    ),
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                    seq_encoder_pb2.SeqEncoderConfig(
                        simple_attention=seq_encoder_pb2.SimpleAttention(
                            input="buy_seq"
                        )
                    ),
                ],
            )
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            ple=multi_task_rank_pb2.PLE(
                extraction_networks=[
                    module_pb2.ExtractionNetwork(
                        network_name="layer1",
                        expert_num_per_task=3,
                        share_num=4,
                        task_expert_net=module_pb2.MLP(hidden_units=[12, 8, 4]),
                        share_expert_net=module_pb2.MLP(hidden_units=[12, 8, 6, 4]),
                    ),
                    module_pb2.ExtractionNetwork(
                        network_name="layer2",
                        expert_num_per_task=3,
                        share_num=3,
                        task_expert_net=module_pb2.MLP(hidden_units=[8, 12, 8]),
                        share_expert_net=module_pb2.MLP(hidden_units=[8, 12, 8]),
                    ),
                    module_pb2.ExtractionNetwork(
                        network_name="layer3",
                        expert_num_per_task=2,
                        share_num=2,
                        task_expert_net=module_pb2.MLP(hidden_units=[12, 6]),
                        share_expert_net=module_pb2.MLP(hidden_units=[12, 6]),
                    ),
                ],
                task_towers=[
                    tower_pb2.TaskTower(
                        tower_name="is_click",
                        label_name="label1",
                        mlp=module_pb2.MLP(hidden_units=[8, 4]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                    tower_pb2.TaskTower(
                        tower_name="is_buy",
                        label_name="label2",
                        # mlp=module_pb2.MLP(hidden_units=[12, 6]),
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                    ),
                    tower_pb2.TaskTower(
                        tower_name="cost_price",
                        label_name="label3",
                        mlp=module_pb2.MLP(hidden_units=[12, 6]),
                        losses=[loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())],
                    ),
                ],
            ),
        )
        ple = PLE(
            model_config=model_config,
            features=features,
            labels=["label1", "label2", "label3"],
        )
        init_parameters(ple, device=torch.device("cpu"))
        ple = create_test_model(ple, graph_type)
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(20))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 2, 3, 2, 2, 1, 2, 1]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(8)], dtype=torch.float32),
            lengths=torch.tensor([3, 2, 2, 1]),
        ).to_dict()
        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features=sequence_dense_feature,
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = ple(batch.to_dict())
        else:
            predictions = ple(batch)
        self.assertEqual(predictions["logits_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["logits_is_buy"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        self.assertEqual(predictions["y_cost_price"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
