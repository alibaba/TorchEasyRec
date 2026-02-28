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
from tzrec.models.pepnet import PEPNet
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


class PEPNetTest(unittest.TestCase):
    """Test PEPNet model."""

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False],
            [TestGraphType.FX_TRACE, False, False],
            [TestGraphType.JIT_SCRIPT, False, False],
            [TestGraphType.NORMAL, True, False],
            [TestGraphType.FX_TRACE, True, False],
            [TestGraphType.JIT_SCRIPT, True, False],
            [TestGraphType.NORMAL, False, True],
            [TestGraphType.FX_TRACE, False, True],
            [TestGraphType.JIT_SCRIPT, False, True],
            [TestGraphType.NORMAL, True, True],
            [TestGraphType.FX_TRACE, True, True],
            [TestGraphType.JIT_SCRIPT, True, True],
        ]
    )
    def test_pepnet(self, graph_type, use_epnet, use_ppnet) -> None:
        """Test PEPNet forward pass."""
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
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="all",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        feature_cfgs.append(
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="domainf", embedding_dim=16, num_buckets=3
                )
            )
        )
        if use_epnet:
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="domain",
                    feature_names=["domainf"],
                    group_type=model_pb2.FeatureGroupType.DEEP,
                )
            )
        if use_ppnet:
            feature_cfgs.append(
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="uia", embedding_dim=16, num_buckets=100
                    )
                )
            )
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="uia",
                    feature_names=["uia"],
                    group_type=model_pb2.FeatureGroupType.DEEP,
                )
            )

        features = create_features(feature_cfgs)

        pepnet_config = multi_task_rank_pb2.PEPNet(
            task_domain_num=3,
            domain_input_name="domainf",
            task_towers=[
                tower_pb2.TaskTower(
                    tower_name="t1",
                    label_name="label1",
                    mlp=module_pb2.MLP(hidden_units=[8, 4]),
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                ),
                tower_pb2.TaskTower(
                    tower_name="t2",
                    label_name="label2",
                    mlp=module_pb2.MLP(hidden_units=[8, 4]),
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                ),
            ],
        )
        if use_epnet:
            pepnet_config.domain_group_name = "domain"
            pepnet_config.epnet_hidden_unit = 8
            pepnet_config.epnet_gamma = 2.0
        if use_ppnet:
            pepnet_config.uia_group_name = "uia"
            pepnet_config.ppnet_hidden_units[:] = [16, 8]
            pepnet_config.ppnet_activation = "nn.ReLU"
            pepnet_config.ppnet_dropout_ratio[:] = [0.1, 0.1]
            pepnet_config.ppnet_gamma = 2.0

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            pepnet=pepnet_config,
        )

        pepnet = PEPNet(
            model_config=model_config,
            features=features,
            labels=["label1", "label2", "domainf"],
        )
        init_parameters(pepnet, device=torch.device("cpu"))
        pepnet = create_test_model(pepnet, graph_type)

        sparse_keys = ["cat_a", "cat_b"]
        sparse_values = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        sparse_lengths = torch.tensor([1, 2, 1, 3])
        sparse_keys.append("domainf")
        domain_values = torch.tensor([1, 2])
        sparse_values = torch.cat([sparse_values, domain_values])
        sparse_lengths = torch.cat([sparse_lengths, torch.tensor([1, 1])])
        if use_ppnet:
            sparse_keys.append("uia")
            sparse_values = torch.cat([sparse_values, torch.tensor([3, 4])])
            sparse_lengths = torch.cat([sparse_lengths, torch.tensor([1, 1])])

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=sparse_keys,
            values=sparse_values,
            lengths=sparse_lengths,
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sample_weights={"domainf": domain_values},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = pepnet(batch.to_dict())
        else:
            predictions = pepnet(batch)

        self.assertEqual(predictions["logits_t1_0"].size(), (2,))
        self.assertEqual(predictions["probs_t1_0"].size(), (2,))
        self.assertEqual(predictions["logits_t2_0"].size(), (2,))
        self.assertEqual(predictions["probs_t2_0"].size(), (2,))
        self.assertEqual(predictions["logits_t1_1"].size(), (2,))
        self.assertEqual(predictions["probs_t1_1"].size(), (2,))
        self.assertEqual(predictions["logits_t2_1"].size(), (2,))
        self.assertEqual(predictions["probs_t2_1"].size(), (2,))
        self.assertEqual(predictions["logits_t1_2"].size(), (2,))
        self.assertEqual(predictions["probs_t1_2"].size(), (2,))
        self.assertEqual(predictions["logits_t2_2"].size(), (2,))
        self.assertEqual(predictions["probs_t2_2"].size(), (2,))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_pepnet_has_sequences(self, graph_type) -> None:
        """Test PEPNet with sequence features."""
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
                id_feature=feature_pb2.IdFeature(
                    feature_name="domainf", embedding_dim=16, num_buckets=10
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="all",
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
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="domain",
                feature_names=["domainf"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]

        pepnet_config = multi_task_rank_pb2.PEPNet(
            epnet_hidden_unit=8,
            task_towers=[
                tower_pb2.TaskTower(
                    tower_name="t1",
                    label_name="label1",
                    mlp=module_pb2.MLP(hidden_units=[8, 4]),
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                ),
                tower_pb2.TaskTower(
                    tower_name="t2",
                    label_name="label2",
                    mlp=module_pb2.MLP(hidden_units=[8, 4]),
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                ),
            ],
        )

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            pepnet=pepnet_config,
        )

        pepnet = PEPNet(
            model_config=model_config,
            features=features,
            labels=["label1", "label2"],
        )
        init_parameters(pepnet, device=torch.device("cpu"))
        pepnet = create_test_model(pepnet, graph_type)

        # Batch size is 2
        # cat_a: [1, 1] -> 2 values
        # cat_b: [1, 1] -> 2 values
        # domain: [1, 1] -> 2 values
        # click_seq__cat_a: [2, 3] -> 5 values (sequence feature)
        # click_seq__cat_b: [2, 3] -> 5 values (sequence feature)
        # Total values: 2+2+2+5+5 = 16
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "domainf",
                "click_seq__cat_a",
                "click_seq__cat_b",
            ],
            values=torch.tensor(list(range(16))),
            lengths=torch.tensor([1, 1, 1, 1, 1, 1, 2, 3, 2, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        # click_seq__int_a: [2, 3] -> 5 values for 2 samples
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([2, 3]),
        ).to_dict()

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features=sequence_dense_feature,
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = pepnet(batch.to_dict())
        else:
            predictions = pepnet(batch)

        self.assertEqual(predictions["logits_t1"].size(), (2,))
        self.assertEqual(predictions["probs_t1"].size(), (2,))
        self.assertEqual(predictions["logits_t2"].size(), (2,))
        self.assertEqual(predictions["probs_t2"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
