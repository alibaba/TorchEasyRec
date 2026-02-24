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
from tzrec.models.pepnet import EPNet, GateNU, PEPNet, PPNet
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
from tzrec.utils.test_util import TestGraphType, create_test_model, create_test_module


class GateNUTest(unittest.TestCase):
    """Test GateNU module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_gatenu(self, graph_type) -> None:
        """Test GateNU forward pass."""
        input_dim = 32
        hidden_dim = 16
        output_dim = 8

        gatenu = GateNU(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, gamma=2.0
        )
        gatenu = create_test_module(gatenu, graph_type)

        batch_size = 4
        x = torch.randn(batch_size, input_dim)

        output = gatenu(x)

        self.assertEqual(output.shape, (batch_size, output_dim))
        # Check that output is in [0, gamma] range due to sigmoid activation
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 2.0))


class EPNetTest(unittest.TestCase):
    """Test EPNet module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_epnet(self, graph_type) -> None:
        """Test EPNet forward pass."""
        domain_dim = 16
        embedding_dim = 32
        epnet = EPNet(
            main_dim=embedding_dim,
            domain_dim=domain_dim,
            hidden_dim=8,
        )
        epnet = create_test_module(epnet, graph_type)
        batch_size = 4
        domain_emb = torch.randn(batch_size, domain_dim)
        main_emb = torch.randn(batch_size, embedding_dim)
        personalized_emb = epnet(main_emb, domain_emb)
        self.assertEqual(personalized_emb.shape, (batch_size, embedding_dim))


class PPNetTest(unittest.TestCase):
    """Test PPNet module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_ppnet_forward(self, graph_type) -> None:
        """Test PPNet forward pass."""
        main_feature = 32
        uia_feature = 16
        num_task = 2

        ppnet = PPNet(
            main_feature=main_feature,
            uia_feature=uia_feature,
            num_task=num_task,
            hidden_units=[16, 8],
        )
        ppnet = create_test_module(ppnet, graph_type)
        batch_size = 4
        main_emb = torch.randn(batch_size, main_feature)
        uia_emb = torch.randn(batch_size, uia_feature)

        task_outputs = ppnet(main_emb, uia_emb)

        self.assertEqual(len(task_outputs), num_task)
        for output in task_outputs:
            self.assertEqual(
                output.shape, (batch_size, 8)
            )  # Output dim from hidden_units


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
                group_name="main",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]

        if use_epnet:
            feature_cfgs.append(
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="domain", embedding_dim=16, num_buckets=10
                    )
                )
            )
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="domain",
                    feature_names=["domain"],
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
            main_group_name="main",
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
            labels=["label1", "label2"],
        )
        init_parameters(pepnet, device=torch.device("cpu"))
        pepnet = create_test_model(pepnet, graph_type)

        sparse_keys = ["cat_a", "cat_b"]
        sparse_values = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        sparse_lengths = torch.tensor([1, 2, 1, 3])
        if use_epnet:
            sparse_keys.append("domain")
            sparse_values = torch.cat([sparse_values, torch.tensor([1, 2])])
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
                    feature_name="domain", embedding_dim=16, num_buckets=10
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="main",
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
                feature_names=["domain"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]

        pepnet_config = multi_task_rank_pb2.PEPNet(
            main_group_name="main",
            domain_group_name="domain",
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
                "domain",
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
