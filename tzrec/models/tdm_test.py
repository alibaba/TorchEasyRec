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
from tzrec.models.tdm import TDM
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class TDMTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_tdm(self, graph_type) -> None:
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
                                feature_name="cat_a", embedding_dim=16, num_buckets=100
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="cat_b", embedding_dim=8, num_buckets=1000
                            )
                        ),
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
                        ),
                    ],
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            tdm=match_model_pb2.TDM(
                multiwindow_din=tower_pb2.MultiWindowDINTower(
                    windows_len=[1, 2, 5], attn_mlp=module_pb2.MLP(hidden_units=[8, 4])
                ),
                final=module_pb2.MLP(hidden_units=[2]),
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        tdm = TDM(model_config=model_config, features=features, labels=["label"])
        init_parameters(tdm, device=torch.device("cpu"))
        tdm = create_test_model(tdm, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b", "click_seq__cat_a", "click_seq__cat_b"],
            values=torch.tensor(list(range(16))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a"],
            values=torch.tensor([[x] for x in range(6)], dtype=torch.float32),
            lengths=torch.tensor([3, 3]),
        ).to_dict()

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            sequence_dense_features=sequence_dense_feature,
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = tdm(batch.to_dict())
        else:
            predictions = tdm(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
