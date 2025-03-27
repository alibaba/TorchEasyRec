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
from tzrec.models.dlrm_hstu import DlrmHSTU
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class DlrmHSTUTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dlrm_hstu(self, graph_type) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_active_degree",
                    embedding_dim=16,
                    num_buckets=1000,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="video_id",
                    embedding_dim=16,
                    embedding_name="video_id_emb",
                    num_buckets=1000,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="item_video_id",
                    embedding_dim=16,
                    embedding_name="video_id_emb",
                    num_buckets=1000,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="action_timestamp"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="item_query_time"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="action_weight"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="item_action_weight"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="watch_time"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.SequenceRawFeature(
                    feature_name="item_target_watchtime"
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="contextual",
                feature_names=["user_id", "user_active_degree"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="uih",
                feature_names=[
                    "video_id",
                    "action_timestamp",
                    "action_weight",
                    "watch_time",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="candidate",
                feature_names=[
                    "item_video_id",
                    "item_query_time",
                    "item_action_weight",
                    "item_target_watchtime",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            ple=multi_task_rank_pb2.DlrmHSTU(
                uih_post_id_feature_name="video_id",
                uih_action_time_feature_name="action_timestamp",
                uih_weight_feature_name="action_weight",
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
        dlrm_hstu = DlrmHSTU(
            model_config=model_config,
            features=features,
            labels=["label1", "label2", "label3"],
        )
        init_parameters(dlrm_hstu, device=torch.device("cpu"))
        dlrm_hstu = create_test_model(dlrm_hstu, graph_type)

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
            predictions = dlrm_hstu(batch.to_dict())
        else:
            predictions = dlrm_hstu(batch)
        self.assertEqual(predictions["logits_is_click"].size(), (2,))
        self.assertEqual(predictions["probs_is_click"].size(), (2,))
        self.assertEqual(predictions["logits_is_buy"].size(), (2,))
        self.assertEqual(predictions["probs_is_buy"].size(), (2,))
        self.assertEqual(predictions["y_cost_price"].size(), (2, 1))


if __name__ == "__main__":
    unittest.main()
