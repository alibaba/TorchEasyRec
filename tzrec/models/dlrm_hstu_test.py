# Copyright (c) 2025, Alibaba Group;
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
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.dlrm_hstu import DlrmHSTU
from tzrec.ops import Kernel
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, gpu_unavailable


class DlrmHSTUTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, torch.device("cuda"), Kernel.PYTORCH],
            [TestGraphType.FX_TRACE, torch.device("cuda"), Kernel.PYTORCH],
            [TestGraphType.JIT_SCRIPT, torch.device("cuda"), Kernel.PYTORCH],
            [TestGraphType.NORMAL, torch.device("cuda"), Kernel.TRITON],
            [TestGraphType.FX_TRACE, torch.device("cuda"), Kernel.TRITON],
            [TestGraphType.JIT_SCRIPT, torch.device("cuda"), Kernel.TRITON],
        ]
    )
    @unittest.skipIf(*gpu_unavailable)
    def test_dlrm_hstu(self, graph_type, device, kernel) -> None:
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
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="action_timestamp"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="item_query_time"
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="action_weight",
                    num_buckets=1000,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_id_feature=feature_pb2.SequenceIdFeature(
                    feature_name="item_action_weight",
                    num_buckets=1000,
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
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="candidate",
                feature_names=[
                    "item_video_id",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]

        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            dlrm_hstu=multi_task_rank_pb2.DlrmHSTU(
                uih_id_feature_name="video_id",
                uih_action_time_feature_name="action_timestamp",
                uih_action_weight_feature_name="action_weight",
                uih_watchtime_feature_name="watch_time",
                candidates_id_feature_name="item_video_id",
                candidates_query_time_feature_name="item_query_time",
                candidates_action_weight_feature_name="item_action_weight",
                candidates_watchtime_feature_name="item_target_watchtime",
                hstu=module_pb2.HSTU(
                    stu=module_pb2.STU(
                        embedding_dim=512,
                        num_heads=4,
                        hidden_dim=128,
                        attention_dim=128,
                        output_dropout_ratio=0.2,
                    ),
                    positional_encoder=module_pb2.GRPositionalEncoder(
                        num_position_buckets=8192,
                        num_time_buckets=2048,
                        use_time_encoding=True,
                    ),
                    input_preprocessor=module_pb2.GRInputPreprocessor(
                        contextual_preprocessor=module_pb2.GRContextualPreprocessor(
                            contextual_feature_to_max_length={
                                "user_id": 1,
                                "user_active_degree": 1,
                            },
                            action_encoder=module_pb2.GRActionEncoder(
                                action_embedding_dim=8,
                                action_feature_name="action_weight",
                                action_weights=[1, 2, 4],
                            ),
                            action_mlp=module_pb2.GRContextualizedMLP(
                                simple_mlp=module_pb2.GRSimpleContextualizedMLP(
                                    hidden_dim=256
                                )
                            ),
                            content_mlp=module_pb2.GRContextualizedMLP(
                                simple_mlp=module_pb2.GRSimpleContextualizedMLP(
                                    hidden_dim=256
                                )
                            ),
                        )
                    ),
                    output_postprocessor=module_pb2.GROutputPostprocessor(
                        layernorm_postprocessor=module_pb2.GRLayerNormPostprocessor()
                    ),
                ),
                fusion_mtl_tower=tower_pb2.FusionMTLTower(
                    mlp=module_pb2.MLP(hidden_units=[512], activation="nn.SiLU"),
                    task_configs=[
                        tower_pb2.FusionSubTaskConfig(
                            task_name="is_click",
                            label_name="item_action_weight",
                            task_bitmask=1,
                            losses=[
                                loss_pb2.LossConfig(
                                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                                )
                            ],
                        ),
                        tower_pb2.FusionSubTaskConfig(
                            task_name="is_like",
                            label_name="item_action_weight",
                            task_bitmask=2,
                            losses=[
                                loss_pb2.LossConfig(
                                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                                )
                            ],
                        ),
                        tower_pb2.FusionSubTaskConfig(
                            task_name="is_comment",
                            label_name="item_action_weight",
                            task_bitmask=4,
                            losses=[
                                loss_pb2.LossConfig(
                                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                                )
                            ],
                        ),
                        tower_pb2.FusionSubTaskConfig(
                            task_name="watchtime",
                            label_name="item_target_watchtime",
                            losses=[loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())],
                        ),
                    ],
                ),
                max_seq_len=100,
            ),
        )
        dlrm_hstu = DlrmHSTU(
            model_config=model_config,
            features=features,
            labels=["item_action_weight", "item_target_watchtime"],
        )
        dlrm_hstu.set_kernel(kernel)
        if graph_type == TestGraphType.JIT_SCRIPT:
            dlrm_hstu.set_is_inference(True)
        init_parameters(dlrm_hstu, device=device)
        dlrm_hstu.to(device)
        # dlrm_hstu = create_test_model(dlrm_hstu, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "user_id",
                "user_active_degree",
                "video_id",
                "item_video_id",
                "action_weight",
                "item_action_weight",
                "action_timestamp",
                "item_query_time",
            ],
            values=torch.tensor(list(range(37))),
            lengths=torch.tensor([1, 1, 1, 1, 2, 3, 2, 4, 2, 3, 2, 4, 2, 3, 2, 4]),
        )
        sequence_dense_features = {
            "watch_time": JaggedTensor(
                values=torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]]),
                lengths=torch.tensor([2, 3]),
            ),
            "item_target_watchtime": JaggedTensor(
                values=torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5], [0.6]]),
                lengths=torch.tensor([2, 4]),
            ),
        }
        batch = Batch(
            sequence_dense_features=sequence_dense_features,
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        ).to(device)

        from tzrec.acc.export_utils import export_pm
        from tzrec.models.model import CudaExportWrapper

        export_pm(CudaExportWrapper(dlrm_hstu), batch.to_dict(), "tmp")

        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = dlrm_hstu(batch.to_dict(), device)
        else:
            predictions = dlrm_hstu(batch)
        self.assertEqual(predictions["logits_is_click"].size(), (6,))
        self.assertEqual(predictions["probs_is_click"].size(), (6,))
        self.assertEqual(predictions["logits_is_like"].size(), (6,))
        self.assertEqual(predictions["probs_is_like"].size(), (6,))
        self.assertEqual(predictions["logits_is_comment"].size(), (6,))
        self.assertEqual(predictions["probs_is_comment"].size(), (6,))


if __name__ == "__main__":
    unittest.main()
