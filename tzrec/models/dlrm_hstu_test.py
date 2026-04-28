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

import os
import shutil
import tempfile
import unittest
from collections import OrderedDict

import torch
from hypothesis import Verbosity, assume, given, settings
from hypothesis import strategies as st
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.dlrm_hstu import DlrmHSTU
from tzrec.models.model import TrainWrapper
from tzrec.ops import Kernel
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    metric_pb2,
    model_pb2,
    module_pb2,
    tower_pb2,
)
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model, gpu_unavailable


def _build_model(
    device: torch.device,
    has_watchtime: bool = False,
    contextual_group_type: int = model_pb2.FeatureGroupType.DEEP,
    enable_global_average_loss: bool = False,
    sequence_timestamp_is_ascending: bool = False,
    task_weight: float = 1.0,
    concat_contextual_features: bool = False,
) -> DlrmHSTU:
    """Build a DlrmHSTU model with standard test configuration."""
    uih_seq_features = [
        feature_pb2.SeqFeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="video_id",
                embedding_dim=16,
                embedding_name="video_id_emb",
                num_buckets=1000,
            )
        ),
        feature_pb2.SeqFeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="video_cat",
                embedding_dim=16,
                embedding_name="video_cat_emb",
                num_buckets=100,
            )
        ),
        feature_pb2.SeqFeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="action_timestamp")
        ),
        feature_pb2.SeqFeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="action_weight")
        ),
    ]
    if has_watchtime:
        uih_seq_features.append(
            feature_pb2.SeqFeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="watch_time")
            )
        )
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
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="uih_seq",
                features=uih_seq_features,
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="cand_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="item_video_id",
                            embedding_dim=16,
                            embedding_name="video_id_emb",
                            num_buckets=1000,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="item_video_cat",
                            embedding_dim=16,
                            embedding_name="video_cat_emb",
                            num_buckets=100,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="item_query_time"
                        )
                    ),
                ],
            )
        ),
    ]
    features = create_features(feature_cfgs)
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="contextual",
            feature_names=["user_id", "user_active_degree"],
            group_type=contextual_group_type,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih",
            feature_names=["uih_seq__video_id", "uih_seq__video_cat"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate",
            feature_names=[
                "cand_seq__item_video_id",
                "cand_seq__item_video_cat",
            ],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_timestamp",
            feature_names=[
                "uih_seq__action_timestamp",
            ],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate_timestamp",
            feature_names=[
                "cand_seq__item_query_time",
            ],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_action",
            feature_names=[
                "uih_seq__action_weight",
            ],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
    ]
    if has_watchtime:
        feature_groups.append(
            model_pb2.FeatureGroupConfig(
                group_name="uih_watchtime",
                feature_names=[
                    "uih_seq__watch_time",
                ],
                group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
            )
        )

    task_configs = [
        tower_pb2.FusionSubTaskConfig(
            task_name="is_click",
            label_name="item_action_weight",
            task_bitmask=1,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            metrics=[metric_pb2.MetricConfig(auc=metric_pb2.AUC())],
        ),
        tower_pb2.FusionSubTaskConfig(
            task_name="is_like",
            label_name="item_action_weight",
            task_bitmask=2,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            metrics=[
                metric_pb2.MetricConfig(
                    grouped_auc=metric_pb2.GroupedAUC(grouping_key="user_id")
                )
            ],
            weight=task_weight,
        ),
        tower_pb2.FusionSubTaskConfig(
            task_name="is_comment",
            label_name="item_action_weight",
            task_bitmask=4,
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            metrics=[metric_pb2.MetricConfig(auc=metric_pb2.AUC())],
        ),
    ]
    labels = ["item_action_weight"]
    if has_watchtime:
        task_configs.append(
            tower_pb2.FusionSubTaskConfig(
                task_name="watchtime",
                label_name="item_target_watchtime",
                losses=[loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())],
                metrics=[
                    metric_pb2.MetricConfig(
                        mean_absolute_error=metric_pb2.MeanAbsoluteError()
                    )
                ],
            )
        )
        labels.append("item_target_watchtime")

    model_config = model_pb2.ModelConfig(
        feature_groups=feature_groups,
        dlrm_hstu=multi_task_rank_pb2.DlrmHSTU(
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
                        action_encoder=module_pb2.GRActionEncoder(
                            # has_watchtime also enables the watchtime->action
                            # bitwise-OR so AOT export traces the cross-sequence
                            # op that constrains uih_action and uih_watchtime nnz.
                            simple_action_encoder=module_pb2.GRSimpleActionEncoder(
                                action_embedding_dim=8,
                                action_weights=[1, 2, 4],
                                watchtime_to_action_thresholds=(
                                    [60, 300] if has_watchtime else []
                                ),
                                watchtime_to_action_weights=(
                                    [256, 512] if has_watchtime else []
                                ),
                            )
                        ),
                        action_mlp=module_pb2.GRContextualizedMLP(
                            simple_mlp=module_pb2.GRSimpleContextualizedMLP(
                                hidden_dim=256
                            )
                        ),
                        content_encoder=module_pb2.GRContentEncoder(
                            slice_content_encoder=module_pb2.GRSliceContentEncoder()
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
                task_configs=task_configs,
            ),
            max_seq_len=100,
            enable_global_average_loss=enable_global_average_loss,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            concat_contextual_features=concat_contextual_features,
        ),
    )
    dlrm_hstu = DlrmHSTU(
        model_config=model_config,
        features=features,
        labels=labels,
    )
    init_parameters(dlrm_hstu, device=device)
    dlrm_hstu.to(device)
    return dlrm_hstu


def _build_batch(
    device: torch.device,
    has_watchtime: bool = False,
) -> Batch:
    """Build a test Batch with standard test data."""
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=[
            "user_id",
            "user_active_degree",
            "uih_seq__video_id",
            "cand_seq__item_video_id",
            "uih_seq__video_cat",
            "cand_seq__item_video_cat",
        ],
        values=torch.tensor(list(range(26))),
        lengths=torch.tensor([1, 1, 1, 1, 2, 3, 2, 4, 2, 3, 2, 4]),
    )
    sequence_dense_features = {
        "uih_seq__action_timestamp": JaggedTensor(
            values=torch.tensor([[1], [2], [3], [4], [5]]),
            lengths=torch.tensor([2, 3]),
        ),
        "cand_seq__item_query_time": JaggedTensor(
            values=torch.tensor([[6], [7], [8], [9], [10], [11]]),
            lengths=torch.tensor([2, 4]),
        ),
        "uih_seq__action_weight": JaggedTensor(
            values=torch.tensor([[0], [1], [0], [1], [0]]),
            lengths=torch.tensor([2, 3]),
        ),
    }
    if has_watchtime:
        sequence_dense_features["uih_seq__watch_time"] = JaggedTensor(
            values=torch.tensor([[0.1], [0.2], [0.3], [0.4], [0.5]]),
            lengths=torch.tensor([2, 3]),
        )
    jagged_labels = {
        "item_action_weight": JaggedTensor(
            values=torch.tensor([0, 1, 0, 0, 1, 0]),
            lengths=torch.tensor([2, 4]),
        ),
        "item_target_watchtime": JaggedTensor(
            values=torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            lengths=torch.tensor([2, 4]),
        ),
    }
    return Batch(
        sequence_dense_features=sequence_dense_features,
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels={},
        jagged_labels=jagged_labels,
    ).to(device)


class DlrmHSTUTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(*gpu_unavailable)
    @given(
        graph_type=st.sampled_from(
            [
                TestGraphType.NORMAL,
                TestGraphType.FX_TRACE,
                TestGraphType.JIT_SCRIPT,
                TestGraphType.AOT_INDUCTOR,
            ]
        ),
        kernel=st.sampled_from([Kernel.PYTORCH, Kernel.TRITON]),
        has_watchtime=st.sampled_from([True, False]),
        enable_global_average_loss=st.sampled_from([True, False]),
        contextual_group_type=st.sampled_from(
            [model_pb2.FeatureGroupType.DEEP, model_pb2.FeatureGroupType.SEQUENCE]
        ),
        sequence_timestamp_is_ascending=st.sampled_from([True, False]),
        task_weight=st.sampled_from([1.0, 0.5]),
        concat_contextual_features=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_dlrm_hstu(
        self,
        graph_type,
        kernel,
        has_watchtime,
        enable_global_average_loss,
        contextual_group_type,
        sequence_timestamp_is_ascending,
        task_weight,
        concat_contextual_features,
    ) -> None:
        # JIT_SCRIPT only support PYTORCH kernel now.
        assume(
            (graph_type == TestGraphType.JIT_SCRIPT and kernel == Kernel.PYTORCH)
            or graph_type != TestGraphType.JIT_SCRIPT
        )

        device = torch.device("cuda")
        dlrm_hstu = _build_model(
            device=device,
            has_watchtime=has_watchtime,
            contextual_group_type=contextual_group_type,
            enable_global_average_loss=enable_global_average_loss,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            task_weight=task_weight,
            concat_contextual_features=concat_contextual_features,
        )
        dlrm_hstu.set_kernel(kernel)
        batch = _build_batch(device=device, has_watchtime=has_watchtime)

        if graph_type == TestGraphType.JIT_SCRIPT:
            dlrm_hstu.set_is_inference(True)
            dlrm_hstu = create_test_model(dlrm_hstu, graph_type)
            predictions = dlrm_hstu(batch.to_dict(), device)
        elif graph_type == TestGraphType.AOT_INDUCTOR:
            data = batch.to_dict()
            data = OrderedDict(sorted(data.items()))
            self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
            dlrm_hstu.set_is_inference(True)
            dlrm_hstu = create_test_model(dlrm_hstu, graph_type, data, self.test_dir)
            predictions = dlrm_hstu(data)
        elif graph_type == TestGraphType.FX_TRACE:
            dlrm_hstu = create_test_model(dlrm_hstu, graph_type)
            predictions = dlrm_hstu(batch)
        else:
            dlrm_hstu = TrainWrapper(dlrm_hstu, device=device).to(device)
            _, (_, predictions, batch) = dlrm_hstu(batch)
            dlrm_hstu.model.update_metric(predictions, batch)
            _ = dlrm_hstu.model.compute_metric()

        self.assertEqual(predictions["logits_is_click"].size(), (6,))
        self.assertEqual(predictions["probs_is_click"].size(), (6,))
        self.assertEqual(predictions["logits_is_like"].size(), (6,))
        self.assertEqual(predictions["probs_is_like"].size(), (6,))
        self.assertEqual(predictions["logits_is_comment"].size(), (6,))
        self.assertEqual(predictions["probs_is_comment"].size(), (6,))

    @unittest.skipIf(*gpu_unavailable)
    def test_dlrm_hstu_task_weight(self) -> None:
        device = torch.device("cuda")
        task_weight = 0.5

        model_unweighted = _build_model(device=device)
        model_weighted = _build_model(device=device, task_weight=task_weight)

        # Copy weights so both models produce the same predictions.
        model_weighted.load_state_dict(model_unweighted.state_dict())

        model_unweighted.init_loss()
        model_weighted.init_loss()

        batch = _build_batch(device=device)

        predictions = model_unweighted.predict(batch)
        losses_unweighted = model_unweighted.loss(predictions, batch)
        losses_weighted = model_weighted.loss(predictions, batch)

        # is_click loss should be identical (weight=1.0 in both).
        self.assertTrue(
            torch.allclose(
                losses_weighted["binary_cross_entropy_is_click"],
                losses_unweighted["binary_cross_entropy_is_click"],
            )
        )
        # is_like loss should be scaled by task_weight.
        self.assertTrue(
            torch.allclose(
                losses_weighted["binary_cross_entropy_is_like"],
                losses_unweighted["binary_cross_entropy_is_like"] * task_weight,
            )
        )


if __name__ == "__main__":
    unittest.main()
