# Copyright (c) 2026, Alibaba Group;
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
from typing import List, Tuple

import torch
from hypothesis import Verbosity, assume, given, settings
from hypothesis import strategies as st
from parameterized import parameterized
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.model import TrainWrapper
from tzrec.models.ultra_hstu import UltraHSTU, _HSTUTransducerStack
from tzrec.modules.gr.hstu_transducer import HSTUTransducer
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


def _hstu_subconfig(channel_name: str, embedding_dim: int) -> module_pb2.HSTU:
    return module_pb2.HSTU(
        name=channel_name,
        stu=module_pb2.STU(
            embedding_dim=embedding_dim,
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
                    simple_action_encoder=module_pb2.GRSimpleActionEncoder(
                        action_embedding_dim=8,
                        action_weights=[1, 2, 4],
                        watchtime_to_action_thresholds=[],
                        watchtime_to_action_weights=[],
                    )
                ),
                action_mlp=module_pb2.GRContextualizedMLP(
                    simple_mlp=module_pb2.GRSimpleContextualizedMLP(hidden_dim=256)
                ),
                content_encoder=module_pb2.GRContentEncoder(
                    slice_content_encoder=module_pb2.GRSliceContentEncoder()
                ),
                content_mlp=module_pb2.GRContextualizedMLP(
                    simple_mlp=module_pb2.GRSimpleContextualizedMLP(hidden_dim=256)
                ),
            )
        ),
        output_postprocessor=module_pb2.GROutputPostprocessor(
            layernorm_postprocessor=module_pb2.GRLayerNormPostprocessor()
        ),
    )


def _build_model(
    device: torch.device,
    channel_specs: List[Tuple[str, int]],
    contextual_group_type: int = model_pb2.FeatureGroupType.DEEP,
    enable_global_average_loss: bool = False,
    sequence_timestamp_is_ascending: bool = False,
    concat_contextual_features: bool = False,
) -> UltraHSTU:
    """Build an UltraHSTU model with the requested MoT channels.

    Each channel reuses a shared underlying embedding table for
    ``video_id`` and ``video_cat`` (the sparse tables are deduped by
    ``embedding_name`` inside ``EmbeddingGroup``); the per-channel STU
    embedding_dim varies per ``channel_specs`` so that
    ``_stu_embedding_dim()`` exercises the sum-across-channels path.
    """
    # One sequence feature per channel; the candidate sequence is
    # shared.  All channels share the same embedding_name on each
    # underlying feature so the embedding tables are deduplicated.
    channel_seq_features = [
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name=f"{channel}_seq",
                features=[
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
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="action_timestamp"
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(feature_name="action_weight")
                    ),
                ],
            )
        )
        for channel, _ in channel_specs
    ]
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
        *channel_seq_features,
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="cand_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="item_video_id",
                            embedding_dim=32,
                            embedding_name="item_video_id_emb",
                            num_buckets=1000,
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
            group_name="candidate",
            feature_names=["cand_seq__item_video_id"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate_timestamp",
            feature_names=["cand_seq__item_query_time"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
    ]
    # Per-channel UIH-side groups; the names must match the HSTU channel
    # name so the preprocessor's name-based key routing finds them.
    for channel, _ in channel_specs:
        feature_groups.extend(
            [
                model_pb2.FeatureGroupConfig(
                    group_name=channel,
                    feature_names=[
                        f"{channel}_seq__video_id",
                        f"{channel}_seq__video_cat",
                    ],
                    group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
                ),
                model_pb2.FeatureGroupConfig(
                    group_name=f"{channel}_timestamp",
                    feature_names=[f"{channel}_seq__action_timestamp"],
                    group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
                ),
                model_pb2.FeatureGroupConfig(
                    group_name=f"{channel}_action",
                    feature_names=[f"{channel}_seq__action_weight"],
                    group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
                ),
            ]
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
            metrics=[metric_pb2.MetricConfig(auc=metric_pb2.AUC())],
        ),
    ]
    labels = ["item_action_weight"]

    model_config = model_pb2.ModelConfig(
        feature_groups=feature_groups,
        ultra_hstu=multi_task_rank_pb2.UltraHSTU(
            hstu=[_hstu_subconfig(name, dim) for name, dim in channel_specs],
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
    ultra_hstu = UltraHSTU(
        model_config=model_config,
        features=features,
        labels=labels,
    )
    init_parameters(ultra_hstu, device=device)
    ultra_hstu.to(device)
    return ultra_hstu


def _build_batch(device: torch.device, channel_names: List[str]) -> Batch:
    """Build a test Batch keyed off the requested MoT channel names.

    Per-row UIH lengths are [2, 3] and candidate lengths are [2, 4].
    Per channel the KJT contributes 10 sparse values (5 video_id + 5
    video_cat across 2 samples) and the dense dict contributes one
    action_timestamp tensor and one action_weight tensor of length 5.
    """
    sparse_keys = ["user_id", "user_active_degree"]
    sparse_lengths = [1, 1, 1, 1]
    sparse_values: List[int] = list(range(4))
    next_value = 4
    for channel in channel_names:
        sparse_keys.extend([f"{channel}_seq__video_id", f"{channel}_seq__video_cat"])
        sparse_lengths.extend([2, 3, 2, 3])
        sparse_values.extend(range(next_value, next_value + 10))
        next_value += 10
    sparse_keys.append("cand_seq__item_video_id")
    sparse_lengths.extend([2, 4])
    sparse_values.extend(range(next_value, next_value + 6))

    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=sparse_keys,
        values=torch.tensor(sparse_values),
        lengths=torch.tensor(sparse_lengths),
    )

    sequence_dense_features = {
        "cand_seq__item_query_time": JaggedTensor(
            values=torch.tensor([[6], [7], [8], [9], [10], [11]]),
            lengths=torch.tensor([2, 4]),
        ),
    }
    for channel in channel_names:
        sequence_dense_features[f"{channel}_seq__action_timestamp"] = JaggedTensor(
            values=torch.tensor([[1], [2], [3], [4], [5]]),
            lengths=torch.tensor([2, 3]),
        )
        sequence_dense_features[f"{channel}_seq__action_weight"] = JaggedTensor(
            values=torch.tensor([[0], [1], [0], [1], [0]]),
            lengths=torch.tensor([2, 3]),
        )

    jagged_labels = {
        "item_action_weight": JaggedTensor(
            values=torch.tensor([0, 1, 0, 0, 1, 0]),
            lengths=torch.tensor([2, 4]),
        ),
    }
    return Batch(
        sequence_dense_features=sequence_dense_features,
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels={},
        jagged_labels=jagged_labels,
    ).to(device)


class UltraHSTUTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @parameterized.expand(
        [
            # Single channel exercises the bare-HSTUTransducer return
            # path (no _HSTUTransducerStack wrapper).
            ("single", [("a", 256)]),
            # Two channels at the same STU dim exercise the wrapper +
            # concat path.
            ("multi_uniform", [("a", 256), ("b", 256)]),
            # Two channels at distinct STU dims so that
            # _stu_embedding_dim's `sum` is the only correct answer
            # (it can't collapse to N*dim or max(dims)).
            ("multi_hetero", [("a", 256), ("b", 512)]),
        ]
    )
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
        contextual_group_type=st.sampled_from(
            [model_pb2.FeatureGroupType.DEEP, model_pb2.FeatureGroupType.SEQUENCE]
        ),
        sequence_timestamp_is_ascending=st.sampled_from([True, False]),
        enable_global_average_loss=st.sampled_from([True, False]),
        concat_contextual_features=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=6,
        deadline=None,
    )
    def test_ultra_hstu(
        self,
        case_name,
        channel_specs,
        graph_type,
        kernel,
        contextual_group_type,
        sequence_timestamp_is_ascending,
        enable_global_average_loss,
        concat_contextual_features,
    ) -> None:
        # JIT_SCRIPT only supports the PyTorch kernel today.
        assume(
            (graph_type == TestGraphType.JIT_SCRIPT and kernel == Kernel.PYTORCH)
            or graph_type != TestGraphType.JIT_SCRIPT
        )

        device = torch.device("cuda")
        ultra_hstu = _build_model(
            device=device,
            channel_specs=channel_specs,
            contextual_group_type=contextual_group_type,
            enable_global_average_loss=enable_global_average_loss,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            concat_contextual_features=concat_contextual_features,
        )

        # Single-channel UltraHSTU returns a bare HSTUTransducer (no
        # _HSTUTransducerStack wrapper) so the predict() path matches
        # DlrmHSTU's exactly.  Multi-channel returns the stack.
        if len(channel_specs) == 1:
            self.assertIsInstance(ultra_hstu._hstu_transducer, HSTUTransducer)
            self.assertNotIsInstance(ultra_hstu._hstu_transducer, _HSTUTransducerStack)
        else:
            self.assertIsInstance(ultra_hstu._hstu_transducer, _HSTUTransducerStack)
        self.assertEqual(
            ultra_hstu._stu_embedding_dim(), sum(d for _, d in channel_specs)
        )

        ultra_hstu.set_kernel(kernel)
        batch = _build_batch(
            device=device, channel_names=[name for name, _ in channel_specs]
        )

        if graph_type == TestGraphType.JIT_SCRIPT:
            ultra_hstu.set_is_inference(True)
            ultra_hstu = create_test_model(ultra_hstu, graph_type)
            predictions = ultra_hstu(batch.to_dict(), device)
        elif graph_type == TestGraphType.AOT_INDUCTOR:
            data = batch.to_dict()
            data = OrderedDict(sorted(data.items()))
            self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
            ultra_hstu.set_is_inference(True)
            ultra_hstu = create_test_model(ultra_hstu, graph_type, data, self.test_dir)
            predictions = ultra_hstu(data)
        elif graph_type == TestGraphType.FX_TRACE:
            ultra_hstu = create_test_model(ultra_hstu, graph_type)
            predictions = ultra_hstu(batch)
        else:
            ultra_hstu = TrainWrapper(ultra_hstu, device=device).to(device)
            _, (_, predictions, batch) = ultra_hstu(batch)
            ultra_hstu.model.update_metric(predictions, batch)
            _ = ultra_hstu.model.compute_metric()

        self.assertEqual(predictions["logits_is_click"].size(), (6,))
        self.assertEqual(predictions["probs_is_click"].size(), (6,))
        self.assertEqual(predictions["logits_is_like"].size(), (6,))
        self.assertEqual(predictions["probs_is_like"].size(), (6,))


if __name__ == "__main__":
    unittest.main()
