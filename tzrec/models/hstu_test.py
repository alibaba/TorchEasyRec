# Copyright (c) 2024-2025, Alibaba Group;
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
from hypothesis import Verbosity, assume, given, settings
from hypothesis import strategies as st
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, CAND_POS_LENGTHS, Batch
from tzrec.features.feature import create_features
from tzrec.models.hstu import HSTUMatch
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
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model, gpu_unavailable


def _build_model(device: torch.device) -> HSTUMatch:
    """Build an HSTUMatch model with standard test configuration."""
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="historical_ids",
                sequence_length=210,
                embedding_dim=64,
                num_buckets=3953,
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_id_feature=feature_pb2.IdFeature(
                feature_name="item_id",
                sequence_length=10,
                sequence_delim=";",
                embedding_dim=64,
                num_buckets=1000,
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_raw_feature=feature_pb2.RawFeature(
                feature_name="historical_ts",
                sequence_length=210,
            )
        ),
    ]
    features = create_features(feature_cfgs)
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="uih",
            feature_names=["historical_ids"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate",
            feature_names=["item_id"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_timestamp",
            feature_names=["historical_ts"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
    ]
    model_config = model_pb2.ModelConfig(
        feature_groups=feature_groups,
        hstu_match=match_model_pb2.HSTUMatch(
            user_tower=tower_pb2.HSTUUserTower(
                input="uih",
                hstu=module_pb2.HSTU(
                    stu=module_pb2.STU(
                        # Power-of-2 dims so the Triton HSTU kernels accept
                        # the shapes.
                        embedding_dim=64,
                        num_heads=2,
                        hidden_dim=32,
                        attention_dim=32,
                        output_dropout_ratio=0.2,
                    ),
                    attn_num_layers=2,
                    positional_encoder=module_pb2.GRPositionalEncoder(
                        num_position_buckets=512,
                    ),
                    input_preprocessor=module_pb2.GRInputPreprocessor(
                        uih_preprocessor=module_pb2.GRUIHPreprocessor(),
                    ),
                    output_postprocessor=module_pb2.GROutputPostprocessor(
                        l2norm_postprocessor=module_pb2.GRL2NormPostprocessor(),
                    ),
                ),
                max_seq_len=210,
            ),
            item_tower=tower_pb2.Tower(input="candidate"),
            output_dim=64,
            temperature=0.05,
        ),
        losses=[
            loss_pb2.LossConfig(softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy())
        ],
        metrics=[metric_pb2.MetricConfig(recall_at_k=metric_pb2.RecallAtK(top_k=1))],
    )
    hstu = HSTUMatch(
        model_config=model_config,
        features=features,
        labels=["label"],
        sampler_type="negative_sampler",
    )
    init_parameters(hstu, device=device)
    hstu.to(device)
    return hstu


def _build_batch(device: torch.device) -> Batch:
    """Build a test Batch with the row-(B-1) suffix candidate layout.

    UIH: user1 has 3 items, user2 has 4 items.
    Candidates: row 0 = [pos_0]; row 1 (last) = [pos_1, simple_neg_0,
    simple_neg_1] -- the shared simple-neg pool sits in the last row's suffix.
    pos_lengths = [1, 1].
    """
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["historical_ids", "item_id"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 100, 200, 101, 201]),
        lengths=torch.tensor([3, 4, 1, 3]),
    )
    sequence_dense_features = {
        "historical_ts": JaggedTensor(
            values=torch.tensor([[1], [2], [3], [4], [5], [6], [7]]),
            lengths=torch.tensor([3, 4]),
        ),
    }
    return Batch(
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        sequence_dense_features=sequence_dense_features,
        labels={"label": torch.tensor([1, 1])},
        additional_infos={CAND_POS_LENGTHS: torch.tensor([1, 1], dtype=torch.int32)},
    ).to(device)


class HSTUMatchTest(unittest.TestCase):
    @given(
        graph_type=st.sampled_from(
            [
                TestGraphType.NORMAL,
                TestGraphType.FX_TRACE,
                TestGraphType.JIT_SCRIPT,
            ]
        ),
        kernel=st.sampled_from([Kernel.PYTORCH, Kernel.TRITON]),
        device_str=st.sampled_from(["cpu", "cuda"]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=6,
        deadline=None,
    )
    def test_hstu_match(self, graph_type, kernel, device_str) -> None:
        # CUDA needs a GPU.
        if device_str == "cuda":
            assume(not gpu_unavailable[0])
        # Triton kernels need CUDA tensors; reject (cpu, TRITON) regardless of host.
        if kernel == Kernel.TRITON:
            assume(device_str == "cuda")
        # JIT_SCRIPT only supports PYTORCH kernel today.
        assume(
            (graph_type == TestGraphType.JIT_SCRIPT and kernel == Kernel.PYTORCH)
            or graph_type != TestGraphType.JIT_SCRIPT
        )

        device = torch.device(device_str)
        hstu = _build_model(device=device)
        hstu.set_kernel(kernel)
        batch = _build_batch(device=device)

        if graph_type == TestGraphType.JIT_SCRIPT:
            hstu.set_is_inference(True)
            hstu = create_test_model(hstu, graph_type)
            predictions = hstu(batch.to_dict(), device)
        elif graph_type == TestGraphType.FX_TRACE:
            hstu = create_test_model(hstu, graph_type)
            predictions = hstu(batch)
        else:
            hstu = TrainWrapper(hstu, device=device).to(device)
            _, (_, predictions, _) = hstu(batch)

        self.assertIn("similarity", predictions)
        # Q = sum(pos_lengths) = 2; column count = 1 (pos) + neg count.
        self.assertEqual(predictions["similarity"].size(0), 2)


if __name__ == "__main__":
    unittest.main()
