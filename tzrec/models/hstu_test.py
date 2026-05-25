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
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor

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
    simi_pb2,
    tower_pb2,
)
from tzrec.protos.models import match_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import TestGraphType, create_test_model, gpu_unavailable


def _build_model(device: torch.device, with_query_time: bool = False) -> HSTUMatch:
    """Build an HSTUMatch model with standard test configuration.

    Mirrors the production grouped-sequence pattern: `uih_seq` and
    `cand_seq` each carry a `video_id` sub-feature with aligned bucket /
    dim / `embedding_name` so the two flattened features share one
    embedding table. `uih_seq` also carries the `historical_ts` raw
    sub-feature for the timestamp dense path.

    When ``with_query_time`` is set, the user tower turns on time encoding
    and a scalar ``request_time`` raw feature is exposed through a
    ``query_time`` DEEP group — the per-row time-bias anchor.
    """
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="uih_seq",
                sequence_length=210,
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="video_id",
                            embedding_dim=64,
                            num_buckets=1000,
                            embedding_name="video_id_emb",
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="historical_ts",
                        )
                    ),
                ],
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="cand_seq",
                sequence_length=10,
                sequence_delim=";",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="video_id",
                            embedding_dim=64,
                            num_buckets=1000,
                            embedding_name="video_id_emb",
                        )
                    ),
                ],
            )
        ),
    ]
    if with_query_time:
        feature_cfgs.append(
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="request_time")
            )
        )
    features = create_features(feature_cfgs)
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="uih",
            feature_names=["uih_seq__video_id"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate",
            feature_names=["cand_seq__video_id"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_timestamp",
            feature_names=["uih_seq__historical_ts"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
    ]
    if with_query_time:
        feature_groups.append(
            model_pb2.FeatureGroupConfig(
                group_name="query_time",
                feature_names=["request_time"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            )
        )
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
                        num_time_buckets=512,
                        use_time_encoding=with_query_time,
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
            item_tower=tower_pb2.Tower(
                input="candidate",
                mlp=module_pb2.MLP(hidden_units=[64], activation=""),
            ),
            similarity=simi_pb2.Similarity.COSINE,
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


def _build_batch(device: torch.device, with_query_time: bool = False) -> Batch:
    """Build a test Batch with the row-(B-1) suffix candidate layout.

    UIH: user1 has 3 items, user2 has 4 items.
    Candidates: row 0 = [pos_0]; row 1 (last) = [pos_1, simple_neg_0,
    simple_neg_1] -- the shared simple-neg pool sits in the last row's suffix.
    pos_lengths = [1, 1].

    When ``with_query_time`` is set, a per-row ``request_time`` dense scalar
    (strictly after each user's last UIH event at ts 3 / 7) is included.
    """
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["uih_seq__video_id", "cand_seq__video_id"],
        values=torch.tensor([1, 2, 3, 4, 5, 6, 7, 100, 200, 101, 201]),
        lengths=torch.tensor([3, 4, 1, 3]),
    )
    sequence_dense_features = {
        "uih_seq__historical_ts": JaggedTensor(
            values=torch.tensor([[1], [2], [3], [4], [5], [6], [7]]),
            lengths=torch.tensor([3, 4]),
        ),
    }
    dense_features = {}
    if with_query_time:
        dense_features[BASE_DATA_GROUP] = KeyedTensor.from_tensor_list(
            keys=["request_time"],
            tensors=[torch.tensor([[100.0], [100.0]])],
        )
    return Batch(
        dense_features=dense_features,
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        sequence_dense_features=sequence_dense_features,
        jagged_labels={
            "label": JaggedTensor(
                values=torch.tensor([1, 1], dtype=torch.int64),
                lengths=torch.tensor([1, 1]),
            ),
        },
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
            hstu_wrapped = create_test_model(hstu, graph_type)
            predictions = hstu_wrapped(batch.to_dict(), device)
        elif graph_type == TestGraphType.FX_TRACE:
            hstu_wrapped = create_test_model(hstu, graph_type)
            predictions = hstu_wrapped(batch)
        else:
            hstu_wrapped = TrainWrapper(hstu, device=device).to(device)
            _, (_, predictions, _) = hstu_wrapped(batch)

        self.assertIn("similarity", predictions)
        # Q = sum(pos_lengths) = 2; column count = 1 (pos) + neg count.
        self.assertEqual(predictions["similarity"].size(0), 2)

        # Scalar-view contract: set_is_inference(True) flips item_tower
        # to the scalar export view (bare sub-feature names).
        hstu.set_is_inference(True)
        self.assertTrue(hstu.item_tower._is_inference)
        scalar_features = hstu.item_tower.features
        scalar_feature_groups = hstu.item_tower.feature_groups
        self.assertEqual(scalar_features[0].name, "video_id")
        self.assertFalse(scalar_features[0].is_grouped_sequence)
        self.assertEqual(scalar_feature_groups[0].feature_names, ["video_id"])
        self.assertEqual(scalar_feature_groups[0].group_name, "candidate")

    @given(
        graph_type=st.sampled_from([TestGraphType.NORMAL, TestGraphType.FX_TRACE]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=2, deadline=None)
    def test_hstu_match_query_time(self, graph_type) -> None:
        """The optional `query_time` DEEP group is the time-bias anchor.

        Detected and threaded into the user tower so it reads a per-row
        request time instead of deriving it from the last UIH event.
        """
        device = torch.device("cpu")
        # No query_time group -> encoder falls back to the last UIH timestamp.
        self.assertEqual(
            _build_model(device).user_tower._hstu_encoder._query_time_key, ""
        )
        # With the group -> encoder reads it as the explicit per-row anchor.
        hstu = _build_model(device, with_query_time=True)
        self.assertEqual(hstu.user_tower._hstu_encoder._query_time_key, "query_time")
        hstu.set_kernel(Kernel.PYTORCH)
        batch = _build_batch(device, with_query_time=True)

        if graph_type == TestGraphType.FX_TRACE:
            predictions = create_test_model(hstu, graph_type)(batch)
        else:
            wrapped = TrainWrapper(hstu, device=device).to(device)
            _, (_, predictions, _) = wrapped(batch)
        self.assertEqual(predictions["similarity"].size(0), 2)


if __name__ == "__main__":
    unittest.main()
