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
from hypothesis import Verbosity, assume, given
from hypothesis import strategies as st
from parameterized import parameterized
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, CAND_POS_LENGTHS, Batch
from tzrec.features.feature import create_features
from tzrec.models.hstu import HSTUMatch
from tzrec.models.match_model import TowerWoEGWrapper
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
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_model,
    gpu_unavailable,
    mark_ci_scope,
    parameterized_name_func,
)
from tzrec.utils.test_util import (
    hypothesis_settings as settings,
)


def _build_model(
    device: torch.device, sequence_timestamp_is_ascending: bool = True
) -> HSTUMatch:
    """Build an HSTUMatch model with standard test configuration.

    Mirrors the production grouped-sequence pattern: `uih_seq` and
    `cand_seq` each carry a `video_id` sub-feature with aligned bucket /
    dim / `embedding_name` so the two flattened features share one
    embedding table. `uih_seq` also carries the `historical_ts` raw
    sub-feature for the timestamp dense path.

    Time encoding is on, with a scalar ``request_time`` raw feature exposed
    through a ``query_time`` DEEP group — the per-row time-bias anchor
    (mirrors the production config).
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
                        use_time_encoding=True,
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
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
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


def _build_batch(
    device: torch.device, sequence_timestamp_is_ascending: bool = True
) -> Batch:
    """Build a test Batch with the row-(B-1) suffix candidate layout.

    UIH: user1 has 3 items, user2 has 4 items.
    Candidates: row 0 = [pos_0]; row 1 (last) = [pos_1, simple_neg_0,
    simple_neg_1] -- the shared simple-neg pool sits in the last row's suffix.
    pos_lengths = [1, 1].

    When ``sequence_timestamp_is_ascending`` is false, only each user's UIH
    values and timestamps are reversed. Candidate sampler order is unchanged.
    Distinct per-row ``request_time`` scalars exercise request-time alignment
    while the user rows are temporarily reversed by the model.
    """
    if sequence_timestamp_is_ascending:
        uih_values = [1, 2, 3, 4, 5, 6, 7]
        uih_timestamps = [1, 2, 3, 4, 5, 6, 7]
    else:
        uih_values = [3, 2, 1, 7, 6, 5, 4]
        uih_timestamps = [3, 2, 1, 7, 6, 5, 4]
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=["uih_seq__video_id", "cand_seq__video_id"],
        values=torch.tensor(uih_values + [100, 200, 101, 201]),
        lengths=torch.tensor([3, 4, 1, 3]),
    )
    sequence_dense_features = {
        "uih_seq__historical_ts": JaggedTensor(
            values=torch.tensor(uih_timestamps).unsqueeze(-1),
            lengths=torch.tensor([3, 4]),
        ),
    }
    dense_features = {
        BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
            keys=["request_time"],
            tensors=[torch.tensor([[100.0], [200.0]])],
        )
    }
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


@mark_ci_scope("gpu")
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
        sequence_timestamp_is_ascending=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=6,
        deadline=None,
    )
    def test_hstu_match(
        self, graph_type, kernel, device_str, sequence_timestamp_is_ascending
    ) -> None:
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
        hstu = _build_model(
            device=device,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
        )
        # The query_time DEEP group is detected and threaded as the per-row
        # time-bias anchor (request-time anchoring, not the last UIH event).
        self.assertEqual(hstu.user_tower._hstu_encoder._query_time_key, "query_time")
        hstu.set_kernel(kernel)
        batch = _build_batch(
            device=device,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
        )

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

    def test_sequence_timestamp_order_parity(self) -> None:
        """Equivalent ascending and descending UIH inputs produce equal outputs."""
        device = torch.device("cpu")
        ascending = _build_model(
            device=device, sequence_timestamp_is_ascending=True
        ).eval()
        descending = _build_model(
            device=device, sequence_timestamp_is_ascending=False
        ).eval()
        descending.load_state_dict(ascending.state_dict())
        ascending.set_kernel(Kernel.PYTORCH)
        descending.set_kernel(Kernel.PYTORCH)

        ascending_batch = _build_batch(
            device=device, sequence_timestamp_is_ascending=True
        )
        descending_batch = _build_batch(
            device=device, sequence_timestamp_is_ascending=False
        )
        with torch.no_grad():
            ascending_predictions = ascending.predict(ascending_batch)
            descending_predictions = descending.predict(descending_batch)
        torch.testing.assert_close(
            descending_predictions["similarity"],
            ascending_predictions["similarity"],
        )

        ascending.set_is_inference(True)
        descending.set_is_inference(True)
        ascending_user_tower = TowerWoEGWrapper(ascending.user_tower).eval()
        descending_user_tower = TowerWoEGWrapper(descending.user_tower).eval()
        init_parameters(ascending_user_tower, device=device)
        init_parameters(descending_user_tower, device=device)
        descending_user_tower.load_state_dict(ascending_user_tower.state_dict())
        with torch.no_grad():
            ascending_user_emb = ascending_user_tower.predict(ascending_batch)[
                "user_tower_emb"
            ]
            descending_user_emb = descending_user_tower.predict(descending_batch)[
                "user_tower_emb"
            ]
        torch.testing.assert_close(descending_user_emb, ascending_user_emb)

    @parameterized.expand(
        [
            (TestGraphType.FX_TRACE, True),
            (TestGraphType.FX_TRACE, False),
            (TestGraphType.JIT_SCRIPT, True),
            (TestGraphType.JIT_SCRIPT, False),
        ],
        name_func=parameterized_name_func,
    )
    def test_sequence_timestamp_order_graph_modes(
        self, graph_type: TestGraphType, sequence_timestamp_is_ascending: bool
    ) -> None:
        """Timestamp-order handling supports FX and JIT user-model graphs."""
        device = torch.device("cpu")
        hstu = _build_model(
            device=device,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
        )
        hstu.set_kernel(Kernel.PYTORCH)
        batch = _build_batch(
            device=device,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
        )
        hstu_wrapped = create_test_model(hstu, graph_type)
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = hstu_wrapped(batch.to_dict(), device)
        else:
            predictions = hstu_wrapped(batch)
        self.assertEqual(predictions["similarity"].size(0), 2)


if __name__ == "__main__":
    unittest.main()
