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

"""Model-level tests for ``tzrec.models.dlrm_hstu_onerank``.

Kernel coverage is ``[PYTORCH, CUTLASS]`` rather than the baseline's
``[PYTORCH, TRITON]``: the OneRank group mask is an NFUNC func tensor and
Triton has no NFUNC path, so CUTLASS is the kernel this model actually
trains with. ``pt_hstu_attention`` implements the same encoding and serves
as the reference.
"""

import unittest
from typing import List, Optional
from unittest import mock

import torch
import torch.distributed as dist
from hypothesis import Verbosity, given
from hypothesis import strategies as st
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.loss.listwise_rank_loss import ListwiseRankLoss
from tzrec.models.dlrm_hstu_onerank import DlrmHSTUOneRank
from tzrec.models.model import TrainWrapper
from tzrec.models.rank_model import TARGET_REPEAT_INTERLEAVE_KEY
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
from tzrec.utils import fx_util
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    TestGraphType,
    cleanup_cuda_memory,
    create_test_model,
    cutlass_hstu_unavailable,
    gpu_unavailable,
    mark_ci_scope,
)
from tzrec.utils.test_util import (
    hypothesis_settings as settings,
)

_TASK_NAMES = ["is_click", "is_like", "is_comment"]
# candidate (cand_seq) counts of _build_batch, in input order.
_NUM_TARGETS = [2, 4]
_TOTAL_TARGETS = sum(_NUM_TARGETS)


def _listwise_loss_cfg(weight: float = 1.0, **kwargs) -> loss_pb2.LossConfig:
    """A ``listwise_rank_loss`` entry, as carried inside a task's losses."""
    return loss_pb2.LossConfig(
        weight=weight, listwise_rank_loss=loss_pb2.ListwiseRankLoss(**kwargs)
    )


def _task_configs(
    task_weight: float = 1.0,
    num_class: int = 1,
    click_loss: str = "binary_cross_entropy",
    listwise_loss: Optional[loss_pb2.LossConfig] = None,
) -> List[tower_pb2.FusionSubTaskConfig]:
    if click_loss == "binary_cross_entropy":
        click_loss_cfg = loss_pb2.LossConfig(
            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
        )
    else:
        click_loss_cfg = loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())
    click_losses = [click_loss_cfg] + (
        [listwise_loss] if listwise_loss is not None else []
    )
    return [
        tower_pb2.FusionSubTaskConfig(
            task_name="is_click",
            label_name="item_action_weight",
            task_bitmask=1,
            num_class=num_class,
            losses=click_losses,
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


def _model_config(
    task_configs: List[tower_pb2.FusionSubTaskConfig],
    contextual_group_type: int = model_pb2.FeatureGroupType.DEEP,
    enable_global_average_loss: bool = False,
    sequence_timestamp_is_ascending: bool = False,
    concat_contextual_features: bool = False,
    with_situation_discernment: bool = True,
    with_cross_task_head: bool = True,
    # 100 history tokens + 8 candidates * (1 + 3 tasks): the inflated
    # sequence bound the model is configured with by default.
    max_seq_len: int = 132,
    output_dropout_ratio: float = 0.0,
    scorer_type: Optional[str] = None,
    scorer_hidden_dim: Optional[int] = None,
    task_bias_init: Optional[List[float]] = None,
    attn_truncation_split_layer: int = 0,
    attn_truncation_tail_len: int = 0,
) -> model_pb2.ModelConfig:
    onerank = multi_task_rank_pb2.OneRankConfig()
    if scorer_type is not None:
        onerank.scorer_type = scorer_type
    if scorer_hidden_dim is not None:
        onerank.scorer_hidden_dim = scorer_hidden_dim
    if task_bias_init is not None:
        onerank.task_bias_init.extend(task_bias_init)
    if with_situation_discernment:
        onerank.situation_discernment.CopyFrom(
            multi_task_rank_pb2.OneRankSituationDiscernment(num_heads=2)
        )
    if with_cross_task_head:
        onerank.cross_task_head.CopyFrom(
            multi_task_rank_pb2.OneRankCrossTaskHead(
                num_heads=2, ffn_hidden_dim=64, gradient_detachment=True
            )
        )
    return model_pb2.ModelConfig(
        feature_groups=_feature_groups(contextual_group_type),
        dlrm_hstu_onerank=multi_task_rank_pb2.DlrmHSTUOneRank(
            hstu=module_pb2.HSTU(
                stu=module_pb2.STU(
                    embedding_dim=64,
                    num_heads=2,
                    # CUTLASS requires attention_dim == hidden_dim.
                    hidden_dim=32,
                    attention_dim=32,
                    output_dropout_ratio=output_dropout_ratio,
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
                            )
                        ),
                        action_mlp=module_pb2.GRContextualizedMLP(
                            simple_mlp=module_pb2.GRSimpleContextualizedMLP(
                                hidden_dim=32
                            )
                        ),
                        content_encoder=module_pb2.GRContentEncoder(
                            slice_content_encoder=module_pb2.GRSliceContentEncoder()
                        ),
                        content_mlp=module_pb2.GRContextualizedMLP(
                            simple_mlp=module_pb2.GRSimpleContextualizedMLP(
                                hidden_dim=32
                            )
                        ),
                    )
                ),
                output_postprocessor=module_pb2.GROutputPostprocessor(
                    layernorm_postprocessor=module_pb2.GRLayerNormPostprocessor()
                ),
                attn_truncation_split_layer=attn_truncation_split_layer,
                attn_truncation_tail_len=attn_truncation_tail_len,
            ),
            # `mlp` is ignored by OneRank but FusionMTLTower requires it.
            fusion_mtl_tower=tower_pb2.FusionMTLTower(
                mlp=module_pb2.MLP(hidden_units=[64], activation="nn.SiLU"),
                task_configs=task_configs,
            ),
            max_seq_len=max_seq_len,
            enable_global_average_loss=enable_global_average_loss,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            concat_contextual_features=concat_contextual_features,
            onerank=onerank,
        ),
    )


def _feature_groups(contextual_group_type: int) -> List[model_pb2.FeatureGroupConfig]:
    return [
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
            feature_names=["cand_seq__item_video_id", "cand_seq__item_video_cat"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_timestamp",
            feature_names=["uih_seq__action_timestamp"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="candidate_timestamp",
            feature_names=["cand_seq__item_query_time"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="uih_action",
            feature_names=["uih_seq__action_weight"],
            group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
        ),
    ]


def _features() -> List:
    return create_features(
        [
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
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="action_weight"
                            )
                        ),
                    ],
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
    )


def _build_model(
    device: torch.device, seed: int = 0, **config_kwargs
) -> DlrmHSTUOneRank:
    """Build a ``DlrmHSTUOneRank`` on ``device`` with initialized parameters."""
    task_configs = config_kwargs.pop("task_configs", None)
    listwise_loss = config_kwargs.pop("listwise_loss", None)
    if task_configs is None:
        task_configs = _task_configs(listwise_loss=listwise_loss)
    model_config = _model_config(task_configs, **config_kwargs)
    model = DlrmHSTUOneRank(
        model_config=model_config,
        features=_features(),
        labels=["item_action_weight"],
    )
    torch.manual_seed(seed)
    init_parameters(model, device=device)
    model.to(device)
    return model


def _build_batch(device: torch.device) -> Batch:
    """A two-request batch with 2 and 4 candidates."""
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
            lengths=torch.tensor(_NUM_TARGETS),
        ),
        "uih_seq__action_weight": JaggedTensor(
            values=torch.tensor([[0], [1], [0], [1], [0]]),
            lengths=torch.tensor([2, 3]),
        ),
    }
    jagged_labels = {
        "item_action_weight": JaggedTensor(
            # Bitmask labels. Chosen so that (a) every task has both a
            # positive and a negative -- otherwise AUC is undefined -- and
            # (b) every request has both, which is what keeps the list-wise
            # term unmasked.
            values=torch.tensor([0, 1, 2, 5, 4, 0]),
            lengths=torch.tensor(_NUM_TARGETS),
        ),
    }
    return Batch(
        sequence_dense_features=sequence_dense_features,
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels={},
        jagged_labels=jagged_labels,
    ).to(device)


def _build_request_permuted_batch(device: torch.device) -> Batch:
    """``_build_batch`` with the two requests swapped (r1 first, then r0).

    Same two requests with identical per-request content -- user state,
    histories, candidates, labels -- only the request order differs.  The
    model is per-request independent, so predictions must come back as
    the same request blocks in the swapped order; anything else means the
    published logits no longer follow the batch's request order.
    """
    sparse_feature = KeyedJaggedTensor.from_lengths_sync(
        keys=[
            "user_id",
            "user_active_degree",
            "uih_seq__video_id",
            "cand_seq__item_video_id",
            "uih_seq__video_cat",
            "cand_seq__item_video_cat",
        ],
        # Key-major, request-major within each key: r1's block first.
        values=torch.tensor(
            [
                1,
                0,
                3,
                2,
                6,
                7,
                8,
                4,
                5,
                11,
                12,
                13,
                14,
                9,
                10,
                17,
                18,
                19,
                15,
                16,
                22,
                23,
                24,
                25,
                20,
                21,
            ]
        ),
        lengths=torch.tensor([1, 1, 1, 1, 3, 2, 4, 2, 3, 2, 4, 2]),
    )
    sequence_dense_features = {
        "uih_seq__action_timestamp": JaggedTensor(
            values=torch.tensor([[3], [4], [5], [1], [2]]),
            lengths=torch.tensor([3, 2]),
        ),
        "cand_seq__item_query_time": JaggedTensor(
            values=torch.tensor([[8], [9], [10], [11], [6], [7]]),
            lengths=torch.tensor([4, 2]),
        ),
        "uih_seq__action_weight": JaggedTensor(
            values=torch.tensor([[0], [1], [0], [0], [1]]),
            lengths=torch.tensor([3, 2]),
        ),
    }
    jagged_labels = {
        "item_action_weight": JaggedTensor(
            values=torch.tensor([2, 5, 4, 0, 0, 1]),
            lengths=torch.tensor([4, 2]),
        ),
    }
    return Batch(
        sequence_dense_features=sequence_dense_features,
        sparse_features={BASE_DATA_GROUP: sparse_feature},
        labels={},
        jagged_labels=jagged_labels,
    ).to(device)


@mark_ci_scope("gpu")
class DlrmHSTUOneRankTest(unittest.TestCase):
    """End-to-end tests of the OneRank model on the PYTORCH/CUTLASS kernels."""

    def teardown_example(self, example):
        cleanup_cuda_memory()

    @unittest.skipIf(*gpu_unavailable)
    @given(
        graph_type=st.sampled_from([TestGraphType.NORMAL, TestGraphType.FX_TRACE]),
        with_situation_discernment=st.sampled_from([True, False]),
        with_cross_task_head=st.sampled_from([True, False]),
        contextual_group_type=st.sampled_from(
            [model_pb2.FeatureGroupType.DEEP, model_pb2.FeatureGroupType.SEQUENCE]
        ),
        concat_contextual_features=st.sampled_from([True, False]),
        sequence_timestamp_is_ascending=st.sampled_from([True, False]),
        enable_global_average_loss=st.sampled_from([True, False]),
        with_listwise_loss=st.sampled_from([True, False]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    def test_dlrm_hstu_onerank(
        self,
        graph_type,
        with_situation_discernment,
        with_cross_task_head,
        contextual_group_type,
        concat_contextual_features,
        sequence_timestamp_is_ascending,
        enable_global_average_loss,
        with_listwise_loss,
    ) -> None:
        """Every ablation switch produces one logit per (candidate, task).

        The SD / cross-task / list-wise flags are exactly the paper's
        ablation axes, so each has to stay independently runnable.
        """
        device = torch.device("cuda")
        listwise_loss = _listwise_loss_cfg(weight=0.5) if with_listwise_loss else None
        model = _build_model(
            device=device,
            with_situation_discernment=with_situation_discernment,
            with_cross_task_head=with_cross_task_head,
            contextual_group_type=contextual_group_type,
            concat_contextual_features=concat_contextual_features,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            enable_global_average_loss=enable_global_average_loss,
            listwise_loss=listwise_loss,
        )
        model.set_kernel(Kernel.PYTORCH)
        batch = _build_batch(device=device)

        if graph_type == TestGraphType.FX_TRACE:
            traced = create_test_model(model, graph_type)
            predictions = traced(batch)
        else:
            wrapper = TrainWrapper(model, device=device).to(device)
            total_loss, (losses, predictions, batch) = wrapper(batch)
            self.assertTrue(torch.isfinite(total_loss))
            expected_listwise = (
                ["listwise_rank_loss_is_click"] if with_listwise_loss else []
            )
            self.assertEqual(
                sorted(k for k in losses if k.startswith("listwise_")),
                expected_listwise,
            )
            wrapper.model.update_metric(predictions, batch)
            self.assertTrue(wrapper.model.compute_metric())

        for task_name in _TASK_NAMES:
            self.assertEqual(
                predictions[f"logits_{task_name}"].size(), (_TOTAL_TARGETS,)
            )
            self.assertEqual(
                predictions[f"probs_{task_name}"].size(), (_TOTAL_TARGETS,)
            )
        self.assertEqual(
            predictions[TARGET_REPEAT_INTERLEAVE_KEY].cpu().tolist(), _NUM_TARGETS
        )

    @unittest.skipIf(*gpu_unavailable)
    def test_predict_num_targets_order_is_deterministic(self) -> None:
        """num_targets must stay in input order for both timestamp orders.

        The hypothesis test samples ``sequence_timestamp_is_ascending``,
        but non-nightly CI derandomizes it and cuts ``max_examples`` 5x
        across 8 sampled axes, so the flipped variant may never be drawn
        there. With ``False``, ``predict()`` flips features (reversing
        request order) and flips predictions back, so reading the split
        key from the still-flipped state would return [4, 2] for the
        [2, 4] batch and misassign whole-request blocks in ``loss()``.
        """
        device = torch.device("cuda")
        for ascending in (True, False):
            model = _build_model(
                device=device, sequence_timestamp_is_ascending=ascending
            )
            model.set_kernel(Kernel.PYTORCH)
            batch = _build_batch(device=device)
            with torch.no_grad():
                predictions = model.predict(batch)
            self.assertEqual(
                predictions[TARGET_REPEAT_INTERLEAVE_KEY].cpu().tolist(),
                _NUM_TARGETS,
                msg=f"num_targets order wrong for ascending={ascending}",
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_trunk_emits_one_vector_per_candidate_and_task(self) -> None:
        """The group layout must survive the transducer as ``(N, K, D)``.

        A wiring slip that returned the candidate slot instead of the
        task slots, or collapsed K, would still give correctly shaped
        logits -- this is where that is caught.
        """
        device = torch.device("cuda")
        model = _build_model(device=device)
        model.set_kernel(Kernel.PYTORCH)
        batch = _build_batch(device=device)

        with torch.no_grad():
            grouped_features = model.build_input(batch)
            task_embeddings, full = model._hstu_transducer(grouped_features)

        self.assertIsNone(full)
        self.assertEqual(task_embeddings.size(), (_TOTAL_TARGETS, len(_TASK_NAMES), 64))
        # Task tokens are distinct parameters under distinct masks, so no two
        # task channels of the same candidate may come out equal.
        for k in range(1, len(_TASK_NAMES)):
            self.assertFalse(
                torch.allclose(task_embeddings[:, 0], task_embeddings[:, k])
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_listwise_loss_is_scaled_by_loss_weight(self) -> None:
        """``LossConfig.weight`` is the knob scaling the list-wise term.

        The task ``weight`` scales every loss of the task alike -- the
        EasyRec convention -- so it cannot trade the list-wise term against
        the point-wise ones; the per-loss ``weight`` is the knob that does,
        and the total is
        ``sum_k task_weight_k * sum_l loss_weight_kl * L_kl``.
        """
        device = torch.device("cuda")
        weight = 0.25
        base = _build_model(
            device=device,
            listwise_loss=_listwise_loss_cfg(),
        )
        scaled = _build_model(
            device=device,
            listwise_loss=_listwise_loss_cfg(weight=weight),
        )
        scaled.load_state_dict(base.state_dict())
        base.set_kernel(Kernel.PYTORCH)
        scaled.set_kernel(Kernel.PYTORCH)
        base.init_loss()
        scaled.init_loss()
        base.eval()
        scaled.eval()

        batch = _build_batch(device=device)
        with torch.no_grad():
            predictions = base.predict(batch)
            base_losses = base.loss(predictions, batch)
            scaled_losses = scaled.loss(predictions, batch)

        key = "listwise_rank_loss_is_click"
        self.assertGreater(base_losses[key].item(), 0.0)
        torch.testing.assert_close(
            scaled_losses[key], base_losses[key] * weight, rtol=1e-5, atol=1e-6
        )
        # Point-wise terms are untouched.
        for task_name in _TASK_NAMES:
            torch.testing.assert_close(
                scaled_losses[f"binary_cross_entropy_{task_name}"],
                base_losses[f"binary_cross_entropy_{task_name}"],
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_global_average_loss_rescales_each_term_by_its_own_denominator(
        self,
    ) -> None:
        """Each loss family takes its own global-average-loss factor.

        The list-wise term is a mean over requests and the point-wise terms
        are means over candidates, so each must be rescaled by the ratio of
        the count it divided by -- mixing them up leaves the loss finite and
        the sign correct, so only a numeric check catches it.  Both counts
        are ragged across ranks under cost-based batching.
        """
        device = torch.device("cuda")
        base = _build_model(device=device, listwise_loss=_listwise_loss_cfg())
        scaled = _build_model(
            device=device,
            listwise_loss=_listwise_loss_cfg(),
            enable_global_average_loss=True,
        )
        scaled.load_state_dict(base.state_dict())
        for model in (base, scaled):
            model.set_kernel(Kernel.PYTORCH)
            model.init_loss()
            model.eval()

        batch = _build_batch(device=device)
        with torch.no_grad():
            predictions = base.predict(batch)
            base_losses = base.loss(predictions, batch)
            # Emulate one peer rank holding 6 requests / 2 candidates against
            # this rank's 2 / 6, so the two ratios differ and neither is 1.0.
            peer = torch.tensor([6.0, 2.0], device=device)
            with mock.patch.object(fx_util, "dist") as dist_mock:
                dist_mock.is_initialized.return_value = True
                dist_mock.ReduceOp.AVG = dist.ReduceOp.AVG
                dist_mock.all_reduce.side_effect = lambda outcome, op: outcome.copy_(
                    (outcome + peer) / 2
                )
                scaled_losses = scaled.loss(predictions, batch)
                # The flag-off model must take no factor even with a live
                # process group, which pins the `global_average` gate: every
                # single-process factor is 1.0, so nothing else would catch
                # the gate being dropped.
                gated_losses = base.loss(predictions, batch)

        # Both axes travel in one reduction, whatever the task or loss count.
        self.assertEqual(dist_mock.all_reduce.call_count, 1)

        request_ratio = len(_NUM_TARGETS) / ((len(_NUM_TARGETS) + 6.0) / 2)
        candidate_ratio = _TOTAL_TARGETS / ((_TOTAL_TARGETS + 2.0) / 2)
        self.assertNotAlmostEqual(request_ratio, candidate_ratio)

        for name, value in base_losses.items():
            torch.testing.assert_close(gated_losses[name], value, msg=name)

        key = "listwise_rank_loss_is_click"
        self.assertGreater(base_losses[key].item(), 0.0)
        torch.testing.assert_close(scaled_losses[key], base_losses[key] * request_ratio)
        for task_name in _TASK_NAMES:
            pointwise = f"binary_cross_entropy_{task_name}"
            torch.testing.assert_close(
                scaled_losses[pointwise], base_losses[pointwise] * candidate_ratio
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_listwise_loss_matches_hand_decoded_reference(self) -> None:
        """The list-wise term must see decoded labels and its own logits.

        A silent wiring regression -- feeding the raw jagged label values
        instead of the bitmask-decoded ``_get_label`` output -- would count
        every nonzero bitmask (2 == is_like, 4 == is_comment) as an
        is_click positive and quietly train a different objective, while
        every "finite / positive / weight-scaled" assertion stays green.
        The reference here decodes the label by hand and evaluates a bare
        ``ListwiseRankLoss`` on the published per-task logits and
        ``num_targets``; only the exact wiring reproduces it, in both
        timestamp directions.
        """
        device = torch.device("cuda")
        for ascending in (True, False):
            model = _build_model(
                device=device,
                sequence_timestamp_is_ascending=ascending,
                listwise_loss=_listwise_loss_cfg(),
            )
            model.set_kernel(Kernel.PYTORCH)
            model.init_loss()
            model.eval()
            batch = _build_batch(device=device)
            with torch.no_grad():
                predictions = model.predict(batch)
                losses = model.loss(predictions, batch)

            # Hand-decoded is_click labels: (value & bitmask=1) > 0.
            raw = batch.jagged_labels["item_action_weight"].values()
            decoded = ((raw.to(torch.int64) & 1) > 0).to(torch.float32)
            reference = ListwiseRankLoss()
            reference.load_state_dict(
                model._loss_modules["listwise_rank_loss_is_click"].state_dict()
            )
            expected = reference(
                predictions["logits_is_click"],
                decoded,
                predictions[TARGET_REPEAT_INTERLEAVE_KEY],
            )
            torch.testing.assert_close(
                losses["listwise_rank_loss_is_click"],
                expected,
                msg=f"list-wise wiring wrong for ascending={ascending}",
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_predict_logits_stay_in_request_order(self) -> None:
        """Published logits must follow the batch's request order.

        ``test_predict_num_targets_order_is_deterministic`` pins the split
        key; this pins the logits themselves.  Under
        ``sequence_timestamp_is_ascending=False`` a regression of the
        ``mt_preds`` flip-back would publish globally reversed logits
        while ``num_targets`` stays ``[2, 4]`` -- whole request blocks
        then pair with the wrong jagged labels in ``loss()`` with no
        error anywhere.  The model is per-request independent, so feeding
        the same two requests in swapped order must return the same
        blocks, swapped.
        """
        device = torch.device("cuda")
        first_request_len = _NUM_TARGETS[0]
        for ascending in (True, False):
            model = _build_model(
                device=device,
                sequence_timestamp_is_ascending=ascending,
                listwise_loss=_listwise_loss_cfg(),
            )
            model.set_kernel(Kernel.PYTORCH)
            model.init_loss()
            model.eval()
            batch = _build_batch(device=device)
            permuted_batch = _build_request_permuted_batch(device=device)
            with torch.no_grad():
                predictions = model.predict(batch)
                permuted_predictions = model.predict(permuted_batch)
                losses = model.loss(predictions, batch)
                permuted_losses = model.loss(permuted_predictions, permuted_batch)

            for task_name in _TASK_NAMES:
                logits = predictions[f"logits_{task_name}"]
                permuted_logits = permuted_predictions[f"logits_{task_name}"]
                torch.testing.assert_close(
                    permuted_logits,
                    torch.cat([logits[first_request_len:], logits[:first_request_len]]),
                    rtol=1e-5,
                    atol=1e-6,
                    msg=f"{task_name} logits not request-permuted for "
                    f"ascending={ascending}",
                )
            torch.testing.assert_close(
                permuted_losses["listwise_rank_loss_is_click"],
                losses["listwise_rank_loss_is_click"],
                rtol=1e-5,
                atol=1e-6,
                msg=f"list-wise loss not request-invariant for ascending={ascending}",
            )

    @unittest.skipIf(*gpu_unavailable)
    def test_listwise_temperature_is_a_trainable_parameter(self) -> None:
        """The temperature must reach the optimizer.

        ``TrainWrapper.__init__`` calls ``init_loss()`` before the optimizer
        is built, so a loss-module parameter is only picked up if it is
        registered there and not later.
        """
        device = torch.device("cuda")
        model = _build_model(
            device=device,
            listwise_loss=_listwise_loss_cfg(learnable_temperature=True),
        )
        model.set_kernel(Kernel.PYTORCH)
        wrapper = TrainWrapper(model, device=device).to(device)
        names = [n for n, _ in wrapper.named_parameters()]
        self.assertIn(
            "model._loss_modules.listwise_rank_loss_is_click.logit_scale", names
        )

        total_loss, _ = wrapper(_build_batch(device=device))
        total_loss.backward()
        scale = wrapper.model._loss_modules["listwise_rank_loss_is_click"].logit_scale
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))

    def test_listwise_loss_is_a_standalone_objective(self) -> None:
        """A list-wise term without a point-wise sibling trains fine.

        ``listwise_rank_loss`` publishes its own ``logits_<task>`` (and
        ``probs_<task>``) in ``_output_to_prediction_impl``, so a task may
        carry it as its only loss.  The term scores over those logits;
        nothing requires a BCE/focal sibling to be present.
        """
        device = torch.device("cpu")
        task_configs = _task_configs(
            click_loss="l2_loss", listwise_loss=_listwise_loss_cfg()
        )
        # The l2 label path yields float values, and float labels cannot
        # pass through task_bitmask (bitwise_and has no Float kernel);
        # the bitmask is orthogonal to what this test pins, so drop it.
        task_configs[0].ClearField("task_bitmask")
        model = _build_model(
            device=device,
            task_configs=task_configs,
        )
        model.set_kernel(Kernel.PYTORCH)
        model.init_loss()
        batch = _build_batch(device=device)
        # Grad-enabled: the backward pass below is part of the contract.
        predictions = model.predict(batch)
        self.assertIn("logits_is_click", predictions)
        self.assertIn("probs_is_click", predictions)
        losses = model.loss(predictions, batch)
        self.assertIn("listwise_rank_loss_is_click", losses)
        total = losses["listwise_rank_loss_is_click"]
        self.assertTrue(torch.isfinite(total))
        total.backward()
        # The scorer's parameters must receive a gradient through the
        # standalone list-wise term alone.
        scored = False
        for name, param in model.named_parameters():
            if "_onerank_head." in name and param.grad is not None:
                self.assertTrue(
                    torch.isfinite(param.grad).all(),
                    msg=f"{name} got a non-finite gradient",
                )
                scored = True
        self.assertTrue(scored, "no prediction-head parameter received a gradient")

    def test_unsupported_task_configs_are_rejected(self) -> None:
        """Inner-product scoring is single-logit and needs at least one task."""
        with self.assertRaisesRegex(ValueError, "num_class"):
            _build_model(
                device=torch.device("cpu"), task_configs=_task_configs(num_class=3)
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            _build_model(device=torch.device("cpu"), task_configs=[])
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            _build_model(device=torch.device("cpu"), max_seq_len=0)

    def test_truncation_tuning_is_rejected_at_construction(self) -> None:
        """A reused ``dlrm_hstu`` block may carry truncation tuning.

        Mid-stack attention truncation is incompatible with the group
        layout; before the construction guard it only surfaced as
        ``OneRankSTULayer.truncate_input``'s ``NotImplementedError`` on
        the first training step, after torchrun, TorchRec sharding and
        the data pipeline were fully up.
        """
        for truncation_kwargs in (
            {"attn_truncation_split_layer": 2},
            {"attn_truncation_tail_len": 64},
        ):
            with self.subTest(**truncation_kwargs):
                with self.assertRaisesRegex(
                    ValueError, "mid-stack attention truncation"
                ):
                    _build_model(device=torch.device("cpu"), **truncation_kwargs)

    def test_scorer_type_wiring(self) -> None:
        """The proto string reaches the head verbatim."""
        for scorer in ("dot_product", "bilinear", "mlp"):
            with self.subTest(scorer=scorer):
                model = _build_model(device=torch.device("cpu"), scorer_type=scorer)
                self.assertEqual(model._onerank_head._scorer_type, scorer)

    def test_scorer_variants_build_exactly_their_parameters(self) -> None:
        """Each variant owns the parameters its docs promise, and no more.

        DOT adds nothing, BILINEAR adds one identity-init ``W_k`` per task,
        MLP adds one private two-layer MLP per task (see the head module
        and the model docs for why they exist).
        """
        device = torch.device("cpu")
        dot = _build_model(device=device)
        self.assertIsNone(dot._onerank_head._bilinear_weight)
        self.assertIsNone(dot._onerank_head._task_mlps)

        bilinear = _build_model(
            device=device,
            scorer_type="bilinear",
        )
        weight = bilinear._onerank_head._bilinear_weight
        self.assertEqual(weight.shape, (len(_TASK_NAMES), 64, 64))
        torch.testing.assert_close(
            weight.detach(), torch.eye(64).repeat(len(_TASK_NAMES), 1, 1)
        )

        mlp = _build_model(
            device=device,
            scorer_type="mlp",
            scorer_hidden_dim=32,
        )
        self.assertEqual(len(mlp._onerank_head._task_mlps), len(_TASK_NAMES))
        self.assertEqual(mlp._onerank_head._task_mlps[0][0].out_features, 32)

    def test_task_bias_init_wiring(self) -> None:
        """``task_bias_init`` lands in the head in ``task_configs`` order."""
        device = torch.device("cpu")
        bias = [0.25, -0.5, 1.0]
        model = _build_model(device=device, task_bias_init=bias)
        torch.testing.assert_close(
            model._onerank_head._task_bias.detach(), torch.tensor(bias)
        )
        with self.assertRaisesRegex(ValueError, "task_bias_init has"):
            _build_model(device=device, task_bias_init=[0.1, 0.2])

    @unittest.skipIf(*gpu_unavailable)
    def test_scorer_variants_train_one_step(self) -> None:
        """Every scorer variant survives a full train step end to end.

        Forward, backward, and the optimizer-bound gradient itself: a
        variant whose parameters never received gradients would silently
        degrade to the dot product.
        """
        device = torch.device("cuda")
        for scorer in (
            "bilinear",
            "mlp",
        ):
            with self.subTest(scorer=scorer):
                model = _build_model(
                    device=device,
                    scorer_type=scorer,
                    listwise_loss=_listwise_loss_cfg(),
                )
                model.set_kernel(Kernel.PYTORCH)
                wrapper = TrainWrapper(model, device=device).to(device)

                total_loss, _ = wrapper(_build_batch(device=device))
                self.assertTrue(torch.isfinite(total_loss))
                total_loss.backward()

                head = wrapper.model._onerank_head
                if scorer == "bilinear":
                    grad = head._bilinear_weight.grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                else:
                    for k, mlp in enumerate(head._task_mlps):
                        grad = mlp[0].weight.grad
                        self.assertIsNotNone(grad, msg=f"task {k} MLP has no grad")
                        self.assertTrue(torch.isfinite(grad).all())

    def test_scaling_seqlen_is_the_configured_max_seq_len(self) -> None:
        """The attention-output divisor is ``max_seq_len``, as in DlrmHSTU.

        ``max_seq_len`` carries the *inflated* sequence bound, so the
        divisor and the autotune bucket follow it verbatim; the group
        inflation factor (K + 1 tokens per candidate) stays private to
        the tokenizer.
        """
        model = _build_model(device=torch.device("cpu"))
        for layer in model._hstu_transducer._stu_module._stu_layers:
            self.assertEqual(layer._scaling_seqlen, 132)

    def test_runtime_rejects_inflated_sequences_above_max_seq_len(self) -> None:
        """Requests whose inflated sequence exceeds ``max_seq_len`` fail loudly.

        The static max sequence length -- and every jagged-kernel autotune
        bucket derived from it -- is the bound the tokenizer guards at
        runtime; a longer request used to overflow those bounds silently.
        """
        device = torch.device("cpu")
        # The batch carries requests of 2 and 4 candidates over 2 and 3
        # history tokens; the largest inflated sequence is 3 + 4 * (1 + 3)
        # = 19, so a bound of 18 must reject it.
        model = _build_model(device=device, max_seq_len=18)
        model.set_kernel(Kernel.PYTORCH)
        model.eval()
        batch = _build_batch(device=device)
        with (
            torch.no_grad(),
            self.assertRaisesRegex(ValueError, r"more than max_seq_len \(18\)"),
        ):
            model.predict(batch)

    # Method-level "h20" in addition to the class-level "gpu": these are
    # the only end-to-end CUTLASS NFUNC checks, and the gpu lane lacks the
    # fbgemm_gpu_hstu wheel -- without the h20 tag they would run on no
    # per-PR lane at all (see rank_integration_test.py for the precedent).
    @mark_ci_scope("h20", "gpu")
    @unittest.skipIf(*cutlass_hstu_unavailable)
    @unittest.skipIf(*gpu_unavailable)
    def test_cutlass_matches_pytorch_kernel(self) -> None:
        """Close the CUTLASS blind spot at model level.

        Triton has no NFUNC path, so CUTLASS is the kernel this model
        trains with, and no existing model-level test exercises it.
        ``pt_hstu_attention`` decodes the same encoding, so a disagreement
        here means the two kernels read the group mask differently.
        Both run under bf16 autocast (CUTLASS supports fp16/bf16 only), so
        the tolerance is bf16-wide on purpose.
        """
        device = torch.device("cuda")
        model = _build_model(device=device)
        batch = _build_batch(device=device)
        model.eval()

        outputs = {}
        for kernel in (Kernel.PYTORCH, Kernel.CUTLASS):
            model.set_kernel(kernel)
            with (
                torch.no_grad(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                outputs[kernel] = {
                    task_name: model.predict(batch)[f"probs_{task_name}"].float()
                    for task_name in _TASK_NAMES
                }

        for task_name in _TASK_NAMES:
            got = outputs[Kernel.CUTLASS][task_name]
            want = outputs[Kernel.PYTORCH][task_name]
            self.assertTrue(torch.isfinite(got).all(), f"{task_name} not finite")
            torch.testing.assert_close(
                got, want, rtol=5e-2, atol=5e-2, msg=f"kernel mismatch on {task_name}"
            )

    @mark_ci_scope("h20", "gpu")
    @unittest.skipIf(*cutlass_hstu_unavailable)
    @unittest.skipIf(*gpu_unavailable)
    def test_cutlass_backward_reaches_the_task_tokens(self) -> None:
        """The CUTLASS NFUNC backward must feed the task-token parameters.

        The task tokens are the only path by which the mask can influence
        learning; a backward that silently dropped them would train an
        ordinary HSTU with extra padding.
        """
        device = torch.device("cuda")
        model = _build_model(
            device=device,
            listwise_loss=_listwise_loss_cfg(),
        )
        model.set_kernel(Kernel.CUTLASS)
        # The CUTLASS kernel accepts fp16/bf16 only, so mixed precision must
        # go through TrainWrapper's own knob -- the production path driven
        # by train_config.mixed_precision.  A manual outer
        # torch.autocast(...) around the call would NOT work:
        # TrainWrapper.forward always enters its own autocast context, and
        # with mixed_precision=None that context is enabled=False, which
        # disables the outer one and hands the CUTLASS kernel fp32 q/k/v.
        wrapper = TrainWrapper(model, device=device, mixed_precision="BF16").to(device)

        total_loss, _ = wrapper(_build_batch(device=device))
        total_loss.float().backward()

        task_tokens = wrapper.model._hstu_transducer._tokenizer._task_tokens
        self.assertIsNotNone(task_tokens.grad)
        self.assertTrue(torch.isfinite(task_tokens.grad).all())
        self.assertGreater(task_tokens.grad.abs().max().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
