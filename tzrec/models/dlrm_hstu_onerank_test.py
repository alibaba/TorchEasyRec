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

"""Model-level tests for ``tzrec.models.dlrm_hstu_onerank``.

Kernel coverage is ``[PYTORCH, CUTLASS]`` rather than the baseline's
``[PYTORCH, TRITON]``: the OneRank group mask is an NFUNC func tensor and
Triton has no NFUNC path, so CUTLASS is the kernel this model actually
trains with. ``pt_hstu_attention`` implements the same encoding and serves
as the reference.
"""

import unittest
from typing import List, Optional

import torch
from hypothesis import Verbosity, given
from hypothesis import strategies as st
from torchrec import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
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
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    TestGraphType,
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


def _task_configs(
    task_weight: float = 1.0,
    num_class: int = 1,
    click_loss: str = "binary_cross_entropy",
) -> List[tower_pb2.FusionSubTaskConfig]:
    if click_loss == "binary_cross_entropy":
        click_loss_cfg = loss_pb2.LossConfig(
            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
        )
    else:
        click_loss_cfg = loss_pb2.LossConfig(l2_loss=loss_pb2.L2Loss())
    return [
        tower_pb2.FusionSubTaskConfig(
            task_name="is_click",
            label_name="item_action_weight",
            task_bitmask=1,
            num_class=num_class,
            losses=[click_loss_cfg],
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
    listwise_losses: Optional[List[multi_task_rank_pb2.OneRankListwiseLoss]] = None,
    max_num_candidates: int = 8,
    output_dropout_ratio: float = 0.0,
    scorer_type: Optional[int] = None,
    scorer_hidden_dim: Optional[int] = None,
    task_bias_init: Optional[List[float]] = None,
) -> model_pb2.ModelConfig:
    onerank = multi_task_rank_pb2.OneRankConfig(
        max_num_candidates=max_num_candidates,
        listwise_losses=listwise_losses or [],
    )
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
            ),
            # `mlp` is ignored by OneRank but FusionMTLTower requires it.
            fusion_mtl_tower=tower_pb2.FusionMTLTower(
                mlp=module_pb2.MLP(hidden_units=[64], activation="nn.SiLU"),
                task_configs=task_configs,
            ),
            max_seq_len=100,
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
    if task_configs is None:
        task_configs = _task_configs()
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


@mark_ci_scope("gpu")
class DlrmHSTUOneRankTest(unittest.TestCase):
    """End-to-end tests of the OneRank model on the PYTORCH/CUTLASS kernels."""

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
        listwise_losses = (
            [multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click", alpha=0.5)]
            if with_listwise_loss
            else None
        )
        model = _build_model(
            device=device,
            with_situation_discernment=with_situation_discernment,
            with_cross_task_head=with_cross_task_head,
            contextual_group_type=contextual_group_type,
            concat_contextual_features=concat_contextual_features,
            sequence_timestamp_is_ascending=sequence_timestamp_is_ascending,
            enable_global_average_loss=enable_global_average_loss,
            listwise_losses=listwise_losses,
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
                ["listwise_infonce_is_click"] if with_listwise_loss else []
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

        A wiring slip that returned the replica slot instead of the task
        slot, or collapsed K, would still give correctly shaped logits --
        this is where that is caught.
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
    def test_listwise_loss_is_scaled_by_alpha(self) -> None:
        """``alpha`` is the only knob multiplying the list-wise term.

        Task ``weight`` scales the point-wise loss instead, so the two must
        not be conflated: the total is
        ``sum_k alpha_k * L_list_k + sum_k weight_k * L_point_k``.
        """
        device = torch.device("cuda")
        alpha = 0.25
        base = _build_model(
            device=device,
            listwise_losses=[
                multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click", alpha=1.0)
            ],
        )
        scaled = _build_model(
            device=device,
            listwise_losses=[
                multi_task_rank_pb2.OneRankListwiseLoss(
                    task_name="is_click", alpha=alpha
                )
            ],
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

        key = "listwise_infonce_is_click"
        self.assertGreater(base_losses[key].item(), 0.0)
        torch.testing.assert_close(
            scaled_losses[key], base_losses[key] * alpha, rtol=1e-5, atol=1e-6
        )
        # Point-wise terms are untouched.
        for task_name in _TASK_NAMES:
            torch.testing.assert_close(
                scaled_losses[f"binary_cross_entropy_{task_name}"],
                base_losses[f"binary_cross_entropy_{task_name}"],
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
            listwise_losses=[
                multi_task_rank_pb2.OneRankListwiseLoss(
                    task_name="is_click", learnable_temperature=True
                )
            ],
        )
        model.set_kernel(Kernel.PYTORCH)
        wrapper = TrainWrapper(model, device=device).to(device)
        names = [n for n, _ in wrapper.named_parameters()]
        self.assertIn(
            "model._loss_modules.listwise_infonce_is_click.logit_scale", names
        )

        total_loss, _ = wrapper(_build_batch(device=device))
        total_loss.backward()
        scale = wrapper.model._loss_modules["listwise_infonce_is_click"].logit_scale
        self.assertIsNotNone(scale.grad)
        self.assertTrue(torch.isfinite(scale.grad))

    def test_listwise_loss_config_is_validated(self) -> None:
        """Misconfiguration fails at ``init_loss()``, not at step 1.

        All three of these would otherwise surface as a ``KeyError`` deep in
        ``loss()`` after the first forward pass.
        """
        device = torch.device("cpu")
        cases = [
            (
                [
                    multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click"),
                    multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click"),
                ],
                None,
                "more than one entry",
            ),
            (
                [multi_task_rank_pb2.OneRankListwiseLoss(task_name="nope")],
                None,
                "not in fusion_mtl_tower.task_configs",
            ),
            (
                [multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click")],
                _task_configs(click_loss="l2_loss"),
                "needs that task to also carry",
            ),
        ]
        for listwise_losses, task_configs, message in cases:
            with self.subTest(message=message):
                model = _build_model(
                    device=device,
                    listwise_losses=listwise_losses,
                    task_configs=task_configs,
                )
                with self.assertRaisesRegex(ValueError, message):
                    model.init_loss()

    def test_unsupported_task_configs_are_rejected(self) -> None:
        """Inner-product scoring is single-logit and needs at least one task."""
        with self.assertRaisesRegex(ValueError, "num_class"):
            _build_model(
                device=torch.device("cpu"), task_configs=_task_configs(num_class=3)
            )
        with self.assertRaisesRegex(ValueError, "at least one"):
            _build_model(device=torch.device("cpu"), task_configs=[])
        with self.assertRaisesRegex(ValueError, "max_num_candidates"):
            _build_model(device=torch.device("cpu"), max_num_candidates=0)

    def test_scorer_type_wiring(self) -> None:
        """The proto enum reaches the head as the documented string."""
        for enum_val, expected in (
            (multi_task_rank_pb2.ONERANK_SCORER_DOT_PRODUCT, "dot_product"),
            (multi_task_rank_pb2.ONERANK_SCORER_BILINEAR, "bilinear"),
            (multi_task_rank_pb2.ONERANK_SCORER_MLP, "mlp"),
        ):
            with self.subTest(scorer=expected):
                model = _build_model(device=torch.device("cpu"), scorer_type=enum_val)
                self.assertEqual(model._onerank_head._scorer_type, expected)

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
            scorer_type=multi_task_rank_pb2.ONERANK_SCORER_BILINEAR,
        )
        weight = bilinear._onerank_head._bilinear_weight
        self.assertEqual(weight.shape, (len(_TASK_NAMES), 64, 64))
        torch.testing.assert_close(
            weight.detach(), torch.eye(64).repeat(len(_TASK_NAMES), 1, 1)
        )

        mlp = _build_model(
            device=device,
            scorer_type=multi_task_rank_pb2.ONERANK_SCORER_MLP,
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
        for enum_val in (
            multi_task_rank_pb2.ONERANK_SCORER_BILINEAR,
            multi_task_rank_pb2.ONERANK_SCORER_MLP,
        ):
            with self.subTest(
                scorer=multi_task_rank_pb2.OneRankScorerType.Name(enum_val)
            ):
                model = _build_model(
                    device=device,
                    scorer_type=enum_val,
                    listwise_losses=[
                        multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click")
                    ],
                )
                model.set_kernel(Kernel.PYTORCH)
                wrapper = TrainWrapper(model, device=device).to(device)

                total_loss, _ = wrapper(_build_batch(device=device))
                self.assertTrue(torch.isfinite(total_loss))
                total_loss.backward()

                head = wrapper.model._onerank_head
                if enum_val == multi_task_rank_pb2.ONERANK_SCORER_BILINEAR:
                    grad = head._bilinear_weight.grad
                    self.assertIsNotNone(grad)
                    self.assertTrue(torch.isfinite(grad).all())
                else:
                    for k, mlp in enumerate(head._task_mlps):
                        grad = mlp[0].weight.grad
                        self.assertIsNotNone(grad, msg=f"task {k} MLP has no grad")
                        self.assertTrue(torch.isfinite(grad).all())

    def test_scaling_seqlen_tracks_the_inflated_sequence(self) -> None:
        """The attention-output divisor must follow the group inflation.

        Leaving it at ``max_seq_len`` would silently change the
        normalization scale relative to the DlrmHSTU baseline and make the
        A/B comparison meaningless.
        """
        model = _build_model(device=torch.device("cpu"), max_num_candidates=8)
        # 100 + 8 candidates * 2 * 3 tasks
        self.assertEqual(model._inflated_max_seq_len(), 100 + 8 * 2 * 3)
        for layer in model._hstu_transducer._stu_module._stu_layers:
            self.assertEqual(layer._scaling_seqlen, 100 + 8 * 2 * 3)

    def test_runtime_rejects_more_candidates_than_configured(self) -> None:
        """Requests above ``onerank.max_num_candidates`` fail loudly.

        The static max sequence length -- and every jagged-kernel autotune
        bucket derived from it -- is sized from ``max_num_candidates``;
        a longer request used to overflow those bounds silently.
        """
        device = torch.device("cpu")
        # The batch carries requests of 2 and 4 candidates.
        model = _build_model(device=device, max_num_candidates=2)
        model.set_kernel(Kernel.PYTORCH)
        model.eval()
        batch = _build_batch(device=device)
        with (
            torch.no_grad(),
            self.assertRaisesRegex(
                ValueError, r"more than onerank\.max_num_candidates \(2\)"
            ),
        ):
            model.predict(batch)

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
            listwise_losses=[
                multi_task_rank_pb2.OneRankListwiseLoss(task_name="is_click")
            ],
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
