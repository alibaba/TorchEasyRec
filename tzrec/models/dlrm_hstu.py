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


from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function
from torchrec import JaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel, _is_classification_loss
from tzrec.modules.embedding import SequenceEmbeddingGroup
from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.modules.norm import LayerNorm, SwishLayerNorm
from tzrec.modules.task_tower import FusionMTLTower
from tzrec.modules.utils import (
    init_linear_xavier_weights_zero_bias,
)
from tzrec.ops.jagged_tensors import concat_2D_jagged
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.protos.tower_pb2 import FusionSubTaskConfig
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.fx_util import fx_infer_max_len, fx_int_item

torch.fx.wrap(fx_infer_max_len)
torch.fx.wrap(fx_int_item)


@torch.fx.wrap
def _fx_odict_jt_vcat(odict_jt: Dict[str, JaggedTensor]) -> torch.Tensor:
    return torch.cat([t.values() for t in odict_jt.values()], dim=-1)


@torch.fx.wrap
def _fx_construct_payload(
    payload_features: Dict[str, torch.Tensor],
    contextual_seq_embeddings: Dict[str, JaggedTensor],
) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    for k, v in contextual_seq_embeddings.items():
        results[k] = v.values()
        results[k + "_offsets"] = v.offsets()
    results.update(payload_features)
    return results


@torch.fx.wrap
def _fx_mark_length_features(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


class DlrmHSTU(RankModel):
    """DLRM HSTU model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        assert model_config.WhichOneof("model") == "dlrm_hstu", (
            "invalid model config: %s" % self._model_config.WhichOneof("model")
        )
        assert isinstance(self._model_config, multi_task_rank_pb2.DlrmHSTU)

        set_static_max_seq_lens([self._model_config.max_seq_len])

        self.embedding_group = SequenceEmbeddingGroup(
            self._features, list(self._base_model_config.feature_groups)
        )

        name_to_feature = {x.name: x for x in features}
        name_to_feature_group = {
            x.group_name: x for x in self._base_model_config.feature_groups
        }
        hstu_embedding_table_dim = name_to_feature[
            self._model_config.uih_id_feature_name
        ]._embedding_dim

        self._merge_uih_candidate_feature_mapping = list(
            zip(
                name_to_feature_group["uih"].feature_names,
                name_to_feature_group["candidate"].feature_names,
            )
        )
        self._merge_uih_candidate_payload_mapping = [
            (
                self._model_config.uih_action_time_feature_name,
                self._model_config.candidates_query_time_feature_name,
                True,  # is_sparse
            ),
            (
                self._model_config.uih_action_weight_feature_name,
                self._model_config.candidates_action_weight_feature_name,
                True,
            ),
            (
                self._model_config.uih_watchtime_feature_name,
                self._model_config.candidates_watchtime_feature_name,
                False,
            ),
        ]

        self._task_configs = self._model_config.fusion_mtl_tower.task_configs
        action_weights = []
        for task_cfg in self._task_configs:
            if task_cfg.HasField("task_bitmask"):
                action_weights.append(task_cfg.task_bitmask)

        # construct HSTU
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            input_embedding_dim=hstu_embedding_table_dim,
            **config_to_kwargs(self._model_config.hstu),
            return_full_embeddings=False,
            listwise=False,
        )

        # item embeddings
        self._item_embedding_mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=self.embedding_group.group_total_dim("candidate"),
                out_features=self._model_config.item_embedding_hidden_dim,
            ),
            SwishLayerNorm(self._model_config.item_embedding_hidden_dim),
            torch.nn.Linear(
                in_features=self._model_config.item_embedding_hidden_dim,
                out_features=self._model_config.hstu.stu.embedding_dim,
            ),
            LayerNorm(self._model_config.hstu.stu.embedding_dim),
        ).apply(init_linear_xavier_weights_zero_bias)

        self._multitask_module = FusionMTLTower(
            tower_feature_in=self._model_config.hstu.stu.embedding_dim,
            **config_to_kwargs(self._model_config.fusion_mtl_tower),
        ).apply(init_linear_xavier_weights_zero_bias)

    def _user_forward(
        self,
        max_uih_len: int,
        max_candidates: int,
        payload_features: Dict[str, torch.Tensor],
        uih_seq_embeddings: Dict[str, JaggedTensor],
        contextual_seq_embeddings: Dict[str, JaggedTensor],
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        source_lengths = uih_seq_embeddings[
            self._model_config.uih_id_feature_name
        ].lengths()
        source_timestamps = concat_2D_jagged(
            values_left=payload_features[
                self._model_config.uih_action_time_feature_name
            ],
            values_right=payload_features[
                self._model_config.candidates_query_time_feature_name
            ],
            max_len_left=max_uih_len,
            max_len_right=max_candidates,
            offsets_left=payload_features["uih_offsets"],
            offsets_right=payload_features["candidate_offsets"],
            kernel=self.kernel(),
        ).squeeze(-1)
        total_targets = fx_int_item(num_candidates.sum())
        candidates_user_embeddings, _ = self._hstu_transducer(
            max_uih_len=max_uih_len,
            max_targets=max_candidates,
            total_uih_len=source_timestamps.numel() - total_targets,
            total_targets=total_targets,
            seq_embeddings=uih_seq_embeddings[
                self._model_config.uih_id_feature_name
            ].values(),
            seq_lengths=source_lengths,
            seq_timestamps=source_timestamps,
            seq_payloads=_fx_construct_payload(
                payload_features=payload_features,
                contextual_seq_embeddings=contextual_seq_embeddings,
            ),
            num_targets=num_candidates,
        )

        return candidates_user_embeddings

    def _item_forward(
        self,
        candidate_seq_embeddings: Dict[str, JaggedTensor],
    ) -> torch.Tensor:  # [L, D]
        all_embeddings = _fx_odict_jt_vcat(candidate_seq_embeddings)
        item_embeddings = self._item_embedding_mlp(all_embeddings)
        return item_embeddings

    def preprocess(
        self,
        batch: Batch,
    ) -> Tuple[
        Dict[str, OrderedDict[str, JaggedTensor]],
        Dict[str, torch.Tensor],
        int,
        torch.Tensor,
        int,
        torch.Tensor,
    ]:
        """Do embedding lookup and prepare payload features."""
        grouped_features = self.embedding_group.jagged_forward(batch)

        sparse_features = batch.sparse_features[BASE_DATA_GROUP].to_dict()
        sequence_dense_features = batch.sequence_dense_features

        num_candidates = _fx_mark_length_features(
            sparse_features[self._model_config.candidates_id_feature_name].lengths()
        )
        max_num_candidates = fx_infer_max_len(num_candidates)

        uih_seq_lengths = sparse_features[
            self._model_config.uih_id_feature_name
        ].lengths()
        max_uih_len = fx_infer_max_len(uih_seq_lengths)

        # prepare payload features
        payload_features: Dict[str, torch.Tensor] = {}
        for (
            uih_feature_name,
            candidate_feature_name,
            is_sparse,
        ) in self._merge_uih_candidate_payload_mapping:
            if is_sparse:
                values_left = sparse_features[uih_feature_name].values().unsqueeze(-1)
            else:
                values_left = sequence_dense_features[uih_feature_name].values()

            if self._is_inference and (
                candidate_feature_name
                == self._model_config.candidates_action_weight_feature_name
                or candidate_feature_name
                == self._model_config.candidates_watchtime_feature_name
            ):
                values_right = torch.zeros_like(
                    sparse_features[
                        self._model_config.candidates_id_feature_name
                    ].values(),
                    dtype=values_left.dtype,
                )
            elif is_sparse:
                values_right = (
                    sparse_features[candidate_feature_name].values().unsqueeze(-1)
                )
            else:
                values_right = sequence_dense_features[candidate_feature_name].values()

            payload_features[uih_feature_name] = values_left
            payload_features[candidate_feature_name] = values_right
        payload_features["uih_offsets"] = torch.ops.fbgemm.asynchronous_complete_cumsum(
            uih_seq_lengths
        )
        payload_features["candidate_offsets"] = (
            torch.ops.fbgemm.asynchronous_complete_cumsum(num_candidates)
        )

        return (
            grouped_features,
            payload_features,
            max_uih_len,
            uih_seq_lengths,
            max_num_candidates,
            num_candidates,
        )

    def main_forward(
        self,
        grouped_features: Dict[str, OrderedDict[str, JaggedTensor]],
        payload_features: Dict[str, torch.Tensor],
        max_uih_len: int,
        uih_seq_lengths: torch.Tensor,
        max_num_candidates: int,
        num_candidates: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward User and Item network."""
        # merge uih and candidates embeddings
        uih_seq_embeddings = OrderedDict()
        for (
            uih_feature_name,
            candidate_feature_name,
        ) in self._merge_uih_candidate_feature_mapping:
            uih_seq_embeddings[uih_feature_name] = JaggedTensor(
                lengths=uih_seq_lengths + num_candidates,
                values=concat_2D_jagged(
                    max_len_left=max_uih_len,
                    offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                        uih_seq_lengths
                    ),
                    values_left=grouped_features["uih"][uih_feature_name].values(),
                    max_len_right=max_num_candidates,
                    offsets_right=torch.ops.fbgemm.asynchronous_complete_cumsum(
                        num_candidates
                    ),
                    values_right=grouped_features["candidate"][
                        candidate_feature_name
                    ].values(),
                    kernel=self.kernel(),
                ),
            )
        candidate_seq_embeddings = grouped_features["candidate"]

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_forward(
                candidate_seq_embeddings,
            )
        with record_function("## user_forward ##"):
            candidates_user_embeddings = self._user_forward(
                max_uih_len=max_uih_len,
                max_candidates=max_num_candidates,
                payload_features=payload_features,
                uih_seq_embeddings=uih_seq_embeddings,
                contextual_seq_embeddings=grouped_features["contextual"],
                num_candidates=num_candidates,
            )
        with record_function("## multitask_module ##"):
            mt_preds = self._multitask_module(
                candidates_user_embeddings, candidates_item_embeddings
            )

        predictions = {}
        for task_cfg in self._task_configs:
            task_name = task_cfg.task_name
            for loss_cfg in task_cfg.losses:
                predictions.update(
                    self._output_to_prediction_impl(
                        mt_preds[task_name],
                        loss_cfg,
                        suffix=f"_{task_name}",
                    )
                )

        return predictions

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        with record_function("## preprocess ##"):
            (
                grouped_features,
                payload_features,
                max_uih_len,
                uih_seq_lengths,
                max_num_candidates,
                num_candidates,
            ) = self.preprocess(batch)

        with record_function("## main_forward ##"):
            return self.main_forward(
                grouped_features=grouped_features,
                payload_features=payload_features,
                max_uih_len=max_uih_len,
                uih_seq_lengths=uih_seq_lengths,
                max_num_candidates=max_num_candidates,
                num_candidates=num_candidates,
            )

    def _get_label(self, batch: Batch, task_cfg: FusionSubTaskConfig) -> torch.Tensor:
        label_name = task_cfg.label_name
        is_sparse_label = any([_is_classification_loss(x) for x in task_cfg.losses])
        if is_sparse_label:
            label = batch.sparse_features[BASE_DATA_GROUP][label_name].values()
        else:
            label = batch.dense_features[BASE_DATA_GROUP][label_name]
        if task_cfg.HasField("task_bitmask"):
            label = (torch.bitwise_and(label, task_cfg.task_bitmask) > 0).to(
                label.dtype
            )
        return label

    def init_loss(self) -> None:
        """Initialize loss modules."""
        for task_cfg in self._task_configs:
            for loss_cfg in task_cfg.losses:
                task_name = task_cfg.task_name
                self._init_loss_impl(loss_cfg, suffix=f"_{task_name}", reduction="mean")

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for task_cfg in self._task_configs:
            task_name = task_cfg.task_name
            label = self._get_label(batch, task_cfg)

            for loss_cfg in task_cfg.losses:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        label,
                        None,
                        loss_cfg,
                        suffix=f"_{task_name}",
                    )
                )
        losses.update(self._loss_collection)
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for task_cfg in self._task_configs:
            task_name = task_cfg.task_name
            for metric_cfg in task_cfg.metrics:
                self._init_metric_impl(
                    metric_cfg,
                    suffix=f"_{task_name}",
                )
            for loss_cfg in task_cfg.losses:
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{task_name}")

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
            losses (dict, optional): a dict of loss.
        """
        for task_cfg in self._task_configs:
            task_name = task_cfg.task_name
            label = self._get_label(batch, task_cfg)

            for metric_cfg in task_cfg.metrics:
                self._update_metric_impl(
                    predictions,
                    batch,
                    label,
                    metric_cfg,
                    suffix=f"_{task_name}",
                )
            if losses is not None:
                for loss_cfg in task_cfg.losses:
                    self._update_loss_metric_impl(
                        losses,
                        batch,
                        label,
                        loss_cfg,
                        suffix=f"_{task_name}",
                    )
