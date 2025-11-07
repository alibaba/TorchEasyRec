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


from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.autograd.profiler import record_function
from torchrec import JaggedTensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import (
    TRAGET_REPEAT_INTERLEAVE_KEY,
    RankModel,
    _is_classification_loss,
)
from tzrec.modules.gr.hstu_transducer import HSTUTransducer
from tzrec.modules.norm import LayerNorm, SwishLayerNorm
from tzrec.modules.task_tower import FusionMTLTower
from tzrec.modules.utils import (
    init_linear_xavier_weights_zero_bias,
)
from tzrec.ops.utils import set_static_max_seq_lens
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.protos.tower_pb2 import FusionSubTaskConfig
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.fx_util import fx_int_item, fx_numel

torch.fx.wrap(fx_int_item)
torch.fx.wrap(fx_numel)


@torch.fx.wrap
def _fx_construct_payload(
    payload_features: Dict[str, torch.Tensor],
    contextual_seq_embeddings: Dict[str, JaggedTensor],
    uih_seq_embeddings: Dict[str, JaggedTensor],
    candidate_seq_embeddings: Dict[str, JaggedTensor],
) -> Dict[str, torch.Tensor]:
    results: Dict[str, torch.Tensor] = {}
    for k, v in contextual_seq_embeddings.items():
        results[k] = v.values()
        results[k + "_offsets"] = v.offsets()
    for k, v in uih_seq_embeddings.items():
        results[k] = v.values()
    for k, v in candidate_seq_embeddings.items():
        results[k] = v.values()
    results.update(payload_features)
    return results


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

        self.init_input()

        contextual_feature_dims = self.embedding_group.group_dims("contextual")
        if len(set(contextual_feature_dims)) > 1:
            raise ValueError(
                "output_dim of features in contextual features_group must be same, "
                f"but now {set(contextual_feature_dims)}."
            )
        contextual_feature_dim = contextual_feature_dims[0]

        self._task_configs = self._model_config.fusion_mtl_tower.task_configs
        action_weights = []
        for task_cfg in self._task_configs:
            if task_cfg.HasField("task_bitmask"):
                action_weights.append(task_cfg.task_bitmask)

        # construct HSTU
        self._hstu_transducer: HSTUTransducer = HSTUTransducer(
            uih_embedding_dim=self.embedding_group.group_total_dim("uih"),
            target_embedding_dim=self.embedding_group.group_total_dim("candidate"),
            contextual_feature_dim=contextual_feature_dim,
            max_contextual_seq_len=len(contextual_feature_dims),
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

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        with record_function("## preprocess ##"):
            grouped_features = self.embedding_group(batch)

        with record_function("## item_forward ##"):
            candidates_item_embeddings = self._item_embedding_mlp(
                grouped_features["candidate.sequence"]
            )

        with record_function("## user_forward ##"):
            candidates_user_embeddings, _ = self._hstu_transducer(grouped_features)
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

    def _get_label(
        self, batch: Batch, task_cfg: FusionSubTaskConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        label_name = task_cfg.label_name
        is_sparse_label = any([_is_classification_loss(x) for x in task_cfg.losses])
        label = batch.sequence_dense_features[label_name]
        label_values = label.values().squeeze(1)
        if is_sparse_label:
            label_values = label_values.to(torch.int64)
        if task_cfg.HasField("task_bitmask"):
            label_values = (
                torch.bitwise_and(label_values, task_cfg.task_bitmask) > 0
            ).to(label_values.dtype)
        return label_values, label.lengths()

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
            label, label_lengths = self._get_label(batch, task_cfg)
            predictions[TRAGET_REPEAT_INTERLEAVE_KEY] = label_lengths

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
            label, label_lengths = self._get_label(batch, task_cfg)
            predictions[TRAGET_REPEAT_INTERLEAVE_KEY] = label_lengths

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
