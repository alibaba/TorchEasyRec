# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from torch import nn

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.jrc_loss import JRCLoss
from tzrec.metrics.grouped_auc import GroupedAUC
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.variational_dropout import VariationalDropout
from tzrec.protos import model_pb2
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.metric_pb2 import MetricConfig
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _update_tensor_dict(
    tensor_dict: Dict[str, torch.Tensor], new_tensor: torch.Tensor, key: str
) -> None:
    tensor_dict[key] = new_tensor


class RankModel(BaseModel):
    """Base model for ranking.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self._num_class = model_config.num_class
        self._label_name = labels[0]
        self._loss_collection = {}
        self.embedding_group = None
        self.group_variational_dropouts = None

    def init_input(self) -> None:
        """Build embedding group and group variational dropout."""
        self.embedding_group = EmbeddingGroup(
            self._features, list(self._base_model_config.feature_groups)
        )

        if self._base_model_config.HasField("variational_dropout"):
            self.group_variational_dropouts = nn.ModuleDict()
            variational_dropout_config = self._base_model_config.variational_dropout
            variational_dropout_config_dict = config_to_kwargs(
                variational_dropout_config
            )
            for feature_group in list(self._base_model_config.feature_groups):
                group_name = feature_group.group_name
                if feature_group.group_type != model_pb2.SEQUENCE:
                    feature_dim = self.embedding_group.group_feature_dims(group_name)
                    if len(feature_dim) > 1:
                        variational_dropout = VariationalDropout(
                            feature_dim, group_name, **variational_dropout_config_dict
                        )
                        self.group_variational_dropouts[group_name] = (
                            variational_dropout
                        )

    def build_input(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Build input feature."""
        feature_dict = self.embedding_group(batch)
        if self.group_variational_dropouts is not None:
            for (
                group_name,
                variational_dropout,
            ) in self.group_variational_dropouts.items():
                feature, variational_dropout_loss = variational_dropout(
                    feature_dict[group_name]
                )
                _update_tensor_dict(feature_dict, feature, group_name)
                _update_tensor_dict(
                    self._loss_collection,
                    variational_dropout_loss,
                    group_name + "_feature_p_loss",
                )
        return feature_dict

    def _output_to_prediction_impl(
        self,
        output: torch.Tensor,
        loss_cfg: LossConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        predictions = {}
        loss_type = loss_cfg.WhichOneof("loss")
        if loss_type == "binary_cross_entropy":
            assert num_class == 1, f"num_class must be 1 when loss type is {loss_type}"
            output = torch.squeeze(output, dim=1)
            predictions["logits" + suffix] = output
            predictions["probs" + suffix] = torch.sigmoid(output)
        elif loss_type == "softmax_cross_entropy":
            assert (
                num_class > 1
            ), f"num_class must be greater than 1 when loss type is {loss_type}"
            probs = torch.softmax(output, dim=1)
            predictions["logits" + suffix] = output
            predictions["probs" + suffix] = probs
            if num_class == 2:
                predictions["probs1" + suffix] = probs[:, 1]
        elif loss_type == "jrc_loss":
            assert num_class == 2, f"num_class must be 2 when loss type is {loss_type}"
            probs = torch.softmax(output, dim=1)
            predictions["logits" + suffix] = output
            predictions["probs" + suffix] = probs
            predictions["probs1" + suffix] = probs[:, 1]
        elif loss_type == "l2_loss":
            predictions["y" + suffix] = output
        else:
            raise NotImplementedError
        return predictions

    def _output_to_prediction(self, output: torch.Tensor) -> Dict[str, torch.Tensor]:
        predictions = {}
        for loss_cfg in self._base_model_config.losses:
            predictions.update(
                self._output_to_prediction_impl(
                    output, loss_cfg, num_class=self._num_class
                )
            )
        return predictions

    def _init_loss_impl(
        self, loss_cfg: LossConfig, num_class: int = 1, suffix: str = ""
    ) -> None:
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        if loss_type == "binary_cross_entropy":
            self._loss_modules[loss_name] = nn.BCEWithLogitsLoss()
        elif loss_type == "softmax_cross_entropy":
            self._loss_modules[loss_name] = nn.CrossEntropyLoss()
        elif loss_type == "jrc_loss":
            assert num_class == 2, f"num_class must be 2 when loss type is {loss_type}"
            self._loss_modules[loss_name] = JRCLoss(
                alpha=loss_cfg.jrc_loss.alpha,
            )
        elif loss_type == "l2_loss":
            self._loss_modules[loss_name] = nn.MSELoss()
        else:
            raise ValueError(f"loss[{loss_type}] is not supported yet.")

    def init_loss(self) -> None:
        """Initialize loss modules."""
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_impl(loss_cfg, self._num_class)

    def _loss_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label_name: str,
        loss_cfg: LossConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        label = batch.labels[label_name]

        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        if loss_type == "binary_cross_entropy":
            pred = predictions["logits" + suffix]
            label = label.to(torch.float32)
            losses[loss_name] = self._loss_modules[loss_name](pred, label)
        elif loss_type == "softmax_cross_entropy":
            pred = predictions["logits" + suffix]
            losses[loss_name] = self._loss_modules[loss_name](pred, label)
        elif loss_type == "jrc_loss":
            assert num_class == 2, f"num_class must be 2 when loss type is {loss_type}"
            pred = predictions["logits" + suffix]
            session_id = batch.sparse_features[BASE_DATA_GROUP][
                loss_cfg.jrc_loss.session_name
            ].values()
            losses[loss_name] = self._loss_modules[loss_name](pred, label, session_id)
        elif loss_type == "l2_loss":
            pred = predictions["y" + suffix]
            losses[loss_name] = self._loss_modules[loss_name](pred, label)
        else:
            raise ValueError(f"loss[{loss_type}] is not supported yet.")
        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for loss_cfg in self._base_model_config.losses:
            losses.update(
                self._loss_impl(
                    predictions,
                    batch,
                    self._label_name,
                    loss_cfg,
                    num_class=self._num_class,
                )
            )
        losses.update(self._loss_collection)
        return losses

    def _init_metric_impl(
        self, metric_cfg: MetricConfig, num_class: int = 1, suffix: str = ""
    ) -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_kwargs = config_to_kwargs(oneof_metric_cfg)
        metric_name = metric_type + suffix
        if metric_type == "auc":
            assert (
                num_class <= 2
            ), f"num_class must less than 2 when metric type is {metric_type}"
            self._metric_modules[metric_name] = torchmetrics.AUROC(
                task="binary", **metric_kwargs
            )
        elif metric_type == "multiclass_auc":
            self._metric_modules[metric_name] = torchmetrics.AUROC(
                task="multiclass", num_class=num_class, **metric_kwargs
            )
        elif metric_type == "mean_absolute_error":
            pass
        elif metric_type == "mean_squared_error":
            pass
        elif metric_type == "accuracy":
            pass
        elif metric_type == "grouped_auc":
            assert (
                num_class <= 2
            ), f"num_class must less than 2 when metric type is {metric_type}"
            self._metric_modules[metric_name] = GroupedAUC()
        else:
            raise ValueError(f"{metric_type} is not supported for this model")

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for metric_cfg in self._base_model_config.metrics:
            self._init_metric_impl(metric_cfg, self._num_class)
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_metric_impl(loss_cfg)

    def _update_metric_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label_name: str,
        metric_cfg: MetricConfig,
        num_class: int = 1,
        suffix: str = "",
    ) -> None:
        label = batch.labels[label_name]

        metric_type = metric_cfg.WhichOneof("metric")
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_name = metric_type + suffix

        base_sparse_feat = None
        if metric_type in ["grouped_auc"]:
            base_sparse_feat = batch.sparse_features[BASE_DATA_GROUP].to_dict()

        if metric_type == "auc":
            pred = (
                predictions["probs" + suffix]
                if num_class == 1
                else predictions["probs1" + suffix]
            )
            self._metric_modules[metric_name].update(pred, label)
        elif metric_type == "multiclass_auc":
            pred = predictions["probs" + suffix]
            self._metric_modules[metric_name].update(pred, label)
        elif metric_type == "mean_absolute_error":
            pass
        elif metric_type == "mean_squared_error":
            pass
        elif metric_type == "grouped_auc":
            pred = (
                predictions["probs" + suffix]
                if num_class == 1
                else predictions["probs1" + suffix]
            )
            # pyre-ignore [16]
            grouping_key = base_sparse_feat[
                oneof_metric_cfg.grouping_key
            ].to_padded_dense(1)[:, 0]
            self._metric_modules[metric_name].update(pred, label, grouping_key)
        else:
            raise ValueError(f"{metric_type} is not supported for this model")

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
        for metric_cfg in self._base_model_config.metrics:
            self._update_metric_impl(
                predictions,
                batch,
                self._label_name,
                metric_cfg,
                num_class=self._num_class,
            )
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                self._update_loss_metric_impl(losses, batch, self._label_name, loss_cfg)
