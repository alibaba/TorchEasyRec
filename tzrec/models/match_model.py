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

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.metrics import recall_at_k
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.utils import div_no_nan
from tzrec.modules.variational_dropout import VariationalDropout
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.metric_pb2 import MetricConfig
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _zero_int_label(pred: torch.Tensor) -> torch.Tensor:
    return torch.zeros((pred.size(0),), dtype=torch.int64, device=pred.device)


@torch.fx.wrap
def _arange_int_label(pred: torch.Tensor) -> torch.Tensor:
    return torch.arange(pred.size(0), dtype=torch.int64, device=pred.device)


@torch.fx.wrap
def _update_tensor_2_dict(
    tensor_dict: Dict[str, torch.Tensor], new_tensor: torch.Tensor, key: str
) -> None:
    tensor_dict[key] = new_tensor


class MatchTower(nn.Module):
    """Base match tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: Union[
            tower_pb2.Tower,
            tower_pb2.DATTower,
            tower_pb2.MINDUserTower,
            tower_pb2.MINDItemTower,
        ],
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__()
        self._tower_config = tower_config
        self._group_name = tower_config.input
        self._output_dim = output_dim
        self._similarity = similarity
        self._feature_group = feature_group
        self._features = features
        self._model_config = model_config
        self.embedding_group = None
        self.group_variational_dropouts = None
        self.group_variational_dropout_loss = {}

    def init_input(self) -> None:
        """Build embedding group and group variational dropout."""
        self.embedding_group = EmbeddingGroup(self._features, [self._feature_group])

        if self._model_config.HasField("variational_dropout"):
            self.group_variational_dropouts = nn.ModuleDict()
            variational_dropout_config = self._model_config.variational_dropout
            variational_dropout_config_dict = config_to_kwargs(
                variational_dropout_config
            )

            if self._feature_group.group_type != model_pb2.SEQUENCE:
                feature_dim = self.embedding_group.group_feature_dims(self._group_name)
                if len(feature_dim) > 1:
                    variational_dropout = VariationalDropout(
                        feature_dim, self._group_name, **variational_dropout_config_dict
                    )
                    self.group_variational_dropouts[self._group_name] = (
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
                    feature_dict[self._group_name]
                )
                _update_tensor_2_dict(feature_dict, feature, self._group_name)
                _update_tensor_2_dict(
                    self.group_variational_dropout_loss,
                    variational_dropout_loss,
                    group_name + "_feature_p_loss",
                )
        return feature_dict


class MatchTowerWoEG(nn.Module):
    """Base match tower without embedding group for share embedding.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
    ) -> None:
        super().__init__()
        self._tower_config = tower_config
        self._group_name = tower_config.input
        self._output_dim = output_dim
        self._similarity = similarity
        self._feature_group = feature_group
        self._features = features


class MatchModel(BaseModel):
    """Base model for match.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names
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
        self._num_class = model_config.num_class
        self._label_name = labels[0]
        self._sample_weight = sample_weights[0] if sample_weights else sample_weights
        self._in_batch_negative = False
        self._loss_collection = {}
        if self._model_config and hasattr(self._model_config, "in_batch_negative"):
            self._in_batch_negative = self._model_config.in_batch_negative

    def sim(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        neg_for_each_sample: bool = False,
    ) -> torch.Tensor:
        """Calculate user and item embedding similarity."""
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)
        else:
            batch_size = user_emb.size(0)
            pos_item_emb = item_emb[:batch_size]
            neg_item_emb = item_emb[batch_size:]
            pos_ui_sim = torch.sum(
                torch.multiply(user_emb, pos_item_emb), dim=-1, keepdim=True
            )
            neg_ui_sim = None
            if not neg_for_each_sample:
                neg_ui_sim = torch.matmul(user_emb, neg_item_emb.transpose(0, 1))
            else:
                # Calculate similarity for each user with corresponding negative items
                num_neg_per_user = neg_item_emb.size(0) // batch_size
                neg_size = batch_size * num_neg_per_user
                neg_item_emb = neg_item_emb[:neg_size]
                neg_item_emb = neg_item_emb.view(batch_size, num_neg_per_user, -1)
                neg_ui_sim = torch.sum(user_emb.unsqueeze(1) * neg_item_emb, dim=-1)
            return torch.cat([pos_ui_sim, neg_ui_sim], dim=-1)

    def _init_loss_impl(self, loss_cfg: LossConfig, suffix: str = "") -> None:
        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        assert (
            loss_type == "softmax_cross_entropy"
        ), "match model only support softmax_cross_entropy loss now."
        reduction = "none" if self._sample_weight else "mean"
        self._loss_modules[loss_name] = nn.CrossEntropyLoss(reduction=reduction)

    def init_loss(self) -> None:
        """Initialize loss modules."""
        assert (
            len(self._base_model_config.losses) == 1
        ), "match model only support single loss now."
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_impl(loss_cfg)

    def _loss_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label_name: str,
        loss_cfg: LossConfig,
        suffix: str = "",
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        label = batch.labels[label_name]
        sample_weight = (
            batch.sample_weights[self._sample_weight]
            if self._sample_weight
            else torch.Tensor([1.0])
        )

        loss_type = loss_cfg.WhichOneof("loss")
        loss_name = loss_type + suffix
        assert (
            loss_type == "softmax_cross_entropy"
        ), "match model only support softmax_cross_entropy loss now."

        pred = predictions["similarity" + suffix]
        if self._in_batch_negative:
            label = _arange_int_label(pred)
        else:
            label = _zero_int_label(pred)
        losses[loss_name] = self._loss_modules[loss_name](pred, label)

        if self._sample_weight:
            losses[loss_name] = div_no_nan(
                torch.mean(losses[loss_name] * sample_weight), torch.mean(sample_weight)
            )

        return losses

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for loss_cfg in self._base_model_config.losses:
            losses.update(
                self._loss_impl(predictions, batch, self._label_name, loss_cfg)
            )
        losses.update(self._loss_collection)
        return losses

    def _init_metric_impl(self, metric_cfg: MetricConfig, suffix: str = "") -> None:
        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        metric_kwargs = config_to_kwargs(oneof_metric_cfg)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            self._metric_modules[metric_name] = recall_at_k.RecallAtK(**metric_kwargs)
        else:
            raise ValueError(f"{metric_type} is not supported for this model")

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for metric_cfg in self._base_model_config.metrics:
            self._init_metric_impl(metric_cfg)
        for loss_cfg in self._base_model_config.losses:
            self._init_loss_metric_impl(loss_cfg)

    def _update_metric_impl(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        label_name: str,
        metric_cfg: MetricConfig,
        suffix: str = "",
    ) -> None:
        label = batch.labels[label_name]

        metric_type = metric_cfg.WhichOneof("metric")
        metric_name = metric_type + suffix
        oneof_metric_cfg = getattr(metric_cfg, metric_type)
        if metric_type == "recall_at_k":
            metric_name = f"recall@{oneof_metric_cfg.top_k}" + suffix
            pred = predictions["similarity" + suffix]
            if self._in_batch_negative:
                label = torch.eye(*pred.size(), dtype=torch.bool, device=pred.device)
            else:
                label = torch.zeros_like(pred, dtype=torch.bool)
                label[:, 0] = True
            self._metric_modules[metric_name].update(pred, label)
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
            self._update_metric_impl(predictions, batch, self._label_name, metric_cfg)
        if losses is not None:
            for loss_cfg in self._base_model_config.losses:
                self._update_loss_metric_impl(losses, batch, self._label_name, loss_cfg)


class TowerWrapper(nn.Module):
    """Tower inference wrapper for jit.script."""

    def __init__(self, module: nn.Module, tower_name: str = "user_tower") -> None:
        super().__init__()
        setattr(self, tower_name, module)
        self._features = module._features
        self._tower_name = tower_name

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (dict): tower output embedding.
        """
        return {f"{self._tower_name}_emb": getattr(self, self._tower_name)(batch)}


class TowerWoEGWrapper(nn.Module):
    """Tower without embedding group inference wrapper for jit.script."""

    def __init__(self, module: nn.Module, tower_name: str = "user_tower") -> None:
        super().__init__()
        self.embedding_group = EmbeddingGroup(module._features, [module._feature_group])
        setattr(self, tower_name, module)
        self._features = module._features
        self._tower_name = tower_name
        self._group_name = module._group_name

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (dict): tower output embedding.
        """
        grouped_features = self.embedding_group(batch)
        return {
            f"{self._tower_name}_emb": getattr(self, self._tower_name)(
                grouped_features[self._group_name]
            )
        }
