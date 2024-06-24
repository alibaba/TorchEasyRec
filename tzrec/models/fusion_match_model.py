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

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List, Optional

import torch

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel
from tzrec.protos.model_pb2 import ModelConfig


class FusionMatchModel(MatchModel):
    """DSSM model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)

    def init_loss(self) -> None:
        """Initialize loss modules."""
        assert (
            len(self._base_model_config.losses) == 1
        ), "match model only support single loss now."
        for tower_cfg in self._model_config.user_tower:
            for loss_cfg in self._base_model_config.losses:
                self._init_loss_impl(loss_cfg, suffix=f"_{tower_cfg.input}")

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for tower_cfg in self._model_config.user_tower:
            for loss_cfg in self._base_model_config.losses:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        self._label_name,
                        loss_cfg,
                        suffix=f"_{tower_cfg.input}",
                    )
                )
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for tower_cfg in self._model_config.user_tower:
            for metric_cfg in self._base_model_config.metrics:
                self._init_metric_impl(metric_cfg, suffix=f"_{tower_cfg.input}")
            for loss_cfg in self._base_model_config.losses:
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{tower_cfg.input}")

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
        for tower_cfg in self._model_config.user_tower:
            for metric_cfg in self._base_model_config.metrics:
                self._update_metric_impl(
                    predictions,
                    batch,
                    self._label_name,
                    metric_cfg,
                    suffix=f"_{tower_cfg.input}",
                )
            if losses is not None:
                for loss_cfg in self._base_model_config.losses:
                    loss_type = loss_cfg.WhichOneof("loss")
                    assert loss_type == "softmax_cross_entropy"
                    suffix = f"_{tower_cfg.input}"
                    loss_name = loss_type + suffix
                    loss = losses[loss_name]
                    pred = predictions["similarity" + suffix]
                    self._metric_modules[loss_name].update(
                        loss, loss.new_tensor(pred.size(0))
                    )
