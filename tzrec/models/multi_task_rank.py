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

from typing import Any, Dict, List, Optional, Union

import torch

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.utils import div_no_nan
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.tower_pb2 import BayesTaskTower, InterventionTaskTower, TaskTower


class MultiTaskRank(RankModel):
    """Multi task model for ranking.

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
        self._task_tower_cfgs = list(self._model_config.task_towers)

    def _multi_task_output_to_prediction(
        self, output: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        predictions = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            for loss_cfg in task_tower_cfg.losses:
                predictions.update(
                    self._output_to_prediction_impl(
                        output[tower_name],
                        loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}",
                    )
                )
        return predictions

    def has_weight(
        self, task_cfg: Union[TaskTower, BayesTaskTower, InterventionTaskTower]
    ) -> bool:
        """Task cfg should have weight."""
        if (
            task_cfg.HasField("sample_weight_name")
            or task_cfg.HasField("weight")
            or task_cfg.HasField("task_space_indicator_label")
        ):
            return True
        else:
            return False

    def init_loss(self) -> None:
        """Initialize loss modules."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            reduction = "none" if self.has_weight(task_tower_cfg) else "mean"
            for loss_cfg in task_tower_cfg.losses:
                self._init_loss_impl(
                    loss_cfg,
                    num_class=task_tower_cfg.num_class,
                    reduction=reduction,
                    suffix=f"_{tower_name}",
                )

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        losses = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            if self.has_weight(task_tower_cfg):
                if task_tower_cfg.sample_weight_name:
                    sample_weight = task_tower_cfg.sample_weight_name
                    loss_weight = batch.sample_weights[sample_weight]
                else:
                    loss_weight = torch.Tensor([1.0]).to(
                        batch.labels[label_name].device
                    )

                if task_tower_cfg.HasField("task_space_indicator_label"):
                    in_task_space = (
                        batch.labels[task_tower_cfg.task_space_indicator_label] > 0
                    ).float()
                    loss_weight = loss_weight * (
                        task_tower_cfg.in_task_space_weight * in_task_space
                        + task_tower_cfg.out_task_space_weight * (1 - in_task_space)
                    )

                loss_weight = div_no_nan(loss_weight, torch.mean(loss_weight))

                loss_weight *= task_tower_cfg.weight
            else:
                loss_weight = None

            for loss_cfg in task_tower_cfg.losses:
                losses.update(
                    self._loss_impl(
                        predictions,
                        batch,
                        batch.labels[label_name],
                        loss_weight,
                        loss_cfg,
                        num_class=task_tower_cfg.num_class,
                        suffix=f"_{tower_name}",
                    )
                )
        losses.update(self._loss_collection)
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules."""
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            for metric_cfg in task_tower_cfg.metrics:
                self._init_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}",
                )
            for metric_cfg in task_tower_cfg.train_metrics:
                self._init_train_metric_impl(
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}",
                )
            for loss_cfg in task_tower_cfg.losses:
                self._init_loss_metric_impl(loss_cfg, suffix=f"_{tower_name}")

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
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            for metric_cfg in task_tower_cfg.metrics:
                self._update_metric_impl(
                    predictions,
                    batch,
                    batch.labels[label_name],
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}",
                )
            if losses is not None:
                for loss_cfg in task_tower_cfg.losses:
                    self._update_loss_metric_impl(
                        losses,
                        batch,
                        batch.labels[label_name],
                        loss_cfg,
                        suffix=f"_{tower_name}",
                    )

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            label_name = task_tower_cfg.label_name
            for metric_cfg in task_tower_cfg.train_metrics:
                self._update_train_metric_impl(
                    predictions,
                    batch,
                    batch.labels[label_name],
                    metric_cfg,
                    num_class=task_tower_cfg.num_class,
                    suffix=f"_{tower_name}",
                )
