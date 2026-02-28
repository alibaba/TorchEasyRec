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
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.personalized_net import EPNet, PPNet
from tzrec.modules.task_tower import TaskTower
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class PEPNet(MultiTaskRank):
    """Parameter and Embedding Personalized Network."""

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()

        self._main_group_name = "all"
        self._domain_group_name = "domain"
        self._uia_group_name = "uia"
        if not self.embedding_group.has_group(self._main_group_name):
            raise Exception("all feature group not found.")
        self._main_group_dim = self.embedding_group.group_total_dim(
            self._main_group_name
        )
        self._task_input_dim = self._main_group_dim

        self.epnet = None
        if self.embedding_group.has_group(self._domain_group_name):
            domain_group_dim = self.embedding_group.group_total_dim(
                self._domain_group_name
            )
            self.epnet = EPNet(
                self._main_group_dim,
                domain_group_dim,
                hidden_dim=self._model_config.epnet_hidden_unit
                if self._model_config.HasField("epnet_hidden_unit")
                else self._main_group_dim,
                gamma=self._model_config.epnet_gamma,
            )
            self._task_input_dim = self.epnet.output_dim()

        self.ppnet = None
        if self.embedding_group.has_group(self._uia_group_name):
            uia_group_dim = self.embedding_group.group_total_dim(self._uia_group_name)
            self.ppnet = PPNet(
                self._main_group_dim,
                uia_group_dim,
                num_task=len(self._task_tower_cfgs),
                hidden_units=list(self._model_config.ppnet_hidden_units),
                activation=self._model_config.ppnet_activation,
                dropout_ratio=list(self._model_config.ppnet_dropout_ratio),
                gamma=self._model_config.ppnet_gamma,
            )
            self._task_input_dim = self.ppnet.task_output_dim()

        self._domain_input_name = None
        if self._model_config.HasField("domain_input_name"):
            self._domain_input_name = self._model_config.domain_input_name
        self._task_domain_num = self._model_config.task_domain_num

        self._task_tower = nn.ModuleList()
        for tower_cfg in self._task_tower_cfgs:
            tower_cfg = config_to_kwargs(tower_cfg)
            mlp = tower_cfg["mlp"] if "mlp" in tower_cfg else None
            if self._domain_input_name:
                for _ in range(self._task_domain_num):
                    self._task_tower.append(
                        TaskTower(self._task_input_dim, tower_cfg["num_class"], mlp=mlp)
                    )
            else:
                self._task_tower.append(
                    TaskTower(self._task_input_dim, tower_cfg["num_class"], mlp=mlp)
                )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        # Get main features
        main_features = grouped_features[self._main_group_name]
        # Apply EPNet if available for embedding personalization
        if self.epnet:
            domain_features = grouped_features[self._domain_group_name]
            final_features = self.epnet(main_features, domain_features)
        else:
            final_features = main_features

        if self.ppnet:
            uia_features = grouped_features[self._uia_group_name]
            task_input_list = self.ppnet(final_features, uia_features)
        else:
            task_input_list = [final_features]

        # Apply task towers
        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            if self.ppnet:
                task_input = task_input_list[i]
            else:
                task_input = task_input_list[0]
            if self._domain_input_name:
                task_all_domain_outputs = []
                for j in range(self._task_domain_num):
                    tower_output = self._task_tower[i * self._task_domain_num + j](
                        task_input
                    )
                    task_all_domain_outputs.append(tower_output)
                tower_outputs[tower_name] = task_all_domain_outputs
            else:
                tower_output = self._task_tower[i](task_input)
                tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)

    def _multi_task_output_to_prediction(
        self, output: Union[Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        predictions = {}
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            task_output = output[tower_name]
            for loss_cfg in task_tower_cfg.losses:
                if self._domain_input_name:
                    for i, domain_output in enumerate(task_output):
                        predictions.update(
                            self._output_to_prediction_impl(
                                domain_output,
                                loss_cfg,
                                num_class=task_tower_cfg.num_class,
                                suffix=f"_{tower_name}_{i}",
                            )
                        )
                else:
                    predictions.update(
                        self._output_to_prediction_impl(
                            task_output,
                            loss_cfg,
                            num_class=task_tower_cfg.num_class,
                            suffix=f"_{tower_name}",
                        )
                    )
        return predictions

    def _select_domain_task_output(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        new_predictions = {}
        if self._domain_input_name:
            for (
                tower_domain_loss_predict_name,
                tower_domain_loss_predict_value,
            ) in predictions.items():
                tower_loss_name = "_".join(
                    tower_domain_loss_predict_name.split("_")[:-1]
                )
                domain_index = int(tower_domain_loss_predict_name.split("_")[-1])
                if tower_loss_name in new_predictions:
                    new_predictions[tower_loss_name].append(
                        (domain_index, tower_domain_loss_predict_value)
                    )
                else:
                    new_predictions[tower_loss_name] = [
                        (domain_index, tower_domain_loss_predict_value)
                    ]
            domain_index = batch.labels[self._domain_input_name]
            for tower_loss_name, tower_loss_predict_values in new_predictions.items():
                tower_loss_domain_predict = torch.stack(
                    [
                        v[1]
                        for v in sorted(tower_loss_predict_values, key=lambda x: x[0])
                    ],
                    dim=1,
                )
                new_predictions[tower_loss_name] = torch.gather(
                    tower_loss_domain_predict, 1, index=domain_index.unsqueeze(1)
                ).squeeze(1)
        return new_predictions

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model."""
        if self._domain_input_name:
            predictions = self._select_domain_task_output(predictions, batch)
        losses = super().loss(predictions, batch)
        return losses

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
        if self._domain_input_name:
            predictions = self._select_domain_task_output(predictions, batch)
        super().update_metric(predictions, batch, losses)

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
        if self._domain_input_name:
            predictions = self._select_domain_task_output(predictions, batch)
        super().update_train_metric(predictions, batch)
