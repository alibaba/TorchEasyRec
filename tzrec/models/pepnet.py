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
from tzrec.modules.activation import create_activation
from tzrec.modules.task_tower import TaskTower
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class GateNU(nn.Module):
    """Gate Neural Unit for PEPNet.

    Implements the Gate Neural Unit from the PEPNet paper with ReLU
    -> Sigmoid activation and optional gamma scaling factor.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, gamma: float = 2.0
    ) -> None:
        super().__init__()
        self._gamma = gamma
        self._output_dim = output_dim
        self.dense_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def output_dim(self) -> int:
        """Get output dimension of the GateNU."""
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for GateNU.

        Args:
            x: Input tensor [B, input_dim]

        Returns:
            Output tensor [B, output_dim] with values scaled by gamma
        """
        return self._gamma * self.dense_layers(x)


class EPNet(nn.Module):
    """Embedding Personalization Network for PEPNet.

    Generates personalized weights to scale the original embeddings
    based on domain and context information, following the original PEPNet paper.
    """

    def __init__(
        self, main_dim: int, domain_dim: int, hidden_dim: int, gamma: float = 2.0
    ) -> None:
        super().__init__()
        self._domain_dim = domain_dim
        self._main_dim = main_dim
        input_dim = domain_dim + main_dim
        self.gate_nu = GateNU(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=main_dim, gamma=gamma
        )

    def output_dim(self) -> int:
        """Get output dimension of the EPNet."""
        return self.gate_nu.output_dim()

    def forward(
        self,
        main_emb: torch.Tensor,
        domain_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for embedding personalization.

        Args:
            main_emb: Main feature embedding tensor [B, embedding_dim]
            domain_emb: Domain embedding tensor [B, domain_dim]

        Returns:
            Personalized embedding tensor [B, embedding_dim]
        """
        gate_input = torch.cat([domain_emb, main_emb.detach()], dim=-1)
        scaling_factors = self.gate_nu(gate_input)
        personalized_emb = scaling_factors * main_emb
        return personalized_emb


class PPNet(nn.Module):
    """Parameter Personalization Network for PEPNet."""

    def __init__(
        self,
        main_feature: int,
        uia_feature: int,
        num_task: int,
        hidden_units: List[int],
        activation: Optional[str] = "nn.ReLU",
        dropout_ratio: Optional[Union[List[float], float]] = None,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.main_feature = main_feature
        self.uia_feature = uia_feature
        self.num_task = num_task
        self.hidden_units = hidden_units
        self.len_hidden = len(hidden_units)
        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropout_ratios = nn.ModuleList()
        self.gate_nus = nn.ModuleList()

        if dropout_ratio is None:
            dropout_ratio = [0.0] * len(hidden_units)
        elif isinstance(dropout_ratio, list):
            if len(dropout_ratio) == 0:
                dropout_ratio = [0.0] * len(hidden_units)
            elif len(dropout_ratio) == 1:
                dropout_ratio = dropout_ratio * len(hidden_units)
            else:
                assert len(dropout_ratio) == len(hidden_units), (
                    "length of dropout_ratio and hidden_units must be same, "
                    f"but got {len(dropout_ratio)} vs {len(hidden_units)}"
                )
        else:
            dropout_ratio = [dropout_ratio] * len(hidden_units)

        for _ in range(self.num_task):
            output = main_feature
            for i, hidden_unit in enumerate(hidden_units):
                self.linears.append(nn.Linear(output, hidden_unit))
                active = create_activation(activation, hidden_size=hidden_unit, dim=2)
                self.activations.append(active)
                self.dropout_ratios.append(nn.Dropout(dropout_ratio[i]))
                self.gate_nus.append(
                    GateNU(
                        input_dim=self.main_feature + self.uia_feature,
                        hidden_dim=hidden_unit,
                        output_dim=hidden_unit,
                        gamma=gamma,
                    )
                )
                output = hidden_unit

    def output_dim(self) -> List[int]:
        """Get output dimension of the PPNet."""
        return [self.hidden_units[-1]] * self.num_task

    def task_output_dim(self) -> int:
        """Get output dimension of the PPNet."""
        return self.hidden_units[-1]

    def forward(
        self, main_emb: torch.Tensor, uia_emb: torch.Tensor
    ) -> List[torch.Tensor]:
        """Forward pass for parameter personalization.

        Args:
            main_emb: Input tensor [B, embedding_dim]
            uia_emb: UI personalization embedding tensor [B, uia_dim]

        Returns:
            List of task-specific outputs [[B, output_dim]] * num_task
        """
        task_outputs = []
        for i in range(self.num_task):
            gate_input = torch.cat([uia_emb, main_emb.detach()], dim=-1)
            input = main_emb
            for j in range(self.len_hidden):
                input = self.linears[i * self.len_hidden + j](input)
                input = self.activations[i * self.len_hidden + j](input)
                input = input * self.gate_nus[i * self.len_hidden + j](gate_input)
                input = self.dropout_ratios[i * self.len_hidden + j](input)
            task_outputs.append(input)
        return task_outputs


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

        self.main_group_name = self._model_config.main_group_name
        self.main_group_dim = self.embedding_group.group_total_dim(self.main_group_name)
        self.task_input_dim = self.main_group_dim
        self.domain_group_name = None
        self.epnet = None
        if self._model_config.HasField("domain_group_name"):
            self.domain_group_name = self._model_config.domain_group_name
            domain_group_dim = self.embedding_group.group_total_dim(
                self.domain_group_name
            )
            self.epnet = EPNet(
                self.main_group_dim,
                domain_group_dim,
                hidden_dim=self._model_config.epnet_hidden_unit,
                gamma=self._model_config.epnet_gamma,
            )
            self.task_input_dim = self.epnet.output_dim()

        self.uia_group_name = None
        self.ppnet = None
        if self._model_config.HasField("uia_group_name"):
            self.uia_group_name = self._model_config.uia_group_name
            uia_group_dim = self.embedding_group.group_total_dim(self.uia_group_name)
            self.ppnet = PPNet(
                self.main_group_dim,
                uia_group_dim,
                num_task=len(self._task_tower_cfgs),
                hidden_units=list(self._model_config.ppnet_hidden_units),
                activation=self._model_config.ppnet_activation,
                dropout_ratio=list(self._model_config.ppnet_dropout_ratio),
                gamma=self._model_config.ppnet_gamma,
            )
            self.task_input_dim = self.ppnet.task_output_dim()

        self._task_tower = nn.ModuleList()
        for tower_cfg in self._task_tower_cfgs:
            tower_cfg = config_to_kwargs(tower_cfg)
            mlp = tower_cfg["mlp"] if "mlp" in tower_cfg else None
            self._task_tower.append(
                TaskTower(self.task_input_dim, tower_cfg["num_class"], mlp=mlp)
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
        main_features = grouped_features[self.main_group_name]
        # Apply EPNet if available for embedding personalization
        if self.domain_group_name:
            domain_features = grouped_features[self.domain_group_name]
            final_features = self.epnet(main_features, domain_features)
        else:
            final_features = main_features

        if self.uia_group_name:
            uia_features = grouped_features[self.uia_group_name]
            task_input_list = self.ppnet(final_features, uia_features)
        else:
            task_input_list = [final_features]

        # Apply task towers
        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            if self.uia_group_name:
                task_input = task_input_list[i]
            else:
                task_input = task_input_list[0]
            tower_output = self._task_tower[i](task_input)
            tower_outputs[tower_name] = tower_output

        return self._multi_task_output_to_prediction(tower_outputs)
