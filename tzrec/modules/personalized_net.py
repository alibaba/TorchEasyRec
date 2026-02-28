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

from typing import List, Optional, Union

import torch
from torch import nn

from tzrec.modules.activation import create_activation


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
