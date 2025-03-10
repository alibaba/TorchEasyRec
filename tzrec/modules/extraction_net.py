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

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from tzrec.modules.mlp import MLP


class ExtractionNet(nn.Module):
    """Multiple multi-gate Mixture-of-Experts module.

    Args:
        in_extraction_networks (list): every task expert input dims.
        in_shared_expert (int): shared expert input dims.
        network_name: ExtractionNet name, not important.
        share_num: number of experts for share.
        expert_num_per_task: number of experts per task.
        share_expert_net: mlp network config of experts share.
        task_expert_net: mlp network config of experts per task.
        final_flag: whether to is last extractionNet or not.
    """

    def __init__(
        self,
        in_extraction_networks: List[int],
        in_shared_expert: int,
        network_name: str,
        share_num: int,
        expert_num_per_task: int,
        share_expert_net: Dict[str, Any],
        task_expert_net: Dict[str, Any],
        final_flag: bool = False,
    ) -> None:
        super().__init__()
        self.name = network_name
        self._final_flag = final_flag
        self._shared_layers = nn.ModuleList()
        self._shared_gate = None
        self._output_dims = []

        share_net_num = share_num
        per_task_num = expert_num_per_task
        share_output_dim = share_expert_net["hidden_units"][-1]
        for _ in range(share_net_num):
            self._shared_layers.append(
                MLP(
                    in_shared_expert,
                    **share_expert_net,
                )
            )
        share_gate_output = len(in_extraction_networks) * per_task_num + share_net_num
        self._shared_gate = None
        if not self._final_flag:
            self._shared_gate = nn.Linear(in_shared_expert, share_gate_output)

        self._task_layers = nn.ModuleList()
        self._task_gates = nn.ModuleList()
        task_gate_output = per_task_num + share_net_num
        task_output_dim = task_expert_net["hidden_units"][-1]
        for in_feature in in_extraction_networks:
            task_model_list = nn.ModuleList()
            for _ in range(per_task_num):
                task_model_list.append(
                    MLP(
                        in_feature,
                        **task_expert_net,
                    )
                )
            self._task_layers.append(task_model_list)
            self._task_gates.append(nn.Linear(in_feature, task_gate_output))
            self._output_dims.append(task_output_dim)
        self._output_dims.append(share_output_dim)

    def output_dim(self) -> List[int]:
        """Output Task expert and shared expert dimension of the module."""
        return self._output_dims

    def _experts_layer_forward(
        self, deep_fea: torch.Tensor, layers: nn.ModuleList
    ) -> List[torch.Tensor]:
        tower_outputs = []
        for layer in layers:
            output = layer(deep_fea)
            tower_outputs.append(output)
        return tower_outputs

    def _gate_forward(
        self,
        selector_fea: torch.Tensor,
        vec_feas: List[torch.Tensor],
        gate_layer: nn.Module,
    ) -> torch.Tensor:
        vec = torch.stack(vec_feas, dim=1)
        gate = gate_layer(selector_fea)
        gate = torch.softmax(gate, dim=1)
        gate = torch.unsqueeze(gate, dim=1)
        output = torch.matmul(gate, vec).squeeze(1)
        return output

    def forward(
        self,
        extraction_network_fea: List[torch.Tensor],
        shared_expert_fea: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        """Forward the module."""
        shared_expert = self._experts_layer_forward(
            shared_expert_fea, self._shared_layers
        )
        all_task_experts = []
        cgc_layer_outs = []
        for i, task_layers in enumerate(self._task_layers):
            task_experts = self._experts_layer_forward(
                extraction_network_fea[i], task_layers
            )
            cgc_task_out = self._gate_forward(
                extraction_network_fea[i],
                task_experts + shared_expert,
                self._task_gates[i],
            )
            all_task_experts.extend(task_experts)
            cgc_layer_outs.append(cgc_task_out)

        shared_layer_out = None
        if self._shared_gate:
            shared_layer_out = self._gate_forward(
                shared_expert_fea, all_task_experts + shared_expert, self._shared_gate
            )
        return cgc_layer_outs, shared_layer_out
