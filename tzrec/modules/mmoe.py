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
from torch import nn
from torch.nn import functional as F

from tzrec.modules.mlp import MLP


class MMoE(nn.Module):
    """Multi-gate Mixture-of-Experts module.

    Args:
        in_features (int): in_size of the input.
        attn_mlp (dict): target attention MLP module parameters.
    """

    def __init__(
        self,
        in_features: int,
        expert_mlp: Dict[str, Any],
        num_expert: int,
        num_task: int,
        gate_mlp: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_expert = num_expert
        self.num_task = num_task

        self.expert_mlps = nn.ModuleList(
            [MLP(in_features=in_features, **expert_mlp) for _ in range(num_expert)]
        )
        gate_final_in = in_features
        self.has_gate_mlp = False
        if gate_mlp is not None:
            self.has_gate_mlp = True
            self.gate_mlps = nn.ModuleList(
                [MLP(in_features=in_features, **gate_mlp) for _ in range(num_task)]
            )
            gate_final_in = self.gate_mlps[0].hidden_units[-1]
        self.gate_finals = nn.ModuleList(
            [nn.Linear(gate_final_in, num_expert) for _ in range(num_task)]
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.expert_mlps[0].hidden_units[-1]

    def forward(self, input: torch.Tensor) -> List[torch.Tensor]:
        """Forward the module."""
        expert_fea_list = []
        for i in range(self.num_expert):
            expert_fea_list.append(self.expert_mlps[i](input))
        expert_feas = torch.stack(expert_fea_list, dim=1)

        result = []
        for i in range(self.num_task):
            if self.has_gate_mlp:
                gate = self.gate_mlps[i](input)
            else:
                gate = input
            gate = self.gate_finals[i](gate)
            gate = F.softmax(gate, dim=1).unsqueeze(1)
            task_input = torch.matmul(gate, expert_feas).squeeze(1)
            result.append(task_input)
        return result
