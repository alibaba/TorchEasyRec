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

from tzrec.modules.mlp import MLP


class TaskTower(nn.Module):
    """General task tower Module.

    Args:
        tower_feature_in (int): task tower input dims.
        num_class (int): num_class for multi-class classification loss.
        mlp (dict):  mlp network config.
    """

    def __init__(
        self,
        tower_feature_in: int,
        num_class: int,
        mlp: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.num_class = num_class
        self.tower_mlp = None
        linear_in = tower_feature_in
        if mlp is not None:
            self.tower_mlp = MLP(tower_feature_in, **mlp)
            linear_in = self.tower_mlp.output_dim()
        self.linear = nn.Linear(linear_in, num_class)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        if self.tower_mlp:
            features = self.tower_mlp(features)
        task_tower_out = self.linear(features)

        return task_tower_out


class FusionMTLTower(nn.Module):
    """Fusion task tower Module.

    Args:
        tower_feature_in (int): task tower input dims.
        mlp:  mlp network config.
    """

    def __init__(
        self,
        tower_feature_in: int,
        mlp: Optional[Dict[str, Any]],
        task_configs: List[Dict[str, Any]],
    ) -> None:
        super().__init__()
        self.task_configs = task_configs
        self.tower_mlp = None
        linear_in = tower_feature_in
        if mlp is not None:
            self.tower_mlp = MLP(tower_feature_in, **mlp)
            linear_in = self.tower_mlp.output_dim()
        self.task_output_dims = []
        for task_config in task_configs:
            self.task_output_dims.append(task_config.get("num_class", 1))
        self.linear = nn.Linear(linear_in, sum(self.task_output_dims))

    def forward(
        self, user_emb: torch.Tensor, item_emb: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Forward the module."""
        features = user_emb * item_emb
        if self.tower_mlp:
            features = self.tower_mlp(features)
        tower_out = self.linear(features)
        tower_outputs = tower_out.split(self.task_output_dims, dim=-1)

        result_dict = {}
        for i, task_cfg in enumerate(self.task_configs):
            result_dict[task_cfg["task_name"]] = tower_outputs[i]
        return result_dict
