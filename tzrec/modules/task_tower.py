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


from typing import Any, Dict, Optional

import torch
from torch import nn

from tzrec.modules.mlp import MLP


class TaskTower(nn.Module):
    """General task tower Module.

    Args:
        tower_feature_in (in): task tower input dims.
        num_class: num_class for multi-class classification loss
        mlp:  mlp network config.
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
