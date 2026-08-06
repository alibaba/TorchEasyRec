# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reconciles a prompt slot's group width with the LM hidden size."""

from typing import Optional

import torch
from torch import nn

from tzrec.modules.mlp import MLP
from tzrec.protos.prompt_pb2 import PromptProjection as PromptProjectionConfig
from tzrec.utils.config_util import config_to_kwargs


class PromptProjection(nn.Module):
    """An optional body followed by a bare Linear to the LM hidden size.

    The final map never carries an activation: every ``Perceptron`` applies one,
    so ending on an MLP would zero half the dimensions feeding the LM input
    space. An empty MLP does not degrade to a linear either -- ``output_dim()``
    raises -- which is why the Linear is structural rather than configurable.

    Args:
        config: the slot's projection config; an empty one is a plain linear.
        in_dim: the slot's ``group_total_dim``, resolved by the model.
        hidden_size: the LM hidden size.
    """

    def __init__(
        self,
        config: PromptProjectionConfig,
        in_dim: int,
        hidden_size: int,
    ) -> None:
        super().__init__()
        dim = in_dim
        self.body: Optional[MLP] = None
        if config.HasField("mlp"):
            self.body = MLP(dim, **config_to_kwargs(config.mlp))
            dim = self.body.output_dim()
        self.head = nn.Linear(dim, hidden_size, bias=config.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Project a slot's group output into the LM input space.

        Args:
            features: ``(..., group_total_dim)``.

        Returns:
            ``(..., hidden_size)``.
        """
        if self.body is not None:
            features = self.body(features)
        return self.head(features)
