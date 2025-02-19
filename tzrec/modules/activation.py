# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F

from tzrec.utils.load_class import load_by_path
from tzrec.utils.logging_util import logger


class Dice(nn.Module):
    """Data Adaptive Activation Function in DIN.

    Args:
        hidden_size (int): hidden dim of input.
        dim: input dims.
    """

    def __init__(self, hidden_size: int, dim: int = 2) -> None:
        super().__init__()
        assert dim in [2, 3]
        self.bn = nn.BatchNorm1d(hidden_size)
        self.alpha = nn.Parameter(torch.empty((hidden_size,)))
        self.dim = dim

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        self.bn.reset_parameters()
        nn.init.zeros_(self.alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        if self.dim == 2:
            x_p = F.sigmoid(self.bn(x))
            out = self.alpha * (1 - x_p) * x + x_p * x
        else:
            x = x.transpose(1, 2)
            x_p = F.sigmoid(self.bn(x))
            out = self.alpha.unsqueeze(1) * (1 - x_p) * x + x_p * x
            out = out.transpose(1, 2)
        return out


def create_activation(act_str: str = "nn.ReLU", **kwargs: Any) -> Optional[nn.Module]:
    """Create activation module."""
    act_str = act_str.strip()

    act_module = None
    if act_str == "Dice":
        assert "hidden_size" in kwargs and "dim" in kwargs, (
            "Dice activation method need hidden_size and dim params."
        )
        hidden_size = kwargs["hidden_size"]
        dim = kwargs["dim"]
        act_module = Dice(hidden_size, dim)
    elif len(act_str) > 0:
        act_strs = act_str.strip(")").split("(", 1)

        act_class = load_by_path(act_strs[0])
        if act_class is None:
            logger.error(f"Unknown activation [{act_str}]")
        else:
            act_params = {}
            if len(act_strs) > 1:
                try:
                    act_params = {
                        kv.split("=")[0]: eval(kv.split("=")[1])
                        for kv in act_strs[1].split(",")
                    }
                except Exception as e:
                    logger.error(f"Can not parse activation [{act_str}]")
                    raise e
            act_module = act_class(**act_params)

    return act_module
