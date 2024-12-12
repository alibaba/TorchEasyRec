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

from enum import Enum
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.fx import GraphModule

from tzrec.models.model import ScriptWrapper
from tzrec.utils.fx_util import symbolic_trace


class TestGraphType(Enum):
    """Graph Type for Tests."""

    NORMAL = 1
    FX_TRACE = 2
    JIT_SCRIPT = 3


def create_test_module(
    module: nn.Module, graph_type: TestGraphType
) -> Union[nn.Module, GraphModule, torch.jit.ScriptModule]:
    """Create module with graph type for tests."""
    if graph_type == TestGraphType.FX_TRACE:
        module = symbolic_trace(module)
    elif graph_type == TestGraphType.JIT_SCRIPT:
        module = symbolic_trace(module)
        module = torch.jit.script(module)
    return module


def create_test_model(
    model: nn.Module, graph_type: TestGraphType
) -> Union[nn.Module, GraphModule, torch.jit.ScriptModule]:
    """Create model with graph type for tests."""
    if graph_type == TestGraphType.JIT_SCRIPT:
        model = ScriptWrapper(model)
    return create_test_module(model, graph_type)


def init_parameters(module: nn.Module, device: torch.device) -> None:
    """Init param for model with meta device type."""

    @torch.no_grad()
    def init_parameters(module: nn.Module) -> None:
        # Allocate parameters and buffers if over 'meta' device.
        has_meta_param = False
        for name, param in module._parameters.items():
            if isinstance(param, torch.Tensor) and param.device.type == "meta":
                module._parameters[name] = nn.Parameter(
                    torch.empty_like(param, device=device),
                    requires_grad=param.requires_grad,
                )
                has_meta_param = True
        for name, buffer in module._buffers.items():
            if isinstance(buffer, torch.Tensor) and buffer.device.type == "meta":
                module._buffers[name] = torch.zeros_like(buffer, device=device)

        # Init parameters if at least one parameter is over 'meta' device.
        if has_meta_param and hasattr(module, "reset_parameters"):
            module.reset_parameters()

    module.apply(init_parameters)


# pyre-ignore [2]
def parameterized_name_func(func, num, p) -> str:
    """Name func for parameterized."""
    base_name = func.__name__
    name_suffix = "_%s" % (num,)
    return base_name + name_suffix


def dicts_are_equal(
    dict1: Dict[str, torch.Tensor], dict2: Dict[str, torch.Tensor]
) -> bool:
    """Compare dict[str,torch.Tensor]."""
    if dict1.keys() != dict2.keys():
        return False

    for key in dict1:
        if not torch.equal(dict1[key], dict2[key]):
            return False

    return True


def lists_are_equal(list1: List[torch.Tensor], list2: List[torch.Tensor]) -> bool:
    """Compare List[torch.Tensor]."""
    if len(list1) != len(list2):
        return False

    for i in range(len(list1)):
        if not torch.equal(list1[i], list2[i]):
            return False
    return True


def dfs_are_close(df1: pd.DataFrame, df2: pd.DataFrame, abs_tol: float) -> bool:
    """Compare DataFrame."""
    if df1.shape != df2.shape:
        return False
    abs_diff = np.abs(df1.values - df2.values)
    result = np.all(abs_diff <= abs_tol)
    # pyre-ignore [7]
    return result
