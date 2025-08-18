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

from typing import Any, Callable, Dict, List, Optional, Union

import torch
from torchrec.fx import symbolic_trace as _symbolic_trace


def symbolic_trace(
    # pyre-ignore[24]
    root: Union[torch.nn.Module, Callable],
    concrete_args: Optional[Dict[str, Any]] = None,
    leaf_modules: Optional[List[str]] = None,
) -> torch.fx.GraphModule:
    """Symbolic tracing API.

    Given an `nn.Module` or function instance `root`, this function will return a
    `GraphModule` constructed by recording operations seen while tracing through `root`.

    `concrete_args` allows you to partially specialize your function, whether it's to
    remove control flow or data structures.

    Args:
        root (Union[torch.nn.Module, Callable]): Module or function to be traced and
            converted into a Graph representation.
        concrete_args (Optional[Dict[str, any]]): Inputs to be partially specialized
        leaf_modules (Optional[List[str]]): modules do not trace

    Returns:
        GraphModule: a Module created from the recorded operations from ``root``.
    """
    # ComputeJTDictToKJT could not be traced
    _leaf_modules = ["ComputeJTDictToKJT"]
    if leaf_modules:
        _leaf_modules.extend(leaf_modules)
    return _symbolic_trace(root, concrete_args, _leaf_modules)


@torch.fx.wrap
def fx_arange(len: int, device: torch.device) -> torch.Tensor:
    """Fx trace wrapper for arange."""
    return torch.arange(len, device=device)


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    """Unwrap optional tensor for trace."""
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


@torch.fx.wrap
def fx_int_item(x: torch.Tensor) -> int:
    """Fx trace wrapper for `int(x.item())`."""
    return int(x.item())


@torch.fx.wrap
def fx_numel(x: torch.Tensor) -> int:
    """Fx trace wrapper for x.numel()."""
    total_len = x.numel()
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check_is_size(total_len)
    return total_len
