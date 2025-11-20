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
from torchrec import KeyedTensor
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
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        int_item = x.item()
        torch._check_is_size(int_item)
    else:
        int_item = int(x.item())
    # pyre-ignore[7]
    return int_item


@torch.fx.wrap
def fx_numel(x: torch.Tensor) -> int:
    """Fx trace wrapper for x.numel()."""
    total_len = x.numel()
    if not torch.jit.is_scripting() and torch.compiler.is_compiling():
        torch._check_is_size(total_len)
    return total_len


@torch.fx.wrap
def fx_mark_keyed_tensor(name: str, x: KeyedTensor) -> None:
    """Mark a KeyedTensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    KeyedTensor.values() will be sparse part output and dense part input.
    """
    return


@torch.fx.wrap
def fx_mark_tensor(
    name: str, x: torch.Tensor, keys: Optional[List[str]] = None
) -> None:
    """Mark a Tensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    Tensor will be sparse part output and dense part input.
    """
    return


@torch.fx.wrap
def fx_mark_seq_tensor(
    name: str,
    x: torch.Tensor,
    keys: Optional[List[str]] = None,
    max_seq_len: Optional[int] = None,
    is_jagged_seq: bool = False,
) -> None:
    """Mark a Sequence Tensor in fx.graph.

    Used in EmbeddingGroup for split sparse part model when export.
    Tensor will be sparse part output and dense part input.
    """
    return


@torch.fx.wrap
def fx_mark_seq_len(name: str, x: torch.Tensor) -> None:
    """Mark a sequence length Tensor in fx.graph."""
    return
