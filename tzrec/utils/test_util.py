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

import os
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from hypothesis import settings as _settings
from hypothesis.utils.conventions import not_set as _not_set
from torch import nn
from torch.fx import GraphModule

from tzrec.models.model import ScriptWrapper
from tzrec.utils.fx_util import symbolic_trace

nv_gpu_unavailable: Tuple[bool, str] = (
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    "CUDA is not available or no GPUs detected",
)
amd_gpu_unavailable: Tuple[bool, str] = (
    not torch.version.hip,
    "AMD HIP not available or no GPUs detected",
)
gpu_unavailable: Tuple[bool, str] = (
    nv_gpu_unavailable[0] and amd_gpu_unavailable[0],
    "CUDA/HIP is not available or no GPUs detected",
)

_settings.register_profile(
    "default", _settings(_settings.get_profile("default"), print_blob=True)
)


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


# pyre-ignore [2]
def parameterized_name_func(func, num, p) -> str:
    """Name func for parameterized."""
    base_name = func.__name__
    name_suffix = "_%s" % (num,)
    return base_name + name_suffix


class hypothesis_settings(_settings):
    """Hypothesis settings for TorchEasyRec."""

    def __init__(
        self,
        parent: Optional[_settings] = None,
        *,
        # pyre-ignore[9]
        max_examples: int = _not_set,
        # pyre-ignore[9]
        derandomize: bool = _not_set,
        **kwargs: Any,
    ) -> None:
        if os.environ.get("CI_HYPOTHESIS", "false").lower() == "true":
            if max_examples != _not_set:
                max_examples = max(1, max_examples // 5)
            if derandomize == _not_set:
                derandomize = True
        else:
            if derandomize == _not_set:
                derandomize = False
        super().__init__(
            parent, max_examples=max_examples, derandomize=derandomize, **kwargs
        )


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


def generate_sparse_seq_len(
    size: int,
    max_seq_len: int,
    sparsity: float,
    device: torch.device,
) -> torch.Tensor:
    """Generate sequence lengths with sparsity."""
    if sparsity == 0.0:
        return torch.zeros(size=(size,), device=device, dtype=torch.int)
    elif sparsity == 1.0:
        return torch.ones(size=(size,), device=device, dtype=torch.int) * max_seq_len
    elif sparsity >= 0.5:
        min_seq_len: int = int((2 * sparsity - 1.0) * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )
    else:
        min_seq_len: int = 0
        max_seq_len: int = int(2 * sparsity * max_seq_len)
        return torch.randint(
            low=min_seq_len,
            high=max_seq_len,
            size=(size,),
            device=device,
            dtype=torch.int,
        )


def get_test_dtypes(dtypes: List[torch.dtype]) -> List[torch.dtype]:
    """Get valid test dtypes."""
    results = []
    for dtype in dtypes:
        if dtype == torch.bfloat16:
            if torch.cuda.is_available():
                if torch.cuda.get_device_capability(torch.device("cuda"))[0] >= 8:
                    results.append(dtype)
        else:
            results.append(dtype)
    return results
