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

# We use the layer_norm ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.


import torch
from torch.fx._symbolic_trace import is_fx_tracing
from torch.utils._triton import has_triton

from tzrec.ops import Kernel
from tzrec.ops.pytorch.pt_layer_norm import (
    pytorch_layer_norm,
    pytorch_rms_norm,
    pytorch_swish_layer_norm,
)

if has_triton():
    from tzrec.ops.triton.triton_layer_norm import (
        triton_layer_norm,
        triton_rms_norm,
        triton_swish_layer_norm,
    )
else:
    triton_layer_norm = pytorch_layer_norm
    triton_rms_norm = pytorch_rms_norm
    triton_swish_layer_norm = pytorch_swish_layer_norm


def layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if kernel == Kernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must not be cpu tensor")
            torch._assert(not weight.is_cpu, "weight must not be cpu tensor")
            torch._assert(not bias.is_cpu, "bias must not be cpu tensor")
        return triton_layer_norm(x, weight, bias, eps)
    else:
        return pytorch_layer_norm(
            x,
            [
                x.shape[-1],
            ],
            weight,
            bias,
            eps,
        )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-5,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if kernel == Kernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must not be cpu tensor")
            torch._assert(not weight.is_cpu, "weight must not be cpu tensor")
        return triton_rms_norm(x, weight, eps)
    else:
        return pytorch_rms_norm(x, weight, eps)


def swish_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float = 1e-5,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if kernel == Kernel.TRITON:
        if not is_fx_tracing():
            torch._assert(not x.is_cpu, "x must not be cpu tensor")
            torch._assert(not weight.is_cpu, "weight must not be cpu tensor")
            torch._assert(not bias.is_cpu, "bias must not be cpu tensor")
        return triton_swish_layer_norm(x, [x.shape[-1]], weight, bias, eps)
    else:
        return pytorch_swish_layer_norm(
            x,
            [
                x.shape[-1],
            ],
            weight,
            bias,
            eps,
        )
