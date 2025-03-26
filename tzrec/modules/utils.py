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

import abc
from typing import Any, Optional

import torch
from torch import nn

from tzrec.ops import Kernel


class BaseModule(nn.Module, abc.ABC):
    def __init__(
        self,
        is_inference: bool = False,
        kernel: Optional[Kernel] = None,
    ) -> None:
        super().__init__()
        self._is_inference = is_inference
        self._kernel = kernel

    def kernel(self) -> Kernel:
        kernel = self._kernel
        if kernel is not None:
            return kernel
        else:
            return Kernel.TRITON

    # pyre-ignore [2]
    def recursive_setattr(self, name: str, value: Any) -> None:
        for _, module in self.named_modules():
            if hasattr(module, name):
                setattr(module, name, value)

    def set_is_inference(self, is_inference: bool) -> None:
        self._is_inference = is_inference
        self.recursive_setattr("_is_inference", is_inference)

    def set_kernel(self, kernel: Kernel) -> None:
        self._kernel = kernel
        self.recursive_setattr("_kernel", kernel)

    @property
    def is_inference(self) -> bool:
        return self._is_inference

    @property
    def is_eval(self) -> bool:
        return (not self._is_inference) and (not self.training)

    @property
    def is_train(self) -> bool:
        return (not self._is_inference) and self.training


class Transpose(nn.Module):
    """Transpose Module.

    Args:
        dim0 (int): the first dimension to be transposed.
        dim1 (int): the second dimension to be transposed
    """

    def __init__(self, dim0: int, dim1: int) -> None:
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        return x.transpose(self.dim0, self.dim1)


def div_no_nan(
    input: torch.Tensor,
    other: torch.Tensor,
    *,
    rounding_mode: Optional[str] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Divides input by other and avoid division by zero.

    Args:
        input (Tensor): the dividend
        other (Tensor): the divisor
        rounding_mode (str, optional):
            Type of rounding applied to the result:
            - None: default behavior. Performs no rounding and, if both input and other
                are integer types, promotes the inputs to the default scalar type.
                Equivalent to true division in Python (the / operator) and NumPy’s
                np.true_divide.
            - "trunc": rounds the results of the division towards zero. Equivalent to
                C-style integer division.
            - "floor": rounds the results of the division down. Equivalent to floor
                division in Python (the // operator) and NumPy’s np.floor_divide.
        out (Tensor, optional): the output tensor.

    Return:
        out (Tensor): the output tensor.
    """
    return torch.nan_to_num(
        torch.div(input, other, rounding_mode=rounding_mode, out=out),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


@torch.fx.wrap
def fx_unwrap_optional_tensor(optional: Optional[torch.Tensor]) -> torch.Tensor:
    """Unwrap optional tensor for trace."""
    assert optional is not None, "Expected optional to be non-None Tensor"
    return optional


def init_linear_xavier_weights_zero_bias(m: torch.nn.Module) -> None:
    """Init nn.Linear module with Xavier weights and zero bias."""
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
