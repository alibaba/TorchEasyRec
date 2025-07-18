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


from typing import List

import torch

from tzrec.modules.utils import BaseModule
from tzrec.ops import Kernel
from tzrec.ops.layer_norm import (
    layer_norm,
    swish_layer_norm,
)
from tzrec.ops.triton.triton_layer_norm import triton_rms_norm


class LayerNorm(BaseModule):
    """LayerNorm module.

    Args:
        dim (int): input dim.
        eps (float): epsilon.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self._eps = eps
        self._weight = torch.nn.Parameter(
            torch.ones(self._normalized_shape),
        )
        self._bias = torch.nn.Parameter(
            torch.zeros(self._normalized_shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        return layer_norm(
            x=x,
            weight=self._weight,
            bias=self._bias,
            eps=self._eps,
            kernel=self.kernel(),
        )


class RMSNorm(BaseModule):
    """RMSNorm module.

    Args:
        dim (int): input dim.
        eps (float): epsilon.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._eps = eps
        self._weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self._eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        if self.kernel() == Kernel.TRITON:
            return triton_rms_norm(x, self._weight, self._eps)
        else:
            output = self._norm(x.float()).type_as(x)
            return output * self._weight


class SwishLayerNorm(BaseModule):
    """SwishLayerNorm module.

    Args:
        dim (int): input dim.
        eps (float): epsilon.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._normalized_shape: List[int] = [dim]
        self._weight = torch.nn.Parameter(torch.ones(self._normalized_shape))
        self._bias = torch.nn.Parameter(torch.zeros(self._normalized_shape))
        self._eps = eps

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Forward the module."""
        return swish_layer_norm(
            x=x,
            weight=self._weight,
            bias=self._bias,
            eps=self._eps,
            kernel=self.kernel(),
        )
