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

import torch
from torch import nn


class RotateLayer(nn.Module):
    """Applies a orthogonal low-rank transformation.

    Args:
        base_dim (int): The dimension of the original space.
        low_rank_dim (int): The dimension of the low-rank space.
    """

    def __init__(self, base_dim, low_rank_dim):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(
            torch.empty(base_dim, low_rank_dim), requires_grad=True
        )
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, base):
        """Forward the module."""
        return torch.matmul(base.to(self.weight.dtype), self.weight)


class Intervention(nn.Module):
    """Deducing the influence of the source on the base representations.

    Args:
        base_dim (int): The dimension of the base space.
        source_dim (int): The dimension of the source space.
        low_rank_dim (int): The dimension of the low-rank space
            (Shared space for the base and source).
        drpout_ratio: dropout rate for the intervented output.
    """

    def __init__(
        self,
        base_dim: int,
        source_dim: int,
        low_rank_dim: int,
        dropout_ratio: float = 0.0,
    ):
        super().__init__()
        assert base_dim > low_rank_dim, "Low-rank dimension should lower than the base"
        self.base_dim = base_dim
        base_rotate_layer = RotateLayer(base_dim, low_rank_dim)
        self.base_rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            base_rotate_layer
        )
        source_rotate_layer = RotateLayer(source_dim, low_rank_dim)
        self.source_rotate_layer = torch.nn.utils.parametrizations.orthogonal(
            source_rotate_layer
        )
        self.dropout = torch.nn.Dropout(dropout_ratio)

    def forward(self, base: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
        """Forward the module."""
        rotated_base = self.base_rotate_layer(base)
        rotated_source = self.source_rotate_layer(source.detach())
        output = (
            torch.matmul(rotated_base - rotated_source, self.base_rotate_layer.weight.T)
            + base
        )
        return self.dropout(output.to(base.dtype))

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.base_dim
