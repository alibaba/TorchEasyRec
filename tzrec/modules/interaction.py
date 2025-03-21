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


from typing import List

import torch
from torch import nn


@torch.fx.wrap
def _new_length_tensor(
    length_per_key: List[int], device_t: torch.Tensor
) -> torch.Tensor:
    return torch.tensor(length_per_key, dtype=torch.int32, device=device_t.device)


class InputSENet(nn.Module):
    """SENet for Input Embedding."""

    def __init__(self, length_per_key: List[int], reduction_ratio: int = 2) -> None:
        super().__init__()
        field_size = len(length_per_key)
        reduction_size = max(1, field_size // reduction_ratio)
        self._length_per_key = length_per_key
        self.excitation = nn.Sequential(
            nn.Linear(field_size, reduction_size, bias=False),
            nn.ReLU(),
            nn.Linear(reduction_size, field_size, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            x (torch.Tensor): a Tensor contains embedding of N features.
        """
        length_per_key = _new_length_tensor(self._length_per_key, x)
        lengths = length_per_key.unsqueeze(0).repeat(x.size(0), 1)
        xx = torch.segment_reduce(x, "mean", lengths=lengths, axis=1)
        xx = self.excitation(xx)
        x = x * torch.repeat_interleave(xx, repeats=length_per_key, dim=1)
        return x


class InteractionArch(nn.Module):
    """Feature interaction module.

    Args:
        feature_num (int): feature_num
    """

    def __init__(self, feature_num: int) -> None:
        super().__init__()
        self.feature_num: int = feature_num
        self.register_buffer(
            "triu_indices",
            torch.triu_indices(self.feature_num, self.feature_num, offset=1),
            persistent=False,
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        dim = 0
        for i in range(1, self.feature_num):
            dim += i
        return dim

    def forward(
        self, dense_features: torch.Tensor, sparse_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            dense_features (torch.Tensor): an input tensor of size B X D.
            sparse_features (torch.Tensor): an input tensor of size B X N X D.
        """
        if self.feature_num <= 0:
            return dense_features

        combined_values = torch.cat(
            (dense_features.unsqueeze(1), sparse_features), dim=1
        )  # B X (N+1) X D

        interactions = torch.bmm(
            combined_values, torch.transpose(combined_values, 1, 2)
        )  # B X (N+1) X (N+1)
        interactions_flat = interactions[:, self.triu_indices[0], self.triu_indices[1]]

        return interactions_flat
