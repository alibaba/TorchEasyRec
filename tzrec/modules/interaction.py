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
