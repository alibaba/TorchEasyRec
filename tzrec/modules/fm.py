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


class FactorizationMachine(nn.Module):
    """Factorization Machine module.

    It covers only the FM part of Factorization Machine model, and is used to
    learn 2nd-order feature interactions.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            feature (torch.Tensor): a Tensor contains embedding of N features,
                shape should be [B, N, D].

        Returns:
            torch.Tensor: output of fm module, shape is [B, D]
        """
        sum_of_input = torch.sum(feature, dim=1)
        sum_of_square = torch.sum(feature * feature, dim=1)
        square_of_sum = sum_of_input * sum_of_input

        y = 0.5 * (square_of_sum - sum_of_square)
        return y
