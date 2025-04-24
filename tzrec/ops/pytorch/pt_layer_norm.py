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


from typing import List

import torch


def pytorch_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    return torch.nn.functional.layer_norm(
        x.to(torch.float32),
        normalized_shape,
        weight.to(torch.float32),
        bias.to(torch.float32),
        eps,
    ).to(dtype)


def pytorch_swish_layer_norm(
    x: torch.Tensor,
    normalized_shape: List[int],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    dtype = x.dtype
    x = x.to(torch.float32)
    return (
        x
        * torch.sigmoid(
            torch.nn.functional.layer_norm(
                x,
                normalized_shape,
                weight.to(torch.float32),
                bias.to(torch.float32),
                eps,
            )
        )
    ).to(dtype)
