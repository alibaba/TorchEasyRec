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


class Cross(nn.Module):
    """Cross Layer in Deep & Cross Network to learn explicit feature interactions.

    Ref: https://arxiv.org/pdf/1708.05123

    Args:
        cross_num(int): number of cross layers
        input_dim(int): input tensor dimension
    """

    def __init__(self, cross_num: int, input_dim: int) -> None:
        super().__init__()
        self.cross_num = cross_num
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for _ in range(cross_num):
            self.w.append(nn.Linear(input_dim, 1, bias=False))
            self.b.append(nn.Parameter(torch.empty(input_dim)))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        for i in range(self.cross_num):
            nn.init.xavier_uniform_(self.w[i].weight)
            nn.init.zeros_(self.b[i])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward."""
        x1 = x
        for i in range(self.cross_num):
            x1 = self.w[i](x1) * x + self.b[i] + x1

        return x1


class CrossV2(nn.Module):
    """Cross network v2.

    Args:
        input_dim (int): input tensor dimension.
        low_rank (int): W dimension
        num_layers (int): number of cross layers.
    """

    def __init__(self, input_dim: int, low_rank=32, cross_num=3):
        super(CrossV2, self).__init__()
        self.cross_num = cross_num
        self._low_rank = low_rank
        self._input_dim = input_dim
        self.u_kernels = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._input_dim, self._low_rank)
                    )
                )
                for _ in range(self.cross_num)
            ]
        )
        self.v_kernels = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.nn.init.xavier_normal_(
                        torch.empty(self._low_rank, self._input_dim)
                    )
                )
                for _ in range(self.cross_num)
            ]
        )

        self.bias = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.nn.init.zeros_(torch.empty(self._input_dim)))
                for _ in range(self.cross_num)
            ]
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._input_dim

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            input (torch.Tensor): tensor with shape [batch_size, input_dim].
        """
        x_0 = input
        x_l = x_0
        for layer in range(self.cross_num):
            x_l_v = torch.nn.functional.linear(x_l, self.v_kernels[layer])
            x_l_w = torch.nn.functional.linear(x_l_v, self.u_kernels[layer])
            x_l = x_0 * (x_l_w + self.bias[layer]) + x_l  # (batch_size, input_dim)

        return x_l


class CIN(nn.Module):
    """CIN module for XDeepFM.

    Args:
        feature_num (int): feature_num
        cin_layer_size(list[int]): cin_layer_size
    """

    def __init__(self, feature_num: int, cin_layer_size: List[int], use_bias=False):
        super(CIN, self).__init__()
        self.feature_num = feature_num
        self.cin_layer_size = cin_layer_size
        self.use_bias = use_bias

        self.cin_layers = nn.ModuleList()
        for i, layer_size in enumerate(cin_layer_size):
            in_channels = (
                feature_num * self.cin_layer_size[i - 1]
                if i > 0
                else feature_num * feature_num
            )
            self.cin_layers.append(
                nn.Conv1d(
                    in_channels=in_channels, out_channels=layer_size, kernel_size=1
                )
            )

        if self.use_bias:
            self.bias = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(cin_layer_size[i]))
                    for i in range(len(cin_layer_size))
                ]
            )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return sum(self.cin_layer_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            input (torch.Tensor): tensor with shape [batch_size, field_num, embed_dim].
        """
        batch_size, field_num, embed_dim = input.shape
        x_vec = input
        x_out = []
        for i, _ in enumerate(self.cin_layer_size):
            z = torch.einsum("bhd,bfd->bhfd", x_vec, input)
            if i > 0:
                h = field_num * self.cin_layer_size[i - 1]
            else:
                h = field_num * field_num
            z = z.view(batch_size, h, embed_dim)
            z = self.cin_layers[i](z)  # (batch_size, cin_layer_size[i], embed_dim)
            if self.use_bias:
                z += self.bias[i]
            z = torch.relu(z)
            x_vec = z
            x_out.append(torch.sum(x_vec, dim=2))

        return torch.cat(x_out, dim=1)
