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


from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.modules.mlp import MLP


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
        input_dim(int): input tensor dimension
        cross_num(int): number of cross layers
    """

    def __init__(self, input_dim: int, cross_num: int = 3) -> None:
        super().__init__()
        self.cross_num = cross_num
        self._input_dim = input_dim
        self.w = nn.ModuleList()
        self.b = nn.ParameterList()
        for _ in range(cross_num):
            self.w.append(nn.Linear(input_dim, 1, bias=False))
            self.b.append(nn.Parameter(torch.empty(input_dim)))

        self.reset_parameters()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._input_dim

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
        cross_num (int): number of cross layers.
        low_rank (int): W dimension
    """

    def __init__(self, input_dim: int, cross_num: int = 3, low_rank: int = 32) -> None:
        super(CrossV2, self).__init__()
        self.cross_num = cross_num
        self._low_rank = low_rank
        self._input_dim = input_dim

        self.u_kernels = nn.ModuleList(
            [
                nn.Linear(self._input_dim, self._low_rank, bias=False)
                for _ in range(cross_num)
            ]
        )
        self.v_kernels = nn.ModuleList(
            [
                nn.Linear(self._low_rank, self._input_dim, bias=True)
                for _ in range(cross_num)
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
        for i in range(self.cross_num):
            x_l_v = self.u_kernels[i](x_l)
            x_l_w = self.v_kernels[i](x_l_v)
            x_l = x_0 * x_l_w + x_l  # (batch_size, input_dim)

        return x_l


class CIN(nn.Module):
    """CIN module for XDeepFM.

    Args:
        feature_num (int): feature_num
        cin_layer_size(list[int]): cin_layer_size
    """

    def __init__(self, feature_num: int, cin_layer_size: List[int]) -> None:
        super(CIN, self).__init__()
        self.feature_num = feature_num
        self.cin_layer_size = cin_layer_size

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
            x_vec = z
            x_out.append(torch.sum(x_vec, dim=2))

        return torch.cat(x_out, dim=1)


class LinearCompressBlock(nn.Module):
    """LinerBlock module for WuKongLayer.

    Args:
        feature_num_in (int): feature_num
        feature_num_out(int): feature_out_num
    """

    def __init__(self, feature_num_in: int, feature_num_out: int) -> None:
        super().__init__()
        self._feature_num_out = feature_num_out
        self._feature_num_in = feature_num_in
        self.weight = nn.Parameter(torch.empty((feature_num_in, feature_num_out)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the weight parameters."""
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            inputs (torch.Tensor): tensor with shape [batch_size, field_num, embed_dim].
        """
        outputs = inputs.permute(0, 2, 1)
        outputs = outputs @ self.weight
        outputs = outputs.permute(0, 2, 1)
        return outputs


class FactorizationMachineBlock(nn.Module):
    """FM Block module for WuKongLayer.

    Args:
        input_dim(int): embedding dimension
        feature_num_in (int): number of input features.
        feature_num_out(int): number of output features.
        compressed_feature_num (int): number of compressed features in optimized FM.
        feature_num_mlp (dict): MLP config
    """

    def __init__(
        self,
        input_dim: int,
        feature_num_in: int,
        feature_num_out: int,
        compressed_feature_num: int,
        feature_num_mlp: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.feature_num_in = feature_num_in
        self.feature_num_out = feature_num_out
        self.input_dim = input_dim
        self.compressed_feature_num = compressed_feature_num
        self.weight = nn.Parameter(
            torch.empty((feature_num_in, compressed_feature_num))
        )
        self.norm = nn.LayerNorm(feature_num_in * compressed_feature_num)

        self.mlp = MLP(
            in_features=feature_num_in * compressed_feature_num, **feature_num_mlp
        )
        self.feature_out_liner = nn.Linear(
            self.mlp.output_dim(), self.feature_num_out * self.input_dim
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset the weight parameters."""
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            inputs (torch.Tensor): tensor with shape [batch_size, field_num, embed_dim].
        """
        outputs = inputs.permute(0, 2, 1)
        outputs = torch.matmul(outputs, self.weight)
        outputs = torch.matmul(inputs, outputs)
        outputs = outputs.view(-1, self.feature_num_in * self.compressed_feature_num)
        outputs = self.mlp(self.norm(outputs))
        outputs = self.feature_out_liner(outputs)
        outputs = outputs.view(-1, self.feature_num_out, self.input_dim)
        return outputs


class WuKongLayer(nn.Module):
    """WuKongLayer module for WuKong model.

    Args:
        input_dim(int): embedding dimension
        feature_num (int): feature_num
        lcb_feature_num (int): LinearCompressBlock output feature num.
        fmb_feature_num (int): FactorizationMachineBlock output feature num.
        compressed_feature_num (int): number of compressed features in optimized FM.
        feature_num_mlp (dict): feature num MLP module parameters.
    """

    def __init__(
        self,
        input_dim: int,
        feature_num: int,
        lcb_feature_num: int,
        fmb_feature_num: int,
        compressed_feature_num: int,
        feature_num_mlp: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.lcb_feature_num = lcb_feature_num
        self.fmb_feature_num = fmb_feature_num
        self.lcb = LinearCompressBlock(feature_num, lcb_feature_num)
        self.fmb = FactorizationMachineBlock(
            input_dim,
            feature_num,
            fmb_feature_num,
            compressed_feature_num,
            feature_num_mlp,
        )
        self.norm = nn.LayerNorm(input_dim)
        if feature_num != lcb_feature_num + fmb_feature_num:
            self.residual_projection = LinearCompressBlock(
                feature_num, lcb_feature_num + fmb_feature_num
            )
        else:
            self.residual_projection = nn.Identity()

    def output_feature_num(self) -> int:
        """Get output feature num."""
        return self.lcb_feature_num + self.fmb_feature_num

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward the module.

        Args:
            inputs (torch.Tensor): tensor with shape [batch_size, field_num, embed_dim].
        """
        lcb = self.lcb(inputs)
        fmb = self.fmb(inputs)
        outputs = torch.concat((fmb, lcb), dim=1)
        outputs = self.norm(outputs + self.residual_projection(inputs))
        return outputs
