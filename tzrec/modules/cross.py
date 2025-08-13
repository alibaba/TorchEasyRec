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

import torch
from torch import nn


class Cross(nn.Module):
    """Cross Layer for DCN (Deep & Cross Network).

    This layer implements the cross layer from DCN, which explicitly learns
    feature interactions of bounded degrees in an efficient way.

    The formula is: x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
    where ⊙ denotes element-wise multiplication.

    Args:
        input_dim (int): Input feature dimension.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        # Weight matrix W_l with shape (input_dim,)
        self.weight = nn.Parameter(torch.empty(input_dim))
        # Bias vector b_l with shape (input_dim,)
        self.bias = nn.Parameter(torch.empty(input_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Xavier uniform initialization for weight
        nn.init.xavier_uniform_(self.weight.unsqueeze(0))
        # Zero initialization for bias
        nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of Cross Layer.

        Args:
            x0 (torch.Tensor): Original input features with shape
                             (batch_size, input_dim)
            xl (torch.Tensor, optional): Input from previous layer with shape
                                       (batch_size, input_dim). If None, will use x0.
                                       Defaults to None.

        Returns:
            torch.Tensor: Output features with shape (batch_size, input_dim)
        """
        if xl is None:
            xl = x0

        # Compute W_l * x_l + b_l
        linear_part = xl * self.weight + self.bias  # (batch_size, input_dim)

        # Compute x_0 ⊙ (W_l * x_l + b_l)
        cross_part = x0 * linear_part  # (batch_size, input_dim)

        # Add residual connection: x_{l+1} = x_0 ⊙ (W_l * x_l + b_l) + x_l
        output = cross_part + xl  # (batch_size, input_dim)

        return output


class CrossNet(nn.Module):
    """Cross Network for DCN (Deep & Cross Network).

    This module stacks multiple Cross Layers to learn high-order feature interactions.

    Args:
        input_dim (int): Input feature dimension.
        num_layers (int): Number of cross layers. Defaults to 3.
    """

    def __init__(self, input_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers

        # Stack multiple cross layers
        self.cross_layers = nn.ModuleList([Cross(input_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of Cross Network.

        Args:
            x (torch.Tensor): Input features with shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output features with shape (batch_size, input_dim)
        """
        x0 = x  # Keep original input for cross operations
        xl = x  # Current layer input

        # Pass through each cross layer
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        return xl

    def output_dim(self) -> int:
        """Output dimension of the Cross Network."""
        return self.input_dim


class DCNv2Layer(nn.Module):
    """Cross Layer for DCN-v2 (Improved Deep & Cross Network).

    This is an improved version of the cross layer that uses a low-rank matrix
    to reduce parameters and computational cost while maintaining expressiveness.

    The formula is: x_{l+1} = x_0 ⊙ (U_l * (V_l^T * x_l) + b_l) + x_l
    where U_l and V_l are low-rank matrices.

    Args:
        input_dim (int): Input feature dimension.
        low_rank (int): Low rank dimension. Defaults to 32.
    """

    def __init__(self, input_dim: int, low_rank: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.low_rank = low_rank

        # Low-rank matrices for DCN-v2
        self.U = nn.Parameter(torch.empty(input_dim, low_rank))  # (input_dim, low_rank)
        self.V = nn.Parameter(torch.empty(input_dim, low_rank))  # (input_dim, low_rank)
        self.bias = nn.Parameter(torch.empty(input_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Xavier uniform initialization for U and V
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        # Zero initialization for bias
        nn.init.zeros_(self.bias)

    def forward(self, x0: torch.Tensor, xl: torch.Tensor = None) -> torch.Tensor:
        """Forward pass of DCN-v2 Layer.

        Args:
            x0 (torch.Tensor): Original input features with shape
                             (batch_size, input_dim)
            xl (torch.Tensor, optional): Input from previous layer with shape
                                       (batch_size, input_dim). If None, will use x0.
                                       Defaults to None.

        Returns:
            torch.Tensor: Output features with shape (batch_size, input_dim)
        """
        if xl is None:
            xl = x0

        # Compute V^T * x_l
        v_xl = torch.matmul(xl, self.V)  # (batch_size, low_rank)

        # Compute U * (V^T * x_l) + b_l
        linear_part = (
            torch.matmul(v_xl, self.U.T) + self.bias
        )  # (batch_size, input_dim)

        # Compute x_0 ⊙ (U * (V^T * x_l) + b_l)
        cross_part = x0 * linear_part  # (batch_size, input_dim)

        # Add residual connection
        output = cross_part + xl  # (batch_size, input_dim)

        return output


class DCNv2Net(nn.Module):
    """Cross Network for DCN-v2 (Improved Deep & Cross Network).

    This module stacks multiple DCN-v2 Layers with low-rank approximation
    to reduce parameters while maintaining model expressiveness.

    Args:
        input_dim (int): Input feature dimension.
        num_layers (int): Number of cross layers. Defaults to 3.
        low_rank (int): Low rank dimension. Defaults to 32.
    """

    def __init__(self, input_dim: int, num_layers: int = 3, low_rank: int = 32) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.low_rank = low_rank

        # Stack multiple DCN-v2 layers
        self.cross_layers = nn.ModuleList(
            [DCNv2Layer(input_dim, low_rank) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of DCN-v2 Network.

        Args:
            x (torch.Tensor): Input features with shape (batch_size, input_dim)

        Returns:
            torch.Tensor: Output features with shape (batch_size, input_dim)
        """
        x0 = x  # Keep original input for cross operations
        xl = x  # Current layer input

        # Pass through each cross layer
        for cross_layer in self.cross_layers:
            xl = cross_layer(x0, xl)

        return xl

    def output_dim(self) -> int:
        """Output dimension of the DCN-v2 Network."""
        return self.input_dim
