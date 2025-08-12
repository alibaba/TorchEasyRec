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
import torch.nn as nn


class Add(nn.Module):
    """Element-wise addition module for multiple tensors.

    This module performs element-wise addition of multiple input tensors.
    It supports a fixed number of tensor inputs for FX tracing and JIT Script compatibility.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, input1, input2, input3=None):
        """Add multiple input tensors element-wise.

        Args:
            input1: First tensor (required)
            input2: Second tensor (required)
            input3: Third tensor (optional)

        Returns:
            torch.Tensor: Sum of all input tensors.
        """
        # Add the first two tensors
        result = input1 + input2
        
        # Add the third tensor if provided
        if input3 is not None:
            result = result + input3
            
        return result


class FM(nn.Module):
    """Factorization Machine module for backbone architecture.

    This module implements the FM interaction computation that learns 2nd-order
    feature interactions. It only supports 3D tensor inputs for better compatibility
    with PyTorch graph compilation modes (FX tracing and JIT Script).

    Args:
        use_variant (bool, optional): Whether to use variant FM calculation.
            Defaults to False.
        l2_regularization (float, optional): L2 regularization coefficient.
            Defaults to 1e-4.

    Input shapes:
        - 3D tensor with shape: ``(batch_size, field_size, embedding_size)``

    Output shape:
        - 2D tensor with shape: ``(batch_size, 1)``
    """

    def __init__(
        self, use_variant: bool = False, l2_regularization: float = 1e-4
    ) -> None:
        super().__init__()
        self.use_variant = use_variant
        self.l2_regularization = l2_regularization

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Forward pass of FM module.

        Args:
            inputs: 3D tensor with shape (batch_size, field_size, embedding_size)

        Returns:
            torch.Tensor: FM interaction output with shape (batch_size, 1)
        """
        # Note: Dimension validation is skipped for FX tracing compatibility
        # Users should ensure inputs are 3D tensors with shape (batch_size, field_size, embedding_size)
        
        feature = inputs
        batch_size, field_size, embedding_size = feature.shape

        if self.use_variant:
            # Variant FM: more computationally efficient for sparse features
            # Sum pooling across fields
            sum_of_features = torch.sum(feature, dim=1)  # (batch_size, embedding_size)
            square_of_sum = sum_of_features.pow(2)  # (batch_size, embedding_size)

            # Sum of squares
            sum_of_squares = torch.sum(
                feature.pow(2), dim=1
            )  # (batch_size, embedding_size)

            # FM interaction: 0.5 * (square_of_sum - sum_of_squares)
            fm_output = 0.5 * (
                square_of_sum - sum_of_squares
            )  # (batch_size, embedding_size)

            # Sum across embedding dimension and add batch dimension
            output = torch.sum(fm_output, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            # Standard FM computation using vectorized operations
            # This is equivalent to pairwise interactions but FX-trace friendly

            # Sum pooling across fields
            sum_of_features = torch.sum(feature, dim=1)  # (batch_size, embedding_size)
            square_of_sum = sum_of_features.pow(2)  # (batch_size, embedding_size)

            # Sum of squares
            sum_of_squares = torch.sum(
                feature.pow(2), dim=1
            )  # (batch_size, embedding_size)

            # FM interaction: 0.5 * (square_of_sum - sum_of_squares)
            fm_interaction = 0.5 * (
                square_of_sum - sum_of_squares
            )  # (batch_size, embedding_size)

            # Sum across embedding dimension to get final output
            output = torch.sum(fm_interaction, dim=1, keepdim=True)  # (batch_size, 1)

        # Apply L2 regularization if specified (add to loss during training)
        if self.training and self.l2_regularization > 0:
            # Store L2 regularization term for potential use in loss calculation
            self.l2_reg_loss = self.l2_regularization * torch.sum(feature.pow(2))

        return output

    def output_dim(self) -> int:
        """Output dimension of the FM module.

        Returns:
            int: Always returns 1 since FM outputs (batch_size, 1)
        """
        return 1
