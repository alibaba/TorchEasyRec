import torch
import torch.nn as nn
from typing import List, Union


class Add(nn.Module):
    def forward(self, *inputs):
        # 支持输入为 list/tuple
        out = inputs[0]
        for i in range(1, len(inputs)):
            out = out + inputs[i]
        return out


class FM(nn.Module):
    """Factorization Machine module for backbone architecture.
    
    This module implements the FM interaction computation that learns 2nd-order
    feature interactions. It supports both list of 2D tensors and 3D tensor inputs.
    
    Args:
        use_variant (bool, optional): Whether to use variant FM calculation. 
            Defaults to False.
        l2_regularization (float, optional): L2 regularization coefficient.
            Defaults to 1e-4.
    
    Input shapes:
        - List of 2D tensors with shape: ``(batch_size, embedding_size)``
        - Or a 3D tensor with shape: ``(batch_size, field_size, embedding_size)``
    
    Output shape:
        - 2D tensor with shape: ``(batch_size, 1)``
    """
    
    def __init__(self, use_variant: bool = False, l2_regularization: float = 1e-4) -> None:
        super().__init__()
        self.use_variant = use_variant
        self.l2_regularization = l2_regularization
        
    def forward(self, inputs: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """Forward pass of FM module.
        
        Args:
            inputs: Either a list of 2D tensors [(batch_size, embedding_size), ...]
                   or a 3D tensor (batch_size, field_size, embedding_size)
        
        Returns:
            torch.Tensor: FM interaction output with shape (batch_size, 1)
        """
        # Convert list of 2D tensors to 3D tensor if needed
        if isinstance(inputs, list):
            # Stack list of 2D tensors to form 3D tensor
            feature = torch.stack(inputs, dim=1)  # (batch_size, field_size, embedding_size)
        else:
            feature = inputs
            
        # Ensure input is 3D
        if feature.dim() != 3:
            raise ValueError(f"Expected 3D tensor after conversion, got {feature.dim()}D")
        
        batch_size, field_size, embedding_size = feature.shape
        
        if self.use_variant:
            # Variant FM: more computationally efficient for sparse features
            # Sum pooling across fields
            sum_of_features = torch.sum(feature, dim=1)  # (batch_size, embedding_size)
            square_of_sum = sum_of_features.pow(2)  # (batch_size, embedding_size)
            
            # Sum of squares
            sum_of_squares = torch.sum(feature.pow(2), dim=1)  # (batch_size, embedding_size)
            
            # FM interaction: 0.5 * (square_of_sum - sum_of_squares)
            fm_output = 0.5 * (square_of_sum - sum_of_squares)  # (batch_size, embedding_size)
            
            # Sum across embedding dimension and add batch dimension
            output = torch.sum(fm_output, dim=1, keepdim=True)  # (batch_size, 1)
        else:
            # Standard FM computation
            # Pairwise interactions: sum over all pairs (i,j) where i<j
            interactions = []
            for i in range(field_size):
                for j in range(i + 1, field_size):
                    # Element-wise product of embeddings
                    interaction = feature[:, i, :] * feature[:, j, :]  # (batch_size, embedding_size)
                    interactions.append(interaction)
            
            if interactions:
                # Stack and sum all interactions
                all_interactions = torch.stack(interactions, dim=1)  # (batch_size, num_pairs, embedding_size)
                fm_output = torch.sum(all_interactions, dim=[1, 2])  # (batch_size,)
                fm_output = fm_output.unsqueeze(1)  # (batch_size, 1)
            else:
                # No interactions possible (less than 2 fields)
                fm_output = torch.zeros(batch_size, 1, device=feature.device, dtype=feature.dtype)
            
            output = fm_output
        
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