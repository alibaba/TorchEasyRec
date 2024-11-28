import torch
from torch import nn

class RotateLayer(nn.Module):
    def __init__(self, 
                base_dim,
                low_rank_dim):
        super().__init__()
        # n > m
        self.weight = torch.nn.Parameter(torch.empty(base_dim, low_rank_dim), requires_grad=True)
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, base):
        return torch.matmul(base.to(self.weight.dtype), self.weight)

class Intervention(nn.Module):
    def __init__(self,
        base_dim,source_dim,
        low_rank_dim,orth=True):
        super().__init__()
        self.base_dim = base_dim
        base_rotate_layer = RotateLayer(base_dim, low_rank_dim)
        self.base_rotate_layer = torch.nn.utils.parametrizations.orthogonal(base_rotate_layer)
        source_rotate_layer = RotateLayer(source_dim, low_rank_dim)
        self.source_rotate_layer = torch.nn.utils.parametrizations.orthogonal(source_rotate_layer)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, base, source):
        rotated_base = self.base_rotate_layer(base)
        rotated_source = self.source_rotate_layer(source.detach())
        output = torch.matmul(rotated_base-rotated_source, self.base_rotate_layer.weight.T) + base
        return self.dropout(output.to(base.dtype))
    
    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.base_dim