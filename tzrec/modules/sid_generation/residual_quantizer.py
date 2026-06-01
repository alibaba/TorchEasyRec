# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ResidualQuantizer: abstract base for multi-layer residual quantizers."""

from typing import List, Union

import torch
from torch import nn


def normalize_n_embed(n_embed: Union[int, List[int]], n_layers: int) -> List[int]:
    """Broadcast a scalar codebook size to a per-layer list (or validate one).

    Args:
        n_embed (int|List[int]): codebook size, shared or per-layer.
        n_layers (int): number of residual quantization layers.

    Returns:
        List[int]: per-layer codebook sizes, length ``n_layers``.
    """
    if isinstance(n_embed, int):
        return [n_embed] * n_layers
    assert len(n_embed) == n_layers, (
        "length of n_embed and n_layers must be same, "
        f"but got {len(n_embed)} vs {n_layers}"
    )
    return list(n_embed)


class ResidualQuantizer(nn.Module):
    """Abstract base for multi-layer residual quantization.

    Shared contract for the two SID quantizer backends — the VQ-based,
    gradient-trained :class:`ResidualVectorQuantizer` and the K-Means-based,
    offline-FAISS-trained :class:`ResidualKMeansQuantizer`. Both quantize the
    residual of the previous layer:

        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1}).

    This base owns the structural invariants (``embed_dim``, ``n_layers``,
    per-layer codebook sizes, residual normalization toggle) and the
    backend-agnostic :meth:`decode_codes` / :meth:`output_dim`. Subclasses
    build ``self.layers`` and implement :meth:`forward`, :meth:`get_codes`,
    :meth:`get_codebook_embeddings`, and :meth:`_lookup_code`.

    Args:
        embed_dim (int): feature / codebook dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        normalize_residuals (bool): L2-normalize residuals before each
            layer. Default: False.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.n_embed_list = normalize_n_embed(n_embed, n_layers)
        # Subclasses MUST populate this with one quantization layer each.
        self.layers: nn.ModuleList = nn.ModuleList()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    def forward(self, input: torch.Tensor):  # noqa: ANN201
        """Assign codes per layer and accumulate the quantized output."""
        raise NotImplementedError

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating the codebook.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get the codebook (centroid) weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        raise NotImplementedError

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up the codebook vectors for ``code_idx`` at ``layer_idx``.

        The single backend-specific primitive :meth:`decode_codes` builds on
        (VQ reads ``embedding(idx)``, K-Means reads ``centroids[idx]``).

        Args:
            layer_idx (int): index of the quantization layer.
            code_idx (Tensor): codebook indices, shape (B,).

        Returns:
            Tensor: looked-up codebook vectors, shape (B, embed_dim).
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes (centroid sum).

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, embed_dim).
        """
        quantized_sum = torch.zeros(
            codes.shape[0],
            self.embed_dim,
            device=codes.device,
            dtype=torch.float,
        )
        for i in range(self.n_layers):
            quantized_sum = quantized_sum + self._lookup_code(i, codes[:, i])
        return quantized_sum
