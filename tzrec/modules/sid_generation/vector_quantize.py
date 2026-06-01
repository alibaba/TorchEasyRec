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

"""Single codebook vector quantization with Sinkhorn uniform assignment."""

from typing import Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.kmeans import _squared_euclidean_distance
from tzrec.modules.sid_generation.types import (
    QuantizeForwardMode,
    QuantizeOutput,
)


def _gumbel_softmax_sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    hard: bool = True,
) -> torch.Tensor:
    """Sample from the Gumbel-Softmax distribution.

    Args:
        logits (Tensor): un-normalized log probabilities, shape (B, N).
        temperature (float): temperature for Gumbel-Softmax.
        hard (bool): if True, return one-hot with straight-through gradient.

    Returns:
        Tensor: soft or hard sample, shape (B, N).
    """
    return F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)


@torch.no_grad()
def _sinkhorn(
    cost: torch.Tensor,
    n_iters: int = 5,
    epsilon: float = 10.0,
    is_distributed: bool = True,
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for optimal-transport based uniform assignment.

    Transforms a distance matrix into a soft assignment matrix via exponential
    kernel and alternating row-column normalization, approximating a doubly
    stochastic matrix to ensure uniform codebook utilization.

    Args:
        cost (Tensor): distance matrix, shape (B, K) where K is codebook size.
            IMPORTANT: must be z-score normalized and shifted to non-negative
            before calling this function to avoid numerical overflow.
        n_iters (int): number of Sinkhorn iterations. Default: 5.
        epsilon (float): sharpness parameter for exp(-cost * epsilon).
            Larger values produce sharper assignments. Default: 10.0.
        is_distributed (bool): whether running in distributed mode.
            If True, row sums are all_reduced across GPUs. Default: True.

    Returns:
        Tensor: assignment matrix, shape (B, K).
            Use Q.argmax(dim=-1) externally to get codebook indices.
    """
    # Step 1: exponential kernel transform  (B, K) -> (K, B)
    Q = torch.exp(-cost * epsilon).t()

    # Global batch size for distributed training
    if is_distributed and dist.is_initialized():
        B = Q.size(1) * dist.get_world_size()
    else:
        B = Q.size(1)
    K = Q.size(0)

    # Step 2: global normalization — make matrix sum to 1
    sum_Q = torch.sum(Q)
    if is_distributed and dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q + 1e-8

    # Step 3: alternating row-column normalization
    for _ in range(n_iters):
        # Row normalization: each prototype's total weight = 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if is_distributed and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows + 1e-8
        Q /= K

        # Column normalization: each sample's total weight = 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-8
        Q /= B

    # Step 4: scale back so columns sum to 1 (assignment)
    Q *= B
    return Q.t()  # (B, K)


class VectorQuantize(nn.Module):
    """Single codebook vector quantization layer.

    Maps continuous input vectors to the nearest codebook entry and returns
    the quantized embeddings + codebook indices. The commitment loss is
    computed at the residual-aggregator level by
    :meth:`ResidualVectorQuantizer._single_commitment_loss` over the cumulative
    quants (matching al_sid's ``RQBottleneck.compute_commitment_loss``);
    this layer is intentionally loss-free.

    During training, Sinkhorn optimal-transport assignment is optionally
    used to encourage uniform codebook utilization.

    Args:
        embed_dim (int): dimension of each codebook embedding.
        n_embed (int): number of codebook entries.
        forward_mode (QuantizeForwardMode): quantization forward mode,
            either GUMBEL_SOFTMAX or STE. Default: STE.
        distance_type (str): distance metric, 'l2' or 'cosine'.
            Default: 'l2'.
        use_sinkhorn (bool): whether to use Sinkhorn uniform assignment
            during training. Default: True.
        sinkhorn_iters (int): number of Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness parameter for
            exp(-cost * epsilon). Default: 10.0.
    """

    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_type: str = "l2",
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.forward_mode = forward_mode
        self.distance_type = distance_type
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

    @torch.no_grad()
    def _compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute distances between input vectors and codebook entries.

        Supports L2 and cosine distance metrics.

        Args:
            x (Tensor): input vectors, shape (B, D).

        Returns:
            Tensor: pairwise distances, shape (B, n_embed).
        """
        codebook = self.embedding.weight  # (n_embed, D)

        if self.distance_type == "l2":
            distances = _squared_euclidean_distance(x, codebook)
        elif self.distance_type == "cosine":
            x_norm = F.normalize(x, p=2, dim=1)
            codebook_norm = F.normalize(codebook, p=2, dim=1)
            distances = -torch.matmul(x_norm, codebook_norm.t())
        else:
            raise ValueError(
                f"Unsupported distance_type '{self.distance_type}', "
                f"choose from ('l2', 'cosine')"
            )
        return distances

    @torch.no_grad()
    def _find_nearest_embedding(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find nearest codebook entry for each input vector.

        During training with use_sinkhorn=True, applies z-score
        normalization + non-negative shift before Sinkhorn assignment.
        Otherwise falls back to argmin.

        Args:
            x (Tensor): input vectors, shape (B, D).

        Returns:
            ids (Tensor): codebook indices, shape (B,).
            distances (Tensor): distance matrix, shape (B, n_embed).
        """
        distances = self._compute_distances(x)  # (B, n_embed)

        if self.training and self.use_sinkhorn:
            # Sinkhorn requires non-negative cost; z-score then shift.
            var, mean = torch.var_mean(distances, unbiased=False)
            distances = (distances - mean) * var.add(1e-12).rsqrt()
            distances = distances - distances.min()

            # Sinkhorn optimal-transport assignment
            Q = _sinkhorn(
                distances,
                n_iters=self.sinkhorn_iters,
                epsilon=self.sinkhorn_epsilon,
                is_distributed=dist.is_initialized(),
            )
            ids = Q.argmax(dim=-1)
        else:
            ids = distances.argmin(dim=-1)

        return ids, distances

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
    ) -> QuantizeOutput:
        """Forward the vector quantization layer.

        Training flow:
            1. compute distances (L2 or cosine)
            2. if use_sinkhorn: z-score normalize + Sinkhorn -> argmax
               else: argmin
            3. compute differentiable embedding (STE or Gumbel-Softmax)

        Commitment loss is computed by the caller
        (:meth:`ResidualVectorQuantizer._single_commitment_loss`).

        Args:
            x (Tensor): input vectors, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.

        Returns:
            QuantizeOutput: named tuple of (embeddings, ids).
        """
        # Step 1-2: find nearest codebook entry
        ids, distances = self._find_nearest_embedding(x)

        # Step 3: differentiable embedding. Gumbel takes a separate path
        # that combines all codebook entries; STE goes through a single
        # embedding lookup.
        if self.training and self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
            weights = _gumbel_softmax_sample(
                -distances, temperature=temperature, hard=True
            )
            emb = weights @ self.embedding.weight
        elif self.training and self.forward_mode == QuantizeForwardMode.STE:
            quantized = self.embedding(ids)
            # Straight-Through Estimator: gradient passes through
            emb = x + (quantized - x).detach()
        elif self.training:
            raise ValueError(f"Unsupported forward mode: {self.forward_mode}")
        else:
            emb = self.embedding(ids)

        return QuantizeOutput(embeddings=emb, ids=ids)
