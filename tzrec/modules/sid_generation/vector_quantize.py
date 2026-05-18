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

"""Single codebook vector quantization with EMA, Sinkhorn, and dead code restart."""

from typing import List, Optional, Tuple

import numpy as np
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
    Q /= (sum_Q + 1e-8)

    # Step 3: alternating row-column normalization
    for _ in range(n_iters):
        # Row normalization: each prototype's total weight = 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if is_distributed and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= (sum_of_rows + 1e-8)
        Q /= K

        # Column normalization: each sample's total weight = 1/B
        Q /= (torch.sum(Q, dim=0, keepdim=True) + 1e-8)
        Q /= B

    # Step 4: scale back so columns sum to 1 (assignment)
    Q *= B
    return Q.t()  # (B, K)


class VectorQuantize(nn.Module):
    """Single codebook vector quantization layer with EMA and Sinkhorn.

    Maps continuous input vectors to the nearest codebook entry and returns
    the quantized embeddings, codebook indices, and commitment loss.

    Supports EMA codebook updates, Sinkhorn uniform assignment, dead code
    restart, and multiple distance metrics.

    Args:
        embed_dim (int): dimension of each codebook embedding.
        n_embed (int): number of codebook entries.
        commitment_weight (float): weight for the commitment loss term.
            Default: 0.25.
        forward_mode (QuantizeForwardMode): quantization forward mode,
            either GUMBEL_SOFTMAX or STE. Default: STE.
        distance_type (str): distance metric, 'l2' or 'cosine'.
            Default: 'l2'.
        use_ema (bool): whether to use EMA to update codebook weights
            instead of gradient descent. Default: True.
        ema_decay (float): EMA decay coefficient. Default: 0.99.
        restart_unused_codes (bool): whether to reset dead codes to random
            batch samples. Only effective when use_ema=True. Default: True.
        use_sinkhorn (bool): whether to use Sinkhorn uniform assignment
            during training. Default: True.
        sinkhorn_iters (int): number of Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness parameter for
            exp(-cost * epsilon). Default: 10.0.
        eps (float): numerical stability term for Laplace smoothing.
            Default: 1e-5.
    """

    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        commitment_weight: float = 0.25,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_type: str = "l2",
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.commitment_weight = commitment_weight
        self.forward_mode = forward_mode
        self.distance_type = distance_type
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.restart_unused_codes = restart_unused_codes
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon
        self.eps = eps

        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

        # EMA buffers (only registered when use_ema=True)
        if self.use_ema:
            self.register_buffer(
                "cluster_size_ema", torch.zeros(n_embed)
            )
            self.register_buffer(
                "embed_ema", self.embedding.weight.detach().clone()
            )


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


    @torch.no_grad()
    def _tile_with_noise(
        self, x: torch.Tensor, target_n: int
    ) -> torch.Tensor:
        """Tile input vectors with small noise to reach target_n rows.

        Used when batch size < n_embed and we need enough candidates
        for dead code replacement.

        Args:
            x (Tensor): input vectors, shape (B, D).
            target_n (int): target number of rows.

        Returns:
            Tensor: tiled vectors, shape (>=target_n, D).
        """
        B, embed_dim = x.shape
        n_repeats = (target_n + B - 1) // B
        std = x.new_ones(embed_dim) * 0.01 / np.sqrt(embed_dim)
        x = x.repeat(n_repeats, 1)
        x = x + torch.rand_like(x) * std
        return x

    @torch.no_grad()
    def _update_ema_buffers(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        ema_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """EMA update cluster_size_ema and embed_ema buffers.

        Includes distributed all_reduce synchronization and dead code
        restart when restart_unused_codes=True.

        Args:
            x (Tensor): input vectors, shape (B, D).
            ids (Tensor): assigned codebook indices, shape (B,).
            ema_mask (Tensor, optional): per-row mask, shape (B,) float.
                1.0 = contribute to EMA, 0.0 = skip. Used by mixed
                mode to prevent recon rows from updating path2 EMA.
        """
        n_embed = self.n_embed
        embed_dim = self.embed_dim

        x_flat = x.reshape(-1, embed_dim)
        ids_flat = ids.reshape(-1)

        # Per-row EMA mask: zero out masked rows before accumulation so they
        # contribute neither to cluster_size nor to vectors_sum.
        if ema_mask is not None:
            ema_mask_flat = ema_mask.reshape(-1)
            x_for_sum = x_flat * ema_mask_flat.unsqueeze(1)
            cluster_size = torch.zeros(
                n_embed, dtype=x_flat.dtype, device=x_flat.device
            ).index_add_(0, ids_flat, ema_mask_flat.to(x_flat.dtype))
        else:
            x_for_sum = x_flat
            cluster_size = torch.bincount(
                ids_flat, minlength=n_embed
            ).to(x_flat.dtype)

        vectors_sum = torch.zeros(
            n_embed, embed_dim, dtype=x_flat.dtype, device=x_flat.device
        ).index_add_(0, ids_flat, x_for_sum)

        # One coalesced all_reduce instead of two: pack cluster_size as a
        # final extra column on vectors_sum, reduce, then split back.
        if dist.is_initialized():
            packed = torch.cat([vectors_sum, cluster_size.unsqueeze(1)], dim=1)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            vectors_sum = packed[:, :embed_dim]
            cluster_size = packed[:, embed_dim]

        # EMA decay update
        self.cluster_size_ema.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self.embed_ema.mul_(self.ema_decay).add_(
            vectors_sum, alpha=1 - self.ema_decay
        )

        # Restart unused codes: only when there is something to restart.
        # Skipping the randperm + broadcast every step is the common case.
        if self.restart_unused_codes and (self.cluster_size_ema < 1).any():
            n_vectors = x_flat.shape[0]
            vectors_for_restart = x_flat
            if n_vectors < n_embed:
                vectors_for_restart = self._tile_with_noise(
                    x_flat, n_embed
                )
            n_avail = vectors_for_restart.shape[0]
            random_vectors = vectors_for_restart[
                torch.randperm(n_avail, device=x.device)
            ][:n_embed]

            # Broadcast from rank 0 for consistency across GPUs
            if dist.is_initialized():
                dist.broadcast(random_vectors, 0)

            usage = (self.cluster_size_ema.view(-1, 1) >= 1).float()
            self.embed_ema.mul_(usage).add_(random_vectors * (1 - usage))
            self.cluster_size_ema.mul_(usage.view(-1))
            self.cluster_size_ema.add_(
                torch.ones_like(self.cluster_size_ema) * (1 - usage).view(-1)
            )

    @torch.no_grad()
    def _update_embedding_from_ema(self) -> None:
        """Refresh codebook weights from EMA buffers.

        Uses Laplace smoothing to avoid division by zero:
            weight = embed_ema / ((n * (c_ema + eps)) / (n + K * eps))
        """
        n = self.cluster_size_ema.sum()
        normalized_cluster_size = (
            n * (self.cluster_size_ema + self.eps)
            / (n + self.n_embed * self.eps)
        )
        self.embedding.weight.data.copy_(
            self.embed_ema / normalized_cluster_size.reshape(-1, 1)
        )


    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        reference_code: Optional[torch.Tensor] = None,
        ema_mask: Optional[torch.Tensor] = None,
    ) -> QuantizeOutput:
        """Forward the vector quantization layer.

        Training flow:
            1. compute distances (L2 or cosine)
            2. if use_sinkhorn: z-score normalize + Sinkhorn -> argmax
               else: argmin
            3. if reference_code: replace ids with reference at prob 0.04
            4. if use_ema: update EMA buffers + refresh embedding weights
            5. compute differentiable embedding (STE or Gumbel-Softmax)
            6. compute commitment loss

        Args:
            x (Tensor): input vectors, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.
            reference_code (Tensor, optional): reference codebook indices,
                shape (B,). If provided, randomly replace assigned ids
                with reference codes at probability 0.04.
            ema_mask (Tensor, optional): per-row EMA mask, shape (B,)
                float. 1.0 = contribute to EMA update, 0.0 = skip.
                Used by mixed mode path2 to exclude recon rows.

        Returns:
            QuantizeOutput: named tuple of (embeddings, ids, loss).
        """
        # Step 1-2: find nearest codebook entry
        ids, distances = self._find_nearest_embedding(x)

        # Step 3: reference_code probabilistic replacement
        if reference_code is not None:
            p = 0.04
            mask = torch.rand(ids.size(0), device=x.device) < p
            ids = torch.where(mask, reference_code, ids)

        # Step 4: EMA codebook update (training only)
        if self.training and self.use_ema:
            self._update_ema_buffers(x, ids, ema_mask)

        # Step 5: compute differentiable embedding
        # Single embedding lookup feeds both the differentiable output and
        # the commitment loss (Gumbel takes a different path entirely).
        if self.training and self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
            weights = _gumbel_softmax_sample(
                -distances, temperature=temperature, hard=True
            )
            emb = weights @ self.embedding.weight
            quantized_for_loss = self.embedding(ids)
        elif self.training and self.forward_mode == QuantizeForwardMode.STE:
            quantized_for_loss = self.embedding(ids)
            # Straight-Through Estimator: gradient passes through
            emb = x + (quantized_for_loss - x).detach()
        elif self.training:
            raise ValueError(
                f"Unsupported forward mode: {self.forward_mode}"
            )
        else:
            quantized_for_loss = self.embedding(ids)
            emb = quantized_for_loss

        # Step 4 continued: refresh weights from EMA (after embedding lookup)
        if self.training and self.use_ema:
            self._update_embedding_from_ema()

        # Step 6: commitment loss
        e_latent_loss = F.mse_loss(x, quantized_for_loss.detach())
        if self.use_ema:
            # EMA mode: codebook not updated via gradient, no q_latent_loss
            loss = self.commitment_weight * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(quantized_for_loss, x.detach())
            loss = self.commitment_weight * e_latent_loss + q_latent_loss

        return QuantizeOutput(embeddings=emb, ids=ids, loss=loss)
