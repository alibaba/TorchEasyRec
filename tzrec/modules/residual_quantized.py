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

from enum import Enum
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class QuantizeForwardMode(Enum):
    """Forward mode for vector quantization.

    Attributes:
        GUMBEL_SOFTMAX: use Gumbel-Softmax reparameterization.
        STE: use Straight-Through Estimator.
    """

    GUMBEL_SOFTMAX = 1
    STE = 2


class QuantizeOutput(NamedTuple):
    """Output of a single vector quantization layer.

    Attributes:
        embeddings (Tensor): quantized embeddings, shape (B, D).
        ids (Tensor): codebook indices, shape (B,).
        loss (Tensor): commitment loss scalar.
    """

    embeddings: torch.Tensor
    ids: torch.Tensor
    loss: torch.Tensor


class ResidualQuantizedOutput(NamedTuple):
    """Output of the residual quantization module.

    Attributes:
        cluster_ids (Tensor): codebook indices per layer, shape (B, n_layers).
        quantized_embeddings (Tensor): sum of quantized embeddings, shape (B, D).
        quantization_loss (Tensor): total commitment loss scalar.
    """

    cluster_ids: torch.Tensor
    quantized_embeddings: torch.Tensor
    quantization_loss: torch.Tensor


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

    Reference: al_sid/SID_generation/rqvae_embed/quantizations.py::sinkhorn

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

    Reference: al_sid/SID_generation/rqvae_embed/quantizations.py::VQEmbedding

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

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------

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
            # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 * x @ e^T
            distances = (
                x.pow(2).sum(dim=1, keepdim=True)
                + codebook.pow(2).sum(dim=1, keepdim=True).t()
                - 2.0 * x @ codebook.t()
            )
        elif self.distance_type == "cosine":
            # Cosine distance: -normalize(x) @ normalize(c)^T
            x_norm = F.normalize(x, p=2, dim=1)
            codebook_norm = F.normalize(codebook, p=2, dim=1)
            distances = -torch.matmul(x_norm, codebook_norm.t())
        else:
            raise ValueError(
                f"Unsupported distance_type '{self.distance_type}', "
                f"choose from ('l2', 'cosine')"
            )
        return distances

    # ------------------------------------------------------------------
    # Codebook assignment (Sinkhorn or argmin)
    # ------------------------------------------------------------------

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
            # z-score normalization + shift to non-negative (critical!)
            distances = (distances - distances.mean()) / (distances.std() + 1e-6)
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

    # ------------------------------------------------------------------
    # EMA update machinery
    # ------------------------------------------------------------------

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
        self, x: torch.Tensor, ids: torch.Tensor
    ) -> None:
        """EMA update cluster_size_ema and embed_ema buffers.

        Includes distributed all_reduce synchronization and dead code
        restart when restart_unused_codes=True.

        Args:
            x (Tensor): input vectors, shape (B, D).
            ids (Tensor): assigned codebook indices, shape (B,).
        """
        n_embed = self.n_embed
        embed_dim = self.embed_dim

        x_flat = x.reshape(-1, embed_dim)
        ids_flat = ids.reshape(-1)
        n_vectors = x_flat.shape[0]

        # One-hot scatter: (n_embed, n_vectors)
        one_hot = x_flat.new_zeros(n_embed, n_vectors)
        one_hot.scatter_(
            dim=0,
            index=ids_flat.unsqueeze(0),
            src=x_flat.new_ones(1, n_vectors),
        )

        cluster_size = one_hot.sum(dim=1)            # (n_embed,)
        vectors_sum = one_hot @ x_flat               # (n_embed, embed_dim)

        # Distributed synchronization
        if dist.is_initialized():
            dist.all_reduce(vectors_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cluster_size, op=dist.ReduceOp.SUM)

        # EMA decay update
        self.cluster_size_ema.mul_(self.ema_decay).add_(
            cluster_size, alpha=1 - self.ema_decay
        )
        self.embed_ema.mul_(self.ema_decay).add_(
            vectors_sum, alpha=1 - self.ema_decay
        )

        # Restart unused codes: replace dead codes with random batch samples
        if self.restart_unused_codes:
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

            # usage mask: 1 = alive, 0 = dead (cluster_size_ema < 1)
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

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        temperature: float = 1.0,
        reference_code: Optional[torch.Tensor] = None,
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
            self._update_ema_buffers(x, ids)

        # Step 5: compute differentiable embedding
        if self.training:
            if self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
                weights = _gumbel_softmax_sample(
                    -distances, temperature=temperature, hard=True
                )
                emb = weights @ self.embedding.weight
            elif self.forward_mode == QuantizeForwardMode.STE:
                emb = self.embedding(ids)
                # Straight-Through Estimator: gradient passes through
                emb = x + (emb - x).detach()
            else:
                raise ValueError(
                    f"Unsupported forward mode: {self.forward_mode}"
                )
        else:
            emb = self.embedding(ids)

        # Step 4 continued: refresh weights from EMA (after embedding lookup)
        if self.training and self.use_ema:
            self._update_embedding_from_ema()

        # Step 6: commitment loss
        quantized_for_loss = self.embedding(ids)
        e_latent_loss = F.mse_loss(x, quantized_for_loss.detach())
        if self.use_ema:
            # EMA mode: codebook not updated via gradient, no q_latent_loss
            loss = self.commitment_weight * e_latent_loss
        else:
            q_latent_loss = F.mse_loss(quantized_for_loss, x.detach())
            loss = self.commitment_weight * e_latent_loss + q_latent_loss

        return QuantizeOutput(embeddings=emb, ids=ids, loss=loss)


# ------------------------------------------------------------------
# Distance helper for KMeans
# ------------------------------------------------------------------


@torch.no_grad()
def _squared_euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Squared L2 distance with chunked computation for memory efficiency.

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::squared_euclidean_distance

    Args:
        x (Tensor): data points, shape (N, D).
        y (Tensor): centroids, shape (K, D).
        chunk_size (int): max rows of x per chunk. Default: 50000.

    Returns:
        Tensor: squared distances, shape (N, K).
    """
    N = x.shape[0]

    if N <= chunk_size:
        x_sq = x.pow(2).sum(dim=1, keepdim=True)
        y_sq = y.pow(2).sum(dim=1, keepdim=True).t()
        return (x_sq + y_sq - 2.0 * x @ y.t()).clamp(min=0.0)

    chunks = []
    for start in range(0, N, chunk_size):
        x_chunk = x[start : start + chunk_size]
        x_sq = x_chunk.pow(2).sum(dim=1, keepdim=True)
        y_sq = y.pow(2).sum(dim=1, keepdim=True).t()
        chunks.append((x_sq + y_sq - 2.0 * x_chunk @ y.t()).clamp(min=0.0))
    return torch.cat(chunks, dim=0)


# ------------------------------------------------------------------
# KMeans++ initialization
# ------------------------------------------------------------------


@torch.no_grad()
def _kmeans_plus_plus(
    data: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """KMeans++ initialization (Arthur & Vassilvitskii 2007).

    Selects initial centroids via distance-weighted probability sampling
    to ensure well-spread starting points.

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::kmeans_plus_plus

    Args:
        data (Tensor): data points, shape (N, D).
        n_clusters (int): number of clusters K.

    Returns:
        Tensor: initial centroids, shape (K, D).
    """
    N, D = data.shape
    centroids = torch.zeros(n_clusters, D, device=data.device, dtype=data.dtype)

    # First centroid: random
    idx = torch.randint(0, N, (1,), device=data.device)
    centroids[0] = data[idx]

    for i in range(1, n_clusters):
        dists = _squared_euclidean_distance(data, centroids[:i])  # (N, i)
        min_dists = dists.min(dim=1)[0]  # (N,)
        if min_dists.sum() == 0:
            # All remaining centroids chosen randomly
            centroids[i:] = data[
                torch.randint(0, N, (n_clusters - i,), device=data.device)
            ]
            break
        next_idx = torch.multinomial(min_dists, num_samples=1)
        centroids[i] = data[next_idx]

    return centroids


# ------------------------------------------------------------------
# MiniBatchKMeans
# ------------------------------------------------------------------


class MiniBatchKMeans(nn.Module):
    """Mini-Batch K-Means single-layer clustering module.

    Online K-Means (Sculley 2010) with KMeans++ initialization.
    Supports distributed training via all-reduce synchronization.

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::MiniBatchKMeans

    Update rule for each cluster c in current batch:
        eta_c = batch_count_c / total_count_c
        centroid_c = (1 - eta_c) * centroid_c + eta_c * batch_mean_c

    Args:
        n_clusters (int): number of clusters (codebook size).
        n_features (int): feature dimension.
        init_buffer_size (int): buffer size for initialization pool.
            After collecting enough samples, initialize centroids via
            KMeans++. Default: 3072.
    """

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
        init_buffer_size: int = 3072,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features
        self.init_buffer_size = max(init_buffer_size, n_clusters)

        # Centroids are manually updated, no gradient needed
        self.register_buffer(
            "centroids", torch.zeros(n_clusters, n_features)
        )
        self.register_buffer("cluster_counts", torch.zeros(n_clusters))
        self.register_buffer("_is_initialized", torch.tensor(False))

        self._init_buffer: List[torch.Tensor] = []

    @property
    def is_initialized(self) -> bool:
        """Whether centroids have been initialized via KMeans++."""
        return self._is_initialized.item()

    # ---- initialization ----

    @torch.no_grad()
    def _buffer_and_maybe_init(self, batch: torch.Tensor) -> bool:
        """Buffer data and initialize when enough is collected.

        Collects incoming batches into an internal buffer. Once the total
        buffered samples reach init_buffer_size, runs KMeans++ on rank 0
        and broadcasts the result to all ranks.

        Args:
            batch (Tensor): data points, shape (B, D).

        Returns:
            bool: True if initialization was performed in this call.
        """
        self._init_buffer.append(batch.detach())
        total = sum(b.shape[0] for b in self._init_buffer)
        if total < self.init_buffer_size:
            return False

        buffer = torch.cat(self._init_buffer, dim=0)[: self.init_buffer_size]

        # Rank 0 does KMeans++, then broadcast to all ranks
        if dist.is_initialized() and dist.get_world_size() > 1:
            if dist.get_rank() == 0:
                init_centroids = _kmeans_plus_plus(buffer, self.n_clusters)
            else:
                init_centroids = torch.zeros_like(self.centroids)
            dist.broadcast(init_centroids, src=0)
        else:
            init_centroids = _kmeans_plus_plus(buffer, self.n_clusters)

        self.centroids.copy_(init_centroids)
        self._is_initialized.fill_(True)
        self.cluster_counts.zero_()
        self._init_buffer = []
        return True

    # ---- predict ----

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest centroid without update.

        Args:
            batch (Tensor): data points, shape (B, D).

        Returns:
            Tensor: cluster indices, shape (B,).
        """
        dists = _squared_euclidean_distance(batch, self.centroids)
        return torch.argmin(dists, dim=-1)

    # ---- train step ----

    @torch.no_grad()
    def train_step(
        self, batch: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """One Mini-Batch K-Means update step.

        Assigns each data point to the nearest centroid, then updates
        centroids using the online update rule:
            eta = batch_count / total_count
            centroid = (1 - eta) * old + eta * batch_mean

        Args:
            batch (Tensor): data points, shape (B, D).

        Returns:
            assignments (Tensor): cluster indices, shape (B,).
            embeddings (Tensor): centroid vectors for assigned clusters,
                shape (B, D).
        """
        batch = batch.detach()

        if not self.is_initialized:
            initialized = self._buffer_and_maybe_init(batch)
            if not initialized:
                dummy = torch.zeros(
                    batch.shape[0], dtype=torch.long, device=batch.device
                )
                return dummy, torch.zeros_like(batch)

        # Assign to nearest centroid
        assignments = self.predict(batch)

        # Accumulate per-cluster statistics
        one_hot = F.one_hot(assignments, self.n_clusters).float()  # (B, K)
        batch_counts = one_hot.sum(dim=0)  # (K,)
        batch_sums = one_hot.t() @ batch  # (K, D)

        # Distributed synchronization
        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(batch_counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(batch_sums, op=dist.ReduceOp.SUM)

        # Update centroids
        self.cluster_counts += batch_counts
        mask = batch_counts > 0
        if mask.any():
            batch_means = batch_sums[mask] / batch_counts[mask].unsqueeze(1)
            eta = (
                batch_counts[mask] / self.cluster_counts[mask]
            ).unsqueeze(1)
            self.centroids[mask] = (
                self.centroids[mask] * (1.0 - eta) + batch_means * eta
            )

        embeddings = self.centroids[assignments]
        return assignments, embeddings


# ------------------------------------------------------------------
# KMeans helper functions for codebook initialization
# ------------------------------------------------------------------


@torch.no_grad()
def _kmeans(
    samples: torch.Tensor,
    n_clusters: int,
    n_iters: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's K-Means algorithm with KMeans++ initialization.

    Reference: al_sid/SID_generation/utils/kmeans.py::kmeans

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters (int): number of clusters K.
        n_iters (int): number of iterations. Default: 100.

    Returns:
        centroids (Tensor): cluster centers, shape (K, D).
        assignments (Tensor): cluster indices, shape (N,).
    """
    N, D = samples.shape
    centroids = _kmeans_plus_plus(samples, n_clusters)

    for _ in range(n_iters):
        dists = _squared_euclidean_distance(samples, centroids)  # (N, K)
        assignments = dists.argmin(dim=-1)  # (N,)

        bins = torch.bincount(assignments, minlength=n_clusters)
        zero_mask = bins == 0
        bins_clamped = bins.masked_fill(zero_mask, 1)

        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(
            0, assignments.unsqueeze(1).expand(-1, D), samples
        )
        new_centroids = new_centroids / bins_clamped.unsqueeze(1)

        # Keep old centroids for empty clusters
        centroids = torch.where(
            zero_mask.unsqueeze(1), centroids, new_centroids
        )

    return centroids, assignments


@torch.no_grad()
def _residual_kmeans(
    samples: torch.Tensor,
    n_clusters_list: List[int],
    n_iters: int = 100,
) -> List[torch.Tensor]:
    """Residual K-Means: sequentially cluster and subtract centroids.

    Reference: al_sid/SID_generation/utils/kmeans.py::residual_kmeans

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters_list (List[int]): per-layer cluster counts.
        n_iters (int): K-Means iterations. Default: 100.

    Returns:
        List[Tensor]: per-layer centroids [(K0, D), (K1, D), ...].
    """
    res_centers = []
    for n_clusters in n_clusters_list:
        centroids, assignments = _kmeans(samples, n_clusters, n_iters)
        res_centers.append(centroids)
        samples = samples - centroids[assignments]
    return res_centers


class ResidualQuantized(nn.Module):
    """Multi-layer residual vector quantization.

    Each layer quantizes the residual from the previous layer:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = quantize(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Reference: al_sid/SID_generation/rqvae_embed/quantizations.py
        ::RQBottleneck

    Args:
        embed_dim (int): dimension of input embeddings.
        n_layers (int): number of quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        forward_mode (str): VQ forward mode ('ste'|'gumbel_softmax').
            Default: 'ste'.
        normalize_residuals (bool): L2-normalize residuals before each
            quantization layer. Default: False.
        shared_codebook (bool): share codebook across all layers.
            Default: False.
        distance_type (str|List[str]): distance metric per layer,
            'l2' or 'cosine'. Supports per-layer list. Default: 'l2'.
        commitment_loss (str): commitment loss type, 'l2' or 'cos'.
            Default: 'l2'.
        latent_weight (List[float]): commitment loss weights [w1, w2].
            w1: x toward quant (always active).
            w2: quant toward x (only when use_ema=False).
            Default: [1.0, 0.5].
        rotation_trick (bool): use rotation trick for improved STE
            gradient estimation (arXiv:2410.06424). Default: False.
        kmeans_init (bool): use residual K-Means codebook initialization
            on first forward. Default: False.
        use_ema (bool): EMA codebook update. Default: True.
        ema_decay (float): EMA decay coefficient. Default: 0.99.
        restart_unused_codes (bool): reset dead codes. Default: True.
        use_sinkhorn (bool): Sinkhorn uniform assignment. Default: True.
        sinkhorn_iters (int): Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness. Default: 10.0.
    """

    _FORWARD_MODE_MAP = {
        "gumbel_softmax": QuantizeForwardMode.GUMBEL_SOFTMAX,
        "ste": QuantizeForwardMode.STE,
    }

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        forward_mode: str = "ste",
        normalize_residuals: bool = False,
        shared_codebook: bool = False,
        distance_type: Union[str, List[str]] = "l2",
        commitment_loss: str = "l2",
        latent_weight: Optional[List[float]] = None,
        rotation_trick: bool = False,
        kmeans_init: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.shared_codebook = shared_codebook
        self.commitment_loss_type = commitment_loss
        self.use_ema = use_ema
        self.rotation_trick = rotation_trick

        if latent_weight is None:
            latent_weight = [1.0, 0.5]
        self.commitment_w1, self.commitment_w2 = latent_weight

        # KMeans initialization control
        self.register_buffer("initted", torch.tensor([not kmeans_init]))

        if forward_mode not in self._FORWARD_MODE_MAP:
            raise ValueError(
                f"Unsupported forward_mode '{forward_mode}', "
                f"choose from {list(self._FORWARD_MODE_MAP.keys())}"
            )
        mode_enum = self._FORWARD_MODE_MAP[forward_mode]

        # Parse n_embed list
        if isinstance(n_embed, int):
            n_embed_list = [n_embed] * n_layers
        else:
            assert len(n_embed) == n_layers, (
                "length of n_embed and n_layers must be same, "
                f"but got {len(n_embed)} vs {n_layers}"
            )
            n_embed_list = list(n_embed)
        self.n_embed_list = n_embed_list

        # Parse distance_type list
        if isinstance(distance_type, str):
            distance_types = [distance_type] * n_layers
        else:
            assert len(distance_type) == n_layers, (
                "length of distance_type and n_layers must be same, "
                f"but got {len(distance_type)} vs {n_layers}"
            )
            distance_types = list(distance_type)

        # Build VQ layers
        if shared_codebook:
            base_layer = VectorQuantize(
                embed_dim=embed_dim,
                n_embed=n_embed_list[0],
                forward_mode=mode_enum,
                distance_type=distance_types[0],
                use_ema=use_ema,
                ema_decay=ema_decay,
                restart_unused_codes=restart_unused_codes,
                use_sinkhorn=use_sinkhorn,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_epsilon=sinkhorn_epsilon,
            )
            self.layers = nn.ModuleList([base_layer] * n_layers)
        else:
            self.layers = nn.ModuleList(
                [
                    VectorQuantize(
                        embed_dim=embed_dim,
                        n_embed=n_embed_list[i],
                        forward_mode=mode_enum,
                        distance_type=distance_types[i],
                        use_ema=use_ema,
                        ema_decay=ema_decay,
                        restart_unused_codes=restart_unused_codes,
                        use_sinkhorn=use_sinkhorn,
                        sinkhorn_iters=sinkhorn_iters,
                        sinkhorn_epsilon=sinkhorn_epsilon,
                    )
                    for i in range(n_layers)
                ]
            )

    # ------------------------------------------------------------------
    # KMeans initialization
    # ------------------------------------------------------------------

    @torch.jit.ignore
    @torch.no_grad()
    def init_embed_(self, data: torch.Tensor) -> None:
        """Initialize codebook weights via residual K-Means.

        Only executed once when kmeans_init=True and not yet initialized.
        Uses the first batch of training data as initialization pool.

        Args:
            data (Tensor): input data, shape (B, D).
        """
        if self.initted:
            return

        centers = _residual_kmeans(data, self.n_embed_list)

        # Distributed sync
        if dist.is_initialized() and dist.get_world_size() > 1:
            for c in centers:
                dist.all_reduce(c, op=dist.ReduceOp.SUM)
                c /= dist.get_world_size()

        for i, layer in enumerate(self.layers):
            layer.embedding.weight.data.copy_(centers[i])

        self.initted.fill_(True)

    # ------------------------------------------------------------------
    # Commitment loss
    # ------------------------------------------------------------------

    def _compute_commitment_loss(
        self,
        x: torch.Tensor,
        quant_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """Compute top-level commitment loss over all layers.

        For each cumulative quantization in quant_list:
          - cos: (1 - cosine_similarity) * weight
          - l2:  (x - quant)^2.mean() * weight
        EMA mode zeros out the codebook-toward-encoder term.

        Reference: al_sid::RQBottleneck.compute_commitment_loss

        Args:
            x (Tensor): original input, shape (B, D).
            quant_list (List[Tensor]): cumulative quantized outputs
                per layer.

        Returns:
            Tensor: scalar commitment loss.
        """
        loss_list = []
        for quant in quant_list:
            if self.commitment_loss_type == "cos":
                loss1 = (
                    (1 - F.cosine_similarity(x, quant.detach(), dim=-1))
                    .mean()
                    * self.commitment_w1
                )
                if self.use_ema:
                    loss2 = torch.tensor(0.0, device=x.device)
                else:
                    loss2 = (
                        (1 - F.cosine_similarity(
                            x.detach(), quant, dim=-1
                        ))
                        .mean()
                        * self.commitment_w2
                    )
            else:  # 'l2'
                loss1 = (
                    (x - quant.detach()).pow(2.0).mean()
                    * self.commitment_w1
                )
                if self.use_ema:
                    loss2 = torch.tensor(0.0, device=x.device)
                else:
                    loss2 = (
                        (x.detach() - quant).pow(2.0).mean()
                        * self.commitment_w2
                    )
            loss_list.append(loss1 + loss2)
        return torch.mean(torch.stack(loss_list))

    # ------------------------------------------------------------------
    # Rotation trick
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rotation_trick(
        x: torch.Tensor,
        quant: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation trick for improved STE gradient estimation.

        Implements equation 4.2 from https://arxiv.org/abs/2410.06424.
        Replaces standard STE with a Householder reflection that rotates
        the gradient direction from x toward quant.

        Reference: GRID/src/components/quantization_strategies.py
            ::RotationTrickQuantization.rotate_and_scale_batch

        Args:
            x (Tensor): original input with gradient, shape (B, D).
            quant (Tensor): quantized output (will be detached),
                shape (B, D).

        Returns:
            Tensor: rotated output with gradient flowing through x.
        """
        quant_detached = quant.detach()
        x_detached = x.detach()

        quant_norms = torch.linalg.vector_norm(
            quant_detached, dim=-1
        ).unsqueeze(1)  # (B, 1)
        x_norms = torch.linalg.vector_norm(
            x_detached, dim=-1
        ).unsqueeze(1)  # (B, 1)
        lambda_ = quant_norms / (x_norms + 1e-8)  # (B, 1)

        x_hat = x_detached / (x_norms + 1e-8)  # (B, D)
        quant_hat = quant_detached / (quant_norms + 1e-8)  # (B, D)

        normalized_sum = F.normalize(
            x_hat + quant_hat, p=2, dim=1
        )  # (B, D)

        x_unsq = x.unsqueeze(1)  # (B, 1, D)

        # Eq 4.2: Householder reflection
        sum_projection = (
            x_unsq
            @ normalized_sum.unsqueeze(2)
            @ normalized_sum.unsqueeze(1)
        )  # (B, 1, D)
        rescaled_embeddings = (
            x_unsq
            @ x_hat.unsqueeze(2)
            @ quant_hat.unsqueeze(1)
        )  # (B, 1, D)
        return (
            lambda_
            * (x_unsq - 2 * sum_projection + 2 * rescaled_embeddings)
            .squeeze(1)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    def forward(
        self,
        input: torch.Tensor,
        temperature: float = 1.0,
        reference_code: Optional[torch.Tensor] = None,
    ) -> ResidualQuantizedOutput:
        """Forward the multi-layer residual quantization.

        Training flow:
            1. If kmeans_init and not initialized -> init_embed_(input)
            2. For each layer: quantize detached residual, collect quant_list
               - pass reference_code[:, i] if provided
            3. Compute commitment loss (cos/l2 with latent_weight)
            4. STE gradient pass-through (or rotation trick)

        Args:
            input (Tensor): input embeddings, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.
            reference_code (Tensor, optional): reference codebook indices,
                shape (B, n_layers). If provided, each layer receives
                reference_code[:, i] for probabilistic replacement.

        Returns:
            ResidualQuantizedOutput: (cluster_ids, quantized_embeddings,
                quantization_loss).
        """
        # Step 1: KMeans initialization (first training forward only)
        if self.training:
            self.init_embed_(input)

        # Detach residual for VQ assignment (gradient flows via STE only)
        residual = input.detach().clone()
        all_ids: List[torch.Tensor] = []
        quant_list: List[torch.Tensor] = []
        aggregated_quants = torch.zeros_like(input)

        # Step 2: per-layer residual quantization
        for i, layer in enumerate(self.layers):
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            ref_code_i = (
                reference_code[:, i]
                if reference_code is not None
                else None
            )

            # VQ forward: assignment + EMA update (internally)
            quantized = layer(
                residual, temperature=temperature,
                reference_code=ref_code_i,
            )
            all_ids.append(quantized.ids)

            # Raw embedding lookup for commitment loss accumulation
            raw_emb = layer.embedding(quantized.ids)

            # Update residual with detached embedding
            residual = residual - raw_emb.detach()

            # Accumulate raw embeddings (preserves gradient to codebook)
            aggregated_quants = aggregated_quants + raw_emb
            quant_list.append(aggregated_quants.clone())

        cluster_ids = torch.stack(all_ids, dim=-1)  # (B, n_layers)

        # Step 3: commitment loss (top-level, over cumulative quant_list)
        commitment_loss = self._compute_commitment_loss(input, quant_list)

        # Step 4: STE or rotation trick
        quants_trunc = quant_list[-1]
        if self.training:
            if self.rotation_trick:
                quants_trunc = self._apply_rotation_trick(
                    input, quants_trunc
                )
            else:
                quants_trunc = input + (quants_trunc - input).detach()

        return ResidualQuantizedOutput(
            cluster_ids=cluster_ids,
            quantized_embeddings=quants_trunc,
            quantization_loss=commitment_loss,
        )

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without gradient computation.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        output = self.forward(input)
        return output.cluster_ids

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get codebook embedding weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].embedding.weight.data

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes.

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, D).
        """
        quantized_sum = torch.zeros(
            codes.shape[0], self.embed_dim,
            device=codes.device, dtype=torch.float,
        )
        for i, layer in enumerate(self.layers):
            emb = layer.embedding(codes[:, i])
            quantized_sum = quantized_sum + emb
        return quantized_sum


class ResidualKMeans(nn.Module):
    """Multi-layer residual Mini-Batch K-Means.

    Each layer quantizes the residual from the previous layer using
    MiniBatchKMeans online clustering:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i.train_step(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::ResidualMiniBatchKMeans

    Args:
        embed_dim (int): feature dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): number of clusters per layer.
            Default: 256.
        normalize_residuals (bool): whether to L2-normalize residuals
            before each layer. Default: False.
        init_buffer_size (int): buffer size for KMeans++ initialization.
            Default: 3072.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        init_buffer_size: int = 3072,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals

        if isinstance(n_embed, int):
            n_embed_list = [n_embed] * n_layers
        else:
            assert len(n_embed) == n_layers, (
                "length of n_embed and n_layers must be same, "
                f"but got {len(n_embed)} vs {n_layers}"
            )
            n_embed_list = list(n_embed)
        self.n_embed_list = n_embed_list

        self.layers = nn.ModuleList(
            [
                MiniBatchKMeans(
                    n_clusters=n_embed_list[i],
                    n_features=embed_dim,
                    init_buffer_size=init_buffer_size,
                )
                for i in range(n_layers)
            ]
        )

    @property
    def all_initialized(self) -> bool:
        """Whether all layers have been initialized via KMeans++."""
        return all(layer.is_initialized for layer in self.layers)

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the multi-layer residual K-Means.

        During training: calls layer.train_step() to update centroids.
        During inference: calls layer.predict() without update.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            codes (Tensor): cluster indices per layer,
                shape (B, n_layers).
            quantized (Tensor): sum of quantized embeddings,
                shape (B, D).
        """
        residual = input
        all_codes: List[torch.Tensor] = []
        quantized_sum = torch.zeros_like(input)

        for layer in self.layers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            if self.training:
                codes, quantized = layer.train_step(residual)
            else:
                codes = layer.predict(residual)
                quantized = layer.centroids[codes]

            all_codes.append(codes)

            # Only update residual if layer is initialized
            # (uninitialized layers return dummy zeros)
            if layer.is_initialized:
                residual = residual - quantized
                quantized_sum = quantized_sum + quantized

        cluster_ids = torch.stack(all_codes, dim=-1)  # (B, n_layers)
        return cluster_ids, quantized_sum

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating centroids.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        residual = input
        all_codes: List[torch.Tensor] = []

        for layer in self.layers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            codes = layer.predict(residual)
            all_codes.append(codes)
            quantized = layer.centroids[codes]
            residual = residual - quantized

        return torch.stack(all_codes, dim=-1)

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get centroid weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: centroids, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].centroids

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes.

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, D).
        """
        quantized_sum = torch.zeros(
            codes.shape[0], self.embed_dim,
            device=codes.device, dtype=torch.float,
        )
        for i, layer in enumerate(self.layers):
            emb = layer.centroids[codes[:, i]]
            quantized_sum = quantized_sum + emb
        return quantized_sum


# ------------------------------------------------------------------
# Distributed gather with gradient support
# ------------------------------------------------------------------


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all workers with gradient support.

    Standard ``dist.all_gather`` detaches gradients; this custom
    ``autograd.Function`` keeps the computation graph connected so
    that contrastive losses can backpropagate through gathered tensors.

    Reference: al_sid/SID_generation/utils/dist_utils.py::GatherLayer
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def _all_gather_with_grad(
    tensors: List[torch.Tensor],
) -> List[torch.Tensor]:
    """All-gather tensors across distributed workers with gradient support.

    In single-process mode, returns input tensors unchanged.
    In multi-process mode, uses GatherLayer for backward-compatible
    all_gather.

    Reference: al_sid/SID_generation/utils/dist_utils.py
        ::all_gather_batch_with_grad

    Args:
        tensors (List[Tensor]): list of tensors to gather.

    Returns:
        List[Tensor]: gathered tensors, each (world_size * B, ...).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensors

    gathered: List[torch.Tensor] = []
    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        gathered.append(torch.cat(tensor_all, dim=0))
    return gathered


# ------------------------------------------------------------------
# CLIPLoss
# ------------------------------------------------------------------


class CLIPLoss(nn.Module):
    """Multi-level CLIP contrastive learning loss.

    Computes three InfoNCE contrastive losses and returns their mean:
    - loss_self:  quantized features vs quantized features
                  (paired items remain similar after quantization)
    - loss_ori:   quantized features vs original features
                  (quantization preserves original semantics)
    - loss_cl:    quantized features vs counterpart original features
                  (cross-modal alignment)

    Supports distributed all_gather to aggregate global batch.

    Reference: al_sid/SID_generation/rqvae_embed/rqvae_clip.py::CLIPLoss

    Input dict keys:
        'image_embed':      (B, D)  quantized output of first feature
        'text_embed':       (B, D)  quantized output of second feature
        'image_embed_ori':  (B, D)  original embedding of first feature
        'text_embed_ori':   (B, D)  original embedding of second feature
        'logit_scale_self': scalar  self-contrast temperature
        'logit_scale_cl':   scalar  cross-modal contrast temperature
        'logit_scale':      scalar  original feature contrast temperature

    Output dict keys:
        'clip_loss':  scalar  mean of three losses
        'loss_self':  scalar
        'loss_ori':   scalar
        'loss_cl':    scalar
        'clip_acc':   scalar  contrast accuracy (%)
    """

    def __init__(self) -> None:
        super().__init__()
        self.labels: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None

    def forward(
        self, outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-level CLIP contrastive loss.

        Args:
            outputs (Dict[str, Tensor]): feature dict, see class docstring.

        Returns:
            Dict[str, Tensor]: losses and accuracy.
        """
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        image_embed_ori = outputs["image_embed_ori"]
        text_embed_ori = outputs["text_embed_ori"]
        logit_scale = outputs["logit_scale"]
        logit_scale_self = outputs["logit_scale_self"]
        logit_scale_cl = outputs["logit_scale_cl"]

        local_batch_size = image_embed.size(0)

        # Update labels when batch size changes (multi-GPU offset)
        if local_batch_size != self.last_local_batch_size:
            rank = dist.get_rank() if dist.is_initialized() else 0
            self.labels = local_batch_size * rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # L2 normalize quantized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # All-gather across GPUs (with gradient support)
        image_embed_all, text_embed_all = _all_gather_with_grad(
            [image_embed, text_embed]
        )
        image_embed_all_ori, text_embed_all_ori = _all_gather_with_grad(
            [image_embed_ori, text_embed_ori]
        )

        # --- loss_self: quantized vs quantized ---
        logits_img_self = logit_scale_self * image_embed @ text_embed_all.t()
        logits_txt_self = logit_scale_self * text_embed @ image_embed_all.t()

        # --- loss_ori: quantized vs original ---
        logits_img_ori = logit_scale * image_embed @ text_embed_all_ori.t()
        logits_txt_ori = logit_scale * text_embed @ image_embed_all_ori.t()

        # --- loss_cl: quantized vs counterpart original ---
        logits_img_cl = logit_scale_cl * image_embed @ image_embed_all_ori.t()
        logits_txt_cl = logit_scale_cl * text_embed @ text_embed_all_ori.t()

        loss_self = (
            F.cross_entropy(logits_img_self, self.labels)
            + F.cross_entropy(logits_txt_self, self.labels)
        ) / 2
        loss_ori = (
            F.cross_entropy(logits_img_ori, self.labels)
            + F.cross_entropy(logits_txt_ori, self.labels)
        ) / 2
        loss_cl = (
            F.cross_entropy(logits_img_cl, self.labels)
            + F.cross_entropy(logits_txt_cl, self.labels)
        ) / 2

        loss = (loss_self + loss_ori + loss_cl) / 3

        # Compute accuracy
        with torch.no_grad():
            pred1 = torch.argmax(logits_img_self, dim=-1)
            correct1 = pred1.eq(self.labels).sum()
            pred2 = torch.argmax(logits_txt_self, dim=-1)
            correct2 = pred2.eq(self.labels).sum()
            pred3 = torch.argmax(logits_img_ori, dim=-1)
            correct3 = pred3.eq(self.labels).sum()
            pred4 = torch.argmax(logits_txt_ori, dim=-1)
            correct4 = pred4.eq(self.labels).sum()
            acc = (
                100
                * (correct1 + correct2 + correct3 + correct4)
                / local_batch_size
                / 4
            )

        return {
            "clip_loss": loss,
            "loss_self": loss_self,
            "loss_ori": loss_ori,
            "loss_cl": loss_cl,
            "clip_acc": acc,
        }


class RQVAE(nn.Module):
    """RQ-VAE: Encoder + ResidualQuantized + Decoder.

    Supports optional CLIP contrastive learning. When use_clip=True,
    forward accepts paired inputs (fea1, fea2) and computes CLIP loss
    via a siamese network (shared parameters).

    Encoder/Decoder are configurable-depth MLPs built via hidden_dims:
        Encoder: input_dim → hidden_dims[0] → ... → hidden_dims[-1] → embed_dim
        Decoder: embed_dim → hidden_dims[-1] → ... → hidden_dims[0] → input_dim
    ReLU activation between hidden layers. Decoder reverses hidden_dims
    for symmetric structure.

    Reference: al_sid/SID_generation/rqvae_embed/rqvae.py → RQVAE_EMBED
               al_sid/SID_generation/rqvae_embed/rqvae_clip.py → RQVAE_EMBED_CLIP

    Args:
        input_dim (int): original embedding dimension. Default: 512.
        embed_dim (int): latent space dimension. Default: 64.
        hidden_dims (List[int]): encoder hidden layer dimensions.
            Decoder automatically reverses for symmetry.
            Default: [input_dim // 2].
        n_layers (int): number of residual quantization layers. Default: 3.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        forward_mode (str): VQ forward mode ('ste'|'gumbel_softmax').
            Default: 'ste'.
        normalize_residuals (bool): L2-normalize residuals. Default: False.
        shared_codebook (bool): share codebook across layers. Default: False.
        distance_type (str|List[str]): distance metric ('l2'|'cosine').
            Default: 'l2'.
        commitment_loss (str|None): commitment loss type ('l2'|'cos').
            Default: follows loss_type (al_sid behavior).
        latent_weight (List[float]): commitment loss weights [w1, w2].
            Default: [1.0, 0.5].
        rotation_trick (bool): STE rotation trick. Default: False.
        kmeans_init (bool): KMeans codebook initialization. Default: True.
        use_ema (bool): EMA codebook update. Default: True.
        ema_decay (float): EMA decay coefficient. Default: 0.99.
        restart_unused_codes (bool): reset dead codes. Default: True.
        use_sinkhorn (bool): Sinkhorn uniform assignment. Default: True.
        sinkhorn_iters (int): Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness. Default: 10.0.
        loss_type (str): reconstruction loss ('mse'|'l1'|'cosine').
            Default: 'mse'.
        use_clip (bool): enable CLIP contrastive learning. Default: False.
    """

    @staticmethod
    def _build_mlp(dims: List[int]) -> nn.Sequential:
        """Build MLP: dims[0] → ... → dims[-1], ReLU between hidden layers."""
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation after last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        n_layers: int = 3,
        n_embed: Union[int, List[int]] = 256,
        forward_mode: str = "ste",
        normalize_residuals: bool = False,
        shared_codebook: bool = False,
        distance_type: Union[str, List[str]] = "l2",
        commitment_loss: Optional[str] = None,
        latent_weight: Optional[List[float]] = None,
        rotation_trick: bool = False,
        kmeans_init: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
        loss_type: str = "mse",
        use_clip: bool = False,
    ) -> None:
        super().__init__()

        assert loss_type in ("mse", "l1", "cosine"), (
            f"loss_type must be 'mse', 'l1' or 'cosine', got '{loss_type}'"
        )
        self.loss_type = loss_type
        self.use_clip = use_clip
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Default hidden_dims
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]

        # commitment_loss defaults to follow loss_type (al_sid behavior:
        # commitment_loss=loss_type, so mse → l2 branch)
        if commitment_loss is None:
            commitment_loss = "l2" if loss_type == "mse" else loss_type

        # Encoder: input_dim → hidden_dims → embed_dim
        enc_dims = [input_dim] + list(hidden_dims) + [embed_dim]
        self.encoder = self._build_mlp(enc_dims)

        # Decoder: embed_dim → reversed(hidden_dims) → input_dim (symmetric)
        dec_dims = [embed_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder = self._build_mlp(dec_dims)

        # Quantizer
        self.quantizer = ResidualQuantized(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_embed=n_embed,
            forward_mode=forward_mode,
            normalize_residuals=normalize_residuals,
            shared_codebook=shared_codebook,
            distance_type=distance_type,
            commitment_loss=commitment_loss,
            latent_weight=latent_weight,
            rotation_trick=rotation_trick,
            kmeans_init=kmeans_init,
            use_ema=use_ema,
            ema_decay=ema_decay,
            restart_unused_codes=restart_unused_codes,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )

        # CLIP contrastive learning (optional)
        self.use_clip = use_clip
        if use_clip:
            self.logit_scale_self = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.logit_scale_cl = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.clip_loss_fn = CLIPLoss()

    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode. (B, input_dim) → (B, embed_dim)."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode. (B, embed_dim) → (B, input_dim)."""
        return self.decoder(z_q)

    def _cosine_loss(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """Cosine distance loss: 1 - mean(cos_sim)."""
        return (1 - F.cosine_similarity(x1, x2, dim=1)).mean()

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss + quantization loss + total loss.

        loss_total = recon_loss + quant_loss
        Note: al_sid latent_loss_weight is declared but unused;
        commitment_loss is added 1:1 with recon_loss. We align with this.

        Args:
            x: original input, shape (B, input_dim).
            x_hat: reconstructed output, shape (B, input_dim).
            quant_loss: quantization (commitment) loss scalar.

        Returns:
            dict with 'reconstruction_loss', 'quantization_loss', 'loss'.
        """
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        elif self.loss_type == "l1":
            recon_loss = F.l1_loss(x_hat, x, reduction="mean")
        elif self.loss_type == "cosine":
            recon_loss = self._cosine_loss(x_hat, x)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        loss_total = recon_loss + quant_loss

        return {
            "reconstruction_loss": recon_loss,
            "quantization_loss": quant_loss,
            "loss": loss_total,
        }

    # ------------------------------------------------------------------
    # Forward interfaces
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Dispatch based on use_clip.

        use_clip=False: forward(x) → forward_rqvae(x)
        use_clip=True:  forward(fea1, fea2) → forward_clip(fea1, fea2)
        """
        if self.use_clip:
            assert len(args) == 2, "CLIP mode requires (fea1, fea2)"
            return self.forward_clip(args[0], args[1], **kwargs)
        else:
            assert len(args) == 1, "Standard mode requires (x,)"
            return self.forward_rqvae(args[0], **kwargs)

    def forward_rqvae(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE forward: encode → quantize → decode → loss.

        Args:
            x: (B, input_dim) original embedding.
            temperature: Gumbel-Softmax temperature.

        Returns:
            dict with keys: 'x_hat', 'codes', 'quantized',
                'reconstruction_loss', 'quantization_loss', 'loss'.
        """
        z_e = self.encode(x)
        quant_output = self.quantizer(z_e, temperature=temperature)
        x_hat = self.decode(quant_output.quantized_embeddings)

        losses = self.compute_loss(
            x, x_hat, quant_output.quantization_loss
        )

        return {
            "x_hat": x_hat,
            "codes": quant_output.cluster_ids,
            "quantized": quant_output.quantized_embeddings,
            **losses,
        }

    def forward_clip(
        self,
        fea1: torch.Tensor,
        fea2: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Siamese RQ-VAE + CLIP contrastive learning.

        fea1, fea2 go through the same RQVAE (shared params),
        then compute CLIP loss + average commitment loss.

        Note: al_sid forward_clip does NOT use reference_code.
        fea1 and fea2 are independently quantized.

        Backward loss = clip_loss + commitment_loss
        recon_loss, pair_code_loss are logged only (no backprop).

        Args:
            fea1: (B, input_dim) first feature.
            fea2: (B, input_dim) second feature (same item, diff modal).
            temperature: Gumbel-Softmax temperature.

        Returns:
            dict with keys: 'clip_loss', 'loss_self', 'loss_ori', 'loss_cl',
                'clip_acc', 'commitment_loss', 'reconstruction_loss',
                'pair_code_loss', 'loss'.
        """
        # Two independent quantization passes (shared params)
        z_e1 = self.encode(fea1)
        quant1 = self.quantizer(z_e1, temperature=temperature)
        fea1_vq = self.decode(quant1.quantized_embeddings)

        z_e2 = self.encode(fea2)
        quant2 = self.quantizer(z_e2, temperature=temperature)
        fea2_vq = self.decode(quant2.quantized_embeddings)

        # CLIP contrastive loss
        features = {
            "image_embed": fea1_vq,
            "text_embed": fea2_vq,
            "image_embed_ori": fea1,
            "text_embed_ori": fea2,
            "logit_scale_self": self.logit_scale_self.exp(),
            "logit_scale_cl": self.logit_scale_cl.exp(),
            "logit_scale": self.logit_scale.exp(),
        }
        clip_result = self.clip_loss_fn(features)

        # Commitment loss (average of two paths)
        commitment_loss = (
            quant1.quantization_loss + quant2.quantization_loss
        ) / 2

        # Reconstruction loss (log only, no backprop)
        feas = torch.cat([fea1, fea2], dim=0)
        recons = torch.cat([fea1_vq, fea2_vq], dim=0)
        with torch.no_grad():
            if self.loss_type == "mse":
                recon_loss = F.mse_loss(recons, feas, reduction="mean")
            elif self.loss_type == "l1":
                recon_loss = F.l1_loss(recons, feas, reduction="mean")
            elif self.loss_type == "cosine":
                recon_loss = self._cosine_loss(recons, feas)
            else:
                recon_loss = torch.tensor(0.0, device=fea1.device)

        # Pair code loss: z_e1 vs z_e2 MSE (log only)
        with torch.no_grad():
            pair_code_loss = F.mse_loss(
                z_e1, z_e2, reduction="mean"
            )

        return {
            "clip_loss": clip_result["clip_loss"],
            "loss_self": clip_result["loss_self"],
            "loss_ori": clip_result["loss_ori"],
            "loss_cl": clip_result["loss_cl"],
            "clip_acc": clip_result["clip_acc"],
            "commitment_loss": commitment_loss,
            "reconstruction_loss": recon_loss,
            "pair_code_loss": pair_code_loss,
            "loss": clip_result["clip_loss"],
        }

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: get semantic IDs.

        Args:
            x: (B, input_dim) original embedding.

        Returns:
            Tensor: codes, shape (B, n_layers).
        """
        z_e = self.encode(x)
        return self.quantizer.get_codes(z_e)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embedding from semantic IDs (through decoder).

        Args:
            codes: (B, n_layers) semantic ID codes.

        Returns:
            Tensor: x_hat, shape (B, input_dim).
        """
        z_q = self.quantizer.decode_codes(codes)
        return self.decode(z_q)


class RQKMeans(nn.Module):
    """RQ-KMeans: multi-layer residual Mini-Batch K-Means.

    No Encoder/Decoder — directly clusters input vectors via residual
    K-Means, updating centroids online without gradient backpropagation.

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::ResidualMiniBatchKMeans (top-level usage)

    Args:
        embed_dim (int): feature dimension. Default: 64.
        n_layers (int): number of residual quantization layers. Default: 3.
        n_embed (int|List[int]): number of clusters per layer. Default: 256.
        normalize_residuals (bool): L2-normalize residuals before each
            layer. Default: False.
        init_buffer_size (int): buffer size for KMeans++ initialization.
            Default: 3072.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        init_buffer_size: int = 3072,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.quantizer = ResidualKMeans(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_embed=n_embed,
            normalize_residuals=normalize_residuals,
            init_buffer_size=init_buffer_size,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward: direct residual K-Means quantization.

        During training: updates centroids via mini-batch steps.
        During inference: assigns to nearest centroids without update.

        Args:
            x: (B, embed_dim) input features.

        Returns:
            dict with keys:
                'codes':     (B, n_layers)  semantic IDs.
                'quantized': (B, embed_dim) quantized vector (sum of centroids).
        """
        codes, quantized = self.quantizer(x)
        return {
            "codes": codes,
            "quantized": quantized,
        }

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: get semantic IDs.

        Args:
            x: (B, embed_dim) input features.

        Returns:
            Tensor: codes, shape (B, n_layers).
        """
        return self.quantizer.get_codes(x)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from semantic IDs (centroid lookup + sum).

        Args:
            codes: (B, n_layers) semantic ID codes.

        Returns:
            Tensor: quantized, shape (B, embed_dim).
        """
        return self.quantizer.decode_codes(codes)

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get centroid weights for a specific layer.

        Args:
            layer_idx: index of the quantization layer.

        Returns:
            Tensor: centroids, shape (n_embed, embed_dim).
        """
        return self.quantizer.get_codebook_embeddings(layer_idx)