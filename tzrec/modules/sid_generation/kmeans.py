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

"""K-Means base utilities: distance, KMeans++ init, and MiniBatchKMeans."""

from typing import List, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


# ------------------------------------------------------------------
# Distance computation
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
