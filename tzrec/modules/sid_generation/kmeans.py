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

"""K-Means utilities for the SID-generation stack.

This module is the single home for torch-native K-Means code used by
SID models:

* :class:`KMeansLayer` — per-layer centroid container used by
  :class:`ResidualKMeans` / :class:`RQKMeans`. Centroids are injected
  by the FAISS backend via ``load_centroids_``; the only forward path
  is ``predict``.
* :func:`_kmeans` / :func:`_residual_kmeans` — pure-torch Lloyd's
  K-Means + residual variant, used by :class:`ResidualQuantized` to
  warm-start the RQ-VAE codebook on the first training batch. They run
  once on a single batch of encoder outputs (typically ~2k × 64), so
  pulling in FAISS here would be all overhead and no benefit.
"""

from typing import List, Tuple

import torch
from torch import nn


@torch.no_grad()
def _squared_euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Squared L2 distance with chunked computation for memory efficiency.

    Chunks the rows of ``x`` so peak memory is bounded by
    ``chunk_size * K * 4 bytes`` (fp32) regardless of ``N``.

    Args:
        x (Tensor): data points, shape (N, D).
        y (Tensor): centroids, shape (K, D).
        chunk_size (int): max rows of x per chunk. Default: 50000.

    Returns:
        Tensor: squared distances, shape (N, K).
    """
    x_sq = x.pow(2).sum(dim=1, keepdim=True)  # (N, 1)
    y_sq = y.pow(2).sum(dim=1, keepdim=True).t()  # (1, K)
    N = x.shape[0]
    if N <= chunk_size:
        return (x_sq + y_sq - 2.0 * x @ y.t()).clamp_(min=0.0)
    out = x.new_empty(N, y.shape[0])
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        out[start:end] = (x_sq[start:end] + y_sq - 2.0 * x[start:end] @ y.t()).clamp_(
            min=0.0
        )
    return out


@torch.no_grad()
def _kmeans_plus_plus(
    data: torch.Tensor,
    n_clusters: int,
) -> torch.Tensor:
    """KMeans++ initialization (Arthur & Vassilvitskii 2007).

    Selects initial centroids via distance-weighted probability sampling
    to ensure well-spread starting points. Used by the RQ-VAE codebook
    init path (``ResidualQuantized.kmeans_init``); RQKMeans itself no
    longer needs it.

    Args:
        data (Tensor): data points, shape (N, D).
        n_clusters (int): number of clusters K.

    Returns:
        Tensor: initial centroids, shape (K, D).
    """
    N, D = data.shape
    centroids = torch.zeros(n_clusters, D, device=data.device, dtype=data.dtype)

    idx = torch.randint(0, N, (1,), device=data.device)
    centroids[0] = data[idx]

    for i in range(1, n_clusters):
        dists = _squared_euclidean_distance(data, centroids[:i])  # (N, i)
        min_dists = dists.min(dim=1)[0]  # (N,)
        if min_dists.sum() == 0:
            centroids[i:] = data[
                torch.randint(0, N, (n_clusters - i,), device=data.device)
            ]
            break
        next_idx = torch.multinomial(min_dists, num_samples=1)
        centroids[i] = data[next_idx]

    return centroids


@torch.no_grad()
def _kmeans(
    samples: torch.Tensor,
    n_clusters: int,
    n_iters: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's K-Means with KMeans++ initialization.

    Used by :class:`ResidualQuantized.init_embed_` to warm-start the
    RQ-VAE codebook on the first training batch.

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters (int): number of clusters K.
        n_iters (int): number of Lloyd iterations. Default: 100.

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
        new_centroids.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), samples)
        new_centroids = new_centroids / bins_clamped.unsqueeze(1)

        # Keep old centroids for empty clusters
        centroids = torch.where(zero_mask.unsqueeze(1), centroids, new_centroids)

    return centroids, assignments


@torch.no_grad()
def _residual_kmeans(
    samples: torch.Tensor,
    n_clusters_list: List[int],
    n_iters: int = 100,
) -> List[torch.Tensor]:
    """Residual K-Means: per-layer cluster then subtract centroids.

    Used by :class:`ResidualQuantized.init_embed_` to seed every RQ
    codebook layer in one pass over the first training batch.

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters_list (List[int]): per-layer cluster counts.
        n_iters (int): K-Means iterations per layer. Default: 100.

    Returns:
        List[Tensor]: per-layer centroids ``[(K0, D), (K1, D), ...]``.
    """
    res_centers = []
    for n_clusters in n_clusters_list:
        centroids, assignments = _kmeans(samples, n_clusters, n_iters)
        res_centers.append(centroids)
        samples = samples - centroids[assignments]
    return res_centers


class KMeansLayer(nn.Module):
    """Single layer of a residual K-Means stack.

    Centroids are populated externally by ``load_centroids_`` (called per
    layer by the FAISS backend in :class:`ResidualKMeans`); ``predict``
    is the only forward path. PyTorch state-dict keys are scoped by
    attribute path (``layers.<i>.centroids``), so renaming the class
    does not break existing checkpoints.

    Args:
        n_clusters (int): number of clusters (codebook size).
        n_features (int): feature dimension.
    """

    def __init__(
        self,
        n_clusters: int,
        n_features: int,
    ) -> None:
        super().__init__()
        self.n_clusters = n_clusters
        self.n_features = n_features

        self.register_buffer("centroids", torch.zeros(n_clusters, n_features))
        self.register_buffer("_is_initialized", torch.tensor(False))

    @property
    def is_initialized(self) -> bool:
        """Whether centroids have been injected via ``load_centroids_``."""
        return self._is_initialized.item()

    @torch.no_grad()
    def load_centroids_(self, centroids: torch.Tensor) -> None:
        """Inject offline-trained centroids.

        Args:
            centroids (Tensor): externally trained centroids,
                shape (n_clusters, n_features).
        """
        assert centroids.shape == self.centroids.shape, (
            f"centroids shape mismatch: expected {tuple(self.centroids.shape)}, "
            f"got {tuple(centroids.shape)}"
        )
        self.centroids.copy_(
            centroids.to(dtype=self.centroids.dtype, device=self.centroids.device)
        )
        self._is_initialized.fill_(True)

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest centroid.

        Args:
            batch (Tensor): data points, shape (B, D).

        Returns:
            Tensor: cluster indices, shape (B,).
        """
        dists = _squared_euclidean_distance(batch, self.centroids)
        return torch.argmin(dists, dim=-1)
