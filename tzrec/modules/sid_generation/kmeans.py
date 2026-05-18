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

"""K-Means single-layer container.

Centroids are owned by an offline backend (FAISS) and injected via
``load_centroids_``. This module no longer performs any online training;
``predict`` is the only forward path.
"""

import torch
from torch import nn


@torch.no_grad()
def _squared_euclidean_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    chunk_size: int = 50000,
) -> torch.Tensor:
    """Squared L2 distance with chunked computation for memory efficiency.

    Args:
        x (Tensor): data points, shape (N, D).
        y (Tensor): centroids, shape (K, D).
        chunk_size (int): max rows of x per chunk. Default: 50000.

    Returns:
        Tensor: squared distances, shape (N, K).
    """
    x_sq = x.pow(2).sum(dim=1, keepdim=True)
    y_sq = y.pow(2).sum(dim=1, keepdim=True).t()
    return (x_sq + y_sq - 2.0 * x @ y.t()).clamp(min=0.0)


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

        self.register_buffer(
            "centroids", torch.zeros(n_clusters, n_features)
        )
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
