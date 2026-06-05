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

"""K-Means utilities for the SID-generation stack.

This module is the single home for torch-native K-Means code used by
SID models:

* :class:`KMeansLayer` — per-layer centroid container used by
  :class:`ResidualKMeansQuantizer`. Centroids are injected
  by the FAISS backend via ``load_centroids_``; the only forward path
  is ``predict``.
* :func:`faiss_residual_kmeans` — FAISS residual K-Means used by
  :class:`ResidualVectorQuantizer` to warm-start the RQ-VAE codebook on the
  first training batch (same FAISS backend as the offline RQ-KMeans fit).
"""

from typing import Dict, List, Optional, Tuple

import torch
from torch import nn


def recon_diagnostics(
    x: torch.Tensor,
    out: torch.Tensor,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MSE + relative-L1 reconstruction diagnostics.

    Shared by :meth:`SidRqkmeans.update_metric` (which wants tensors for
    ``torchmetrics.MeanMetric``) and :meth:`ResidualKMeansQuantizer.train_offline`'s
    per-layer log line (which converts to Python floats via ``.item()``).

    Args:
        x: ground-truth embedding, shape (B, D).
        out: quantized reconstruction, shape (B, D).
        epsilon: numerical stabilizer for the relative-L1 denominator.

    Returns:
        mse:  scalar ``((out - x) ** 2).mean()``.
        rel:  scalar relative-L1 ``mean(|x - out| / (max(|x|, |out|) + eps))``.
    """
    mse = ((out - x) ** 2).mean()
    rel = (
        torch.abs(x - out) / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)
    ).mean()
    return mse, rel


@torch.no_grad()
def _squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance between rows of ``x`` and ``y``.

    Args:
        x (Tensor): data points, shape (N, D).
        y (Tensor): centroids, shape (K, D).

    Returns:
        Tensor: squared distances, shape (N, K).

    Called per-batch from :meth:`KMeansLayer.predict`, so ``N`` is the batch
    size and the full (N, K) product is small. Kept branch-free (no
    data-dependent chunking on ``N``) so the predict forward stays
    FX-traceable: torchrec's inference pipeline symbolically traces the
    model, and a ``if N <= chunk_size`` on the traced batch dim raises a
    ``torch.fx`` TraceError.
    """
    x_sq = x.pow(2).sum(dim=1, keepdim=True)  # (N, 1)
    y_sq = y.pow(2).sum(dim=1, keepdim=True).t()  # (1, K)
    return (x_sq + y_sq - 2.0 * x @ y.t()).clamp_(min=0.0)


@torch.no_grad()
def faiss_residual_kmeans(
    samples: torch.Tensor,
    n_clusters_list: List[int],
    faiss_kmeans_kwargs: Optional[Dict] = None,
) -> List[torch.Tensor]:
    """Residual K-Means warm-start via FAISS, one pass per layer.

    Clusters ``samples`` with FAISS K-Means, subtracts each point's assigned
    centroid, and repeats on the residual for every layer. Used by
    :meth:`ResidualVectorQuantizer.init_embed_` to seed the RQ-VAE codebook
    from the first training batch — the same FAISS backend the offline
    RQ-KMeans model uses, instead of a separate torch-native Lloyd's loop.

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters_list (List[int]): per-layer cluster counts.
        faiss_kmeans_kwargs (Dict|None): extra kwargs for ``faiss.Kmeans``
            (e.g. ``{'niter': 10, 'seed': 123}``).

    Returns:
        List[Tensor]: per-layer centroids ``[(K0, D), ...]`` on samples.device.

    Raises:
        ImportError: if ``faiss`` is not installed.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is required for RQ-VAE kmeans_init. Install via "
            "`pip install faiss-cpu` or `pip install faiss-gpu`."
        ) from e

    kwargs = dict(faiss_kmeans_kwargs or {})
    device = samples.device
    _, D = samples.shape
    # Own a contiguous fp32 numpy copy we mutate in place to form residuals.
    x = samples.detach().cpu().float().numpy().copy()

    res_centers: List[torch.Tensor] = []
    for n_clusters in n_clusters_list:
        kmeans = faiss.Kmeans(D, n_clusters, **kwargs)
        kmeans.train(x)
        centroids = kmeans.centroids.copy()  # (K, D)
        res_centers.append(torch.from_numpy(centroids).to(device))
        _, idx = kmeans.index.search(x, 1)
        x -= centroids[idx.ravel()]  # residual, in place
    return res_centers


class KMeansLayer(nn.Module):
    """Single layer of a residual K-Means stack.

    Centroids are populated externally by ``load_centroids_`` (called per
    layer by the FAISS backend in :class:`ResidualKMeansQuantizer`); ``predict``
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
        # Flipped by ``load_centroids_`` after the FAISS fit. Persistent
        # so a normal post-fit checkpoint round-trips; mid-fit poisoning
        # (True flag + still-zero centroids) is caught in _load_from_state_dict.
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

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Reject mid-fit-checkpoint state dicts (True flag + zero centroids)."""
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if bool(self._is_initialized.item()) and self.centroids.abs().sum() == 0:
            error_msgs.append(
                f"KMeansLayer at '{prefix}': _is_initialized=True but centroids "
                "are all zero — checkpoint was likely taken mid-FAISS-fit. "
                "Re-run on_train_end to produce a valid checkpoint."
            )

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
