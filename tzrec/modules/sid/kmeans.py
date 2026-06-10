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
"""

from typing import Tuple

import torch
from torch import nn


def recon_diagnostics(
    x: torch.Tensor,
    out: torch.Tensor,
    epsilon: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MSE + relative-L1 reconstruction diagnostics.

    Shared by :meth:`SidRqkmeans.update_metric` and
    :meth:`ResidualKMeansQuantizer.train_offline`'s per-layer log.

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


class KMeansLayer(nn.Module):
    """Single layer of a residual K-Means stack.

    Centroids are populated externally by ``load_centroids_`` (the FAISS
    backend in :class:`ResidualKMeansQuantizer`); ``predict`` is the only
    forward path.

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
        # Persistent so a post-fit checkpoint round-trips; a mid-fit poison
        # (True flag + zero centroids) is caught in _load_from_state_dict.
        self.register_buffer("_is_initialized", torch.tensor(False))
        # Plain-Python mirror of the buffer, read on the per-batch forward
        # path to avoid a .item() GPU->CPU sync. Synced only via
        # mark_initialized_ and _load_from_state_dict.
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Whether centroids have been injected via ``load_centroids_``."""
        return self._initialized

    def mark_initialized_(self) -> None:
        """Flag centroids populated, syncing buffer + cached mirror.

        For callers that fill ``centroids`` in place (e.g. the DDP broadcast
        in :meth:`SidRqkmeans.on_train_end`) rather than via ``load_centroids_``.
        """
        self._is_initialized.fill_(True)
        self._initialized = True

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
        self.mark_initialized_()

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
        # Mirror the restored buffer into the cached flag (one load-time sync).
        self._initialized = bool(self._is_initialized.item())
        if self._initialized and self.centroids.abs().sum() == 0:
            error_msgs.append(
                f"KMeansLayer at '{prefix}': _is_initialized=True but centroids "
                "are all zero — checkpoint was likely taken mid-FAISS-fit. "
                "Re-run on_train_end to produce a valid checkpoint."
            )

    @torch.no_grad()
    def predict(self, batch: torch.Tensor) -> torch.Tensor:
        """Assign points to nearest centroid.

        Uses ``torch.cdist`` (L2); argmin is invariant to the monotonic sqrt,
        so assignments match squared-L2 except at exact equidistant ties
        (measure zero for real embeddings), where either centroid is valid.

        Args:
            batch (Tensor): data points, shape (B, D).

        Returns:
            Tensor: cluster indices, shape (B,).
        """
        return torch.cdist(batch, self.centroids).argmin(dim=-1)
