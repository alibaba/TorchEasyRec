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
* :class:`ReservoirSampler` — bounded uniform stream sample (Vitter
  Algorithm R) that :class:`~tzrec.models.sid_rqkmeans.SidRqkmeans`
  fills during training to feed the one-shot FAISS fit.
"""

from typing import Optional, Tuple

import torch
from torch import nn

from tzrec.utils.logging_util import logger


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
    return ((out - x) ** 2).mean(), relative_l1(x, out, epsilon)


def relative_l1(
    x: torch.Tensor,
    out: torch.Tensor,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Relative-L1 error ``mean(|x - out| / (max(|x|, |out|) + eps))``.

    Symmetric relative error in [0, 1] (verbatim port of OpenOneRec's
    ``calc_loss``). Used standalone by :meth:`SidRqkmeans.update_metric` (which
    needs only ``rel``, not the MSE :meth:`recon_diagnostics` also computes).

    Args:
        x: ground-truth embedding, shape (B, D).
        out: quantized reconstruction, shape (B, D).
        epsilon: numerical stabilizer for the denominator.
    """
    return (
        torch.abs(x - out) / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)
    ).mean()


class ReservoirSampler:
    """Bounded uniform sample of a stream (Vitter Algorithm R).

    Keeps a uniform ``capacity``-row sample of all rows passed to ``add``, in
    O(capacity) host (CPU) memory — used to subsample the training corpus for
    the one-shot FAISS fit without buffering the whole corpus. The buffer is a
    CPU float32 tensor, allocated lazily on the first ``add``.

    Args:
        capacity (int): max rows retained.
        dim (int): row width (feature dimension).
    """

    def __init__(self, capacity: int, dim: int) -> None:
        self._cap = capacity
        self._dim = dim
        # Allocated lazily on the first add. _n_filled = used slots;
        # _n_seen = running count for the accept prob.
        self._buf: Optional[torch.Tensor] = None
        self._n_filled = 0
        self._n_seen = 0
        logger.info("[ReservoirSampler] capacity=%d, dim=%d", capacity, dim)

    @property
    def capacity(self) -> int:
        """Max rows retained."""
        return self._cap

    @property
    def n_seen(self) -> int:
        """Total rows passed to ``add`` so far."""
        return self._n_seen

    @property
    def n_filled(self) -> int:
        """Rows currently held (<= capacity)."""
        return self._n_filled

    @torch.no_grad()
    def add(self, x: torch.Tensor) -> None:
        """Stream a batch of rows into the reservoir.

        Args:
            x (Tensor): rows to add, shape (B, dim).
        """
        x = x.detach()
        cap = self._cap
        if self._buf is None:
            self._buf = torch.empty(cap, self._dim, dtype=torch.float32)

        # Phase 1: fill empty slots first. x is already on the host (CPU-only
        # model), so this is a dtype cast into the buffer, not a device copy.
        if self._n_filled < cap:
            take = min(x.shape[0], cap - self._n_filled)
            self._buf[self._n_filled : self._n_filled + take] = x[:take].to(
                torch.float32
            )
            self._n_filled += take
            self._n_seen += take
            x = x[take:]
            if x.shape[0] == 0:
                return

        # Phase 2: row j enters with prob cap/(n_seen+j+1), displacing a random
        # slot. float64 keeps n_seen+j+1 exact past 2**24.
        r = x.shape[0]
        pos = self._n_seen + torch.arange(r)
        accept = torch.rand(r) < (cap / (pos + 1).to(torch.float64))
        idx = accept.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            slots = torch.randint(0, cap, (idx.numel(),))
            # Slot collisions are last-write-wins; O(B/cap) bias, negligible here.
            self._buf[slots] = x[idx].to(torch.float32)
        self._n_seen += r

    def sample(self) -> torch.Tensor:
        """Return the filled portion of the reservoir, shape (n_filled, dim)."""
        if self._buf is None or self._n_filled == 0:
            return torch.empty(0, self._dim, dtype=torch.float32)
        return self._buf[: self._n_filled]

    def reset(self) -> None:
        """Drop the buffer and counters to free host memory."""
        self._buf = None
        self._n_filled = 0
        self._n_seen = 0


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
