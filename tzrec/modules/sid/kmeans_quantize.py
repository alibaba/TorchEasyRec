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

* :class:`KMeansQuantizeLayer` — the K-Means :class:`QuantizeLayer`: a
  centroid container populated by the FAISS backend via ``load_centroids_``.
* :class:`ReservoirSampler` — bounded uniform stream sample (Vitter
  Algorithm R) that :class:`~tzrec.models.sid_rqkmeans.SidRqkmeans`
  fills during training to feed the one-shot FAISS fit.
* :func:`faiss_kmeans_fit` — the shared one-layer FAISS fit behind both SID
  residual-K-Means loops (RQ-VAE warm-start and offline RQ-K-Means).
"""

from typing import Any, Dict, Optional

import torch

from tzrec.modules.sid.quantize_layer import QuantizeLayer
from tzrec.modules.sid.types import QuantizeOutput
from tzrec.utils.logging_util import logger


def faiss_kmeans_fit(
    x: Any,
    dim: int,
    n_clusters: int,
    faiss_kmeans_kwargs: Optional[Dict] = None,
) -> Any:
    """Train one ``faiss.Kmeans(dim, n_clusters)`` on ``x`` and return it.

    The shared one-layer FAISS fit behind both SID residual-K-Means loops (the
    RQ-VAE warm-start and the offline RQ-K-Means); the caller reads
    ``km.centroids`` and assigns via ``km.index.search``. Strips a ``gpu`` kwarg
    (faiss honors it and would move the fit to GPU, breaking the CPU-only
    contract) and guards ``N >= n_clusters`` before faiss's opaque C++ throw.
    ``x`` may be a numpy array or a torch tensor.

    Args:
        x: data points, shape (N, dim) — numpy array or torch tensor.
        dim (int): feature dimension.
        n_clusters (int): number of centroids (codebook size).
        faiss_kmeans_kwargs (Dict|None): extra kwargs for ``faiss.Kmeans``.

    Returns:
        The trained ``faiss.Kmeans`` (read ``.centroids`` / ``.index``).

    Raises:
        ImportError: if ``faiss`` is not installed.
        RuntimeError: if ``x`` has fewer than ``n_clusters`` rows.
    """
    try:
        import faiss
    except ImportError as e:
        raise ImportError(
            "faiss is required for SID residual K-Means. Install via "
            "`pip install faiss-cpu` or `pip install faiss-gpu`."
        ) from e

    # Drop any `gpu` key: faiss.Kmeans honors it and would move the fit off CPU.
    kwargs = dict(faiss_kmeans_kwargs or {})
    kwargs.pop("gpu", None)
    n = int(x.shape[0])
    if n < n_clusters:
        raise RuntimeError(
            f"need >= {n_clusters} points to fit the codebook, got N={n}"
        )
    km = faiss.Kmeans(dim, n_clusters, **kwargs)
    km.train(x)
    return km


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

        # float64 keeps n_seen+j+1 exact past 2**24.
        r = x.shape[0]
        pos = self._n_seen + torch.arange(r)
        accept = torch.rand(r) < (cap / (pos + 1).to(torch.float64))
        idx = accept.nonzero(as_tuple=True)[0]
        if idx.numel() > 0:
            slots = torch.randint(0, cap, (idx.numel(),))
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


class KMeansQuantizeLayer(QuantizeLayer):
    """K-Means :class:`QuantizeLayer`: a centroid codebook + nearest assignment.

    Centroids are populated externally by ``load_centroids_`` (the FAISS
    backend in :class:`ResidualKMeansQuantizer`); ``quantize`` is the only
    forward path. (The k-means *fit* lives in the quantizer; this layer just
    holds the resulting centroids.)

    Args:
        n_embed (int): number of centroids (codebook size).
        embed_dim (int): feature dimension.
    """

    def __init__(self, n_embed: int, embed_dim: int) -> None:
        super().__init__(n_embed, embed_dim)
        self.register_buffer("centroids", torch.zeros(n_embed, embed_dim))
        # Persistent so post-fit checkpoints round-trip; mid-fit poison caught on load.
        self.register_buffer("_is_initialized", torch.tensor(False))
        # Plain-Python mirror of the buffer, avoiding a per-forward .item() GPU sync.
        self._initialized: bool = False

    @property
    def is_initialized(self) -> bool:
        """Whether centroids have been injected via ``load_centroids_``."""
        return self._initialized

    def mark_initialized_(self) -> None:
        """Flag centroids populated, syncing buffer + cached mirror."""
        self._is_initialized.fill_(True)
        self._initialized = True

    @torch.no_grad()
    def load_centroids_(self, centroids: torch.Tensor) -> None:
        """Inject offline-trained centroids.

        Args:
            centroids (Tensor): externally trained centroids,
                shape (n_embed, embed_dim).
        """
        # raise (not assert): `python -O` would drop it, letting a bad shape broadcast.
        if centroids.shape != self.centroids.shape:
            raise RuntimeError(
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
        self._initialized = bool(self._is_initialized.item())
        if self._initialized and self.centroids.abs().sum() == 0:
            error_msgs.append(
                f"KMeansQuantizeLayer at '{prefix}': _is_initialized=True but "
                "centroids are all zero — checkpoint was likely taken "
                "mid-FAISS-fit. Re-run on_train_end to produce a valid checkpoint."
            )

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, topk: int = 1) -> QuantizeOutput:
        """Assign points to the nearest centroid and gather them.

        Uses ``torch.cdist`` (L2); argmin is invariant to the monotonic sqrt,
        so assignments match squared-L2 except at exact equidistant ties
        (measure zero for real embeddings), where either centroid is valid.
        Before the FAISS fit (uninitialized) this returns all-zero codes +
        embeddings so the residual walk stays a no-op and the model is callable.

        Args:
            x (Tensor): data points, shape (B, D).
            topk (int): number of nearest centroids to return.

        Returns:
            QuantizeOutput: selected centroid/id plus top-k nearest ids/scores.
        """
        if not self.is_initialized:
            ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return QuantizeOutput(embeddings=torch.zeros_like(x), ids=ids)
        # Match centroid dtype: cdist rejects mismatched dtypes.
        distances = torch.cdist(x.to(self.centroids.dtype), self.centroids)
        if self.training:
            ids = distances.argmin(dim=-1)
            return QuantizeOutput(embeddings=self.centroids[ids], ids=ids)
        return self._topk_output(distances.pow(2), topk)

    def get_codebook_embeddings(self) -> torch.Tensor:
        """Return the centroid table, shape (n_embed, embed_dim)."""
        return self.centroids
