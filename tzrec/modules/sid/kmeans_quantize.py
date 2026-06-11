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
* :func:`faiss_residual_kmeans` — FAISS residual K-Means used by
  :class:`~tzrec.modules.sid.residual_vector_quantizer.ResidualVectorQuantizer`
  to warm-start the RQ-VAE codebook on the first training batch (same FAISS
  backend as the offline RQ-KMeans fit). Fits on CPU and returns centroids on
  the input device, so it is safe to call from a GPU-resident RQ-VAE.
"""

from typing import Dict, List, Optional

import torch

from tzrec.modules.sid.quantize_layer import QuantizeLayer
from tzrec.modules.sid.types import QuantizeOutput
from tzrec.utils.logging_util import logger


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

    Device handling (CPU + GPU): the FAISS fit is always CPU (``samples`` is
    copied to host as fp32 numpy), and the returned centroids are moved back to
    ``samples.device``. So an RQ-VAE training on GPU gets GPU centroids while
    the fit itself stays on CPU — no faiss-gpu build required.

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

        # Phase 1: fill empty slots first. x is on the host, so ``.to`` is a
        # dtype cast into the buffer, not a device copy.
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
        # raise (not assert): under ``python -O`` a dropped assert would let a
        # (1, D) tensor broadcast-replicate into all K centroid rows silently.
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
        # Mirror the restored buffer into the cached flag (one load-time sync).
        self._initialized = bool(self._is_initialized.item())
        if self._initialized and self.centroids.abs().sum() == 0:
            error_msgs.append(
                f"KMeansQuantizeLayer at '{prefix}': _is_initialized=True but "
                "centroids are all zero — checkpoint was likely taken "
                "mid-FAISS-fit. Re-run on_train_end to produce a valid checkpoint."
            )

    @torch.no_grad()
    def quantize(self, x: torch.Tensor, temperature: float = 1.0) -> QuantizeOutput:
        """Assign points to the nearest centroid and gather them.

        Uses ``torch.cdist`` (L2); argmin is invariant to the monotonic sqrt,
        so assignments match squared-L2 except at exact equidistant ties
        (measure zero for real embeddings), where either centroid is valid.
        Before the FAISS fit (uninitialized) this returns all-zero codes +
        embeddings so the residual walk stays a no-op and the model is callable.
        ``temperature`` is unused (no soft assignment).

        Args:
            x (Tensor): data points, shape (B, D).
            temperature (float): unused.

        Returns:
            QuantizeOutput: ``ids`` (B,) and ``embeddings`` (B, D).
        """
        if not self.is_initialized:
            ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return QuantizeOutput(embeddings=torch.zeros_like(x), ids=ids)
        ids = torch.cdist(x, self.centroids).argmin(dim=-1)
        return QuantizeOutput(embeddings=self.centroids[ids], ids=ids)

    def get_codebook_embeddings(self) -> torch.Tensor:
        """Return the centroid table, shape (n_embed, embed_dim)."""
        return self.centroids
