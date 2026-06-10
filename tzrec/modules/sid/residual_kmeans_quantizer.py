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

"""Multi-layer residual K-Means: ResidualKMeansQuantizer.

Training is FAISS-only: the codebook is built once via ``train_offline``
over the full embedding matrix; ``forward`` is read-only (predict + lookup).
"""

from typing import Dict, List, Optional, Tuple, Union

import faiss
import faiss.contrib.torch_utils  # noqa: F401  (registers torch tensor I/O)
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.kmeans import KMeansLayer, recon_diagnostics
from tzrec.modules.sid.residual_quantizer import ResidualQuantizer
from tzrec.utils.logging_util import logger


class ResidualKMeansQuantizer(ResidualQuantizer):
    """Multi-layer residual K-Means with offline FAISS training.

    Each layer quantizes the residual from the previous layer:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i = layer_i.predict(residual_i)
            quantized_i = layer_i.centroids[code_i]
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Args:
        embed_dim (int): feature dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): number of clusters per layer. Default: 256.
            May differ per layer (non-uniform codebooks such as
            ``[256, 512, 1024]`` are supported) — ``train_offline`` builds a
            separate ``faiss.Kmeans`` per layer.
        normalize_residuals (bool): whether to L2-normalize residuals
            before each layer. Default: False, matching the ``SidRqkmeans``
            proto default (and OpenOneRec's residual k-means, which fits raw
            residuals with no per-layer normalization).
        faiss_kmeans_kwargs (Dict|None): extra kwargs forwarded to
            ``faiss.Kmeans(D, K, **kwargs)`` (e.g. {'niter': 20,
            'verbose': True, 'spherical': False}). A ``gpu`` key is ignored —
            the fit is CPU-only.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        faiss_kmeans_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__(embed_dim, n_layers, n_embed, normalize_residuals)
        self.faiss_kmeans_kwargs = dict(faiss_kmeans_kwargs or {})

        self.layers = nn.ModuleList(
            [
                KMeansLayer(
                    n_clusters=self.n_embed_list[i],
                    n_features=embed_dim,
                )
                for i in range(n_layers)
            ]
        )

    def _quantize_layer(
        self,
        layer_idx: int,
        residual: torch.Tensor,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Nearest-centroid assignment for one layer.

        Uninitialized layers (before ``train_offline``) return zeros, so the
        residual walk is a no-op and the model stays callable. ``temperature``
        is unused (no soft assignment).

        Args:
            layer_idx (int): quantization layer index.
            residual (Tensor): current residual, shape (B, D).
            temperature (float): unused.

        Returns:
            codes (Tensor): cluster indices, shape (B,).
            quantized (Tensor): selected centroids, shape (B, D).
        """
        layer = self.layers[layer_idx]
        if not layer.is_initialized:
            codes = torch.zeros(
                residual.shape[0], dtype=torch.long, device=residual.device
            )
            return codes, torch.zeros_like(residual)
        codes = layer.predict(residual)
        return codes, layer.centroids[codes]

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign codes per layer and sum the centroids.

        Codebook is read-only here; training happens in ``train_offline``.
        Uninitialized layers contribute zeros (see :meth:`_quantize_layer`) so
        the model is callable before the one-shot FAISS fit completes.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            codes (Tensor): cluster indices per layer, shape (B, n_layers).
            quantized (Tensor): sum of quantized embeddings, shape (B, D).
        """
        cluster_ids, quantized_sum, _ = self._residual_pass(input)
        return cluster_ids, quantized_sum

    @property
    def is_fitted(self) -> bool:
        """Whether ``train_offline`` has populated every layer's codebook.

        ``forward`` is callable before the fit (uninitialized layers emit
        zeros), so reconstruction outputs are meaningful only once this is True.
        """
        return all(layer.is_initialized for layer in self.layers)

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get centroid weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: centroids, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].centroids

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors via the layer's centroid table."""
        return self.layers[layer_idx].centroids[code_idx]

    def default_fit_sample_size(self) -> int:
        """Points the FAISS fit subsamples to: max(K) * max_points_per_centroid.

        ``faiss.Kmeans`` caps each layer's training set at
        ``K * max_points_per_centroid`` (default 256), so fitting on more is
        wasted. Callers use this to size their training-sample reservoir.
        """
        max_ppc = int(self.faiss_kmeans_kwargs.get("max_points_per_centroid", 256))
        return max(self.n_embed_list) * max_ppc

    @torch.no_grad()
    def train_offline(
        self,
        inputs: torch.Tensor,
        verbose: bool = True,
    ) -> None:
        """Train the multi-layer codebook via offline FAISS K-Means.

        CPU-only: ``inputs`` is already a host tensor (SidRqkmeans refuses to
        run when CUDA is visible) and the FAISS fit runs on CPU. The post-fit
        ``index.search`` assignment streams all N rows through in
        ``SEARCH_CHUNK``-sized chunks to cap peak memory.

        Args:
            inputs (Tensor): embedding matrix (N, D) on CPU. Copied once to an
                owned float32 tensor; not mutated.
            verbose (bool): print per-layer reconstruction loss. Default: True.
        """
        # CPU-only: SidRqkmeans refuses to init when CUDA is visible, but this
        # quantizer is a standalone module — assert the host-tensor contract it
        # relies on so misuse fails here, not deep inside faiss.
        assert not inputs.is_cuda, "train_offline is CPU-only; got a CUDA tensor"
        assert inputs.dim() == 2 and inputs.shape[1] == self.embed_dim, (
            f"inputs must be (N, {self.embed_dim}), got {tuple(inputs.shape)}"
        )
        # Own one contiguous float32 copy to update in place as the residual.
        x = inputs.detach().to(dtype=torch.float32, copy=True).contiguous()
        N = x.shape[0]
        # Fail loudly on a too-small corpus: faiss.Kmeans only warns (not
        # errors) when N < K and returns a degenerate codebook, which the
        # all-zero poison guard in KMeansLayer would not catch.
        max_k = max(self.n_embed_list)
        assert N >= max_k, (
            f"need >= {max_k} points to fit the codebook (largest layer K), got N={N}"
        )
        out = torch.zeros_like(x)
        # The per-layer log reports the cumulative recon error of the original
        # input x0 by the centroid sum. Without normalization the invariant
        # ``out + x == x0`` holds, so x0 is reconstructed on the fly below and we
        # skip the persistent (N, D) clone; with normalization x is rescaled each
        # layer, breaking the invariant, so the clone is required.
        x0 = x.clone() if (verbose and self.normalize_residuals) else None

        # CPU-only fit: SidRqkmeans refuses to initialize when CUDA is visible,
        # so the codebook is always built on CPU. Drop any stale ``gpu`` request
        # from the config so a faiss-gpu build can't try to use an absent GPU.
        kwargs = dict(self.faiss_kmeans_kwargs)
        kwargs.pop("gpu", None)
        if verbose:
            logger.info(
                "[ResidualKMeansQuantizer] fitting %d-layer codebook on CPU "
                "(N=%d, D=%d).",
                self.n_layers,
                N,
                self.embed_dim,
            )

        # Chunk index.search to cap peak memory (~1 GB at 500K × 512 × 4B).
        SEARCH_CHUNK = 500_000

        for layer_idx in range(self.n_layers):
            if self.normalize_residuals:
                x = F.normalize(x, dim=-1)

            # Fresh Kmeans per layer so each can use its own K (non-uniform
            # codebooks).
            kmeans = faiss.Kmeans(
                self.embed_dim, self.n_embed_list[layer_idx], **kwargs
            )
            kmeans.train(x)
            centroids = torch.as_tensor(kmeans.centroids, dtype=torch.float32)

            for start in range(0, N, SEARCH_CHUNK):
                end = min(start + SEARCH_CHUNK, N)
                _, idx = kmeans.index.search(x[start:end], 1)
                idx = torch.as_tensor(idx).reshape(-1).long()
                q = centroids[idx]  # (chunk, D)
                out[start:end] += q
                x[start:end] -= q  # residual
                del idx, q

            if verbose:
                # x0 == out + x without normalization (see above).
                ref = x0 if x0 is not None else out + x
                logger.info(
                    "[ResidualKMeansQuantizer][offline_faiss][layer %d] %s",
                    layer_idx,
                    self._calc_loss(ref, out),  # cumulative recon of original input
                )

            self.layers[layer_idx].load_centroids_(centroids)
            if verbose:
                logger.info(
                    "[ResidualKMeansQuantizer][offline_faiss] layer %d finished",
                    layer_idx,
                )

    @staticmethod
    def _calc_loss(
        x: torch.Tensor, out: torch.Tensor, epsilon: float = 1e-4
    ) -> Dict[str, float]:
        """Reconstruction loss diagnostics (MSE + relative L1)."""
        loss, rel_loss = recon_diagnostics(x, out, epsilon=epsilon)
        return {"loss": float(loss.item()), "rel_loss": float(rel_loss.item())}
