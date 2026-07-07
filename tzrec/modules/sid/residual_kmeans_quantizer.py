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

from typing import Any, Dict, List, Mapping, Optional, Union

import faiss.contrib.torch_utils  # noqa: F401  (registers torch tensor I/O)
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.kmeans_quantize import KMeansQuantizeLayer, faiss_kmeans_fit
from tzrec.modules.sid.residual_quantizer import ResidualQuantizer
from tzrec.modules.sid.types import ResidualQuantizerOutput
from tzrec.utils.logging_util import logger


class ResidualKMeansQuantizer(ResidualQuantizer):
    """Multi-layer residual K-Means with offline FAISS training.

    Each layer quantizes the residual from the previous layer:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i.quantize(residual_i)
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
        candidate_output_config (Mapping|None): optional inference-time candidate
            SID settings (``enabled`` / ``topk`` / ``strategy``); candidates are
            emitted only once the codebook is fit. Default: None (disabled).
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        faiss_kmeans_kwargs: Optional[Dict] = None,
        candidate_output_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(
            embed_dim,
            n_layers,
            n_embed,
            normalize_residuals,
            candidate_output_config=candidate_output_config,
        )
        self.faiss_kmeans_kwargs = dict(faiss_kmeans_kwargs or {})

        self.layers = nn.ModuleList(
            [
                KMeansQuantizeLayer(
                    n_embed=self.n_embed_list[i],
                    embed_dim=embed_dim,
                )
                for i in range(n_layers)
            ]
        )

    def forward(
        self,
        input: torch.Tensor,
    ) -> ResidualQuantizerOutput:
        """Assign codes per layer and sum the centroids.

        Codebook is read-only here; training happens in ``train_offline``.
        Uninitialized layers contribute zeros (see :meth:`_quantize_layer`) so
        the model is callable before the one-shot FAISS fit completes.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            ResidualQuantizerOutput: named output with optional candidate tensors.
        """
        walk = self._residual_pass(input)
        return self._residual_output(walk, walk.aggregated, with_latents=False)

    @property
    def is_fitted(self) -> bool:
        """Whether ``train_offline`` has populated every layer's codebook.

        ``forward`` is callable before the fit (uninitialized layers emit
        zeros), so reconstruction outputs are meaningful only once this is True.
        """
        return all(layer.is_initialized for layer in self.layers)

    def _candidates_available(self) -> bool:
        """Candidate SIDs need a fit codebook; skip (don't crash) before the fit."""
        return self.is_fitted

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get centroid weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: centroids, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].get_codebook_embeddings()

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors via the layer's centroid table."""
        return self.layers[layer_idx].lookup(code_idx)

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
            inputs (Tensor): embedding matrix (N, D) on CPU. CONSUMED: the
                residual pass may mutate it in place, so the caller must not
                rely on its contents afterward (copy first if it needs them).
            verbose (bool): print per-layer reconstruction loss. Default: True.
        """
        # raise (not assert): these host-tensor guards must survive `python -O`.
        if inputs.is_cuda:
            raise RuntimeError("train_offline is CPU-only; got a CUDA tensor")
        if inputs.dim() != 2 or inputs.shape[1] != self.embed_dim:
            raise RuntimeError(
                f"inputs must be (N, {self.embed_dim}), got {tuple(inputs.shape)}"
            )
        # The loop mutates x in place into the caller's buffer (see Args: CONSUMED).
        x = inputs.detach().to(dtype=torch.float32).contiguous()
        N = x.shape[0]
        # Clear N<K error before faiss's opaque throw (K<=N<K*min_points unguarded).
        max_k = max(self.n_embed_list)
        if N < max_k:
            raise RuntimeError(
                f"need >= {max_k} points to fit the codebook (largest layer K), "
                f"got N={N}"
            )
        out = torch.zeros_like(x)
        x0 = x.clone() if (verbose and self.normalize_residuals) else None

        if verbose:
            logger.info(
                "[ResidualKMeansQuantizer] fitting %d-layer codebook on CPU "
                "(N=%d, D=%d).",
                self.n_layers,
                N,
                self.embed_dim,
            )

        SEARCH_CHUNK = 500_000

        for layer_idx in range(self.n_layers):
            if self.normalize_residuals:
                x = F.normalize(x, dim=-1)

            km = faiss_kmeans_fit(
                x,
                self.embed_dim,
                self.n_embed_list[layer_idx],
                self.faiss_kmeans_kwargs,
            )
            centroids = torch.as_tensor(km.centroids, dtype=torch.float32)

            for start in range(0, N, SEARCH_CHUNK):
                end = min(start + SEARCH_CHUNK, N)
                _, idx = km.index.search(x[start:end], 1)
                idx = torch.as_tensor(idx).reshape(-1).long()
                q = centroids[idx]
                out[start:end] += q
                x[start:end] -= q
                del idx, q

            if verbose:
                ref = x0 if self.normalize_residuals else out + x
                logger.info(
                    "[ResidualKMeansQuantizer][offline_faiss][layer %d] %s",
                    layer_idx,
                    self._calc_loss(ref, out),
                )

            self.layers[layer_idx].load_centroids_(centroids)
            if verbose:
                logger.info(
                    "[ResidualKMeansQuantizer][offline_faiss] layer %d finished",
                    layer_idx,
                )

    @staticmethod
    def _calc_loss(x: torch.Tensor, out: torch.Tensor) -> Dict[str, float]:
        """Per-layer reconstruction MSE for the offline-fit log."""
        return {"loss": float(((out - x) ** 2).mean().item())}
