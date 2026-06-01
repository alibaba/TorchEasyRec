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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.kmeans import KMeansLayer, recon_diagnostics
from tzrec.modules.sid_generation.residual_quantizer import ResidualQuantizer
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
            before each layer. Default: False.
        faiss_kmeans_kwargs (Dict|None): extra kwargs forwarded to
            ``faiss.Kmeans(D, K, **kwargs)`` (e.g. {'niter': 20,
            'gpu': True, 'verbose': True, 'spherical': False}).
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

    @property
    def all_initialized(self) -> bool:
        """Whether all layers have been initialized via offline FAISS."""
        return all(layer.is_initialized for layer in self.layers)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign codes per layer and sum the centroids.

        Codebook is read-only here; training happens in ``train_offline``.
        Uninitialized layers return dummy zeros so the model is callable
        before the one-shot FAISS fit completes.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            codes (Tensor): cluster indices per layer, shape (B, n_layers).
            quantized (Tensor): sum of quantized embeddings, shape (B, D).
        """
        residual = input
        all_codes: List[torch.Tensor] = []
        quantized_sum = torch.zeros_like(input)

        for layer in self.layers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            if layer.is_initialized:
                codes = layer.predict(residual)
                quantized = layer.centroids[codes]
                residual = residual - quantized
                quantized_sum = quantized_sum + quantized
            else:
                codes = torch.zeros(
                    input.shape[0], dtype=torch.long, device=input.device
                )
            all_codes.append(codes)

        cluster_ids = torch.stack(all_codes, dim=-1)  # (B, n_layers)
        return cluster_ids, quantized_sum

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating centroids."""
        residual = input
        all_codes: List[torch.Tensor] = []

        for layer in self.layers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            codes = layer.predict(residual)
            all_codes.append(codes)
            quantized = layer.centroids[codes]
            residual = residual - quantized

        return torch.stack(all_codes, dim=-1)

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

    @torch.no_grad()
    def train_offline(
        self,
        inputs: Union[torch.Tensor, "np.ndarray"],
        verbose: bool = True,
    ) -> None:
        """Train the multi-layer codebook via offline FAISS K-Means.

        FAISS consumes torch tensors directly (via ``faiss.contrib.
        torch_utils``) — no numpy round-trips. The residual matrix stays a
        host (CPU) tensor; when a faiss-gpu build is present, ``gpu=<dev>``
        moves only FAISS's internal, subsampled working set to the GPU, so we
        never hold (N, D) in VRAM. On a faiss-cpu build it runs on CPU
        unchanged. Either way the code path is identical.

        Args:
            inputs: full embedding matrix, shape (N, D), ``torch.Tensor`` or
                ``np.ndarray``. Copied once to an owned CPU float32 tensor;
                the caller's input is not mutated.
            verbose (bool): whether to print per-layer reconstruction
                loss. Default: True.

        Raises:
            ImportError: if ``faiss`` is not installed.
        """
        try:
            import faiss
            import faiss.contrib.torch_utils  # noqa: F401  (torch tensor I/O)
        except ImportError as e:
            raise ImportError(
                "faiss is required for ResidualKMeansQuantizer training. Install via "
                "`pip install faiss-cpu` or `pip install faiss-gpu`."
            ) from e

        # Own a contiguous CPU float32 tensor we can update in place for
        # residuals, without mutating the caller's input.
        if isinstance(inputs, torch.Tensor):
            assert inputs.dim() == 2 and inputs.shape[1] == self.embed_dim, (
                f"inputs must be (N, {self.embed_dim}), got {tuple(inputs.shape)}"
            )
            x = inputs.detach().to("cpu", torch.float32).contiguous().clone()
        else:
            assert inputs.ndim == 2 and inputs.shape[1] == self.embed_dim, (
                f"inputs must be (N, {self.embed_dim}), got {tuple(inputs.shape)}"
            )
            x = torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)).clone()
        N = x.shape[0]
        out = torch.zeros_like(x)

        # Use FAISS GPU compute when a GPU build is available (data stays on
        # host; FAISS streams only its subsampled training set to the device).
        # An explicit ``gpu`` in faiss_kmeans_kwargs always wins.
        kwargs = dict(self.faiss_kmeans_kwargs)
        if "gpu" not in kwargs:
            kwargs["gpu"] = (
                torch.cuda.current_device()
                if faiss.get_num_gpus() > 0 and torch.cuda.is_available()
                else False
            )

        # Chunk size for index.search to limit peak memory.
        # 500K × 512 × 4B ≈ 1 GB per chunk.
        SEARCH_CHUNK = 500_000

        for layer_idx in range(self.n_layers):
            if self.normalize_residuals:
                x = F.normalize(x, dim=-1)

            # Fresh Kmeans per layer so each layer can use its own K
            # (non-uniform codebooks supported). Index construction is a cheap
            # O(K*D) allocation next to train(), so this is effectively free.
            kmeans = faiss.Kmeans(
                self.embed_dim, self.n_embed_list[layer_idx], **kwargs
            )
            kmeans.train(x)
            centroids = torch.as_tensor(kmeans.centroids, dtype=torch.float32).cpu()

            for start in range(0, N, SEARCH_CHUNK):
                end = min(start + SEARCH_CHUNK, N)
                _, idx = kmeans.index.search(x[start:end], 1)
                idx = torch.as_tensor(idx, device="cpu").reshape(-1).long()
                q = centroids[idx]  # (chunk, D)
                out[start:end] += q
                x[start:end] -= q  # residual
                del idx, q

            if verbose:
                logger.info(
                    "[ResidualKMeansQuantizer][offline_faiss][layer %d] %s",
                    layer_idx,
                    self._calc_loss(out + x, out),  # x_in = out + residual
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
