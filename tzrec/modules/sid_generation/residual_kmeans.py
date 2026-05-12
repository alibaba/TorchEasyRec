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

"""Multi-layer residual K-Means: ResidualKMeans and RQKMeans wrapper."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.kmeans import MiniBatchKMeans

VALID_TRAIN_MODES = ("online", "offline_faiss")


class ResidualKMeans(nn.Module):
    """Multi-layer residual Mini-Batch K-Means.

    Each layer quantizes the residual from the previous layer using
    MiniBatchKMeans online clustering:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i.train_step(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::ResidualMiniBatchKMeans

    Args:
        embed_dim (int): feature dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): number of clusters per layer.
            Default: 256.
        normalize_residuals (bool): whether to L2-normalize residuals
            before each layer. Default: False.
        init_buffer_size (int): buffer size for KMeans++ initialization
            (only used in 'online' mode). Default: 3072.
        train_mode (str): training backend, one of {'online',
            'offline_faiss'}. Default: 'online'.
            * 'online'        : original Mini-Batch K-Means; centroids are
                                updated incrementally inside ``forward``.
            * 'offline_faiss' : codebook is built once via ``train_offline``
                                using FAISS over the full embedding matrix;
                                ``forward`` becomes effectively read-only
                                (predict + lookup).
        faiss_kmeans_kwargs (Dict|None): extra kwargs forwarded to
            ``faiss.Kmeans(D, K, **kwargs)`` when ``train_mode ==
            'offline_faiss'`` (e.g. {'niter': 20, 'gpu': True,
            'verbose': True, 'spherical': False}). Ignored otherwise.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        init_buffer_size: int = 3072,
        train_mode: str = "online",
        faiss_kmeans_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        assert train_mode in VALID_TRAIN_MODES, (
            f"train_mode must be one of {VALID_TRAIN_MODES}, got {train_mode}"
        )
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.train_mode = train_mode
        self.faiss_kmeans_kwargs = dict(faiss_kmeans_kwargs or {})

        if isinstance(n_embed, int):
            n_embed_list = [n_embed] * n_layers
        else:
            assert len(n_embed) == n_layers, (
                "length of n_embed and n_layers must be same, "
                f"but got {len(n_embed)} vs {n_layers}"
            )
            n_embed_list = list(n_embed)
        self.n_embed_list = n_embed_list

        self.layers = nn.ModuleList(
            [
                MiniBatchKMeans(
                    n_clusters=n_embed_list[i],
                    n_features=embed_dim,
                    init_buffer_size=init_buffer_size,
                )
                for i in range(n_layers)
            ]
        )

    @property
    def all_initialized(self) -> bool:
        """Whether all layers have been initialized via KMeans++."""
        return all(layer.is_initialized for layer in self.layers)

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim


    def forward(
        self, input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward the multi-layer residual K-Means.

        During training: calls layer.train_step() to update centroids.
        During inference: calls layer.predict() without update.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            codes (Tensor): cluster indices per layer,
                shape (B, n_layers).
            quantized (Tensor): sum of quantized embeddings,
                shape (B, D).
        """
        residual = input
        all_codes: List[torch.Tensor] = []
        quantized_sum = torch.zeros_like(input)

        for layer in self.layers:
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            if self.training and self.train_mode == "online":
                codes, quantized = layer.train_step(residual)
            else:
                codes = layer.predict(residual)
                quantized = layer.centroids[codes]

            all_codes.append(codes)

            # Only update residual if layer is initialized
            # (uninitialized layers return dummy zeros)
            if layer.is_initialized:
                residual = residual - quantized
                quantized_sum = quantized_sum + quantized

        cluster_ids = torch.stack(all_codes, dim=-1)  # (B, n_layers)
        return cluster_ids, quantized_sum


    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating centroids.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
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

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes.

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, D).
        """
        quantized_sum = torch.zeros(
            codes.shape[0], self.embed_dim,
            device=codes.device, dtype=torch.float,
        )
        for i, layer in enumerate(self.layers):
            emb = layer.centroids[codes[:, i]]
            quantized_sum = quantized_sum + emb
        return quantized_sum


    @torch.no_grad()
    def train_offline(
        self,
        inputs: Union[torch.Tensor, "np.ndarray"],
        verbose: bool = True,
    ) -> None:
        """Train the multi-layer codebook via offline FAISS K-Means.

        Args:
            inputs: full embedding matrix, shape (N, D). Either a
                ``torch.Tensor`` (will be copied to numpy) or a
                ``np.ndarray`` (ownership transferred; caller MUST
                release any outside reference — the array is mutated
                in-place to compute residuals layer by layer).
            verbose (bool): whether to print per-layer reconstruction
                loss. Default: True.

        Raises:
            AssertionError: if ``train_mode != 'offline_faiss'``.
            ImportError: if ``faiss`` is not installed.
        """
        assert self.train_mode == "offline_faiss", (
            "train_offline() requires train_mode='offline_faiss', "
            f"got {self.train_mode}"
        )
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "faiss is required for offline_faiss mode. Install via "
                "`pip install faiss-cpu` or `pip install faiss-gpu`."
            ) from e

        # Materialise to a float32 contiguous numpy array that we own
        # (so in-place residual updates are safe).
        if isinstance(inputs, torch.Tensor):
            assert inputs.dim() == 2 and inputs.shape[1] == self.embed_dim, (
                f"inputs must be (N, {self.embed_dim}), "
                f"got {tuple(inputs.shape)}"
            )
            # Tensor path still requires a copy; caller will hold a
            # reference until we return, so we must not alias it.
            x = inputs.detach().cpu().float().numpy().copy()
        else:
            assert inputs.ndim == 2 and inputs.shape[1] == self.embed_dim, (
                f"inputs must be (N, {self.embed_dim}), "
                f"got {tuple(inputs.shape)}"
            )
            # Numpy path: take ownership — no extra copy. Caller promises
            # the array is no longer used outside. Only ensure dtype
            # + contiguity (zero-copy when already satisfied).
            x = np.ascontiguousarray(inputs, dtype=np.float32)
        N, D = x.shape
        out = np.zeros((N, D), dtype=np.float32)

        # 对齐OneRec: 循环外创建一次 kmeans 实例复用
        n_embed = self.n_embed_list[0]
        kmeans = faiss.Kmeans(
            self.embed_dim, n_embed, **self.faiss_kmeans_kwargs
        )

        # Chunk size for index.search to limit peak memory.
        # 500K × 512 × 4B ≈ 1 GB per chunk.
        SEARCH_CHUNK = 500_000

        for layer_idx in range(self.n_layers):
            # ---- normalise residuals in-place (saves N×D allocation) ----
            if self.normalize_residuals:
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                np.maximum(norms, 1e-8, out=norms)
                x /= norms                             # in-place

            # ---- FAISS train (internally subsamples) ----
            kmeans.train(x)

            # ---- chunked search + quantize + residual (in-place) ----
            for start in range(0, N, SEARCH_CHUNK):
                end = min(start + SEARCH_CHUNK, N)
                _, idx = kmeans.index.search(x[start:end], 1)
                q = kmeans.centroids[idx.ravel()]       # (chunk, D)
                out[start:end] += q
                x[start:end] -= q                       # residual
                del idx, q

            # ---- verbose diagnostic ----
            if verbose:
                out_t = torch.from_numpy(out)
                ref_t = torch.from_numpy(out + x)       # x_in = out + residual
                print(
                    f"[ResidualKMeans][offline_faiss][layer {layer_idx}] "
                    f"{self._calc_loss(ref_t, out_t)}"
                )
                del out_t, ref_t

            centroids_t = torch.from_numpy(kmeans.centroids.copy())
            self.layers[layer_idx].load_centroids_(centroids_t)
            if verbose:
                print(
                    f"[ResidualKMeans][offline_faiss] layer {layer_idx} finished"
                )

    @staticmethod
    def _calc_loss(
        x: torch.Tensor, out: torch.Tensor, epsilon: float = 1e-4
    ) -> Dict[str, float]:
        """Reconstruction loss diagnostics (MSE + relative L1).

        Mirrors ``OpenOneRec/tokenizer/res_kmeans.py::ResKmeans.calc_loss``.
        """
        loss = ((out - x) ** 2).mean()
        rel_loss = (
            torch.abs(x - out)
            / (torch.maximum(torch.abs(x), torch.abs(out)) + epsilon)
        ).mean()
        return {"loss": float(loss.item()), "rel_loss": float(rel_loss.item())}




class RQKMeans(nn.Module):
    """RQ-KMeans: multi-layer residual Mini-Batch K-Means.

    No Encoder/Decoder — directly clusters input vectors via residual
    K-Means, updating centroids online without gradient backpropagation.

    Reference: al_sid/SID_generation/rqvae_embed/grid_kmeans.py
        ::ResidualMiniBatchKMeans (top-level usage)

    Args:
        embed_dim (int): feature dimension. Default: 64.
        n_layers (int): number of residual quantization layers. Default: 3.
        n_embed (int|List[int]): number of clusters per layer. Default: 256.
        normalize_residuals (bool): L2-normalize residuals before each
            layer. Default: False.
        init_buffer_size (int): buffer size for KMeans++ initialization
            (only used in 'online' mode). Default: 3072.
        train_mode (str): training backend, one of {'online',
            'offline_faiss'}. Default: 'online'. See
            :class:`ResidualKMeans` for semantics.
        faiss_kmeans_kwargs (Dict|None): extra kwargs forwarded to
            ``faiss.Kmeans(...)`` when ``train_mode == 'offline_faiss'``.
            Ignored otherwise.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        n_layers: int = 3,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
        init_buffer_size: int = 3072,
        train_mode: str = "online",
        faiss_kmeans_kwargs: Optional[Dict] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.train_mode = train_mode
        self.quantizer = ResidualKMeans(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_embed=n_embed,
            normalize_residuals=normalize_residuals,
            init_buffer_size=init_buffer_size,
            train_mode=train_mode,
            faiss_kmeans_kwargs=faiss_kmeans_kwargs,
        )

    def train_offline(
        self,
        inputs: Union[torch.Tensor, "np.ndarray"],
        verbose: bool = True,
    ) -> None:
        """Build codebook offline via FAISS (only in 'offline_faiss' mode).

        Args:
            inputs: full embedding matrix, shape (N, embed_dim). Either
                a ``torch.Tensor`` or an ``np.ndarray`` (ownership
                transferred — array is mutated in-place).
            verbose (bool): print per-layer reconstruction loss.
        """
        assert self.train_mode == "offline_faiss", (
            "RQKMeans.train_offline() only valid in offline_faiss mode, "
            f"got {self.train_mode}"
        )
        self.quantizer.train_offline(inputs, verbose=verbose)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward: direct residual K-Means quantization.

        During training: updates centroids via mini-batch steps.
        During inference: assigns to nearest centroids without update.

        Args:
            x: (B, embed_dim) input features.

        Returns:
            dict with keys:
                'codes':     (B, n_layers)  semantic IDs.
                'quantized': (B, embed_dim) quantized vector (sum of centroids).
        """
        codes, quantized = self.quantizer(x)
        return {
            "codes": codes,
            "quantized": quantized,
        }

    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: get semantic IDs.

        Args:
            x: (B, embed_dim) input features.

        Returns:
            Tensor: codes, shape (B, n_layers).
        """
        return self.quantizer.get_codes(x)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct vectors from semantic IDs (centroid lookup + sum).

        Args:
            codes: (B, n_layers) semantic ID codes.

        Returns:
            Tensor: quantized, shape (B, embed_dim).
        """
        return self.quantizer.decode_codes(codes)

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get centroid weights for a specific layer.

        Args:
            layer_idx: index of the quantization layer.

        Returns:
            Tensor: centroids, shape (n_embed, embed_dim).
        """
        return self.quantizer.get_codebook_embeddings(layer_idx)
