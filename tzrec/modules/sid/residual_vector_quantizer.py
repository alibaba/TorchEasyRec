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

"""ResidualVectorQuantizer: multi-layer residual VQ with gradient training."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.kmeans_quantize import faiss_kmeans_fit
from tzrec.modules.sid.residual_quantizer import ResidualQuantizer
from tzrec.modules.sid.types import (
    QuantizeForwardMode,
    ResidualQuantizerOutput,
)
from tzrec.modules.sid.vector_quantize import VectorQuantizeLayer
from tzrec.utils.logging_util import logger


@torch.no_grad()
def faiss_residual_kmeans(
    samples: torch.Tensor,
    n_clusters_list: List[int],
    faiss_kmeans_kwargs: Optional[Dict] = None,
) -> List[torch.Tensor]:
    """Residual K-Means warm-start via FAISS, one pass per layer.

    Clusters ``samples``, subtracts each point's assigned centroid, and repeats
    on the residual per layer. Seeds the RQ-VAE codebook (via
    :meth:`ResidualVectorQuantizer.init_embed_`) from the first training batch.
    The fit is always CPU (host fp32 numpy copy); centroids return on
    ``samples.device`` — no faiss-gpu build needed.

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters_list (List[int]): per-layer cluster counts.
        faiss_kmeans_kwargs (Dict|None): extra kwargs for ``faiss.Kmeans``
            (e.g. ``{'niter': 10, 'seed': 123}``).

    Returns:
        List[Tensor]: per-layer centroids ``[(K0, D), ...]`` on samples.device.

    Raises:
        ImportError: if ``faiss`` is not installed.
        RuntimeError: if a layer has fewer points than its cluster count.
    """
    device = samples.device
    _, D = samples.shape
    # Own a contiguous fp32 numpy copy we mutate in place to form residuals.
    x = samples.detach().cpu().float().numpy().copy()

    res_centers: List[torch.Tensor] = []
    for n_clusters in n_clusters_list:
        km = faiss_kmeans_fit(x, D, n_clusters, faiss_kmeans_kwargs)
        centroids = km.centroids.copy()  # (K, D)
        res_centers.append(torch.from_numpy(centroids).to(device))
        _, idx = km.index.search(x, 1)
        x -= centroids[idx.ravel()]  # residual, in place
    return res_centers


class ResidualVectorQuantizer(ResidualQuantizer):
    """Multi-layer residual vector quantization.

    Each layer quantizes the residual from the previous layer:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = quantize(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Args:
        embed_dim (int): dimension of input embeddings.
        n_layers (int): number of quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        forward_mode (str): VQ forward mode ('ste'|'gumbel_softmax').
            Default: 'ste'.
        normalize_residuals (bool): L2-normalize residuals before each
            quantization layer. Default: False.
        distance_type (str): distance metric, 'l2' or 'cosine'. Default: 'l2'.
        rotation_trick (bool): use rotation trick for improved STE
            gradient estimation (arXiv:2410.06424). Default: False.
        kmeans_init (bool): use residual K-Means codebook initialization
            on first forward. Default: False.
        use_sinkhorn (bool): Sinkhorn uniform assignment. Default: True.
        sinkhorn_iters (int): Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness. Default: 10.0.
        gumbel_temperature (float): Gumbel-Softmax temperature. Default: 1.0.
    """

    _FORWARD_MODE_MAP = {
        "gumbel_softmax": QuantizeForwardMode.GUMBEL_SOFTMAX,
        "ste": QuantizeForwardMode.STE,
    }

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        forward_mode: str = "ste",
        normalize_residuals: bool = False,
        distance_type: str = "l2",
        rotation_trick: bool = False,
        kmeans_init: bool = False,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
        gumbel_temperature: float = 1.0,
    ) -> None:
        super().__init__(embed_dim, n_layers, n_embed, normalize_residuals)
        self.rotation_trick = rotation_trick

        # ``initted`` is the kmeans_init guard: True means "codebook has
        # been seeded", so init_embed_() becomes a no-op on later forwards.
        self.register_buffer("initted", torch.tensor([not kmeans_init]))

        if forward_mode not in self._FORWARD_MODE_MAP:
            raise ValueError(
                f"Unsupported forward_mode '{forward_mode}', "
                f"choose from {list(self._FORWARD_MODE_MAP.keys())}"
            )
        mode_enum = self._FORWARD_MODE_MAP[forward_mode]
        self._forward_mode = mode_enum
        is_gumbel = mode_enum == QuantizeForwardMode.GUMBEL_SOFTMAX
        # Sinkhorn is incompatible with Gumbel; auto-disable (the proto default
        # is on) instead of crashing.
        if is_gumbel and use_sinkhorn:
            logger.warning("gumbel_softmax: disabling incompatible use_sinkhorn.")
            use_sinkhorn = False
        # Gumbel skips the aggregate STE, so the rotation trick is unused.
        if is_gumbel and rotation_trick:
            logger.warning("gumbel_softmax: rotation_trick has no effect; ignoring.")

        distance_types = [distance_type] * n_layers

        self.layers = nn.ModuleList(
            [
                VectorQuantizeLayer(
                    embed_dim=embed_dim,
                    n_embed=self.n_embed_list[i],
                    forward_mode=mode_enum,
                    distance_type=distance_types[i],
                    use_sinkhorn=use_sinkhorn,
                    sinkhorn_iters=sinkhorn_iters,
                    sinkhorn_epsilon=sinkhorn_epsilon,
                    gumbel_temperature=gumbel_temperature,
                )
                for i in range(n_layers)
            ]
        )

        logger.info(
            "ResidualVectorQuantizer init: embed_dim=%d, n_layers=%d, "
            "n_embed=%s, forward_mode=%s, normalize_residuals=%s, "
            "distance_type=%s, rotation_trick=%s, kmeans_init=%s, "
            "use_sinkhorn=%s, sinkhorn_iters=%d, sinkhorn_epsilon=%s",
            embed_dim,
            n_layers,
            n_embed,
            forward_mode,
            normalize_residuals,
            distance_type,
            rotation_trick,
            kmeans_init,
            use_sinkhorn,
            sinkhorn_iters,
            sinkhorn_epsilon,
        )

    @torch.jit.ignore
    @torch.no_grad()
    def init_embed_(self, data: torch.Tensor) -> None:
        """Initialize codebook weights via FAISS residual K-Means.

        Runs once (kmeans_init=True, not yet initialized), seeding from the first
        training batch. Under DDP the fit happens on rank 0 and is broadcast, so
        every rank starts from the same codebook (averaging per-rank centroids
        would mix permutation-misaligned clusters into a near-random start).

        Args:
            data (Tensor): input data, shape (B, D).
        """
        if self.initted:
            return

        is_ddp = dist.is_initialized() and dist.get_world_size() > 1
        # The fit runs on rank 0 only, then broadcasts. faiss needs N >= max(K),
        # so a too-small rank-0 first batch would raise on rank 0 while the other
        # ranks block forever on the centroid broadcast. Broadcast rank 0's verdict
        # first so every rank aborts together with a clear error instead.
        max_k = max(self.n_embed_list)
        enough = torch.tensor([1 if data.shape[0] >= max_k else 0], device=data.device)
        if is_ddp:
            dist.broadcast(enough, src=0)
        if enough.item() == 0:
            raise RuntimeError(
                f"kmeans_init: rank-0 first training batch has fewer rows than the "
                f"largest codebook ({max_k}); raise batch_size or disable kmeans_init."
            )

        if (not is_ddp) or dist.get_rank() == 0:
            # TODO(follow-up): accumulate samples across multiple batches for the
            # warm-start fit instead of seeding from only the first training batch.
            centers = faiss_residual_kmeans(
                data,
                self.n_embed_list,
                {"niter": 10, "seed": 123, "verbose": False},
            )
        else:
            centers = [
                torch.empty(k, self.embed_dim, dtype=torch.float32, device=data.device)
                for k in self.n_embed_list
            ]
        if is_ddp:
            for c in centers:
                dist.broadcast(c, src=0)

        for i, layer in enumerate(self.layers):
            layer.embedding.weight.data.copy_(centers[i])

        self.initted.fill_(True)

    @staticmethod
    def _apply_rotation_trick(
        x: torch.Tensor,
        quant: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation trick for improved STE gradient estimation.

        Implements equation 4.2 from https://arxiv.org/abs/2410.06424.
        Replaces standard STE with a Householder reflection that rotates
        the gradient direction from x toward quant.

        Args:
            x (Tensor): original input with gradient, shape (B, D).
            quant (Tensor): quantized output (will be detached),
                shape (B, D).

        Returns:
            Tensor: rotated output with gradient flowing through x.
        """
        quant_detached = quant.detach()
        x_detached = x.detach()

        quant_norms = torch.linalg.vector_norm(quant_detached, dim=-1).unsqueeze(
            1
        )  # (B, 1)
        x_norms = torch.linalg.vector_norm(x_detached, dim=-1).unsqueeze(1)  # (B, 1)
        lambda_ = quant_norms / (x_norms + 1e-8)  # (B, 1)

        x_hat = x_detached / (x_norms + 1e-8)  # (B, D)
        quant_hat = quant_detached / (quant_norms + 1e-8)  # (B, D)

        normalized_sum = F.normalize(x_hat + quant_hat, p=2, dim=1)  # (B, D)

        x_unsq = x.unsqueeze(1)  # (B, 1, D)

        # Eq 4.2: Householder reflection
        sum_projection = (
            x_unsq @ normalized_sum.unsqueeze(2) @ normalized_sum.unsqueeze(1)
        )  # (B, 1, D)
        rescaled_embeddings = (
            x_unsq @ x_hat.unsqueeze(2) @ quant_hat.unsqueeze(1)
        )  # (B, 1, D)
        return lambda_ * (
            x_unsq - 2 * sum_projection + 2 * rescaled_embeddings
        ).squeeze(1)

    def _quantize_layer(
        self,
        layer_idx: int,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize one layer's residual via its ``VectorQuantizeLayer`` layer.

        STE: raw codebook vector (STE applied on the aggregate in :meth:`forward`).
        Gumbel: the soft embedding (carries grad directly).

        Args:
            layer_idx (int): quantization layer index.
            residual (Tensor): current residual, shape (B, D).

        Returns:
            ids (Tensor): per-layer cluster ids, shape (B,).
            emb (Tensor): the raw codebook vector (STE/eval) or the soft
                embedding (Gumbel), with grad, shape (B, D).
        """
        # On the STE residual walk the residual is detached, so the layer's
        # straight-through wrap is a numeric no-op; the real STE gradient comes
        # from the aggregate STE in :meth:`forward`. Gumbel returns the soft
        # embedding that carries grad directly.
        out = self.layers[layer_idx].quantize(residual)
        return out.ids, out.embeddings

    def forward(
        self,
        input: torch.Tensor,
    ) -> ResidualQuantizerOutput:
        """Forward the multi-layer residual quantization.

        Encoder gradient by ``forward_mode``: STE walks the DETACHED input and
        re-attaches grad via the aggregate STE below (codebook trains via the
        commitment loss); Gumbel's soft assignment is differentiable, so it walks
        the LIVE input and skips the aggregate STE.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            ResidualQuantizerOutput: (cluster_ids, quantized_embeddings,
                latents).
        """
        if self.training:
            self.init_embed_(input)  # first training forward only

        train_gumbel = (
            self.training and self._forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX
        )

        # cumulative[i] = sum after layer i.
        walk_input = input if train_gumbel else input.detach()
        cluster_ids, aggregated_quants, cumulative = self._residual_pass(walk_input)

        # Expose the per-layer cumulative quantized vectors (grad-carrying on the
        # codebook side) so the model-side CommitmentLoss can consume them; the
        # commitment loss is no longer computed inside the quantizer.
        latents = torch.stack(cumulative, dim=1)  # (B, n_layers, D)

        # Aggregate STE (STE only; Gumbel already carries grad).
        quants_trunc = aggregated_quants
        if self.training and not train_gumbel:
            if self.rotation_trick:
                quants_trunc = self._apply_rotation_trick(input, quants_trunc)
            else:
                quants_trunc = input + (quants_trunc - input).detach()

        return ResidualQuantizerOutput(
            cluster_ids=cluster_ids,
            quantized_embeddings=quants_trunc,
            latents=latents,
        )

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get codebook embedding weights for a specific layer.

        Detached read-only view for export/inspection (the layer's weight is a
        grad leaf, needed by the training ``lookup`` path).

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].get_codebook_embeddings().detach()

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors via the layer's embedding table."""
        return self.layers[layer_idx].lookup(code_idx)
