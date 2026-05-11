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

"""ResidualQuantized: multi-layer residual vector quantization with VQ layers."""

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.kmeans import (
    _kmeans_plus_plus,
    _squared_euclidean_distance,
)
from tzrec.modules.sid_generation.types import (
    QuantizeForwardMode,
    QuantizeOutput,
    ResidualQuantizedOutput,
)
from tzrec.modules.sid_generation.vector_quantize import VectorQuantize




@torch.no_grad()
def _kmeans(
    samples: torch.Tensor,
    n_clusters: int,
    n_iters: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Lloyd's K-Means algorithm with KMeans++ initialization.

    Reference: al_sid/SID_generation/utils/kmeans.py::kmeans

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters (int): number of clusters K.
        n_iters (int): number of iterations. Default: 100.

    Returns:
        centroids (Tensor): cluster centers, shape (K, D).
        assignments (Tensor): cluster indices, shape (N,).
    """
    N, D = samples.shape
    centroids = _kmeans_plus_plus(samples, n_clusters)

    for _ in range(n_iters):
        dists = _squared_euclidean_distance(samples, centroids)  # (N, K)
        assignments = dists.argmin(dim=-1)  # (N,)

        bins = torch.bincount(assignments, minlength=n_clusters)
        zero_mask = bins == 0
        bins_clamped = bins.masked_fill(zero_mask, 1)

        new_centroids = torch.zeros_like(centroids)
        new_centroids.scatter_add_(
            0, assignments.unsqueeze(1).expand(-1, D), samples
        )
        new_centroids = new_centroids / bins_clamped.unsqueeze(1)

        # Keep old centroids for empty clusters
        centroids = torch.where(
            zero_mask.unsqueeze(1), centroids, new_centroids
        )

    return centroids, assignments


@torch.no_grad()
def _residual_kmeans(
    samples: torch.Tensor,
    n_clusters_list: List[int],
    n_iters: int = 100,
) -> List[torch.Tensor]:
    """Residual K-Means: sequentially cluster and subtract centroids.

    Reference: al_sid/SID_generation/utils/kmeans.py::residual_kmeans

    Args:
        samples (Tensor): data points, shape (N, D).
        n_clusters_list (List[int]): per-layer cluster counts.
        n_iters (int): K-Means iterations. Default: 100.

    Returns:
        List[Tensor]: per-layer centroids [(K0, D), (K1, D), ...].
    """
    res_centers = []
    for n_clusters in n_clusters_list:
        centroids, assignments = _kmeans(samples, n_clusters, n_iters)
        res_centers.append(centroids)
        samples = samples - centroids[assignments]
    return res_centers




class ResidualQuantized(nn.Module):
    """Multi-layer residual vector quantization.

    Each layer quantizes the residual from the previous layer:
        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = quantize(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1})

    Reference: al_sid/SID_generation/rqvae_embed/quantizations.py
        ::RQBottleneck

    Args:
        embed_dim (int): dimension of input embeddings.
        n_layers (int): number of quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        forward_mode (str): VQ forward mode ('ste'|'gumbel_softmax').
            Default: 'ste'.
        normalize_residuals (bool): L2-normalize residuals before each
            quantization layer. Default: False.
        shared_codebook (bool): share codebook across all layers.
            Default: False.
        distance_type (str|List[str]): distance metric per layer,
            'l2' or 'cosine'. Supports per-layer list. Default: 'l2'.
        commitment_loss (str): commitment loss type, 'l2' or 'cos'.
            Default: 'l2'.
        latent_weight (List[float]): commitment loss weights [w1, w2].
            w1: x toward quant (always active).
            w2: quant toward x (only when use_ema=False).
            Default: [1.0, 0.5].
        rotation_trick (bool): use rotation trick for improved STE
            gradient estimation (arXiv:2410.06424). Default: False.
        kmeans_init (bool): use residual K-Means codebook initialization
            on first forward. Default: False.
        use_ema (bool): EMA codebook update. Default: True.
        ema_decay (float): EMA decay coefficient. Default: 0.99.
        restart_unused_codes (bool): reset dead codes. Default: True.
        use_sinkhorn (bool): Sinkhorn uniform assignment. Default: True.
        sinkhorn_iters (int): Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness. Default: 10.0.
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
        shared_codebook: bool = False,
        distance_type: Union[str, List[str]] = "l2",
        commitment_loss: str = "l2",
        latent_weight: Optional[List[float]] = None,
        rotation_trick: bool = False,
        kmeans_init: bool = False,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.shared_codebook = shared_codebook
        self.commitment_loss_type = commitment_loss
        self.use_ema = use_ema
        self.rotation_trick = rotation_trick

        if latent_weight is None:
            latent_weight = [1.0, 0.5]
        self.commitment_w1, self.commitment_w2 = latent_weight

        # KMeans initialization control
        self.register_buffer("initted", torch.tensor([not kmeans_init]))

        if forward_mode not in self._FORWARD_MODE_MAP:
            raise ValueError(
                f"Unsupported forward_mode '{forward_mode}', "
                f"choose from {list(self._FORWARD_MODE_MAP.keys())}"
            )
        mode_enum = self._FORWARD_MODE_MAP[forward_mode]

        # Parse n_embed list
        if isinstance(n_embed, int):
            n_embed_list = [n_embed] * n_layers
        else:
            assert len(n_embed) == n_layers, (
                "length of n_embed and n_layers must be same, "
                f"but got {len(n_embed)} vs {n_layers}"
            )
            n_embed_list = list(n_embed)
        self.n_embed_list = n_embed_list

        # Parse distance_type list
        if isinstance(distance_type, str):
            distance_types = [distance_type] * n_layers
        else:
            assert len(distance_type) == n_layers, (
                "length of distance_type and n_layers must be same, "
                f"but got {len(distance_type)} vs {n_layers}"
            )
            distance_types = list(distance_type)

        # Build VQ layers
        if shared_codebook:
            base_layer = VectorQuantize(
                embed_dim=embed_dim,
                n_embed=n_embed_list[0],
                forward_mode=mode_enum,
                distance_type=distance_types[0],
                use_ema=use_ema,
                ema_decay=ema_decay,
                restart_unused_codes=restart_unused_codes,
                use_sinkhorn=use_sinkhorn,
                sinkhorn_iters=sinkhorn_iters,
                sinkhorn_epsilon=sinkhorn_epsilon,
            )
            self.layers = nn.ModuleList([base_layer] * n_layers)
        else:
            self.layers = nn.ModuleList(
                [
                    VectorQuantize(
                        embed_dim=embed_dim,
                        n_embed=n_embed_list[i],
                        forward_mode=mode_enum,
                        distance_type=distance_types[i],
                        use_ema=use_ema,
                        ema_decay=ema_decay,
                        restart_unused_codes=restart_unused_codes,
                        use_sinkhorn=use_sinkhorn,
                        sinkhorn_iters=sinkhorn_iters,
                        sinkhorn_epsilon=sinkhorn_epsilon,
                    )
                    for i in range(n_layers)
                ]
            )


    @torch.jit.ignore
    @torch.no_grad()
    def init_embed_(self, data: torch.Tensor) -> None:
        """Initialize codebook weights via residual K-Means.

        Only executed once when kmeans_init=True and not yet initialized.
        Uses the first batch of training data as initialization pool.

        Args:
            data (Tensor): input data, shape (B, D).
        """
        if self.initted:
            return

        centers = _residual_kmeans(data, self.n_embed_list)

        # Distributed sync
        if dist.is_initialized() and dist.get_world_size() > 1:
            for c in centers:
                dist.all_reduce(c, op=dist.ReduceOp.SUM)
                c /= dist.get_world_size()

        for i, layer in enumerate(self.layers):
            layer.embedding.weight.data.copy_(centers[i])

        self.initted.fill_(True)


    def _single_commitment_loss(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
    ) -> torch.Tensor:
        """Commitment loss for a single cumulative quantization tensor.

          - cos: (1 - cosine_similarity) * weight
          - l2:  (x - quant)^2.mean() * weight
        EMA mode zeros out the codebook-toward-encoder term.

        Reference: al_sid::RQBottleneck.compute_commitment_loss

        Args:
            x (Tensor): original input, shape (B, D).
            quant (Tensor): cumulative quantized output at one layer,
                shape (B, D).

        Returns:
            Tensor: scalar commitment loss for this layer.
        """
        if self.commitment_loss_type == "cos":
            loss1 = (
                (1 - F.cosine_similarity(x, quant.detach(), dim=-1))
                .mean()
                * self.commitment_w1
            )
            if self.use_ema:
                loss2 = torch.tensor(0.0, device=x.device)
            else:
                loss2 = (
                    (1 - F.cosine_similarity(
                        x.detach(), quant, dim=-1
                    ))
                    .mean()
                    * self.commitment_w2
                )
        else:  # 'l2'
            loss1 = (
                (x - quant.detach()).pow(2.0).mean()
                * self.commitment_w1
            )
            if self.use_ema:
                loss2 = torch.tensor(0.0, device=x.device)
            else:
                loss2 = (
                    (x.detach() - quant).pow(2.0).mean()
                    * self.commitment_w2
                )
        return loss1 + loss2


    @staticmethod
    def _apply_rotation_trick(
        x: torch.Tensor,
        quant: torch.Tensor,
    ) -> torch.Tensor:
        """Apply rotation trick for improved STE gradient estimation.

        Implements equation 4.2 from https://arxiv.org/abs/2410.06424.
        Replaces standard STE with a Householder reflection that rotates
        the gradient direction from x toward quant.

        Reference: GRID/src/components/quantization_strategies.py
            ::RotationTrickQuantization.rotate_and_scale_batch

        Args:
            x (Tensor): original input with gradient, shape (B, D).
            quant (Tensor): quantized output (will be detached),
                shape (B, D).

        Returns:
            Tensor: rotated output with gradient flowing through x.
        """
        quant_detached = quant.detach()
        x_detached = x.detach()

        quant_norms = torch.linalg.vector_norm(
            quant_detached, dim=-1
        ).unsqueeze(1)  # (B, 1)
        x_norms = torch.linalg.vector_norm(
            x_detached, dim=-1
        ).unsqueeze(1)  # (B, 1)
        lambda_ = quant_norms / (x_norms + 1e-8)  # (B, 1)

        x_hat = x_detached / (x_norms + 1e-8)  # (B, D)
        quant_hat = quant_detached / (quant_norms + 1e-8)  # (B, D)

        normalized_sum = F.normalize(
            x_hat + quant_hat, p=2, dim=1
        )  # (B, D)

        x_unsq = x.unsqueeze(1)  # (B, 1, D)

        # Eq 4.2: Householder reflection
        sum_projection = (
            x_unsq
            @ normalized_sum.unsqueeze(2)
            @ normalized_sum.unsqueeze(1)
        )  # (B, 1, D)
        rescaled_embeddings = (
            x_unsq
            @ x_hat.unsqueeze(2)
            @ quant_hat.unsqueeze(1)
        )  # (B, 1, D)
        return (
            lambda_
            * (x_unsq - 2 * sum_projection + 2 * rescaled_embeddings)
            .squeeze(1)
        )


    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    def forward(
        self,
        input: torch.Tensor,
        temperature: float = 1.0,
        reference_code: Optional[torch.Tensor] = None,
        ema_mask: Optional[torch.Tensor] = None,
    ) -> ResidualQuantizedOutput:
        """Forward the multi-layer residual quantization.

        Training flow:
            1. If kmeans_init and not initialized -> init_embed_(input)
            2. For each layer: quantize detached residual, accumulate
               into aggregated_quants and compute per-layer commitment loss
               in-place (avoids storing a quant_list of clones).
               - pass reference_code[:, i] if provided
            3. Mean of per-layer commitment losses (cos/l2 with latent_weight)
            4. STE gradient pass-through (or rotation trick)

        Args:
            input (Tensor): input embeddings, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.
            reference_code (Tensor, optional): reference codebook indices,
                shape (B, n_layers). If provided, each layer receives
                reference_code[:, i] for probabilistic replacement.
            ema_mask (Tensor, optional): per-row EMA mask, shape (B,)
                float. Passed through to each VectorQuantize layer.

        Returns:
            ResidualQuantizedOutput: (cluster_ids, quantized_embeddings,
                quantization_loss).
        """
        # Step 1: KMeans initialization (first training forward only)
        if self.training:
            self.init_embed_(input)

        # Detach residual for VQ assignment (gradient flows via STE only)
        residual = input.detach().clone()
        all_ids: List[torch.Tensor] = []
        commitment_loss_list: List[torch.Tensor] = []
        aggregated_quants = torch.zeros_like(input)

        # Step 2: per-layer residual quantization
        for i, layer in enumerate(self.layers):
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)

            ref_code_i = (
                reference_code[:, i]
                if reference_code is not None
                else None
            )

            # VQ forward: assignment + EMA update (internally)
            quantized = layer(
                residual, temperature=temperature,
                reference_code=ref_code_i,
                ema_mask=ema_mask,
            )
            all_ids.append(quantized.ids)

            # Raw embedding lookup for commitment loss accumulation
            raw_emb = layer.embedding(quantized.ids)

            # Update residual with detached embedding
            residual = residual - raw_emb.detach()

            # Accumulate raw embeddings (preserves gradient to codebook)
            aggregated_quants = aggregated_quants + raw_emb

            # Compute per-layer commitment loss in-place (no quant_list clone)
            commitment_loss_list.append(
                self._single_commitment_loss(input, aggregated_quants)
            )

        cluster_ids = torch.stack(all_ids, dim=-1)  # (B, n_layers)

        # Step 3: aggregate per-layer commitment loss
        commitment_loss = torch.mean(torch.stack(commitment_loss_list))

        # Step 4: STE or rotation trick (quants_trunc = final accumulated)
        quants_trunc = aggregated_quants
        if self.training:
            if self.rotation_trick:
                quants_trunc = self._apply_rotation_trick(
                    input, quants_trunc
                )
            else:
                quants_trunc = input + (quants_trunc - input).detach()

        return ResidualQuantizedOutput(
            cluster_ids=cluster_ids,
            quantized_embeddings=quants_trunc,
            quantization_loss=commitment_loss,
        )

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without gradient computation.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        output = self.forward(input)
        return output.cluster_ids

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get codebook embedding weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].embedding.weight.data

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
            emb = layer.embedding(codes[:, i])
            quantized_sum = quantized_sum + emb
        return quantized_sum
