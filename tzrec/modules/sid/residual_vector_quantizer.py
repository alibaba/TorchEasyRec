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

from typing import List, Sequence, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.kmeans_quantize import faiss_residual_kmeans
from tzrec.modules.sid.residual_quantizer import ResidualQuantizer
from tzrec.modules.sid.types import (
    QuantizeForwardMode,
    ResidualQuantizerOutput,
)
from tzrec.modules.sid.vector_quantize import VectorQuantize
from tzrec.utils.logging_util import logger


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
        distance_type (str|List[str]): distance metric per layer,
            'l2' or 'cosine'. Supports per-layer list. Default: 'l2'.
        commitment_loss (str): commitment loss type, 'l2', 'l1' or 'cos'.
            Default: 'l2'.
        latent_weight (List[float]): commitment loss weights [w1, w2].
            w1: x toward quant (encoder side).
            w2: quant toward x (codebook side).
            Default: [1.0, 0.5].
        rotation_trick (bool): use rotation trick for improved STE
            gradient estimation (arXiv:2410.06424). Default: False.
        kmeans_init (bool): use residual K-Means codebook initialization
            on first forward. Default: False.
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
        distance_type: Union[str, List[str]] = "l2",
        commitment_loss: str = "l2",
        latent_weight: Sequence[float] = (1.0, 0.5),
        rotation_trick: bool = False,
        kmeans_init: bool = False,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__(embed_dim, n_layers, n_embed, normalize_residuals)
        assert commitment_loss in ("l2", "l1", "cos"), (
            f"commitment_loss must be 'l2', 'l1' or 'cos', got {commitment_loss!r}"
        )
        self.commitment_loss_type = commitment_loss
        self.rotation_trick = rotation_trick

        self.commitment_w1, self.commitment_w2 = latent_weight

        # ``initted`` is the kmeans_init guard: True means "codebook has
        # been seeded", so init_embed_() becomes a no-op on later forwards.
        self.register_buffer("initted", torch.tensor([not kmeans_init]))

        if forward_mode not in self._FORWARD_MODE_MAP:
            raise ValueError(
                f"Unsupported forward_mode '{forward_mode}', "
                f"choose from {list(self._FORWARD_MODE_MAP.keys())}"
            )
        mode_enum = self._FORWARD_MODE_MAP[forward_mode]

        if isinstance(distance_type, str):
            distance_types = [distance_type] * n_layers
        else:
            assert len(distance_type) == n_layers, (
                "length of distance_type and n_layers must be same, "
                f"but got {len(distance_type)} vs {n_layers}"
            )
            distance_types = list(distance_type)

        self.layers = nn.ModuleList(
            [
                VectorQuantize(
                    embed_dim=embed_dim,
                    n_embed=self.n_embed_list[i],
                    forward_mode=mode_enum,
                    distance_type=distance_types[i],
                    use_sinkhorn=use_sinkhorn,
                    sinkhorn_iters=sinkhorn_iters,
                    sinkhorn_epsilon=sinkhorn_epsilon,
                )
                for i in range(n_layers)
            ]
        )

        logger.info(
            "ResidualVectorQuantizer init: embed_dim=%d, n_layers=%d, "
            "n_embed=%s, forward_mode=%s, normalize_residuals=%s, "
            "distance_type=%s, commitment_loss=%s, latent_weight=%s, "
            "rotation_trick=%s, kmeans_init=%s, use_sinkhorn=%s, "
            "sinkhorn_iters=%d, sinkhorn_epsilon=%s",
            embed_dim,
            n_layers,
            n_embed,
            forward_mode,
            normalize_residuals,
            distance_type,
            commitment_loss,
            list(latent_weight),
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

        Only executed once when kmeans_init=True and not yet initialized.
        Uses the first batch of training data as the initialization pool.

        Under DDP the codebook is fit on rank 0 only and broadcast, so every
        rank starts from the SAME codebook. (Averaging per-rank centroids —
        the previous behavior — mixes permutation-misaligned clusters across
        ranks and yields a near-random warm start.)

        Args:
            data (Tensor): input data, shape (B, D).
        """
        if self.initted:
            return

        is_ddp = dist.is_initialized() and dist.get_world_size() > 1
        if (not is_ddp) or dist.get_rank() == 0:
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

    def _single_commitment_loss(
        self,
        x: torch.Tensor,
        quant: torch.Tensor,
    ) -> torch.Tensor:
        """Commitment loss for a single cumulative quantization tensor.

          - cos: (1 - cosine_similarity) * weight
          - l2:  (x - quant)^2.mean() * weight
          - l1:  |x - quant|.mean() * weight

        Both directions are always summed:
            loss1 = encoder-toward-quant (gradient flows into encoder)
            loss2 = quant-toward-encoder (gradient flows into codebook)

        Args:
            x (Tensor): original input, shape (B, D).
            quant (Tensor): cumulative quantized output at one layer,
                shape (B, D).

        Returns:
            Tensor: scalar commitment loss for this layer.
        """
        if self.commitment_loss_type == "cos":
            loss1 = (
                1 - F.cosine_similarity(x, quant.detach(), dim=-1)
            ).mean() * self.commitment_w1
            loss2 = (
                1 - F.cosine_similarity(x.detach(), quant, dim=-1)
            ).mean() * self.commitment_w2
        elif self.commitment_loss_type == "l1":
            loss1 = (x - quant.detach()).abs().mean() * self.commitment_w1
            loss2 = (x.detach() - quant).abs().mean() * self.commitment_w2
        else:  # 'l2'
            loss1 = (x - quant.detach()).pow(2.0).mean() * self.commitment_w1
            loss2 = (x.detach() - quant).pow(2.0).mean() * self.commitment_w2
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
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize one layer's residual via its ``VectorQuantize`` layer.

        Returns the raw (un-STE'd) codebook vector so gradient still flows into
        the codebook; STE is applied once on the aggregate in :meth:`forward`.

        Args:
            layer_idx (int): quantization layer index.
            residual (Tensor): current residual, shape (B, D).
            temperature (float): Gumbel-Softmax temperature.

        Returns:
            ids (Tensor): per-layer cluster ids, shape (B,).
            raw_emb (Tensor): raw codebook vectors (with grad), shape (B, D).
        """
        layer = self.layers[layer_idx]
        out = layer.quantize(residual, temperature)
        # ``lookup`` (QuantizeLayer base) returns the raw codebook vector
        # embedding.weight[ids] — gradient still flows into the codebook; STE is
        # applied once on the aggregate in :meth:`forward`. The soft STE/Gumbel
        # ``out.embeddings`` is intentionally not used in the residual walk.
        return out.ids, layer.lookup(out.ids)

    def forward(
        self,
        input: torch.Tensor,
        temperature: float = 1.0,
    ) -> ResidualQuantizerOutput:
        """Forward the multi-layer residual quantization.

        Training flow:
            1. If kmeans_init and not initialized -> init_embed_(input)
            2. Shared residual walk (:meth:`_residual_pass`) over the detached
               input: per-layer assign + grad-carrying accumulation.
            3. Mean of per-layer commitment losses over the cumulative quants
               (cos/l2 with latent_weight).
            4. STE gradient pass-through (or rotation trick).

        Args:
            input (Tensor): input embeddings, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.

        Returns:
            ResidualQuantizerOutput: (cluster_ids, quantized_embeddings,
                quantization_loss).
        """
        # Step 1: KMeans initialization (first training forward only)
        if self.training:
            self.init_embed_(input)

        # Step 2: shared residual walk on the detached input (encoder grad
        # flows only via the STE in step 4; the accumulated quants keep grad
        # so the codebook still trains). cumulative[i] = sum after layer i.
        cluster_ids, aggregated_quants, cumulative = self._residual_pass(
            input.detach(), temperature
        )

        # Step 3: aggregate per-layer commitment loss
        commitment_loss = torch.mean(
            torch.stack([self._single_commitment_loss(input, c) for c in cumulative])
        )

        # Step 4: STE or rotation trick (quants_trunc = final accumulated)
        quants_trunc = aggregated_quants
        if self.training:
            if self.rotation_trick:
                quants_trunc = self._apply_rotation_trick(input, quants_trunc)
            else:
                quants_trunc = input + (quants_trunc - input).detach()

        return ResidualQuantizerOutput(
            cluster_ids=cluster_ids,
            quantized_embeddings=quants_trunc,
            quantization_loss=commitment_loss,
        )

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get codebook embedding weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        return self.layers[layer_idx].get_codebook_embeddings()

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up codebook vectors via the layer's embedding table."""
        return self.layers[layer_idx].lookup(code_idx)
