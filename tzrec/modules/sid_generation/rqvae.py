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

"""RQVAE: Encoder + ResidualQuantized + Decoder top-level wrapper."""

from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.clip_loss import CLIPLoss, MaskedCLIPLoss
from tzrec.modules.sid_generation.residual_quantized import ResidualQuantized
from tzrec.utils.logging_util import logger


class RQVAE(nn.Module):
    """RQ-VAE: Encoder + ResidualQuantized + Decoder.

    Supports optional CLIP contrastive learning. When use_clip=True,
    forward accepts paired inputs (fea1, fea2) and computes CLIP loss
    via a siamese network (shared parameters).

    Encoder/Decoder are configurable-depth MLPs built via hidden_dims:
        Encoder: input_dim -> hidden_dims[0] -> ... -> hidden_dims[-1] -> embed_dim
        Decoder: embed_dim -> hidden_dims[-1] -> ... -> hidden_dims[0] -> input_dim
    ReLU activation between hidden layers. Decoder reverses hidden_dims
    for symmetric structure.

    Args:
        input_dim (int): original embedding dimension. Default: 512.
        embed_dim (int): latent space dimension. Default: 64.
        hidden_dims (List[int]): encoder hidden layer dimensions.
            Decoder automatically reverses for symmetry.
            Default: [input_dim // 2].
        n_layers (int): number of residual quantization layers. Default: 3.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        forward_mode (str): VQ forward mode ('ste'|'gumbel_softmax').
            Default: 'ste'.
        normalize_residuals (bool): L2-normalize residuals. Default: False.
        shared_codebook (bool): share codebook across layers. Default: False.
        distance_type (str|List[str]): distance metric ('l2'|'cosine').
            Default: 'l2'.
        commitment_loss (str|None): commitment loss type ('l2'|'cos').
            Default: follows loss_type (al_sid behavior).
        latent_weight (List[float]): commitment loss weights [w1, w2].
            Default: [1.0, 0.5].
        rotation_trick (bool): STE rotation trick. Default: False.
        kmeans_init (bool): KMeans codebook initialization. Default: True.
        use_ema (bool): EMA codebook update. Default: True.
        ema_decay (float): EMA decay coefficient. Default: 0.99.
        restart_unused_codes (bool): reset dead codes. Default: True.
        use_sinkhorn (bool): Sinkhorn uniform assignment. Default: True.
        sinkhorn_iters (int): Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness. Default: 10.0.
        loss_type (str): reconstruction loss ('mse'|'l1'|'cosine').
            Default: 'mse'.
        use_clip (bool): enable CLIP contrastive learning. Default: False.
    """

    @staticmethod
    def _build_mlp(dims: List[int]) -> nn.Sequential:
        """Build MLP: dims[0] -> ... -> dims[-1], ReLU between hidden layers."""
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation after last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __init__(
        self,
        input_dim: int = 512,
        embed_dim: int = 64,
        hidden_dims: Optional[List[int]] = None,
        n_layers: int = 3,
        n_embed: Union[int, List[int]] = 256,
        forward_mode: str = "ste",
        normalize_residuals: bool = False,
        shared_codebook: bool = False,
        distance_type: Union[str, List[str]] = "l2",
        commitment_loss: Optional[str] = None,
        latent_weight: Sequence[float] = (1.0, 0.5),
        rotation_trick: bool = False,
        kmeans_init: bool = True,
        use_ema: bool = True,
        ema_decay: float = 0.99,
        restart_unused_codes: bool = True,
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
        loss_type: str = "mse",
        use_clip: bool = False,
    ) -> None:
        super().__init__()

        assert loss_type in ("mse", "l1", "cosine"), (
            f"loss_type must be 'mse', 'l1' or 'cosine', got '{loss_type}'"
        )
        self.loss_type = loss_type
        self.use_clip = use_clip
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self._is_inference = False

        if hidden_dims is None:
            hidden_dims = [input_dim // 2]

        # commitment_loss defaults to follow loss_type (al_sid behavior:
        # commitment_loss=loss_type, so mse -> l2 branch)
        if commitment_loss is None:
            commitment_loss = "l2" if loss_type == "mse" else loss_type

        enc_dims = [input_dim] + list(hidden_dims) + [embed_dim]
        self.encoder = self._build_mlp(enc_dims)

        # Decoder is the symmetric reverse of the encoder.
        dec_dims = [embed_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder = self._build_mlp(dec_dims)

        self.quantizer = ResidualQuantized(
            embed_dim=embed_dim,
            n_layers=n_layers,
            n_embed=n_embed,
            forward_mode=forward_mode,
            normalize_residuals=normalize_residuals,
            shared_codebook=shared_codebook,
            distance_type=distance_type,
            commitment_loss=commitment_loss,
            latent_weight=latent_weight,
            rotation_trick=rotation_trick,
            kmeans_init=kmeans_init,
            use_ema=use_ema,
            ema_decay=ema_decay,
            restart_unused_codes=restart_unused_codes,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )

        # CLIP contrastive learning (optional)
        if use_clip:
            self.logit_scale_self = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.logit_scale_cl = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.logit_scale = nn.Parameter(
                torch.ones([]) * np.log(1 / 0.07)
            )
            self.masked_clip_loss_fn = MaskedCLIPLoss()

        logger.info("RQVAE init: %s", {
            k: v for k, v in vars(self).items()
            if not k.startswith("_") and k != "training"
        })

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode. (B, input_dim) -> (B, embed_dim)."""
        return self.encoder(x)

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode. (B, embed_dim) -> (B, input_dim)."""
        return self.decoder(z_q)

    def _cosine_loss(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """Cosine distance loss: 1 - mean(cos_sim)."""
        return (1 - F.cosine_similarity(x1, x2, dim=1)).mean()

    def compute_loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        quant_loss: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute reconstruction loss + quantization loss + total loss.

        loss_total = recon_loss + quant_loss
        Note: al_sid latent_loss_weight is declared but unused;
        commitment_loss is added 1:1 with recon_loss. We align with this.

        Args:
            x: original input, shape (B, input_dim).
            x_hat: reconstructed output, shape (B, input_dim).
            quant_loss: quantization (commitment) loss scalar.

        Returns:
            dict with 'reconstruction_loss', 'quantization_loss', 'loss'.
        """
        if self.loss_type == "mse":
            recon_loss = F.mse_loss(x_hat, x, reduction="mean")
        elif self.loss_type == "l1":
            recon_loss = F.l1_loss(x_hat, x, reduction="mean")
        elif self.loss_type == "cosine":
            recon_loss = self._cosine_loss(x_hat, x)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")

        loss_total = recon_loss + quant_loss

        return {
            "reconstruction_loss": recon_loss,
            "quantization_loss": quant_loss,
            "loss": loss_total,
        }

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Dispatch based on use_clip.

        use_clip=False: forward(x) -> forward_rqvae(x)
        use_clip=True:  forward(fea1, fea2, clip_mask) -> forward_mixed(...)
        """
        if self._is_inference or not self.use_clip:
            assert len(args) >= 1, "Standard mode requires (x,)"
            return self.forward_rqvae(args[0], **kwargs)
        else:
            assert len(args) == 3, "Mixed mode requires (fea1, fea2, clip_mask)"
            return self.forward_mixed(args[0], args[1], args[2], **kwargs)

    def forward_rqvae(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE forward: encode -> quantize -> decode -> loss.

        Args:
            x: (B, input_dim) original embedding.
            temperature: Gumbel-Softmax temperature.

        Returns:
            dict with keys: 'x_hat', 'codes', 'quantized',
                'reconstruction_loss', 'quantization_loss', 'loss'.
        """
        z_e = self.encode(x)
        quant_output = self.quantizer(z_e, temperature=temperature)
        x_hat = self.decode(quant_output.quantized_embeddings)

        losses = self.compute_loss(
            x, x_hat, quant_output.quantization_loss
        )

        return {
            "x_hat": x_hat,
            "codes": quant_output.cluster_ids,
            "quantized": quant_output.quantized_embeddings,
            **losses,
        }

    def _compute_masked_recon_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        recon_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-sample recon loss, masked to recon rows only.

        No boolean indexing, no data-dependent branching,
        compatible with torch.compile.

        Args:
            x_hat: (B, D) reconstructed output.
            x: (B, D) original input.
            recon_mask: (B,) bool, True = recon row.
        """
        if self.loss_type == "mse":
            per_sample = F.mse_loss(
                x_hat, x, reduction="none"
            ).mean(dim=-1)
        elif self.loss_type == "l1":
            per_sample = F.l1_loss(
                x_hat, x, reduction="none"
            ).mean(dim=-1)
        elif self.loss_type == "cosine":
            per_sample = 1 - F.cosine_similarity(x_hat, x, dim=-1)
        else:
            raise ValueError(f"Unsupported loss_type: {self.loss_type}")
        n_recon = recon_mask.float().sum().clamp(min=1)
        return (per_sample * recon_mask.float()).sum() / n_recon

    def forward_mixed(
        self,
        fea1: torch.Tensor,
        fea2: torch.Tensor,
        clip_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Mixed recon + CLIP forward.

        All samples go through dual paths; mask separates recon and clip
        loss contributions.

        Args:
            fea1: (B, input_dim) main embedding (all rows valid).
            fea2: (B, input_dim) clip embedding (recon rows == fea1).
            clip_mask: (B,) bool, True = clip sample.
            temperature: Gumbel-Softmax temperature.
        """
        # Step 1: dual-path encode -> quantize -> decode
        z_e1 = self.encode(fea1)
        quant1 = self.quantizer(z_e1, temperature=temperature)
        x_hat1 = self.decode(quant1.quantized_embeddings)

        z_e2 = self.encode(fea2)
        quant2 = self.quantizer(
            z_e2, temperature=temperature,
            ema_mask=clip_mask.float(),
        )
        x_hat2 = self.decode(quant2.quantized_embeddings)

        # Step 2: recon loss (only recon rows, no branching)
        recon_mask = ~clip_mask
        recon_loss = self._compute_masked_recon_loss(x_hat1, fea1, recon_mask)

        # Step 3: masked CLIP loss (only clip rows)
        features = {
            "image_embed": x_hat1,
            "text_embed": x_hat2,
            "image_embed_ori": fea1,
            "text_embed_ori": fea2,
            "logit_scale_self": self.logit_scale_self.exp(),
            "logit_scale_cl": self.logit_scale_cl.exp(),
            "logit_scale": self.logit_scale.exp(),
        }
        clip_result = self.masked_clip_loss_fn(features, clip_mask)

        # Step 4: commitment loss (average of two paths)
        commitment = (
            quant1.quantization_loss + quant2.quantization_loss
        ) / 2

        return {
            "codes": quant1.cluster_ids,
            "quantized": quant1.quantized_embeddings,
            "x_hat": x_hat1,
            "recon_loss": recon_loss,
            "clip_loss": clip_result["clip_loss"],
            "clip_acc": clip_result["clip_acc"],
            "loss_self": clip_result["loss_self"],
            "loss_ori": clip_result["loss_ori"],
            "loss_cl": clip_result["loss_cl"],
            "commitment_loss": commitment,
            "loss": recon_loss + clip_result["clip_loss"] + commitment,
        }


    @torch.no_grad()
    def get_codes(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: get semantic IDs.

        Args:
            x: (B, input_dim) original embedding.

        Returns:
            Tensor: codes, shape (B, n_layers).
        """
        z_e = self.encode(x)
        return self.quantizer.get_codes(z_e)

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embedding from semantic IDs (through decoder).

        Args:
            codes: (B, n_layers) semantic ID codes.

        Returns:
            Tensor: x_hat, shape (B, input_dim).
        """
        z_q = self.quantizer.decode_codes(codes)
        return self.decode(z_q)
