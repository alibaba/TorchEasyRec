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

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid_generation.clip_loss import CLIPLoss
from tzrec.modules.sid_generation.residual_quantized import ResidualQuantized


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

    Reference: al_sid/SID_generation/rqvae_embed/rqvae.py -> RQVAE_EMBED
               al_sid/SID_generation/rqvae_embed/rqvae_clip.py -> RQVAE_EMBED_CLIP

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
        latent_weight: Optional[List[float]] = None,
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

        # Default hidden_dims
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]

        # commitment_loss defaults to follow loss_type (al_sid behavior:
        # commitment_loss=loss_type, so mse -> l2 branch)
        if commitment_loss is None:
            commitment_loss = "l2" if loss_type == "mse" else loss_type

        # Encoder: input_dim -> hidden_dims -> embed_dim
        enc_dims = [input_dim] + list(hidden_dims) + [embed_dim]
        self.encoder = self._build_mlp(enc_dims)

        # Decoder: embed_dim -> reversed(hidden_dims) -> input_dim (symmetric)
        dec_dims = [embed_dim] + list(reversed(hidden_dims)) + [input_dim]
        self.decoder = self._build_mlp(dec_dims)

        # Quantizer
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
        self.use_clip = use_clip
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
            self.clip_loss_fn = CLIPLoss()

    # ------------------------------------------------------------------
    # Basic methods
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Forward interfaces
    # ------------------------------------------------------------------

    def forward(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Dispatch based on use_clip.

        use_clip=False: forward(x) -> forward_rqvae(x)
        use_clip=True:  forward(fea1, fea2) -> forward_clip(fea1, fea2)
        """
        if self.use_clip:
            assert len(args) == 2, "CLIP mode requires (fea1, fea2)"
            return self.forward_clip(args[0], args[1], **kwargs)
        else:
            assert len(args) == 1, "Standard mode requires (x,)"
            return self.forward_rqvae(args[0], **kwargs)

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

    def forward_clip(
        self,
        fea1: torch.Tensor,
        fea2: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Siamese RQ-VAE + CLIP contrastive learning.

        fea1, fea2 go through the same RQVAE (shared params),
        then compute CLIP loss + average commitment loss.

        Note: al_sid forward_clip does NOT use reference_code.
        fea1 and fea2 are independently quantized.

        Backward loss = clip_loss + commitment_loss
        recon_loss, pair_code_loss are logged only (no backprop).

        Args:
            fea1: (B, input_dim) first feature.
            fea2: (B, input_dim) second feature (same item, diff modal).
            temperature: Gumbel-Softmax temperature.

        Returns:
            dict with keys: 'clip_loss', 'loss_self', 'loss_ori', 'loss_cl',
                'clip_acc', 'commitment_loss', 'reconstruction_loss',
                'pair_code_loss', 'loss'.
        """
        # Two independent quantization passes (shared params)
        z_e1 = self.encode(fea1)
        quant1 = self.quantizer(z_e1, temperature=temperature)
        fea1_vq = self.decode(quant1.quantized_embeddings)

        z_e2 = self.encode(fea2)
        quant2 = self.quantizer(z_e2, temperature=temperature)
        fea2_vq = self.decode(quant2.quantized_embeddings)

        # CLIP contrastive loss
        features = {
            "image_embed": fea1_vq,
            "text_embed": fea2_vq,
            "image_embed_ori": fea1,
            "text_embed_ori": fea2,
            "logit_scale_self": self.logit_scale_self.exp(),
            "logit_scale_cl": self.logit_scale_cl.exp(),
            "logit_scale": self.logit_scale.exp(),
        }
        clip_result = self.clip_loss_fn(features)

        # Commitment loss (average of two paths)
        commitment_loss = (
            quant1.quantization_loss + quant2.quantization_loss
        ) / 2

        # Reconstruction loss (log only, no backprop)
        feas = torch.cat([fea1, fea2], dim=0)
        recons = torch.cat([fea1_vq, fea2_vq], dim=0)
        with torch.no_grad():
            if self.loss_type == "mse":
                recon_loss = F.mse_loss(recons, feas, reduction="mean")
            elif self.loss_type == "l1":
                recon_loss = F.l1_loss(recons, feas, reduction="mean")
            elif self.loss_type == "cosine":
                recon_loss = self._cosine_loss(recons, feas)
            else:
                recon_loss = torch.tensor(0.0, device=fea1.device)

        # Pair code loss: z_e1 vs z_e2 MSE (log only)
        with torch.no_grad():
            pair_code_loss = F.mse_loss(
                z_e1, z_e2, reduction="mean"
            )

        return {
            "clip_loss": clip_result["clip_loss"],
            "loss_self": clip_result["loss_self"],
            "loss_ori": clip_result["loss_ori"],
            "loss_cl": clip_result["loss_cl"],
            "clip_acc": clip_result["clip_acc"],
            "commitment_loss": commitment_loss,
            "reconstruction_loss": recon_loss,
            "pair_code_loss": pair_code_loss,
            "loss": clip_result["clip_loss"],
        }

    # ------------------------------------------------------------------
    # Inference methods
    # ------------------------------------------------------------------

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
