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

"""SidRqvae: SID generation model using RQ-VAE (Encoder + VQ + Decoder).

End-to-end differentiable training with reconstruction loss and commitment
loss. Optionally supports CLIP contrastive learning. The encoder/decoder,
residual vector quantizer, and CLIP head all live directly on the model —
there is no intermediate ``RQVAE`` module wrapper.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.clip_loss import MaskedCLIPLoss
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.logging_util import logger

# Cap the CLIP temperatures before ``exp`` (reference CLIP clamps to ln(100)):
# an unbounded ``logit_scale`` overflows to +Inf -> NaN grad -> corrupt param.
_LOGIT_SCALE_MAX = float(np.log(100))


class SidRqvae(BaseSidModel):
    """SID generation model using RQ-VAE (Encoder + VQ + Decoder).

    Encoder/Decoder are configurable-depth MLPs built from ``hidden_dims``:
        Encoder: input_dim -> hidden_dims[0] -> ... -> embed_dim
        Decoder: embed_dim -> ... -> hidden_dims[0] -> input_dim
    (ReLU between hidden layers; the decoder mirrors the encoder.)

    When ``clip_config`` is set, ``predict`` runs a dual path and a masked
    CLIP contrastive loss is added for the CLIP-pair rows.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    @staticmethod
    def _build_mlp(dims: List[int]) -> nn.Sequential:
        """Build MLP: dims[0] -> ... -> dims[-1], ReLU between hidden layers."""
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # no activation after the last layer
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)

        cfg = self._model_config  # SidRqvae proto message
        self._loss_type = cfg.loss_type
        assert self._loss_type in ("mse", "l1", "cosine"), (
            f"loss_type must be 'mse', 'l1' or 'cosine', got '{self._loss_type}'"
        )
        self._use_clip = cfg.HasField("clip_config")
        self._clip_feature_name = (
            cfg.clip_config.clip_feature_name if self._use_clip else None
        )
        self._is_clip_pair_feature_name = (
            cfg.clip_config.is_clip_pair_feature_name if self._use_clip else None
        )

        embed_dim = cfg.embed_dim
        # Fail fast (parity with BaseSidModel's codebook/input_dim checks): a zero
        # dim only errors opaquely deep in nn.Linear/Embedding otherwise.
        if embed_dim < 1:
            raise ValueError(f"embed_dim must be >= 1, got {embed_dim}")
        hidden_dims = (
            list(cfg.hidden_dims) if cfg.hidden_dims else [self._input_dim // 2]
        )
        if any(h < 1 for h in hidden_dims):
            raise ValueError(f"every hidden_dims entry must be >= 1, got {hidden_dims}")
        # Empty -> default (1.0, 0.5); the quantizer validates the arity.
        latent_weight = list(cfg.latent_weight) if cfg.latent_weight else (1.0, 0.5)

        use_sinkhorn = True
        sinkhorn_iters = 5
        sinkhorn_epsilon = 10.0
        if cfg.HasField("sinkhorn_config"):
            use_sinkhorn = cfg.sinkhorn_config.enabled
            sinkhorn_iters = cfg.sinkhorn_config.iters
            sinkhorn_epsilon = cfg.sinkhorn_config.epsilon

        self._encoder = self._build_mlp([self._input_dim, *hidden_dims, embed_dim])
        # Decoder is the symmetric reverse of the encoder.
        self._decoder = self._build_mlp(
            [embed_dim, *reversed(hidden_dims), self._input_dim]
        )

        self._quantizer = ResidualVectorQuantizer(
            embed_dim=embed_dim,
            n_layers=self._n_layers,
            n_embed=self._n_embed_list,
            forward_mode=cfg.forward_mode,
            normalize_residuals=self._normalize_residuals,
            distance_type=cfg.distance_type,
            commitment_loss=cfg.commitment_loss,
            latent_weight=latent_weight,
            rotation_trick=cfg.rotation_trick,
            kmeans_init=cfg.kmeans_init,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
        )

        # CLIP contrastive head (optional).
        if self._use_clip:
            self._logit_scale_self = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self._logit_scale_cl = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self._logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
            self._masked_clip_loss_fn = MaskedCLIPLoss()

        logger.info(
            "SidRqvae init: input_dim=%d, embed_dim=%d, hidden_dims=%s, "
            "n_layers=%d, n_embed=%s, loss_type=%s, use_clip=%s",
            self._input_dim,
            embed_dim,
            hidden_dims,
            self._n_layers,
            self._n_embed_list,
            self._loss_type,
            self._use_clip,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode. (B, input_dim) -> (B, embed_dim)."""
        return self._encoder(x)

    def _decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode. (B, embed_dim) -> (B, input_dim)."""
        return self._decoder(z_q)

    def _recon_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruction loss for the configured ``loss_type``.

        Returns the mean over all rows, or — when ``mask`` (a per-row bool)
        is given — the mean over only the masked-in rows (the mixed
        recon+CLIP path applies recon loss to recon rows only). No
        data-dependent branching, so it stays ``torch.compile``-friendly.

        Args:
            x_hat (Tensor): reconstructed output, shape (B, D).
            x (Tensor): original input, shape (B, D).
            mask (Tensor, optional): per-row bool; rows to include.
        """
        if self._loss_type == "mse":
            per_sample = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        elif self._loss_type == "l1":
            per_sample = F.l1_loss(x_hat, x, reduction="none").mean(dim=-1)
        else:  # 'cosine'
            per_sample = 1 - F.cosine_similarity(x_hat, x, dim=-1)
        if mask is None:
            return per_sample.mean()
        mask = mask.float()
        return (per_sample * mask).sum() / mask.sum().clamp(min=1)

    def _forward_rqvae(
        self, x: torch.Tensor, temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE forward: encode -> quantize -> decode -> loss."""
        z_e = self._encode(x)
        quant = self._quantizer(z_e, temperature=temperature)
        x_hat = self._decode(quant.quantized_embeddings)

        recon_loss = self._recon_loss(x_hat, x)
        quant_loss = quant.quantization_loss
        return {
            "x_hat": x_hat,
            "codes": quant.cluster_ids,
            "quantized": quant.quantized_embeddings,
            "reconstruction_loss": recon_loss,
            "quantization_loss": quant_loss,
            "loss": recon_loss + quant_loss,
        }

    def _forward_mixed(
        self,
        fea1: torch.Tensor,
        fea2: torch.Tensor,
        clip_mask: torch.Tensor,
        temperature: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """Mixed recon + CLIP forward (all rows dual-pathed; mask splits loss)."""
        z_e1 = self._encode(fea1)
        quant1 = self._quantizer(z_e1, temperature=temperature)
        x_hat1 = self._decode(quant1.quantized_embeddings)

        z_e2 = self._encode(fea2)
        quant2 = self._quantizer(z_e2, temperature=temperature)
        x_hat2 = self._decode(quant2.quantized_embeddings)

        recon_mask = ~clip_mask
        recon_loss = self._recon_loss(x_hat1, fea1, recon_mask)

        features = {
            "image_embed": x_hat1,
            "text_embed": x_hat2,
            "image_embed_ori": fea1,
            "text_embed_ori": fea2,
            "logit_scale_self": self._logit_scale_self.clamp(
                max=_LOGIT_SCALE_MAX
            ).exp(),
            "logit_scale_cl": self._logit_scale_cl.clamp(max=_LOGIT_SCALE_MAX).exp(),
            "logit_scale": self._logit_scale.clamp(max=_LOGIT_SCALE_MAX).exp(),
        }
        clip_result = self._masked_clip_loss_fn(features, clip_mask)

        commitment = (quant1.quantization_loss + quant2.quantization_loss) / 2
        return {
            "codes": quant1.cluster_ids,
            "quantized": quant1.quantized_embeddings,
            "x_hat": x_hat1,
            "reconstruction_loss": recon_loss,
            "clip_loss": clip_result["clip_loss"],
            "quantization_loss": commitment,
        }

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        embedding = self._extract_feature(batch)

        if self._use_clip:
            return self._predict_mixed(embedding, batch)
        else:
            return self._predict_rqvae(embedding)

    def _predict_rqvae(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE: encode -> quantize -> decode -> loss."""
        result = self._forward_rqvae(embedding)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
        }

        if self.is_train or self.is_eval:
            predictions["quantized"] = result["quantized"]
            predictions["x_hat"] = result["x_hat"]
            predictions["reconstruction_loss"] = result["reconstruction_loss"]
            predictions["quantization_loss"] = result["quantization_loss"]

        return predictions

    def _predict_mixed(
        self, embedding: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Mixed recon + CLIP: extract fea2 and clip_mask, run the dual path."""
        # Inference skips the dual path: fea2 / clip_mask aren't needed
        # when we only emit codes.
        if self._is_inference:
            result = self._forward_rqvae(embedding)
            return {"codes": result["codes"]}

        fea2 = self._extract_feature(batch, self._clip_feature_name)

        is_clip_pair_raw = self._extract_feature(batch, self._is_clip_pair_feature_name)
        clip_mask = is_clip_pair_raw.view(is_clip_pair_raw.shape[0], -1)[:, 0] > 0.5

        result = self._forward_mixed(embedding, fea2, clip_mask)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
            "quantized": result["quantized"],
            "x_hat": result["x_hat"],
            "reconstruction_loss": result["reconstruction_loss"],
            "clip_loss": result["clip_loss"],
            "quantization_loss": result["quantization_loss"],
        }
        return predictions

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor.
        """
        losses: Dict[str, torch.Tensor] = {}
        losses["reconstruction_loss"] = predictions["reconstruction_loss"]
        losses["quantization_loss"] = predictions["quantization_loss"]
        if self._use_clip:
            losses["clip_loss"] = predictions["clip_loss"]
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules (shared eval metrics + train-path mse)."""
        super().init_metric()

        # Only the train-path reconstruction needs a metric here; unique_sid_ratio
        # is eval-only (its torch.unique forces a per-step GPU->host sync).
        self._train_metric_modules["mse"] = torchmetrics.MeanSquaredError()

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state.

        Overrides the BaseSidModel no-op: RQ-VAE has a train-time reconstruction
        (the decoder output), so it reports a train-path mse. Eval metrics are
        handled by ``BaseSidModel.update_metric`` (SidRqvae emits ``x_hat``).

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        if "x_hat" in predictions:
            embedding = self._extract_feature(batch)
            self._train_metric_modules["mse"].update(predictions["x_hat"], embedding)
