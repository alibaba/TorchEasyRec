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

"""SidRqvae: SID generation model using RQ-VAE (Encoder + VQ + Decoder).

End-to-end differentiable training with reconstruction loss
and commitment loss. Optionally supports CLIP contrastive learning.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models._sid_helpers import parse_float_list, parse_int_list
from tzrec.models.model import BaseModel
from tzrec.modules.sid_generation import RQVAE
from tzrec.protos.model_pb2 import ModelConfig


class SidRqvae(BaseModel):
    """SID generation model using RQ-VAE (Encoder + VQ + Decoder).

    End-to-end differentiable training with reconstruction loss
    and commitment loss. Optionally supports CLIP contrastive learning.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

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
        self._embedding_feature_name = cfg.embedding_feature_name
        self._loss_type = cfg.loss_type
        self._use_clip = cfg.HasField("clip_config")
        self._clip_feature_name = (
            cfg.clip_config.clip_feature_name if self._use_clip else None
        )

        hidden_dims = parse_int_list(cfg.hidden_dims) if cfg.hidden_dims else None
        # Only forward latent_weight when proto sets it; otherwise let
        # RQVAE / ResidualQuantized apply their signature default (1.0, 0.5).
        rqvae_extra: Dict[str, Any] = {}
        if cfg.latent_weight:
            rqvae_extra["latent_weight"] = parse_float_list(cfg.latent_weight)

        assert cfg.codebook, (
            "codebook must be set, e.g. '256,256,256'"
        )
        n_embed_list = parse_int_list(cfg.codebook)
        n_layers = len(n_embed_list)

        # Parse EMA sub-config (disabled unless ema_config is explicitly set)
        use_ema = cfg.HasField("ema_config")
        ema_decay = 0.99
        restart_unused_codes = True
        if use_ema:
            ema_decay = cfg.ema_config.decay
            restart_unused_codes = cfg.ema_config.restart_unused_codes

        # Parse Sinkhorn sub-config (defaults: enabled, iters=5, epsilon=10.0)
        use_sinkhorn = True
        sinkhorn_iters = 5
        sinkhorn_epsilon = 10.0
        if cfg.HasField("sinkhorn_config"):
            sinkhorn_iters = cfg.sinkhorn_config.iters
            sinkhorn_epsilon = cfg.sinkhorn_config.epsilon

        self._rqvae = RQVAE(
            input_dim=cfg.input_dim,
            embed_dim=cfg.embed_dim,
            hidden_dims=hidden_dims,
            n_layers=n_layers,
            n_embed=n_embed_list,
            forward_mode=cfg.forward_mode,
            normalize_residuals=cfg.normalize_residuals,
            shared_codebook=cfg.shared_codebook,
            distance_type=cfg.distance_type,
            commitment_loss=cfg.commitment_loss,
            rotation_trick=cfg.rotation_trick,
            kmeans_init=cfg.kmeans_init,
            use_ema=use_ema,
            ema_decay=ema_decay,
            restart_unused_codes=restart_unused_codes,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
            loss_type=cfg.loss_type,
            use_clip=self._use_clip,
            **rqvae_extra,
        )

    def _extract_feature(
        self, batch: Batch, feature_name: Optional[str] = None
    ) -> torch.Tensor:
        """Extract a named feature from Batch.dense_features.

        Args:
            batch (Batch): input batch data.
            feature_name (str, optional): feature name to extract.
                Defaults to self._embedding_feature_name.
        """
        if feature_name is None:
            feature_name = self._embedding_feature_name
        kt = batch.dense_features[BASE_DATA_GROUP]
        return kt[feature_name]

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

    def _predict_rqvae(
        self, embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE: encode -> quantize -> decode -> loss."""
        result = self._rqvae.forward_rqvae(embedding)

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
        """Mixed recon + CLIP: extract fea2 and clip_mask, call forward_mixed."""
        # Inference: only path 1, only return codes
        if self._is_inference:
            result = self._rqvae.forward_rqvae(embedding)
            return {"codes": result["codes"]}

        fea2 = self._extract_feature(batch, self._clip_feature_name)

        # Derive clip_mask: recon rows have fea2 == fea1 (bit-identical)
        clip_mask = ~torch.all(embedding == fea2, dim=-1)  # (B,) bool

        # Train / eval: mixed forward
        result = self._rqvae.forward_mixed(embedding, fea2, clip_mask)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
            "quantized": result["quantized"],
            "x_hat": result["x_hat"],
            "recon_loss": result["recon_loss"],
            "clip_loss": result["clip_loss"],
            "commitment_loss": result["commitment_loss"],
        }
        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules.

        Reconstruction loss and commitment loss are computed internally
        by RQVAE and passed through predictions. No external loss module needed.
        """
        pass

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
        if self._use_clip:
            losses["recon_loss"] = predictions["recon_loss"]
            losses["clip_loss"] = predictions["clip_loss"]
            losses["commitment_loss"] = predictions["commitment_loss"]
        else:
            losses["reconstruction_loss"] = predictions["reconstruction_loss"]
            losses["quantization_loss"] = predictions["quantization_loss"]
        return losses

    def init_metric(self) -> None:
        """Initialize metric modules."""
        # Eval metrics
        self._metric_modules["mse"] = torchmetrics.MeanMetric()
        self._metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

        # Train metrics: mse + unique_sid_ratio only
        # (loss values are already logged by the framework via loss() return)
        self._train_metric_modules["mse"] = torchmetrics.MeanMetric()
        self._train_metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
        """
        # Reconstruction MSE
        if "x_hat" in predictions:
            embedding = self._extract_feature(batch)
            mse = F.mse_loss(predictions["x_hat"], embedding, reduction="mean")
            self._train_metric_modules["mse"].update(mse)

        # Unique SID ratio
        codes = predictions["codes"]
        B = codes.shape[0]
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._train_metric_modules["unique_sid_ratio"].update(
            torch.tensor(unique_sids / B, device=codes.device)
        )

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update metric state.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
            losses (dict, optional): a dict of loss.
        """
        codes = predictions["codes"]
        B = codes.shape[0]

        # Reconstruction MSE
        if "x_hat" in predictions:
            embedding = self._extract_feature(batch)
            mse = F.mse_loss(predictions["x_hat"], embedding, reduction="mean")
            self._metric_modules["mse"].update(mse)

        # Unique SID ratio
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._metric_modules["unique_sid_ratio"].update(
            torch.tensor(unique_sids / B, device=codes.device)
        )
