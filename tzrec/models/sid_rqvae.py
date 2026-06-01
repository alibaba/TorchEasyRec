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

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid_generation import RQVAE
from tzrec.protos.model_pb2 import ModelConfig


class SidRqvae(BaseSidModel):
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
        self._loss_type = cfg.loss_type
        self._use_clip = cfg.HasField("clip_config")
        self._clip_feature_name = (
            cfg.clip_config.clip_feature_name if self._use_clip else None
        )
        self._is_clip_pair_feature_name = (
            cfg.clip_config.is_clip_pair_feature_name if self._use_clip else None
        )

        hidden_dims = list(cfg.hidden_dims) if cfg.hidden_dims else None
        # Only forward latent_weight when the user set it (repeated field is
        # empty when unset); otherwise let RQVAE / ResidualVectorQuantizer
        # apply their signature default (1.0, 0.5).
        rqvae_extra: Dict[str, Any] = {}
        if cfg.latent_weight:
            rqvae_extra["latent_weight"] = list(cfg.latent_weight)

        use_sinkhorn = True
        sinkhorn_iters = 5
        sinkhorn_epsilon = 10.0
        if cfg.HasField("sinkhorn_config"):
            use_sinkhorn = cfg.sinkhorn_config.enabled
            sinkhorn_iters = cfg.sinkhorn_config.iters
            sinkhorn_epsilon = cfg.sinkhorn_config.epsilon

        self._rqvae = RQVAE(
            input_dim=cfg.input_dim,
            embed_dim=cfg.embed_dim,
            hidden_dims=hidden_dims,
            n_layers=self._n_layers,
            n_embed=self._n_embed_list,
            forward_mode=cfg.forward_mode,
            normalize_residuals=cfg.normalize_residuals,
            distance_type=cfg.distance_type,
            commitment_loss=cfg.commitment_loss,
            rotation_trick=cfg.rotation_trick,
            kmeans_init=cfg.kmeans_init,
            use_sinkhorn=use_sinkhorn,
            sinkhorn_iters=sinkhorn_iters,
            sinkhorn_epsilon=sinkhorn_epsilon,
            loss_type=cfg.loss_type,
            use_clip=self._use_clip,
            **rqvae_extra,
        )

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
        # Inference skips the dual path: fea2 / clip_mask aren't needed
        # when we only emit codes.
        if self._is_inference:
            result = self._rqvae.forward_rqvae(embedding)
            return {"codes": result["codes"]}

        fea2 = self._extract_feature(batch, self._clip_feature_name)

        is_clip_pair_raw = self._extract_feature(batch, self._is_clip_pair_feature_name)
        clip_mask = is_clip_pair_raw.view(is_clip_pair_raw.shape[0], -1)[:, 0] > 0.5

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
        """Initialize metric modules (shared eval metrics + train-path mse)."""
        super().init_metric()

        # Loss values are already logged by the framework via loss(); only
        # quantization quality needs the train-path metric. unique_sid_ratio
        # is intentionally eval-only: torch.unique(codes, dim=0).shape[0]
        # forces a GPU->host sync every step, and codebook coverage is a
        # diagnostic, not a training signal.
        self._train_metric_modules["mse"] = torchmetrics.MeanMetric()

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
        if "x_hat" in predictions:
            embedding = self._extract_feature(batch)
            mse = F.mse_loss(predictions["x_hat"], embedding, reduction="mean")
            self._train_metric_modules["mse"].update(mse)

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
        if "x_hat" in predictions:
            embedding = self._extract_feature(batch)
            mse = F.mse_loss(predictions["x_hat"], embedding, reduction="mean")
            self._metric_modules["mse"].update(mse)

        self._update_unique_sid_ratio(predictions["codes"])
