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

End-to-end differentiable training. The reconstruction, commitment and optional
CLIP contrastive losses are configured via ``ModelConfig.losses`` (the
``LossConfig`` ``sid_loss`` oneof) and computed centrally in
:meth:`BaseSidModel.loss`; :meth:`predict` only produces the raw tensors those
losses consume. The encoder/decoder and residual vector quantizer live directly
on the model — there is no intermediate ``RQVAE`` module wrapper.
"""

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.logging_util import logger


class SidRqvae(BaseSidModel):
    """SID generation model using RQ-VAE (Encoder + VQ + Decoder).

    Encoder/Decoder are configurable-depth MLPs built from ``hidden_dims``:
        Encoder: input_dim -> hidden_dims[0] -> ... -> embed_dim
        Decoder: embed_dim -> ... -> hidden_dims[0] -> input_dim
    (ReLU between hidden layers; the decoder mirrors the encoder.)

    Losses are config-driven (``ModelConfig.losses`` / ``sid_loss`` oneof). When a
    ``sid_clip_loss`` is configured, ``predict`` runs a dual (image/text) path and
    the masked CLIP contrastive loss is applied to the CLIP-pair rows.

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

        # CLIP is enabled by a `sid_clip_loss` entry in ModelConfig.losses, which
        # also carries the paired-feature names (data wiring).
        self._clip_feature_name: Optional[str] = None
        self._is_clip_pair_feature_name: Optional[str] = None
        for loss_cfg in self._base_model_config.losses:
            if loss_cfg.WhichOneof("sid_loss") == "sid_clip_loss":
                self._clip_feature_name = loss_cfg.sid_clip_loss.clip_feature_name
                self._is_clip_pair_feature_name = (
                    loss_cfg.sid_clip_loss.is_clip_pair_feature_name
                )
        self._use_clip = self._clip_feature_name is not None

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

        # Sinkhorn params from the proto: config_to_kwargs flows the proto
        # defaults (enabled=True, iters=5, epsilon=10.0) so the model never
        # restates them; keys map to the quantizer's use_sinkhorn/iters/epsilon.
        sinkhorn_cfg = config_to_kwargs(cfg.sinkhorn_config)

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
            rotation_trick=cfg.rotation_trick,
            kmeans_init=cfg.kmeans_init,
            use_sinkhorn=sinkhorn_cfg["enabled"],
            sinkhorn_iters=sinkhorn_cfg["iters"],
            sinkhorn_epsilon=sinkhorn_cfg["epsilon"],
        )

        logger.info(
            "SidRqvae init: input_dim=%d, embed_dim=%d, hidden_dims=%s, "
            "n_layers=%d, n_embed=%s, use_clip=%s",
            self._input_dim,
            embed_dim,
            hidden_dims,
            self._n_layers,
            self._n_embed_list,
            self._use_clip,
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode. (B, input_dim) -> (B, embed_dim)."""
        return self._encoder(x)

    def _decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """Decode. (B, embed_dim) -> (B, input_dim)."""
        return self._decoder(z_q)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Returns the raw tensors the configured losses consume (computed in
        :meth:`BaseSidModel.loss`); inference emits codes only.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        embedding = self._extract_feature(batch)
        if self._use_clip:
            return self._predict_mixed(embedding, batch)
        return self._predict_rqvae(embedding)

    def _predict_rqvae(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE: encode -> quantize -> decode."""
        z_e = self._encode(embedding)
        quant = self._quantizer(z_e)
        if self._is_inference:
            return {"codes": quant.cluster_ids}
        return {
            "codes": quant.cluster_ids,
            "x_hat": self._decode(quant.quantized_embeddings),
            "recon_target": embedding,
            "encoder_out": z_e,
            "latents": quant.latents,
        }

    def _predict_mixed(
        self, embedding: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Mixed recon + CLIP: dual path over the embedding + its paired feature.

        ``encoder_out`` / ``latents`` stack both paths so the commitment loss
        averages over them; ``recon_mask`` (= non-CLIP rows) restricts the recon
        loss to reconstruction-only rows.
        """
        if self._is_inference:
            z_e = self._encode(embedding)
            return {"codes": self._quantizer(z_e).cluster_ids}

        fea2 = self._extract_feature(batch, self._clip_feature_name)
        is_clip_pair_raw = self._extract_feature(batch, self._is_clip_pair_feature_name)
        clip_mask = is_clip_pair_raw.view(is_clip_pair_raw.shape[0], -1)[:, 0] > 0.5

        z_e1 = self._encode(embedding)
        quant1 = self._quantizer(z_e1)
        x_hat1 = self._decode(quant1.quantized_embeddings)

        z_e2 = self._encode(fea2)
        quant2 = self._quantizer(z_e2)
        x_hat2 = self._decode(quant2.quantized_embeddings)

        return {
            "codes": quant1.cluster_ids,
            "x_hat": x_hat1,
            "recon_target": embedding,
            "recon_mask": ~clip_mask,
            "encoder_out": torch.cat([z_e1, z_e2], dim=0),
            "latents": torch.cat([quant1.latents, quant2.latents], dim=0),
            "clip_image": x_hat1,
            "clip_text": x_hat2,
            "clip_image_ori": embedding,
            "clip_text_ori": fea2,
            "clip_mask": clip_mask,
        }
