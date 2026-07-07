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
contrastive losses are configured via ``ModelConfig.losses`` (the
``LossConfig`` ``sid_loss`` oneof) and computed centrally in
:meth:`BaseSidModel.loss`; :meth:`predict` only produces the raw tensors those
losses consume. The encoder/decoder and residual vector quantizer live directly
on the model — there is no intermediate ``RQVAE`` module wrapper.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.sid_model import BaseSidModel
from tzrec.modules.mlp import MLP
from tzrec.modules.sid.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
from tzrec.modules.sid.types import ResidualQuantizerOutput
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
    ``contrastive_loss`` is configured, ``predict`` runs a dual (paired) path and
    the masked contrastive loss is applied to the contrastive-pair rows.

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

        cfg = self._model_config

        self._init_contrastive()

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

        sinkhorn_cfg = config_to_kwargs(cfg.sinkhorn_config)

        # MLP activates its last layer; the trailing bare Linear keeps the
        # latent / reconstruction unbounded.
        self._encoder = nn.Sequential(
            MLP(self._input_dim, hidden_units=hidden_dims),
            nn.Linear(hidden_dims[-1], embed_dim),
        )
        self._decoder = nn.Sequential(
            MLP(embed_dim, hidden_units=list(reversed(hidden_dims))),
            nn.Linear(hidden_dims[0], self._input_dim),
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
            candidate_output_config=self._candidate_output_kwargs,
        )

        logger.info(
            "SidRqvae init: input_dim=%d, embed_dim=%d, hidden_dims=%s, "
            "n_layers=%d, n_embed=%s, use_contrastive=%s",
            self._input_dim,
            embed_dim,
            hidden_dims,
            self._n_layers,
            self._n_embed_list,
            self._use_contrastive,
        )

    def _init_contrastive(self) -> None:
        """Read and validate the pair-contrastive wiring (``contrastive_config``).

        Sets ``_use_contrastive`` and the paired / pair-flag group names, and
        enforces: ``contrastive_config`` (structure) and a ``contrastive_loss``
        entry (objective) are set together; the paired group exists and matches
        ``input_dim`` (it shares the encoder); the pair-flag group is a single
        dim-1 raw flag. Must run after ``super().__init__()`` — it needs
        ``embedding_group`` / ``_input_dim``.
        """
        cfg = self._model_config
        self._pair_feature_group = None
        self._pair_flag_feature_group = None
        self._use_contrastive = cfg.HasField("contrastive_config")
        has_contrastive_obj = any(
            lc.WhichOneof("sid_loss") == "contrastive_loss"
            for lc in self._base_model_config.losses
        )
        if self._use_contrastive != has_contrastive_obj:
            raise ValueError(
                "contrastive_config (model structure) and a contrastive_loss entry "
                "in losses (the objective) must be set together; got "
                f"contrastive_config={self._use_contrastive}, "
                f"contrastive_loss={has_contrastive_obj}"
            )
        if not self._use_contrastive:
            return
        self._pair_feature_group = cfg.contrastive_config.pair_feature_group
        self._pair_flag_feature_group = cfg.contrastive_config.pair_flag_feature_group
        for grp in (self._pair_feature_group, self._pair_flag_feature_group):
            if not self.embedding_group.has_group(grp):
                raise ValueError(
                    f"contrastive group {grp!r} is not in model_config.feature_groups"
                    f" {self.embedding_group.group_names()}"
                )
        pair_dim = self.embedding_group.group_total_dim(self._pair_feature_group)
        if pair_dim != self._input_dim:
            raise ValueError(
                f"pair_feature_group {self._pair_feature_group!r} has total "
                f"dim {pair_dim}, but it is encoded by the same encoder as the "
                f"main feature_group (dim {self._input_dim}); the two must match"
            )
        flag_dim = self.embedding_group.group_total_dim(self._pair_flag_feature_group)
        if flag_dim != 1:
            raise ValueError(
                f"pair_flag_feature_group {self._pair_flag_feature_group!r} must "
                f"be a single dim-1 raw flag, got total dim {flag_dim}"
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
        grouped = self.build_input(batch)
        embedding = grouped[self._feature_group]
        if self._is_inference:
            quant = self._quantizer(self._encode(embedding))
            return self._sid_predictions(quant)
        if self._use_contrastive:
            return self._predict_mixed(grouped)
        return self._predict_rqvae(embedding)

    def _rqvae_pass(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, ResidualQuantizerOutput, torch.Tensor]:
        """One RQ-VAE pass over ``x``: encode -> quantize -> decode.

        Returns the encoder output ``z_e`` (commitment operand), the quantizer
        output ``quant`` (cluster_ids / latents / quantized_embeddings) and the
        decoded reconstruction ``x_hat``.
        """
        z_e = self._encode(x)
        quant = self._quantizer(z_e)
        return z_e, quant, self._decode(quant.quantized_embeddings)

    def _predict_rqvae(self, embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Standard RQ-VAE: a single reconstruction pass."""
        z_e, quant, x_hat = self._rqvae_pass(embedding)
        return {
            "codes": quant.cluster_ids,
            "x_hat": x_hat,
            "recon_target": embedding,
            "encoder_out": z_e,
            "latents": quant.latents,
        }

    def _predict_mixed(
        self, grouped: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Mixed recon + contrastive: dual path over the main + paired groups.

        ``encoder_out`` / ``latents`` stack both paths so the commitment loss
        averages over them; ``recon_mask`` (= non-pair rows) restricts the recon
        loss to reconstruction-only rows.

        Args:
            grouped (dict): the EmbeddingGroup output (group name -> tensor).
        """
        embedding = grouped[self._feature_group]
        fea2 = grouped[self._pair_feature_group]
        is_pair_raw = grouped[self._pair_flag_feature_group]
        pair_mask = is_pair_raw.view(is_pair_raw.shape[0], -1)[:, 0] > 0.5

        z_e1, quant1, x_hat1 = self._rqvae_pass(embedding)
        z_e2, quant2, x_hat2 = self._rqvae_pass(fea2)

        return {
            "codes": quant1.cluster_ids,
            "x_hat": x_hat1,
            "recon_target": embedding,
            "recon_mask": ~pair_mask,
            "encoder_out": torch.cat([z_e1, z_e2], dim=0),
            "latents": torch.cat([quant1.latents, quant2.latents], dim=0),
            "embed_a": x_hat1,
            "embed_b": x_hat2,
            "embed_a_ori": embedding,
            "embed_b_ori": fea2,
            "pair_mask": pair_mask,
        }
