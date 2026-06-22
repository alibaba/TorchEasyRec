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

"""BaseSidModel: shared base for semantic-ID generation models."""

from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.clip_loss import MaskedCLIPLoss
from tzrec.loss.commitment_loss import CommitmentLoss
from tzrec.metrics.relative_l1 import RelativeL1
from tzrec.metrics.unique_ratio import UniqueRatio
from tzrec.models.model import BaseModel
from tzrec.modules.utils import div_no_nan
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.model_pb2 import ModelConfig

# Cap the CLIP temperatures before ``exp`` (reference CLIP clamps to ln(100)):
# an unbounded ``logit_scale`` overflows to +Inf -> NaN grad -> corrupt param.
_LOGIT_SCALE_MAX = float(np.log(100))
# CLIP temperature init (reference CLIP: log(1 / 0.07)).
_LOGIT_SCALE_INIT = float(np.log(1 / 0.07))

# sid_loss reconstruction variants (``_recon_loss`` branches on these directly).
_RECON_LOSSES = frozenset(("recon_l2_loss", "recon_l1_loss", "recon_cosine_loss"))


class BaseSidModel(BaseModel):
    """Shared base for semantic-ID (SID) generation models.

    Factors the structure common to :class:`SidRqvae` (RQ-VAE) and
    :class:`SidRqkmeans` (residual K-Means):

    - the shared config fields every SID proto carries —
      ``embedding_feature_name`` (``_embedding_feature_name``), ``input_dim``
      (``_input_dim``), ``normalize_residuals`` (``_normalize_residuals``),
      and the per-layer ``codebook`` (``_n_embed_list`` / ``_n_layers``),
    - reading the item-embedding feature out of ``Batch.dense_features``,
    - the eval metrics every SID model reports — reconstruction ``mse`` and
      ``unique_sid_ratio`` (mean per-batch unique-SID ratio, a diversity
      proxy).

    Subclasses build their quantizer in ``__init__`` (after calling
    ``super().__init__``) and implement :meth:`predict` and :meth:`loss`.
    :meth:`predict` exposes the reconstruction under ``predictions["x_hat"]``
    (only when meaningful) so the shared :meth:`update_metric` can score it.
    (:meth:`update_train_metric` defaults to a no-op.)

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
        # Config fields shared by every SID model (present on each SID proto
        # message): the item-embedding feature, the input dimension, the
        # residual-normalization toggle, and the per-layer codebook.
        self._embedding_feature_name = cfg.embedding_feature_name
        self._input_dim = cfg.input_dim
        self._normalize_residuals = cfg.normalize_residuals

        if not cfg.codebook:
            raise ValueError("codebook must be set, e.g. [256, 256, 256]")
        self._n_embed_list = list(cfg.codebook)
        # Fail fast: a zero codebook entry / input_dim==0 only errors opaquely
        # deep inside faiss, after the whole training pass.
        if any(k < 1 for k in self._n_embed_list):
            raise ValueError(
                f"every codebook entry must be >= 1, got {self._n_embed_list}"
            )
        if self._input_dim < 1:
            raise ValueError(f"input_dim must be >= 1, got {self._input_dim}")
        self._n_layers = len(self._n_embed_list)

    def _extract_feature(
        self, batch: Batch, feature_name: Optional[str] = None
    ) -> torch.Tensor:
        """Extract a named dense feature from ``Batch.dense_features``.

        Args:
            batch (Batch): input batch data.
            feature_name (str, optional): feature name to extract.
                Defaults to ``self._embedding_feature_name``.
        """
        if feature_name is None:
            feature_name = self._embedding_feature_name
        kt = batch.dense_features[BASE_DATA_GROUP]
        return kt[feature_name]

    def init_loss(self) -> None:
        """Initialize SID loss modules from ``ModelConfig.losses``.

        Each ``LossConfig`` sets one ``sid_loss`` oneof variant (a reconstruction
        loss, the commitment loss, or the CLIP loss). Mirrors ``RankModel``: the
        config drives which loss modules are registered, and :meth:`loss`
        computes them from ``predictions``.
        """
        for loss_cfg in self._base_model_config.losses:
            self._init_sid_loss_impl(loss_cfg)

    def _init_sid_loss_impl(self, loss_cfg: LossConfig) -> None:
        """Register the module (if any) for one ``sid_loss`` config."""
        loss_type = loss_cfg.WhichOneof("sid_loss")
        if loss_type in _RECON_LOSSES:
            return  # reconstruction losses are functional (no module)
        elif loss_type == "commitment_loss":
            cfg = loss_cfg.commitment_loss
            latent_weight = list(cfg.latent_weight) if cfg.latent_weight else (1.0, 0.5)
            self._loss_modules["commitment_loss"] = CommitmentLoss(
                latent_weight=latent_weight,
                commitment_type=cfg.commitment_type,
            )
        elif loss_type == "sid_clip_loss":
            # The three learnable CLIP temperatures + the masked-CLIP module.
            self._logit_scale_self = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)
            self._logit_scale_cl = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)
            self._logit_scale = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)
            self._loss_modules["sid_clip_loss"] = MaskedCLIPLoss()
        else:
            raise ValueError(
                f"LossConfig for a SID model must set a sid_loss variant, "
                f"got {loss_type!r}"
            )

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute the configured SID losses from ``predictions``.

        Args:
            predictions (dict): a dict of predicted result (the raw tensors the
                losses consume — ``x_hat``/``recon_target`` for reconstruction,
                ``encoder_out``/``latents`` for commitment, and the CLIP embeds).
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor keyed by the sid_loss variant.
        """
        losses: Dict[str, torch.Tensor] = {}
        for loss_cfg in self._base_model_config.losses:
            losses.update(self._sid_loss_impl(predictions, loss_cfg))
        return losses

    def _sid_loss_impl(
        self, predictions: Dict[str, torch.Tensor], loss_cfg: LossConfig
    ) -> Dict[str, torch.Tensor]:
        """Compute one ``sid_loss`` term from ``predictions``."""
        loss_type = loss_cfg.WhichOneof("sid_loss")
        if loss_type in _RECON_LOSSES:
            loss = self._recon_loss(
                predictions["x_hat"],
                predictions["recon_target"],
                loss_type,
                predictions.get("recon_mask"),
            )
            return {loss_type: loss}
        elif loss_type == "commitment_loss":
            loss = self._loss_modules["commitment_loss"](
                predictions["encoder_out"], predictions["latents"]
            )
            return {"commitment_loss": loss}
        elif loss_type == "sid_clip_loss":

            def scaled(p: torch.Tensor) -> torch.Tensor:
                # clamp before exp so a large temperature can't overflow to +Inf.
                return p.clamp(max=_LOGIT_SCALE_MAX).exp()

            feats = {
                "image_embed": predictions["clip_image"],
                "text_embed": predictions["clip_text"],
                "image_embed_ori": predictions["clip_image_ori"],
                "text_embed_ori": predictions["clip_text_ori"],
                "logit_scale_self": scaled(self._logit_scale_self),
                "logit_scale_cl": scaled(self._logit_scale_cl),
                "logit_scale": scaled(self._logit_scale),
            }
            out = self._loss_modules["sid_clip_loss"](feats, predictions["clip_mask"])
            return {"sid_clip_loss": out["clip_loss"]}
        else:
            raise ValueError(f"unsupported sid_loss variant: {loss_type!r}")

    def _recon_loss(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        recon_loss: str,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reconstruction loss for a ``sid_loss`` recon variant.

        Returns the mean over all rows, or — when ``mask`` (a per-row bool) is
        given — the mean over only the masked-in rows (the mixed recon+CLIP path
        applies recon loss to recon rows only). No data-dependent branching, so
        it stays ``torch.compile``-friendly.

        Args:
            x_hat (Tensor): reconstructed output, shape (B, D).
            x (Tensor): original input, shape (B, D).
            recon_loss (str): the recon variant, one of ``_RECON_LOSSES``
                (``recon_l2_loss`` | ``recon_l1_loss`` | ``recon_cosine_loss``).
            mask (Tensor, optional): per-row bool; rows to include.
        """
        if recon_loss == "recon_l2_loss":
            per_sample = F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)
        elif recon_loss == "recon_l1_loss":
            per_sample = F.l1_loss(x_hat, x, reduction="none").mean(dim=-1)
        else:  # "recon_cosine_loss"
            per_sample = 1 - F.cosine_similarity(x_hat, x, dim=-1)
        if mask is None:
            return per_sample.mean()
        mask = mask.float()
        return div_no_nan((per_sample * mask).sum(), mask.sum())

    def init_metric(self) -> None:
        """Initialize the eval metrics shared by all SID models.

        - ``mse``: reconstruction error (input vs. quantized / decoded).
        - ``rel_loss``: symmetric relative-L1 reconstruction error
          (:class:`~tzrec.metrics.relative_l1.RelativeL1`); meaningful only with
          ``normalize_residuals=False`` (else the reconstruction and the input
          live on different scales).
        - ``unique_sid_ratio``: mean per-batch unique-SID ratio (distinct rows /
          batch size; a batch-size-sensitive diversity proxy, not global
          coverage).

        Subclasses that add extras call ``super().init_metric()`` first.
        """
        self._metric_modules["mse"] = torchmetrics.MeanSquaredError()
        self._metric_modules["rel_loss"] = RelativeL1()
        self._metric_modules["unique_sid_ratio"] = UniqueRatio()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update eval metrics from the reconstruction + the re-extracted input.

        ``predictions["x_hat"]`` is the model's reconstruction of the input
        embedding (the centroid sum for RQ-KMeans, the decoder output for
        RQ-VAE). Subclasses expose it only when it is meaningful, so a
        not-yet-fitted model omits it and this logs nothing. The target
        embedding is re-extracted from ``batch`` (it is an input, not an output).

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
            losses (dict, optional): a dict of loss.
        """
        if "x_hat" not in predictions:
            return
        recon = predictions["x_hat"]
        embedding = self._extract_feature(batch)
        self._metric_modules["mse"].update(recon, embedding)
        self._metric_modules["rel_loss"].update(recon, embedding)
        self._metric_modules["unique_sid_ratio"].update(predictions["codes"])

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train-path metric state.

        Default is a no-op: K-Means has no train-time codes, so only models
        with a meaningful train signal (RQ-VAE) override this.
        """
        return
