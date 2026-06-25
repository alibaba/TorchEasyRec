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

import torch
import torchmetrics

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.sid_commitment_loss import SidCommitmentLoss
from tzrec.loss.sid_contrastive_loss import SidContrastiveLoss
from tzrec.loss.sid_recon_loss import SidReconLoss
from tzrec.metrics.relative_l1 import RelativeL1
from tzrec.metrics.unique_ratio import UniqueRatio
from tzrec.models.model import BaseModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.utils import div_no_nan
from tzrec.protos.loss_pb2 import LossConfig
from tzrec.protos.model_pb2 import ModelConfig


def _masked_mean(
    per_sample: torch.Tensor, mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Mean of a per-row loss over the masked-in rows (all rows if ``mask`` None).

    The mixed recon+contrastive path applies the reconstruction loss to recon rows
    only; the masked mean divides by the valid-row count (``div_no_nan`` keeps an
    empty mask at 0). No data-dependent branching → ``torch.compile``-friendly.

    Args:
        per_sample (Tensor): per-row loss, shape (B,).
        mask (Tensor, optional): per-row bool; rows to include.
    """
    if mask is None:
        return per_sample.mean()
    mask = mask.float()
    return div_no_nan((per_sample * mask).sum(), mask.sum())


class BaseSidModel(BaseModel):
    """Shared base for semantic-ID (SID) generation models.

    Factors the structure common to :class:`SidRqvae` (RQ-VAE) and
    :class:`SidRqkmeans` (residual K-Means):

    - the shared config fields every SID proto carries — ``feature_group``
      (``_feature_group``), ``normalize_residuals`` (``_normalize_residuals``),
      and the per-layer ``codebook`` (``_n_embed_list`` / ``_n_layers``),
    - building the main input through the framework's :class:`EmbeddingGroup`
      (:meth:`init_input` / :meth:`build_input`), so a SID model consumes the
      same grouped/concatenated feature tensor as every other model and
      ``_input_dim`` is *derived* from the group's total dimension (supporting
      multiple content embeddings + side-info in one group),
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
        self._normalize_residuals = cfg.normalize_residuals

        if not cfg.codebook:
            raise ValueError("codebook must be set, e.g. [256, 256, 256]")
        self._n_embed_list = list(cfg.codebook)
        # Fail fast: a zero entry only errors opaquely deep in faiss later.
        if any(k < 1 for k in self._n_embed_list):
            raise ValueError(
                f"every codebook entry must be >= 1, got {self._n_embed_list}"
            )
        self._n_layers = len(self._n_embed_list)

        self.init_input()
        self._feature_group = self._resolve_feature_group()
        self._input_dim = self.embedding_group.group_total_dim(self._feature_group)
        if self._input_dim < 1:
            raise ValueError(
                f"feature group {self._feature_group!r} has total dim "
                f"{self._input_dim}; it must be >= 1"
            )

    def init_input(self) -> None:
        """Build the :class:`EmbeddingGroup` from features + feature groups."""
        self.embedding_group = EmbeddingGroup(self._features, self._feature_groups)

    def build_input(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Build grouped input features: ``{group_name: (B, group_total_dim)}``."""
        return self.embedding_group(batch)

    def _resolve_feature_group(self) -> str:
        """Resolve the main input feature group name.

        Uses ``feature_group`` when set; otherwise, when exactly one group is
        declared, that sole group (DLRM-style auto-detect); otherwise fails as
        ambiguous. The resolved name must exist in the model's feature groups.
        """
        groups = self.embedding_group.group_names()
        if self._model_config.HasField("feature_group"):
            name = self._model_config.feature_group
            if name not in groups:
                raise ValueError(
                    f"feature_group {name!r} is not in model_config.feature_groups "
                    f"{groups}"
                )
            return name
        if len(groups) == 1:
            return groups[0]
        raise ValueError(
            "feature_group must be set when multiple feature_groups are declared, "
            f"got groups {groups}"
        )

    def init_loss(self) -> None:
        """Initialize SID loss modules from ``ModelConfig.losses``.

        Each ``LossConfig`` sets one ``sid_loss`` oneof variant (a reconstruction
        loss, the commitment loss, or the contrastive loss). Mirrors ``RankModel``:
        the config drives what is registered here, and :meth:`loss` computes them
        from ``predictions``. All three are registered as ``_loss_modules`` entries.
        """
        for loss_cfg in self._base_model_config.losses:
            self._init_sid_loss_impl(loss_cfg)

    def _init_sid_loss_impl(self, loss_cfg: LossConfig) -> None:
        """Register the loss module for one ``sid_loss`` config."""
        loss_type = loss_cfg.WhichOneof("sid_loss")
        if loss_type == "recon_loss":
            self._loss_modules["recon_loss"] = SidReconLoss(
                loss_cfg.recon_loss.recon_type
            )
        elif loss_type == "commitment_loss":
            cfg = loss_cfg.commitment_loss
            latent_weight = list(cfg.latent_weight) if cfg.latent_weight else (1.0, 0.5)
            self._loss_modules["commitment_loss"] = SidCommitmentLoss(
                latent_weight=latent_weight,
                commitment_type=cfg.commitment_type,
            )
        elif loss_type == "contrastive_loss":
            # The contrastive module owns its learnable temperatures.
            self._loss_modules["contrastive_loss"] = SidContrastiveLoss()
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
                ``encoder_out``/``latents`` for commitment, and the contrastive
                embeds).
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
        if loss_type == "recon_loss":
            per_sample = self._loss_modules["recon_loss"](
                predictions["x_hat"], predictions["recon_target"]
            )
            return {
                "recon_loss": _masked_mean(per_sample, predictions.get("recon_mask"))
            }
        elif loss_type == "commitment_loss":
            loss = self._loss_modules["commitment_loss"](
                predictions["encoder_out"], predictions["latents"]
            )
            return {"commitment_loss": loss}
        elif loss_type == "contrastive_loss":
            loss = self._loss_modules["contrastive_loss"](
                predictions["embed_a"],
                predictions["embed_b"],
                predictions["embed_a_ori"],
                predictions["embed_b_ori"],
                predictions["pair_mask"],
            )
            return {"contrastive_loss": loss}
        else:
            raise ValueError(f"unsupported sid_loss variant: {loss_type!r}")

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
        """Update eval metrics from the reconstruction vs. the input embedding.

        ``predictions["x_hat"]`` is the model's reconstruction of the input
        embedding (the centroid sum for RQ-KMeans, the decoder output for
        RQ-VAE); ``predictions["recon_target"]`` is the input it reconstructs.
        Subclasses expose both only when meaningful, so a not-yet-fitted model
        omits them and this logs nothing. (Reading the target from
        ``predictions`` avoids a second ``build_input`` pass over ``batch``.)
        For the mixed contrastive path the reconstruction is scored only on the
        non-pair rows (``recon_mask``), matching the masked training recon loss
        so the eval mse/rel_loss stay comparable to the optimized objective.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.
            losses (dict, optional): a dict of loss.
        """
        if "x_hat" not in predictions:
            return
        recon = predictions["x_hat"]
        embedding = predictions["recon_target"]
        recon_mask = predictions.get("recon_mask")
        if recon_mask is not None:
            recon = recon[recon_mask]
            embedding = embedding[recon_mask]
        self._metric_modules["mse"].update(recon, embedding)
        self._metric_modules["rel_loss"].update(recon, embedding)
        self._metric_modules["unique_sid_ratio"].update(predictions["codes"])

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> None:
        """Update train-path metric state.

        Default no-op: the current SID models report metrics at eval (after the
        codebook is fit / the decoder is trained), not during training. A
        subclass with a meaningful train-time signal may override this.
        """
        return
