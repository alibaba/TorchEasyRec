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

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.metrics.relative_l1 import RelativeL1
from tzrec.metrics.unique_ratio import UniqueRatio
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


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
        embedding = kt[feature_name]
        # Guard a misconfigured feature width: a (B, 1) tensor (raw_feature
        # missing value_dim, which defaults to 1) would otherwise broadcast
        # silently downstream and fit a degenerate rank-1 codebook.
        if embedding.dim() != 2 or embedding.shape[1] != self._input_dim:
            raise ValueError(
                f"feature '{feature_name}' has shape {tuple(embedding.shape)}, "
                f"expected (B, {self._input_dim}); check that its value_dim "
                "matches the SID input_dim."
            )
        return embedding

    def init_loss(self) -> None:
        """Initialize loss modules.

        SID models compute their losses internally and pass them through
        ``predictions``; there is no external loss module to register.
        """
        pass

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
