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

from typing import Any, List, Optional

import torch
import torchmetrics

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
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
      ``unique_sid_ratio`` (codebook coverage).

    Subclasses build their quantizer in ``__init__`` (after calling
    ``super().__init__``) and implement :meth:`predict` and :meth:`loss`.
    They extend :meth:`init_metric` / :meth:`update_metric` with any
    backend-specific metrics.

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

        assert cfg.codebook, "codebook must be set, e.g. [256, 256, 256]"
        self._n_embed_list = list(cfg.codebook)
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
        """Initialize loss modules.

        SID models compute their losses internally and pass them through
        ``predictions``; there is no external loss module to register.
        """
        pass

    def init_metric(self) -> None:
        """Initialize the eval metrics shared by all SID models.

        ``mse``: reconstruction error (input vs. quantized / decoded).
        ``unique_sid_ratio``: codebook coverage = unique SIDs / batch size.
        Subclasses call ``super().init_metric()`` then add their extras.
        """
        self._metric_modules["mse"] = torchmetrics.MeanMetric()
        self._metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

    def update_train_metric(
        self,
        predictions: dict,
        batch: Batch,
    ) -> None:
        """Update train-path metric state.

        Default is a no-op: K-Means has no train-time codes, so only models
        with a meaningful train signal (RQ-VAE) override this.
        """
        return

    def _update_unique_sid_ratio(self, codes: torch.Tensor) -> None:
        """Update the codebook-coverage metric (unique SIDs / batch size).

        Args:
            codes (Tensor): semantic-ID codes, shape (B, n_layers).
        """
        B = codes.shape[0]
        if B == 0:  # empty final shard under DDP/TorchRec
            return
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._metric_modules["unique_sid_ratio"].update(unique_sids / B)
