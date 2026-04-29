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

"""SidRqkmeans: SID generation model using residual Mini-Batch KMeans.

No gradient-based training. Centroids are updated online via
train_step() during the predict() call in training mode.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import torchmetrics
from torch import nn

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.modules.sid_generation import RQKMeans
from tzrec.protos.model_pb2 import ModelConfig


def _parse_int_list(s: str) -> List[int]:
    """Parse comma-separated int string, e.g. '256,128' -> [256, 128]."""
    return [int(x.strip()) for x in s.split(",") if x.strip()]


class SidRqkmeans(BaseModel):
    """SID generation model using residual Mini-Batch KMeans.

    No gradient-based training. Centroids are updated online via
    train_step() during the predict() call in training mode.

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

        cfg = self._model_config  # SidRqkmeans proto message
        self._embedding_feature_name = cfg.embedding_feature_name

        assert cfg.codebook, (
            "codebook must be set, e.g. '256,256,256'"
        )
        n_embed_list = _parse_int_list(cfg.codebook)
        n_layers = len(n_embed_list)

        self._rqkmeans = RQKMeans(
            embed_dim=cfg.input_dim,
            n_layers=n_layers,
            n_embed=n_embed_list,
            normalize_residuals=cfg.normalize_residuals,
            init_buffer_size=cfg.init_buffer_size,
        )

        # KMeans has no learnable parameters (centroids use register_buffer).
        # Add dummy param to keep optimizer/DDP happy.
        self._dummy_param = nn.Parameter(torch.zeros(1), requires_grad=True)

    def _extract_embedding(self, batch: Batch) -> torch.Tensor:
        """Extract item embedding from Batch.dense_features."""
        kt = batch.dense_features[BASE_DATA_GROUP]
        return kt[self._embedding_feature_name]

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        RQKMeans.forward() internally distinguishes training/eval:
          training: calls layer.train_step() to update centroids
          eval:     calls layer.predict() for assignment only

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        embedding = self._extract_embedding(batch)

        # RQKMeans forward returns {'codes': (B, n_layers), 'quantized': (B, D)}
        result = self._rqkmeans(embedding)

        predictions: Dict[str, torch.Tensor] = {
            "codes": result["codes"],
        }

        if self.is_train or self.is_eval:
            predictions["quantized"] = result["quantized"]
            predictions["input_embedding"] = embedding

        return predictions

    def init_loss(self) -> None:
        """Initialize loss modules.

        KMeans has no gradient loss. Centroids are updated
        in predict() via train_step().
        """
        pass

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Compute loss of the model.

        Returns zero loss to keep TrainWrapper backward happy.
        _dummy_param * 0.0 ensures a compute graph exists so DDP
        does not complain about unused parameters.

        Args:
            predictions (dict): a dict of predicted result.
            batch (Batch): input batch data.

        Return:
            losses (dict): a dict of loss tensor.
        """
        return {"dummy_loss": self._dummy_param.sum() * 0.0}

    def init_metric(self) -> None:
        """Initialize metric modules."""
        # Eval metrics
        self._metric_modules["mse"] = torchmetrics.MeanMetric()
        self._metric_modules["unique_sid_ratio"] = torchmetrics.MeanMetric()

        # Train metrics (loss is dummy, only track mse + unique_sid_ratio)
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
        # Quantization MSE
        if "input_embedding" in predictions:
            mse = F.mse_loss(
                predictions["quantized"],
                predictions["input_embedding"],
                reduction="mean",
            )
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

        # Quantization MSE: ||input - quantized||^2
        if "input_embedding" in predictions:
            mse = F.mse_loss(
                predictions["quantized"],
                predictions["input_embedding"],
                reduction="mean",
            )
            self._metric_modules["mse"].update(mse)

        # Unique SID ratio
        unique_sids = torch.unique(codes, dim=0).shape[0]
        self._metric_modules["unique_sid_ratio"].update(
            torch.tensor(unique_sids / B, device=codes.device)
        )
