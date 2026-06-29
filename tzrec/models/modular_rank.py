# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.backbone import Backbone
from tzrec.protos.model_pb2 import ModelConfig


class ModularRank(RankModel):
    """Ranking backbone model."""

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self._feature_dict = features
        self._backbone_output = None
        self._backbone_net = self.build_backbone_network()

        # Use the final output dimension of backbone and consider the impact of top_mlp
        output_dims = self._backbone_net.output_dim()

        self.output_mlp = nn.Linear(output_dims, self._num_class)

    def build_backbone_network(self) -> Backbone:
        """Build backbone."""
        wide_embedding_dim = (
            int(self.wide_embedding_dim)
            if hasattr(self, "wide_embedding_dim")
            else None
        )
        wide_init_fn = self.wide_init_fn if hasattr(self, "wide_init_fn") else None
        feature_groups = list(self._base_model_config.feature_groups)
        return Backbone(
            config=self._base_model_config.rank_backbone.backbone,
            features=self._feature_dict,
            embedding_group=None,  # Backbone create the EmbeddingGroup itself
            feature_groups=feature_groups,
            wide_embedding_dim=wide_embedding_dim,
            wide_init_fn=wide_init_fn,
        )

    def backbone(
        self,
        batch: Batch,
    ) -> Optional[nn.Module]:
        """Get backbone."""
        if self._backbone_output:
            return self._backbone_output
        if self._backbone_net:
            kwargs = {
                "loss_modules": self._loss_modules,
                "metric_modules": self._metric_modules,
                "labels": self._labels,
            }
            return self._backbone_net(
                batch=batch,
                **kwargs,
            )
        return None

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        output = self.backbone(batch=batch)
        y = self.output_mlp(output)
        return self._output_to_prediction(y)
