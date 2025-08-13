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
from tzrec.modules.interaction import CrossV2
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class DCNV2(RankModel):
    """Deep cross network v1.

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
        self.init_input()
        self.group_name = self.embedding_group.group_names()[0]
        feature_dim = self.embedding_group.group_total_dim(self.group_name)
        self.cross = CrossV2(
            input_dim=feature_dim, **config_to_kwargs(self._model_config.cross)
        )
        final_input_dim = self.cross.output_dim()
        self.deep = None
        if self._model_config.HasField("deep"):
            self.deep = MLP(
                in_features=feature_dim, **config_to_kwargs(self._model_config.deep)
            )
            final_input_dim += self.deep.output_dim()
        self.final = MLP(
            in_features=final_input_dim,
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(
            self.final.output_dim(), self._num_class, bias=False
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward method."""
        feature_dict = self.build_input(batch)
        features = feature_dict[self.group_name]
        net = self.cross(features)

        if self.deep:
            deep_net = self.deep(features)
            net = torch.concat([net, deep_net], dim=-1)

        out = self.output_mlp(self.final(net))
        return self._output_to_prediction(out)
