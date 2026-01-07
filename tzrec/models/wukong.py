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

from typing import Any, Dict, List, Optional

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.interaction import WuKongLayer
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class WuKong(RankModel):
    """DLRM model.

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
        group_dims = self.embedding_group.group_dims("deep")
        self._feature_num = len(group_dims)
        self._emb_dim = group_dims[0]
        self._layers = nn.ModuleList()
        for layer_cgf in self._model_config.layers:
            layer = WuKongLayer(
                self._emb_dim, self._feature_num, **config_to_kwargs(layer_cgf)
            )
            self._layers.append(layer)
        self.final_mlp = MLP(
            self._feature_num * self._emb_dim,
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(self.final_mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        # dense
        feat = grouped_features["deep"]
        feat = feat.reshape(-1, self._feature_num, self._emb_dim)
        for layer in self._layers:
            feat = layer(feat)
        feat = feat.view(feat.size(0), -1)
        y_final = self.final_mlp(feat)
        # output
        y = self.output_mlp(y_final)
        return self._output_to_prediction(y)
