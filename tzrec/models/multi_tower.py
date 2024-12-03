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

from typing import Dict, List

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MultiTower(RankModel):
    """Multi Tower model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str], sample_weights: List[str] = None
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights)

        self.init_input()
        self.towers = nn.ModuleDict()
        total_tower_dim = 0
        for tower in self._model_config.towers:
            group_name = tower.input
            tower_feature_in = self.embedding_group.group_total_dim(group_name)
            tower_mlp = MLP(tower_feature_in, **config_to_kwargs(tower.mlp))
            self.towers[group_name] = tower_mlp
            total_tower_dim += tower_mlp.output_dim()

        final_dim = total_tower_dim
        if self._model_config.HasField("final"):
            self.final_mlp = MLP(
                in_features=total_tower_dim,
                **config_to_kwargs(self._model_config.final),
            )
            final_dim = self.final_mlp.output_dim()

        self.output_mlp = nn.Linear(final_dim, self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        tower_outputs = []
        for k, tower_mlp in self.towers.items():
            tower_outputs.append(tower_mlp(grouped_features[k]))
        tower_output = torch.cat(tower_outputs, dim=-1)

        if self._model_config.HasField("final"):
            tower_output = self.final_mlp(tower_output)

        y = self.output_mlp(tower_output)

        return self._output_to_prediction(y)
