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

# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import Dict, List

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.modules.sequence import DINEncoder
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _get_dict(
    grouped_features_keys: List[str], args: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    if len(grouped_features_keys) != len(args):
        raise ValueError(
            "The number of grouped_features_keys must match " "the number of arguments."
        )
    grouped_features = {key: value for key, value in zip(grouped_features_keys, args)}
    return grouped_features


class MultiTowerDINDense(RankModel):
    """DIN Dense model.

    Args:
        embedding_group(EmbeddingGroup): Embedding Group
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self,
        embedding_group: EmbeddingGroup,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
    ) -> None:
        super().__init__(model_config, features, labels)

        self.grouped_features_keys = embedding_group.grouped_features_keys()

        self.towers = nn.ModuleDict()
        total_tower_dim = 0
        for tower in self._model_config.towers:
            group_name = tower.input
            tower_feature_in = embedding_group.group_total_dim(group_name)
            tower_mlp = MLP(tower_feature_in, **config_to_kwargs(tower.mlp))
            self.towers[group_name] = tower_mlp
            total_tower_dim += tower_mlp.output_dim()

        self.din_towers = nn.ModuleList()
        for tower in self._model_config.din_towers:
            group_name = tower.input
            sequence_dim = embedding_group.group_total_dim(f"{group_name}.sequence")
            query_dim = embedding_group.group_total_dim(f"{group_name}.query")
            tower_din = DINEncoder(
                sequence_dim,
                query_dim,
                group_name,
                attn_mlp=config_to_kwargs(tower.attn_mlp),
            )
            self.din_towers.append(tower_din)
            total_tower_dim += tower_din.output_dim()

        final_dim = total_tower_dim
        if self._model_config.HasField("final"):
            self.final_mlp = MLP(
                in_features=total_tower_dim,
                **config_to_kwargs(self._model_config.final),
            )
            final_dim = self.final_mlp.output_dim()

        self.output_mlp = nn.Linear(final_dim, self._num_class)

    # pyre-ignore [14]
    def predict(self, args: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward the module."""
        grouped_features = _get_dict(self.grouped_features_keys, args)
        tower_outputs = []
        for k, tower_mlp in self.towers.items():
            tower_outputs.append(tower_mlp(grouped_features[k]))

        for tower_din in self.din_towers:
            tower_outputs.append(tower_din(grouped_features))

        tower_output = torch.cat(tower_outputs, dim=-1)
        if self._model_config.HasField("final"):
            tower_output = self.final_mlp(tower_output)

        y = self.output_mlp(tower_output)
        return self._output_to_prediction(y)

    # pyre-ignore [14]
    def forward(self, args: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward the module."""
        # use predict to avoid trace error in self._output_to_prediction(y)
        return self.predict(args)


class MultiTowerDINTRT(RankModel):
    """DIN model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)
        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )
        self.dense = MultiTowerDINDense(
            self.embedding_group, model_config, features, labels
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.embedding_group.predict(batch)
        y = self.dense.predict(grouped_features)
        return y
