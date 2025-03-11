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
import torch.nn.functional as F
from torch import nn
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTower
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _update_dict_tensor(
    tensor_dict: Dict[str, torch.Tensor],
    new_tensor_dict: Optional[Dict[str, Optional[torch.Tensor]]],
) -> None:
    if new_tensor_dict:
        for k, v in new_tensor_dict.items():
            if v is not None:
                tensor_dict[k] = v


class DSSMTower(MatchTower):
    """DSSM user/item tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_groups (list[FeatureGroupConfig]): feature group configs.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        feature_groups: List[model_pb2.FeatureGroupConfig],
        features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config, output_dim, similarity, feature_groups, features, model_config
        )
        self.init_input()
        tower_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.mlp = MLP(tower_feature_in, **config_to_kwargs(tower_config.mlp))
        if self._output_dim > 0:
            self.output = nn.Linear(self.mlp.output_dim(), output_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (Tensor): tower output embedding.
        """
        grouped_features = self.build_input(batch)
        output = self.mlp(grouped_features[self._group_name])
        if self._output_dim > 0:
            output = self.output(output)
        if self._similarity == simi_pb2.Similarity.COSINE:
            output = F.normalize(output, p=2.0, dim=1)
        return output


class DSSM(MatchModel):
    """DSSM model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        user_features = self.get_features_in_feature_groups([user_group])
        item_features = self.get_features_in_feature_groups([item_group])

        self.user_tower = DSSMTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            [user_group],
            user_features,
            model_config,
        )

        self.item_tower = DSSMTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            [item_group],
            item_features,
            model_config,
        )

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        user_tower_emb = self.user_tower(batch)
        item_tower_emb = self.item_tower(batch)
        _update_dict_tensor(
            self._loss_collection, self.user_tower.group_variational_dropout_loss
        )
        _update_dict_tensor(
            self._loss_collection, self.item_tower.group_variational_dropout_loss
        )

        ui_sim = (
            self.sim(user_tower_emb, item_tower_emb) / self._model_config.temperature
        )
        return {"similarity": ui_sim}
