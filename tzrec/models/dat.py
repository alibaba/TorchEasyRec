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

from collections import OrderedDict
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTower
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
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


class DATTower(MatchTower):
    """DAT user/item tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.DATTower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        augment_feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
        augment_features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config, output_dim, similarity, feature_group, features, model_config
        )
        self.init_input()
        self._augment_features = augment_features
        self._augment_feature_group = augment_feature_group
        self._augment_group_name = tower_config.augment_input
        self.augment_embedding_group = EmbeddingGroup(
            augment_features, [augment_feature_group]
        )

        tower_feature_in = self.embedding_group.group_total_dim(self._group_name)
        tower_augment_feature_in = self.augment_embedding_group.group_total_dim(
            tower_config.augment_input
        )

        self.mlp = MLP(
            tower_feature_in + tower_augment_feature_in,
            **config_to_kwargs(tower_config.mlp),
        )
        if self._output_dim > 0:
            self.output = nn.Linear(self.mlp.output_dim(), output_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (dict): tower output embedding.
        """
        grouped_features = self.build_input(batch)
        input_features = grouped_features[self._group_name]

        augmented_feature = self.augment_embedding_group(batch)[
            self._augment_group_name
        ]

        output = self.mlp(torch.concat([input_features, augmented_feature], dim=1))
        if self._output_dim > 0:
            output = self.output(output)
        if self._similarity == match_model_pb2.Similarity.COSINE:
            output = F.normalize(output, p=2.0, dim=1)
        return output, augmented_feature


class DAT(MatchModel):
    """Dual Augmented Two-Towers model.

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
        user_augment_group = name_to_feature_group[
            self._model_config.user_tower.augment_input
        ]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        item_augment_group = name_to_feature_group[
            self._model_config.item_tower.augment_input
        ]

        name_to_feature = {x.name: x for x in features}
        user_features = OrderedDict(
            [(x, name_to_feature[x]) for x in user_group.feature_names]
        )
        user_augment_features = OrderedDict(
            [(x, name_to_feature[x]) for x in user_augment_group.feature_names]
        )

        for sequence_group in user_group.sequence_groups:
            for x in sequence_group.feature_names:
                user_features[x] = name_to_feature[x]
        item_features = [name_to_feature[x] for x in item_group.feature_names]
        item_augment_features = [
            name_to_feature[x] for x in item_augment_group.feature_names
        ]

        self.user_tower = DATTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            user_augment_group,
            list(user_features.values()),
            list(user_augment_features.values()),
            model_config,
        )

        self.item_tower = DATTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            item_group,
            item_augment_group,
            item_features,
            item_augment_features,
            model_config,
        )

        self.amm_u_weight = self._model_config.amm_u_weight
        self.amm_i_weight = self._model_config.amm_i_weight

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        user_tower_emb, user_augment = self.user_tower(batch)
        item_tower_emb, item_augment = self.item_tower(batch)
        _update_dict_tensor(
            self._loss_collection, self.user_tower.group_variational_dropout_loss
        )
        _update_dict_tensor(
            self._loss_collection, self.item_tower.group_variational_dropout_loss
        )

        ui_sim = (
            self.sim(user_tower_emb, item_tower_emb) / self._model_config.temperature
        )
        return {
            "similarity": ui_sim,
            "augmented_p_u": user_tower_emb.detach(),
            "augmented_p_i": item_tower_emb.detach(),
            "augmented_a_u": user_augment,
            "augmented_a_i": item_augment,
        }
