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
import torch.nn.functional as F

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTower
from tzrec.modules.capsule import CapsuleLayer
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs


class MINDUserTower(MatchTower):
    """MIND user tower."""

    def __init__(
        self,
        tower_config: tower_pb2.MINDUserTower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        user_feature_group: model_pb2.FeatureGroupConfig,
        hist_feature_group: model_pb2.FeatureGroupConfig,
        user_features: List[BaseFeature],
        hist_features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config,
            output_dim,
            similarity,
            user_feature_group,
            user_features,
            model_config,
        )

        self._hist_group_name = tower_config.history_input
        self._hist_feature_group = hist_feature_group
        self._hist_features = hist_features

        user_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.user_mlp = MLP(user_feature_in, **config_to_kwargs(tower_config.user_mlp))

        hist_feature_in = self.hist_embedding_group.group_total_dim(
            self._hist_group_name
        )
        if self._tower_config.hist_seq_mlp:
            self.hist_seq_mlp = MLP(
                hist_feature_in, **config_to_kwargs(tower_config.hist_seq_mlp)
            )

        self._capsule_layer = CapsuleLayer()

    def init_input(self) -> None:
        """Initialize input."""
        super().init_input()
        self.hist_embedding_group = EmbeddingGroup(
            self._hist_features, [self._hist_feature_group]
        )

    def build_input(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Build input."""
        feature_dict = super().build_input(batch)
        hist_feature_dict = self.hist_embedding_group(batch)
        return feature_dict, hist_feature_dict

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower."""
        grp_user, grp_hist = self.build_input(batch)

        user_feature = self.user_mlp(grp_user[self._group_name])

        hist_seq_feas = torch.concat(grp_hist[self._hist_group_name], dim=-1)  # todo

        if self._tower_config.user_seq_mlp:
            hist_seq_feas = self.hist_seq_mlp(hist_seq_feas)

        high_capsules = self._capsule_layer(hist_seq_feas)  # todo

        # concatenate user feature and high_capsules
        user_feature_tile = torch.tile(
            user_feature, [1, high_capsules.shape[1]]
        )  # todo
        user_interests = torch.cat([user_feature_tile, high_capsules], dim=-1)

        if self._similarity == match_model_pb2.Similarity.COSINE:
            user_interests = F.normalize(user_interests, p=2.0, dim=1)
        return user_interests


class MINDItemTower(MatchTower):
    """MIND item tower."""

    def __init__(
        self,
        tower_config: tower_pb2.MINDItemTower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        item_feature_group: model_pb2.FeatureGroupConfig,
        item_features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config,
            output_dim,
            similarity,
            item_feature_group,
            item_features,
            model_config,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower."""
        pass


class MIND(MatchModel):
    """MIND model."""

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
        # user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        # hist_group = name_to_feature_group[
        #     self._model_config.user_tower.history_input
        # ]

        name_to_feature = {x.name: x for x in features}
        # user_features = [name_to_feature[x] for x in user_group.feature_names]
        item_features = [name_to_feature[x] for x in item_group.feature_names]
        # hist_features = [name_to_feature[x] for x in hist_group.feature_names]

        self.user_tower = MINDUserTower(self._model_config.user_tower)
        self.item_tower = MINDItemTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            item_group,
            item_features,
            model_config,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the model."""
        pass
