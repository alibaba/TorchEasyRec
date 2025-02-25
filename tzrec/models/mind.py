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
from torch._tensor import Tensor

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
            user_features + hist_features,
            model_config,
        )

        self._hist_group_name = tower_config.history_input
        self._hist_feature_group = hist_feature_group
        self._hist_features = hist_features

        self.init_input()

        user_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.user_mlp = MLP(user_feature_in, **config_to_kwargs(tower_config.user_mlp))

        hist_feature_dim = self.hist_embedding_group.group_total_dim(
            self._hist_group_name + ".sequence"
        )
        if self._tower_config.hist_seq_mlp:
            self._hist_seq_mlp = MLP(
                in_features=hist_feature_dim,
                dim=3,
                **config_to_kwargs(tower_config.hist_seq_mlp),
            )
            capsule_input_dim = tower_config.hist_seq_mlp.hidden_units[-1]
        else:
            capsule_input_dim = hist_feature_dim

        self._capsule_layer = CapsuleLayer(
            capsule_config=tower_config.capsule_config,
            input_dim=capsule_input_dim,
        )
        self._concat_mlp = MLP(
            in_features=tower_config.user_mlp.hidden_units[-1]
            + tower_config.capsule_config.high_dim,
            dim=3,
            **config_to_kwargs(tower_config.concat_mlp),
        )

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
        grp_hist_seq = grp_hist[self._hist_group_name + ".sequence"]
        grp_hist_len = grp_hist[self._hist_group_name + ".sequence_length"]

        user_feature = self.user_mlp(grp_user[self._group_name])

        if self._tower_config.hist_seq_mlp:
            hist_seq_feas = self._hist_seq_mlp(grp_hist_seq)
        else:
            hist_seq_feas = grp_hist[self._hist_group_name]

        high_capsules = self._capsule_layer(hist_seq_feas, grp_hist_len)

        # concatenate user feature and high_capsules
        user_feature = torch.unsqueeze(user_feature, dim=1)
        user_feature_tile = torch.tile(user_feature, [1, high_capsules.shape[1], 1])
        user_interests = torch.cat([user_feature_tile, high_capsules], dim=-1)
        user_interests = self._concat_mlp(user_interests)

        if self._similarity == match_model_pb2.Similarity.COSINE:
            user_interests = F.normalize(user_interests, p=2.0, dim=-1)
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
        self.init_input()
        tower_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.mlp = MLP(tower_feature_in, **config_to_kwargs(tower_config.mlp))

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower."""
        grouped_features = self.build_input(batch)
        item_emb = self.mlp(grouped_features[self._group_name])

        if self._similarity == match_model_pb2.Similarity.COSINE:
            item_emb = F.normalize(item_emb, p=2.0, dim=1)
        return item_emb


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
        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]
        hist_group = name_to_feature_group[self._model_config.user_tower.history_input]

        name_to_feature = {x.name: x for x in features}
        user_features = [name_to_feature[x] for x in user_group.feature_names]
        item_features = [name_to_feature[x] for x in item_group.feature_names]
        hist_features = [name_to_feature[x] for x in hist_group.feature_names]

        self.user_tower = MINDUserTower(
            self._model_config.user_tower,
            0,
            self._model_config.similarity,
            user_group,
            hist_group,
            user_features,
            hist_features,
            model_config,
        )
        self.item_tower = MINDItemTower(
            self._model_config.item_tower,
            0,
            self._model_config.similarity,
            item_group,
            item_features,
            model_config,
        )

    def label_aware_attention(
        self,
        user_interests: torch.Tensor,
        item_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute label-aware attention for user interests."""
        batch_size = user_interests.size(0)
        pos_item_emb = item_emb[:batch_size]

        simi_pow = self._model_config.simi_pow
        interest_weight = torch.einsum("bkd, bd->bk", user_interests, pos_item_emb)
        interest_weight = interest_weight.unsqueeze(-1)
        interest_weight = torch.nn.functional.softmax(
            torch.pow(interest_weight, simi_pow), dim=1
        )

        user_emb = torch.sum(torch.multiply(interest_weight, user_interests), dim=1)
        return user_emb

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model."""
        user_interests = self.user_tower(batch)
        item_emb = self.item_tower(batch)

        user_emb = self.label_aware_attention(user_interests, item_emb)
        ui_sim = self.sim(user_emb, item_emb)
        return {"similarity": ui_sim}
