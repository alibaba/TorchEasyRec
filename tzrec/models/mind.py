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
from torch import nn
from torch._tensor import Tensor
from torch.linalg import norm

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTower
from tzrec.modules.capsule import CapsuleLayer
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
from tzrec.utils.config_util import config_to_kwargs


class MINDUserTower(MatchTower):
    """MIND user tower.

    Args:
        tower_config (Tower): mind user tower config.
        output_dim (int): user output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        user_feature_group (FeatureGroupConfig): user feature group config.
        hist_feature_group (FeatureGroupConfig): history sequence feature group config.
        user_features (list): list of user features.
        hist_features (list): list of history sequence features.
        model_config (ModelConfig): model config.
    """

    def __init__(
        self,
        tower_config: tower_pb2.MINDUserTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
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
            [user_feature_group, hist_feature_group],
            user_features + hist_features,
            model_config,
        )

        self._hist_group_name = tower_config.history_input

        self.init_input()

        user_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.user_mlp = MLP(
            in_features=user_feature_in,
            hidden_units=tower_config.user_mlp.hidden_units[0:-1],
            activation=tower_config.user_mlp.activation,
            use_bn=tower_config.user_mlp.use_bn,
            dropout_ratio=tower_config.user_mlp.dropout_ratio[0],
        )
        self.user_mlp_out = nn.Linear(
            self.user_mlp.output_dim(), tower_config.user_mlp.hidden_units[-1]
        )

        hist_feature_dim = self.embedding_group.group_total_dim(
            self._hist_group_name + ".sequence"
        )

        if (
            tower_config.hist_seq_mlp
            and len(tower_config.hist_seq_mlp.hidden_units) > 0
        ):
            self._hist_seq_mlp = MLP(
                in_features=hist_feature_dim,
                dim=3,
                hidden_units=tower_config.hist_seq_mlp.hidden_units[0:-1],
                activation=tower_config.hist_seq_mlp.activation,
                use_bn=tower_config.hist_seq_mlp.use_bn,
                dropout_ratio=tower_config.hist_seq_mlp.dropout_ratio[0],
            )
            self._hist_seq_mlp_out = nn.Linear(
                self._hist_seq_mlp.output_dim(),
                tower_config.hist_seq_mlp.hidden_units[-1],
                bias=False,
            )
            capsule_input_dim = tower_config.hist_seq_mlp.hidden_units[-1]

        else:
            self._hist_seq_mlp = None
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
        if self._output_dim > 0:
            self.output = nn.Linear(
                self._concat_mlp.output_dim(), output_dim, bias=False
            )

    def forward(self, batch: Batch) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.

        Returns:
            user_interests (Tensor): user interests.
        """
        user_feature_dict = self.build_input(batch)
        grp_hist_seq = user_feature_dict[self._hist_group_name + ".sequence"]
        grp_hist_len = user_feature_dict[self._hist_group_name + ".sequence_length"]

        user_feature = self.user_mlp_out(
            self.user_mlp(user_feature_dict[self._group_name])
        )

        if self._hist_seq_mlp:
            hist_seq_feas = self._hist_seq_mlp_out(self._hist_seq_mlp(grp_hist_seq))
        else:
            hist_seq_feas = grp_hist_seq

        high_capsules = self._capsule_layer(hist_seq_feas, grp_hist_len)
        high_capsules_mask = norm(high_capsules, dim=-1) != 0.0

        # concatenate user feature and high_capsules
        user_feature = torch.unsqueeze(user_feature, dim=1)
        user_feature_tile = torch.tile(user_feature, [1, high_capsules.shape[1], 1])
        user_interests = torch.cat([user_feature_tile, high_capsules], dim=-1)
        user_interests = user_interests * high_capsules_mask.unsqueeze(-1).float()
        user_interests = self._concat_mlp(user_interests)
        user_interests = user_interests * high_capsules_mask.unsqueeze(-1).float()

        if self._output_dim > 0:
            user_interests = self.output(user_interests)

        if self._similarity == simi_pb2.Similarity.COSINE:
            user_interests = F.normalize(user_interests, p=2.0, dim=-1)

        return user_interests


class MINDItemTower(MatchTower):
    """MIND item tower.

    Args:
        tower_config (Tower): mind item tower config.
        output_dim (int): item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        item_feature_group (FeatureGroupConfig): item feature group config.
        item_features (list): list of item features.
        model_config (ModelConfig): model config.

    """

    def __init__(
        self,
        tower_config: tower_pb2.MINDItemTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
        item_feature_group: model_pb2.FeatureGroupConfig,
        item_features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(
            tower_config,
            output_dim,
            similarity,
            [item_feature_group],
            item_features,
            model_config,
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
            item_emb (Tensor): item embedding.

        """
        grouped_features = self.build_input(batch)
        item_emb = self.mlp(grouped_features[self._group_name])

        if self._output_dim > 0:
            item_emb = self.output(item_emb)

        if self._similarity == simi_pb2.Similarity.COSINE:
            item_emb = F.normalize(item_emb, p=2.0, dim=1)
        return item_emb


class MIND(MatchModel):
    """MIND model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): list of sample weight names.

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
        hist_group = name_to_feature_group[self._model_config.user_tower.history_input]

        user_features = self.get_features_in_feature_groups([user_group])
        item_features = self.get_features_in_feature_groups([item_group])
        hist_features = self.get_features_in_feature_groups([hist_group])

        self.user_tower = MINDUserTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            hist_group,
            user_features,
            hist_features,
            model_config,
        )
        self.item_tower = MINDItemTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
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
        """Compute label-aware attention for user interests.

        Args:
            user_interests (Tensor): user interests.
            item_emb (Tensor): item embedding.

        Returns:
            user_emb (Tensor): user embedding.
        """
        batch_size = user_interests.size(0)
        pos_item_emb = item_emb[:batch_size]

        interest_mask = norm(user_interests, dim=-1) != 0.0

        simi_pow = self._model_config.simi_pow
        interest_weight = torch.einsum("bkd, bd->bk", user_interests, pos_item_emb)

        threshold = (interest_mask.float() * 2 - 1) * 1e32
        interest_weight = torch.minimum(interest_weight, threshold)

        interest_weight = interest_weight.unsqueeze(-1)
        interest_weight = interest_weight * simi_pow
        interest_weight = torch.nn.functional.softmax(interest_weight, dim=1)

        user_emb = torch.sum(torch.multiply(interest_weight, user_interests), dim=1)
        return user_emb

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Returns:
            simi (dict): a dict of predicted result.

        """
        user_interests = self.user_tower(batch)

        item_emb = self.item_tower(batch)

        user_emb = self.label_aware_attention(user_interests, item_emb)
        if self._model_config.similarity == simi_pb2.Similarity.COSINE:
            user_emb = F.normalize(user_emb, p=2.0, dim=1)

        ui_sim = self.sim(user_emb, item_emb) / self._model_config.temperature
        return {"similarity": ui_sim}
