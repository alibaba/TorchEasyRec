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
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs


class DSSMTower(MatchTowerWoEG):
    """DSSM user/item tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        feature_group_dims (list): feature dimension for each feature.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        feature_group_dims: List[int],
        features: List[BaseFeature],
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        tower_feature_in = sum(feature_group_dims)
        self.mlp = MLP(tower_feature_in, **config_to_kwargs(tower_config.mlp))
        if self._output_dim > 0:
            self.output = nn.Linear(self.mlp.output_dim(), output_dim)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward the tower.

        Args:
            feature (torch.Tensor): input batch data.

        Return:
            embedding (dict): tower output embedding.
        """
        output = self.mlp(feature)
        if self._output_dim > 0:
            output = self.output(output)
        if self._similarity == match_model_pb2.Similarity.COSINE:
            output = F.normalize(output, p=2.0, dim=1)
        return output


class DSSMV2(MatchModel):
    """DSSM model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: model_pb2.ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]

        name_to_feature = {x.name: x for x in features}
        user_features = OrderedDict(
            [(x, name_to_feature[x]) for x in user_group.feature_names]
        )
        for sequence_group in user_group.sequence_groups:
            for x in sequence_group.feature_names:
                user_features[x] = name_to_feature[x]
        item_features = [name_to_feature[x] for x in item_group.feature_names]

        self.user_tower = DSSMTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            self.embedding_group.group_dims(self._model_config.user_tower.input),
            list(user_features.values()),
        )

        self.item_tower = DSSMTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            item_group,
            self.embedding_group.group_dims(self._model_config.item_tower.input),
            item_features,
        )

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.embedding_group(batch)

        batch_size = batch.labels[self._labels[0]].size(0)
        user_feat = grouped_features[self._model_config.user_tower.input][:batch_size]
        item_feat = grouped_features[self._model_config.item_tower.input]
        user_tower_emb = self.user_tower(user_feat)
        item_tower_emb = self.item_tower(item_feat)

        ui_sim = (
            self.sim(user_tower_emb, item_tower_emb) / self._model_config.temperature
        )
        return {"similarity": ui_sim}
