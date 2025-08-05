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

from tzrec.datasets.utils import HARD_NEG_INDICES, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.dssm import _update_dict_tensor
from tzrec.models.match_model import MatchModel, MatchTowerWoEG, _update_tensor_2_dict
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.modules.variational_dropout import VariationalDropout
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
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
        similarity: simi_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        feature_dims: Dict[str, int],
        features: List[BaseFeature],
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self._model_config = model_config
        self._feature_dims = feature_dims
        self.group_variational_dropouts = None
        self.group_variational_dropout_loss = {}
        self.init_variational_dropouts()
        tower_feature_in = sum(feature_dims.values())
        self.mlp = MLP(tower_feature_in, **config_to_kwargs(tower_config.mlp))
        if self._output_dim > 0:
            self.output = nn.Linear(self.mlp.output_dim(), output_dim)

    def init_variational_dropouts(self) -> None:
        """Build embedding group and group variational dropout."""
        if self._model_config.HasField("variational_dropout"):
            self.group_variational_dropouts = nn.ModuleDict()
            variational_dropout_config = self._model_config.variational_dropout
            variational_dropout_config_dict = config_to_kwargs(
                variational_dropout_config
            )
            if self._feature_group.group_type != model_pb2.SEQUENCE:
                if len(self._feature_dims) > 1:
                    variational_dropout = VariationalDropout(
                        self._feature_dims,
                        self._feature_group.group_name,
                        **variational_dropout_config_dict,
                    )
                    self.group_variational_dropouts[self._feature_group.group_name] = (
                        variational_dropout
                    )

    def run_variational_dropout(self, feature: torch.Tensor) -> torch.Tensor:
        """Run the variational dropout."""
        if self.group_variational_dropouts is not None:
            variational_dropout = self.group_variational_dropouts[self._group_name]
            feature, variational_dropout_loss = variational_dropout(feature)
            _update_tensor_2_dict(
                self.group_variational_dropout_loss,
                variational_dropout_loss,
                self._group_name + "_feature_p_loss",
            )
        return feature

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        """Forward the tower.

        Args:
            feature (torch.Tensor): input batch data.

        Return:
            embedding (dict): tower output embedding.
        """
        feature = self.run_variational_dropout(feature)
        output = self.mlp(feature)
        if self._output_dim > 0:
            output = self.output(output)
        if self._similarity == simi_pb2.Similarity.COSINE:
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
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )
        user_group = name_to_feature_group[self._model_config.user_tower.input]
        item_group = name_to_feature_group[self._model_config.item_tower.input]

        user_features = self.get_features_in_feature_groups([user_group])
        item_features = self.get_features_in_feature_groups([item_group])

        self.user_tower = DSSMTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            self.embedding_group.group_feature_dims(
                self._model_config.user_tower.input
            ),
            user_features,
            model_config,
        )

        self.item_tower = DSSMTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            item_group,
            self.embedding_group.group_feature_dims(
                self._model_config.item_tower.input
            ),
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
        grouped_features = self.embedding_group(batch)

        batch_size = batch.labels[self._labels[0]].size(0)
        user_feat = grouped_features[self._model_config.user_tower.input][:batch_size]
        item_feat = grouped_features[self._model_config.item_tower.input]
        user_tower_emb = self.user_tower(user_feat)
        item_tower_emb = self.item_tower(item_feat)
        _update_dict_tensor(
            self._loss_collection, self.user_tower.group_variational_dropout_loss
        )
        _update_dict_tensor(
            self._loss_collection, self.item_tower.group_variational_dropout_loss
        )
        ui_sim = (
            self.sim(
                user_tower_emb,
                item_tower_emb,
                batch.additional_infos.get(HARD_NEG_INDICES, None),
            )
            / self._model_config.temperature
        )
        return {"similarity": ui_sim}
