# Copyright (c) 2024-2025, Alibaba Group;
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
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.sequence import HSTUEncoder
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils import config_util


@torch.fx.wrap
def _update_dict_tensor(
    tensor_dict: Dict[str, torch.Tensor],
    new_tensor_dict: Optional[Dict[str, Optional[torch.Tensor]]],
) -> None:
    if new_tensor_dict:
        for k, v in new_tensor_dict.items():
            if v is not None:
                tensor_dict[k] = v


class HSTUMatchUserTower(MatchTowerWoEG):
    """HSTU Match model user tower.

    Args:
        tower_config (Tower): user tower config.
        output_dim (int): user output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
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
        model_config: model_pb2.ModelConfig,
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self.tower_config = tower_config
        encoder_config = tower_config.hstu_encoder
        seq_config_dict = config_util.config_to_kwargs(encoder_config)
        sequence_dim = sum(feature_group_dims)
        seq_config_dict["sequence_dim"] = sequence_dim
        self.seq_encoder = HSTUEncoder(**seq_config_dict)

    def forward(self, grouped_features: Dict, is_train: bool = False) -> torch.Tensor:
        """Forward the tower.

        Args:
            grouped_features: Dictionary containing grouped feature tensors
            is_train: Boolean flag indicating whether the model is in training mode

        Returns:
            torch.Tensor: The output tensor from the tower
        """
        output = self.seq_encoder(grouped_features, is_train=is_train)

        return output


class HSTUMatchItemTower(MatchTowerWoEG):
    """HSTU Match model item tower.

    Args:
        tower_config (Tower): item tower config.
        output_dim (int): item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self.tower_config = tower_config

    def forward(self, grouped_features: Dict, is_train: bool = False) -> torch.Tensor:
        """Forward the tower.

        Args:
            grouped_features: Dictionary containing grouped feature tensors
            is_train: Boolean flag indicating whether the model is in training mode

        Returns:
            torch.Tensor: The output tensor from the tower
        """
        # print(grouped_features.keys(), self._group_name)
        output = grouped_features[f"{self._group_name}.sequence"]
        output = F.normalize(output, p=2.0, dim=1, eps=1e-6)

        return output


class HSTUMatch(MatchModel):
    """HSTU Match model.

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
        self.embedding_group = EmbeddingGroup(
            [features[0]], list([model_config.feature_groups[0]])
        )
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_group = name_to_feature_group[self._model_config.user_tower.input]

        name_to_feature = {x.name: x for x in features}
        user_features = OrderedDict(
            [(x, name_to_feature[x]) for x in user_group.feature_names]
        )
        for sequence_group in user_group.sequence_groups:
            for x in sequence_group.feature_names:
                user_features[x] = name_to_feature[x]

        self.user_tower = HSTUMatchUserTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            self.embedding_group.group_dims(
                self._model_config.user_tower.input + ".sequence"
            ),
            list(user_features.values()),
            model_config,
        )

        self.item_tower = HSTUMatchItemTower(
            self._model_config.user_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            user_group,
            list(user_features.values()),
        )

        self.seq_tower_input = self._model_config.user_tower.input

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        batch_sparse_features = batch.sparse_features["__BASE__"]
        nonzero_indices = torch.where(
            batch_sparse_features.lengths()[1:] != batch_sparse_features.lengths()[:-1]
        )[0]
        default_value = torch.tensor([-1]).to(nonzero_indices.device)
        batch_size = torch.cat([nonzero_indices, default_value]).max() + 1
        neg_sample_size = batch_sparse_features.lengths()[-1] - 1
        grouped_features = self.embedding_group(batch)

        item_group_features = {
            self.seq_tower_input + ".sequence": grouped_features[
                self.seq_tower_input + ".sequence"
            ][batch_size:, : neg_sample_size + 1],
        }
        item_tower_emb = self.item_tower(item_group_features, is_train=self.training)
        user_group_features = {
            self.seq_tower_input + ".sequence": grouped_features[
                self.seq_tower_input + ".sequence"
            ][:batch_size],
            self.seq_tower_input + ".sequence_length": grouped_features[
                self.seq_tower_input + ".sequence_length"
            ][:batch_size],
        }
        user_tower_emb = self.user_tower(user_group_features, is_train=self.training)
        ui_sim = (
            self.sim(
                user_tower_emb, item_tower_emb, neg_for_each_sample=True, is_hstu=True
            )
            / self._model_config.temperature
        )
        return {"similarity": ui_sim}
