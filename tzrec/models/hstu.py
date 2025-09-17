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

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch._tensor import Tensor

from tzrec.datasets.utils import NEG_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel, MatchTowerWoEG
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.sequence import HSTUEncoder
from tzrec.protos import model_pb2, simi_pb2, tower_pb2
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
        tower_config: tower_pb2.HSTUMatchTower,
        output_dim: int,
        similarity: simi_pb2.Similarity,
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

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the tower.

        Args:
            grouped_features: Dictionary containing grouped feature tensors

        Returns:
            torch.Tensor: The output tensor from the tower
        """
        output = self.seq_encoder(grouped_features)

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
        similarity: simi_pb2.Similarity,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self.tower_config = tower_config

    def forward(self, grouped_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward the tower.

        Args:
            grouped_features: Dictionary containing grouped feature tensors

        Returns:
            torch.Tensor: The output tensor from the tower
        """
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
        assert len(model_config.feature_groups) == 1
        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}
        feature_group = name_to_feature_group[self._model_config.hstu_tower.input]

        used_features = self.get_features_in_feature_groups([feature_group])

        self.user_tower = HSTUMatchUserTower(
            self._model_config.hstu_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            feature_group,
            self.embedding_group.group_dims(
                self._model_config.hstu_tower.input + ".sequence"
            ),
            used_features,
            model_config,
        )

        self.item_tower = HSTUMatchItemTower(
            self._model_config.hstu_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            feature_group,
            used_features,
        )

        self.seq_tower_input = self._model_config.hstu_tower.input

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        batch_sparse_features = batch.sparse_features[NEG_DATA_GROUP]
        # Get batch_size and neg_sample_size from batch_sparse_features
        batch_size = batch.labels[self._label_name].shape[0]
        neg_sample_size = batch_sparse_features.lengths()[batch_size] - 1
        grouped_features = self.embedding_group(batch)

        item_group_features = {
            self.seq_tower_input + ".sequence": grouped_features[
                self.seq_tower_input + ".sequence"
            ][batch_size:, : neg_sample_size + 1],
        }
        item_tower_emb = self.item_tower(item_group_features)
        user_group_features = {
            self.seq_tower_input + ".sequence": grouped_features[
                self.seq_tower_input + ".sequence"
            ][:batch_size],
            self.seq_tower_input + ".sequence_length": grouped_features[
                self.seq_tower_input + ".sequence_length"
            ][:batch_size],
        }
        user_tower_emb = self.user_tower(user_group_features)
        ui_sim = (
            self.simi(user_tower_emb, item_tower_emb, neg_for_each_sample=True)
            / self._model_config.temperature
        )
        return {"similarity": ui_sim}

    def simi(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        neg_for_each_sample: bool = False,
    ) -> torch.Tensor:
        """Override the sim method in MatchModel to calculate similarity."""
        if self._in_batch_negative:
            return torch.mm(user_emb, item_emb.T)
        else:
            batch_size = user_emb.size(0)
            pos_item_emb = item_emb[:, 0]
            neg_item_emb = item_emb[:, 1:].reshape(-1, item_emb.shape[-1])
            pos_ui_sim = torch.sum(
                torch.multiply(user_emb, pos_item_emb), dim=-1, keepdim=True
            )
            neg_ui_sim = None
            if not neg_for_each_sample:
                neg_ui_sim = torch.matmul(user_emb, neg_item_emb.transpose(0, 1))
            else:
                num_neg_per_user = neg_item_emb.size(0) // batch_size
                neg_size = batch_size * num_neg_per_user
                neg_item_emb = neg_item_emb[:neg_size]
                neg_item_emb = neg_item_emb.view(batch_size, num_neg_per_user, -1)
                neg_ui_sim = torch.sum(user_emb.unsqueeze(1) * neg_item_emb, dim=-1)
            return torch.cat([pos_ui_sim, neg_ui_sim], dim=-1)
