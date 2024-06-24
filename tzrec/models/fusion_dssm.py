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
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import fx, nn
from torch._tensor import Tensor

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.fusion_match_model import FusionMatchModel
from tzrec.models.match_model import MatchTower
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.interaction import InputSENet
from tzrec.modules.mlp import MLP
from tzrec.protos import model_pb2, tower_pb2
from tzrec.protos.models import match_model_pb2
from tzrec.utils.config_util import config_to_kwargs


@fx.wrap
def _perm_feat(
    feature: torch.Tensor,
    feat_dims: List[int],
    perm_feats_index: List[int],
    perm_nflods: int,
) -> torch.Tensor:
    feature_list = list(torch.split(feature, feat_dims, dim=-1))
    features = []
    feature_list_tmp = []
    for _ in range(perm_nflods):
        perm = torch.randperm(feature_list[0].size(0))
        feature_list_tmp = []
        for idx, f in enumerate(feature_list):
            if idx in perm_feats_index:
                feature_list_tmp.append(f[perm])
            else:
                feature_list_tmp.append(f)
        features.append(torch.cat(feature_list_tmp, dim=-1))
    return torch.cat(features)


class DSSMTower(MatchTower):
    """DSSM user/item tower.

    Args:
        tower_config (Tower): user/item tower config.
        output_dim (int): user/item output embedding dimension.
        similarity (Similarity): when use COSINE similarity,
            will norm the output embedding.
        use_senet (bool): use input senet or not.
        feature_group (FeatureGroupConfig): feature group config.
        features (list): list of features.
    """

    def __init__(
        self,
        tower_config: tower_pb2.Tower,
        output_dim: int,
        similarity: match_model_pb2.Similarity,
        use_senet: bool,
        feature_group: model_pb2.FeatureGroupConfig,
        features: List[BaseFeature],
        perm_features: Optional[List[str]] = None,
        perm_nflods: int = 0,
    ) -> None:
        super().__init__(tower_config, output_dim, similarity, feature_group, features)
        self._use_senet = use_senet
        self.embedding_group = EmbeddingGroup(features, [feature_group])
        if self._use_senet:
            self.senet = InputSENet(
                length_per_key=self.embedding_group.group_dims(self._group_name)
            )
        tower_feature_in = self.embedding_group.group_total_dim(self._group_name)
        self.mlp = MLP(tower_feature_in, **config_to_kwargs(tower_config.mlp))
        self.output = nn.Linear(self.mlp.output_dim(), output_dim)

        feature_names = list(feature_group.feature_names)
        self._perm_feats_index = []
        if perm_features:
            for feat_name in perm_features:
                self._perm_feats_index.append(feature_names.index(feat_name))
        self._perm_nflods = perm_nflods

    def forward(
        self,
        batch: Batch,
        tower_index: Optional[torch.Tensor] = None,
        permute: bool = False,
    ) -> torch.Tensor:
        """Forward the tower.

        Args:
            batch (Batch): input batch data.
            tower_index (torch.Tensor, optional): valid tower row.
            permute (bool): permute feat or not.

        Return:
            embedding (dict): tower output embedding.
        """
        grouped_features = self.embedding_group(batch)
        feature = grouped_features[self._group_name]
        if tower_index is not None:
            feature = feature[tower_index]
        if permute and len(self._perm_feats_index) > 0:
            feature = _perm_feat(
                feature,
                self.embedding_group.group_dims(self._group_name),
                self._perm_feats_index,
                self._perm_nflods,
            )
        if self._use_senet:
            feature = self.senet(feature)
        mlp_output = self.mlp(feature)
        output = self.output(mlp_output)
        if self._similarity == match_model_pb2.Similarity.COSINE:
            output = F.normalize(output, p=2.0, dim=1)
        return output


class FusionDSSM(FusionMatchModel):
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
    ) -> None:
        super().__init__(model_config, features, labels)
        name_to_feature_group = {x.group_name: x for x in model_config.feature_groups}

        user_groups = [
            name_to_feature_group[tower.input]
            for tower in self._model_config.user_tower
        ]
        item_group = name_to_feature_group[self._model_config.item_tower.input]

        name_to_feature = {x.name: x for x in features}
        user_features = [
            [name_to_feature[x] for x in g.feature_names] for g in user_groups
        ]
        item_features = [name_to_feature[x] for x in item_group.feature_names]

        self._user_tower_perm_feat = {
            x.tower_name: x for x in self._model_config.user_tower_perm_feat
        }

        self.user_tower = nn.ModuleList()
        for i, tower_cfg in enumerate(self._model_config.user_tower):
            perm_features = []
            perm_nflods = 0
            if tower_cfg.input in self._user_tower_perm_feat:
                perm_feat_cfg = self._user_tower_perm_feat[tower_cfg.input]
                perm_features = perm_feat_cfg.features
                perm_nflods = perm_feat_cfg.nflods
            setattr(
                self,
                f"{tower_cfg.input}_tower",
                DSSMTower(
                    tower_cfg,
                    self._model_config.output_dim,
                    self._model_config.similarity,
                    self._model_config.use_senet,
                    user_groups[i],
                    user_features[i],
                    perm_features,
                    perm_nflods,
                ),
            )
        self.item_tower = DSSMTower(
            self._model_config.item_tower,
            self._model_config.output_dim,
            self._model_config.similarity,
            self._model_config.use_senet,
            item_group,
            item_features,
        )

    def predict(self, batch: Batch) -> Dict[str, Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        item_tower_emb = self.item_tower(batch)

        prediction_dict = {}
        for i, tower_cfg in enumerate(self._model_config.user_tower):
            tower_index = batch.labels["channel"] == i
            batch_size = batch.labels["channel"].size(0)
            user_emb = getattr(self, f"{tower_cfg.input}_tower")(batch, tower_index)

            pos_item_emb = torch.index_select(
                item_tower_emb, 0, torch.where(tower_index)[0]
            )
            neg_item_emb = item_tower_emb[batch_size:]

            ui_sim = self.sim(user_emb, torch.cat([pos_item_emb, neg_item_emb]))

            if tower_cfg.input in self._user_tower_perm_feat and self.training:
                user_emb = getattr(self, f"{tower_cfg.input}_tower")(
                    batch, tower_index, permute=True
                )
                user_emb = user_emb.reshape(
                    (
                        self._user_tower_perm_feat[tower_cfg.input].nflods,
                        -1,
                        user_emb.size(1),
                    )
                )
                neg_ui_sim = torch.einsum("bij,ij->ib", user_emb, pos_item_emb)
                ui_sim = torch.cat([ui_sim, neg_ui_sim], dim=-1)

            prediction_dict[f"similarity_{tower_cfg.input}"] = (
                ui_sim / self._model_config.temperature
            )
        return prediction_dict
