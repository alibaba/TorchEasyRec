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
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.interaction import InteractionArch
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class DLRM(RankModel):
    """DLRM model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        self.init_input()

        if len(self.embedding_group.group_names()) == 1:
            self._sparse_group_name = self.embedding_group.group_names()[0]
        else:
            self._sparse_group_name = "sparse"
        self.dense_mlp = None
        self._dense_group_name = "dense"
        if len(
            self.embedding_group.group_names()
        ) > 1 and self.embedding_group.has_group(self._dense_group_name):
            dense_dim = self.embedding_group.group_total_dim(self._dense_group_name)
            dense_feature_dims = self.embedding_group.group_feature_dims(
                self._dense_group_name
            )
            for feature_name in dense_feature_dims.keys():
                if "seq_encoder" in feature_name:
                    raise Exception("dense group not have sequence features.")
            self.dense_mlp = MLP(
                dense_dim, **config_to_kwargs(self._model_config.dense_mlp)
            )

        sparse_feature_dims = self.embedding_group.group_feature_dims(
            self._sparse_group_name
        )
        sparse_dim = self.embedding_group.group_total_dim(self._sparse_group_name)
        self._per_sparse_dim = 0
        for feature_name, dim in sparse_feature_dims.items():
            self._per_sparse_dim = dim
            if "seq_encoder" in feature_name:
                raise Exception("sparse group not have sequence features.")

        self._sparse_num = len(sparse_feature_dims)
        sparse_dims = set(sparse_feature_dims.values())
        if len(sparse_dims) > 1:
            raise Exception(
                f"sparse group feature dims must be the same, but we find {sparse_dims}"
            )
        if self.dense_mlp and self._per_sparse_dim != self.dense_mlp.output_dim():
            raise Exception(
                "dense mlp last hidden_unit must be the same sparse feature dim"
            )
        self._feature_num = self._sparse_num
        if self.dense_mlp:
            self._feature_num += 1
        self.interaction = InteractionArch(self._feature_num)

        feature_dim = self.interaction.output_dim()
        if self.dense_mlp:
            feature_dim += self.dense_mlp.output_dim()
        if self._model_config.arch_with_sparse:
            feature_dim += sparse_dim

        self.final_mlp = MLP(feature_dim, **config_to_kwargs(self._model_config.final))
        self.output_mlp = nn.Linear(self.final_mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        # sparse
        sparse_group_feat = grouped_features["sparse"]
        sparse_feat = sparse_group_feat.reshape(
            -1, self._sparse_num, self._per_sparse_dim
        )
        feat = sparse_feat
        # dense
        dense_feat = None
        if self.dense_mlp:
            dense_group_feat = grouped_features["dense"]
            dense_feat = self.dense_mlp(dense_group_feat)
            feat = torch.cat([dense_feat.unsqueeze(1), feat], dim=1)
        # interaction
        all_feat = self.interaction(feat)
        # final mlp
        if self.dense_mlp:
            all_feat = torch.cat([all_feat, dense_feat], dim=-1)
        if self._model_config.arch_with_sparse:
            all_feat = torch.cat([all_feat, sparse_group_feat], dim=-1)
        y_final = self.final_mlp(all_feat)

        # output
        y = self.output_mlp(y_final)
        return self._output_to_prediction(y)
