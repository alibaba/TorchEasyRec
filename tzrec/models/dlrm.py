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

        assert self.embedding_group.has_group("dense"), "dense group is not specified"
        dense_dim = self.embedding_group.group_total_dim("dense")
        dense_feature_dims = self.embedding_group.group_feature_dims("dense")
        for feature_name in dense_feature_dims.keys():
            if "seq_encoder" in feature_name:
                raise Exception("dense group not have sequence features.")
        self.bot_mlp = MLP(dense_dim, **config_to_kwargs(self._model_config.bot_mlp))

        assert self.embedding_group.has_group("sparse"), "sparse group is not specified"
        sparse_feature_dims = self.embedding_group.group_feature_dims("sparse")
        sparse_dim = self.embedding_group.group_total_dim("sparse")
        self.per_sparse_dim = 0
        for feature_name, dim in sparse_feature_dims.items():
            self.per_sparse_dim = dim
            if "seq_encoder" in feature_name:
                raise Exception("sparse group not have sequence features.")
        self.sparse_num = len(sparse_feature_dims)
        sparse_dims = set(sparse_feature_dims.values())
        if len(sparse_dims) > 1:
            raise Exception(
                f"sparse group feature dims must be the same, but we find {sparse_dims}"
            )
        self.interaction = InteractionArch(self.sparse_num + 1)

        feature_dim = self.bot_mlp.output_dim() + self.interaction.output_dim()
        if self._model_config.arch_with_sparse:
            feature_dim += sparse_dim

        self.top_mlp = MLP(feature_dim, **config_to_kwargs(self._model_config.top_mlp))

        self.output_mlp = nn.Linear(self.top_mlp.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        # dense
        dense_group_feat = grouped_features["dense"]
        dense_feat = self.bot_mlp(dense_group_feat)

        # sparse
        sparse_group_feat = grouped_features["sparse"]
        sparse_feat = sparse_group_feat.reshape(
            -1, self.sparse_num, self.per_sparse_dim
        )

        # interaction
        interaction_feat = self.interaction(dense_feat, sparse_feat)

        # top mlp
        all_feat = torch.cat([interaction_feat, dense_feat], dim=-1)
        if self._model_config.arch_with_sparse:
            all_feat = torch.cat([all_feat, sparse_group_feat], dim=-1)
        top_feat = self.top_mlp(all_feat)

        # output
        y = self.output_mlp(top_feat)
        return self._output_to_prediction(y)
