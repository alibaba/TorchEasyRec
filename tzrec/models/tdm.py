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

from typing import Dict, List

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.rank_model import RankModel
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.modules.mlp import MLP
from tzrec.modules.sequence import MultiWindowDINEncoder
from tzrec.protos import model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class TDM(RankModel):
    """TDM model.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        features (list): list of features.
        labels (list): list of label names.
        sample_weights (list): sample weight names.
    """

    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str], sample_weights: List[str] = None
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights)
        self.embedding_group = EmbeddingGroup(
            features, list(model_config.feature_groups)
        )

        non_seq_fea_dim = 0
        self.seq_group_name = ""
        self.non_seq_group_name = []
        for feature_group in model_config.feature_groups:
            if feature_group.group_type == model_pb2.SEQUENCE:
                self.seq_group_name = feature_group.group_name
            else:
                non_seq_fea_dim += self.embedding_group.group_total_dim(
                    feature_group.group_name
                )
                self.non_seq_group_name.append(feature_group.group_name)

        self.multiwindow_din = MultiWindowDINEncoder(
            self.embedding_group.group_total_dim(f"{self.seq_group_name}.sequence"),
            self.embedding_group.group_total_dim(f"{self.seq_group_name}.query"),
            self.seq_group_name,
            list(self._model_config.multiwindow_din.windows_len),
            config_to_kwargs(self._model_config.multiwindow_din.attn_mlp),
        )

        self.deep_mlp = MLP(
            in_features=self.multiwindow_din.output_dim() + non_seq_fea_dim,
            **config_to_kwargs(self._model_config.final),
        )

        final_dim = self.deep_mlp.output_dim()
        self.output_mlp = nn.Linear(final_dim, self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_feature = self.embedding_group(batch)

        multiwindow_output = self.multiwindow_din(grouped_feature)
        mlp_input = multiwindow_output

        for group_name in self.non_seq_group_name:
            mlp_input = torch.concat([mlp_input, grouped_feature[group_name]], dim=1)
        mlp_output = self.deep_mlp(mlp_input)

        y = self.output_mlp(mlp_output)
        return self._output_to_prediction(y)


class TDMEmbedding(nn.Module):
    """TDMEmbedding inference wrapper for jit.script."""

    def __init__(
        self, module: nn.Module, embedding_group_name: str = "embedding_group"
    ) -> None:
        super().__init__()
        self._embedding_group_name = embedding_group_name

        seq_feature_group = None
        for feature_group in module._feature_groups:
            if feature_group.group_type == model_pb2.SEQUENCE:
                seq_feature_group = feature_group
                self.seq_feature_group_name = feature_group.group_name
                break

        name_to_fea = {x.name: x for x in module._features}
        seq_group_query_fea = []
        seq_feature_group_feature_names = list(seq_feature_group.feature_names)
        for feature_name in seq_feature_group_feature_names:
            feature = name_to_fea[feature_name]
            if feature.is_sequence:
                seq_feature_group.feature_names.remove(feature_name)
            else:
                seq_group_query_fea.append(feature)
        self._features = seq_group_query_fea
        setattr(
            self,
            embedding_group_name,
            EmbeddingGroup(seq_group_query_fea, [seq_feature_group]),
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the embedding.

        Args:
            batch (Batch): input batch data.

        Return:
            embedding (dict): embedding group output.
        """
        grouped_feature = getattr(self, self._embedding_group_name)(batch)
        result = {"item_emb": grouped_feature[f"{self.seq_feature_group_name}.query"]}
        return result
