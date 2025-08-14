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
from tzrec.modules.interaction import CIN
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class xDeepFM(RankModel):
    """XDeepFM model.

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
        self.wide_embedding_dim = self._model_config.wide_embedding_dim
        self.wide_init_fn = self._model_config.wide_init_fn
        self.init_input()
        deep_feature_dim = self.embedding_group.group_total_dim("deep")
        self.deep = MLP(
            in_features=deep_feature_dim, **config_to_kwargs(self._model_config.deep)
        )
        self.feature_num = len(self.embedding_group.group_dims("wide"))
        self.cin = CIN(
            feature_num=self.feature_num, **config_to_kwargs(self._model_config.cin)
        )

        self.final = MLP(
            in_features=self.deep.output_dim() + self.cin.output_dim(),
            **config_to_kwargs(self._model_config.final),
        )
        self.output_mlp = nn.Linear(self.final.output_dim(), self._num_class)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)

        # Wide
        wide_feat = grouped_features["wide"]
        wide_feat = wide_feat.reshape(-1, self.feature_num, self.wide_embedding_dim)
        cin_feat = self.cin(wide_feat)

        # Deep
        deep_feat = grouped_features["deep"]
        deep_feat = self.deep(deep_feat)

        all_feat = torch.cat([cin_feat, deep_feat], dim=1)
        y = self.final(all_feat)
        y = self.output_mlp(y)
        return self._output_to_prediction(y)
