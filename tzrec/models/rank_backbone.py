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
from tzrec.layers.backbone import Backbone
from tzrec.models.rank_model import RankModel
from tzrec.protos.model_pb2 import ModelConfig


class RankBackbone(RankModel):
    """Ranking backbone model."""

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
        self._feature_dict = features
        self._backbone_output = None
        self._l2_reg = None
        self._backbone_net = self.build_backbone_network()

    def build_backbone_network(self):
        """Build backbone."""
        # if self.has_backbone:
        if True:
            return Backbone(
                self._base_model_config.rank_backbone.backbone,
                self._feature_dict,
                embedding_group=self.embedding_group,
                # input_layer=self._input_layer,
                l2_reg=self._l2_reg,
            )
        return None

    def backbone(
        self, group_features: Dict[str, torch.Tensor], batch: Batch
    ) -> Optional[nn.Module]:
        # -> torch.Tensor:
        """Get backbone."""
        if self._backbone_output:
            return self._backbone_output
        if self._backbone_net:
            kwargs = {
                "loss_modules": self._loss_modules,
                "metric_modules": self._metric_modules,
                # 'prediction_modules': self._prediction_modules,
                "labels": self._labels,
            }
            return self._backbone_net(
                is_training=self.training,
                group_features=group_features,
                batch=batch,
                **kwargs,
            )
        return None

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        grouped_features = self.build_input(batch)
        output = self.backbone(group_features=grouped_features, batch=batch)
        if output.shape[-1] != self.num_class:
            # logging.info('add head logits layer for rank model')
            output = self.head_layer(output)

        # 返回预测结果
        prediction_dict = {"output": output}
        return prediction_dict
