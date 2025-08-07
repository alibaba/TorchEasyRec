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
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import model_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.modules.variational_dropout import VariationalDropout

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
        # self.init_input()
        self._feature_dict = features
        self._backbone_output = None
        self._l2_reg = None
        self._backbone_net = self.build_backbone_network()
        
        # 使用backbone的最终输出维度，考虑top_mlp的影响
        output_dims = self._backbone_net.get_final_output_dim()
        # 如果有多个 package（如 Package.__packages 里），如何Í拿到output_dims，暂未实现
        # for pkg_name, pkg in Package._Package__packages.items():
        #     print(f"Package: {pkg_name}")
        #     print("  输出block列表:", pkg.get_output_block_names())
        #     print("  输出block维度:", pkg.output_block_dims())
        #     print("  总输出维度:", pkg.total_output_dim())
        self.output_mlp = nn.Linear(output_dims, self._num_class)
    
    def init_input(self) -> None:
        """Build embedding group and group variational dropout."""
        self.embedding_group = EmbeddingGroup(
            self._features,
            list(self._base_model_config.feature_groups),
            wide_embedding_dim=int(self.wide_embedding_dim)
            if hasattr(self, "wide_embedding_dim")
            else None,
            wide_init_fn=self.wide_init_fn if hasattr(self, "wide_init_fn") else None,
        )

        if self._base_model_config.HasField("variational_dropout"):
            self.group_variational_dropouts = nn.ModuleDict()
            variational_dropout_config = self._base_model_config.variational_dropout
            variational_dropout_config_dict = config_to_kwargs(
                variational_dropout_config
            )
            for feature_group in list(self._base_model_config.feature_groups):
                group_name = feature_group.group_name
                if feature_group.group_type != model_pb2.SEQUENCE:
                    feature_dim = self.embedding_group.group_feature_dims(group_name)
                    if len(feature_dim) > 1:
                        variational_dropout = VariationalDropout(
                            feature_dim, group_name, **variational_dropout_config_dict
                        )
                        self.group_variational_dropouts[group_name] = (
                            variational_dropout
                        )
        

    def build_backbone_network(self):
        """Build backbone."""
        # return Backbone(
        #     self._base_model_config.rank_backbone.backbone,
        #     self._feature_dict,
        #     embedding_group=self.embedding_group,
        #     # input_layer=self._input_layer,
        #     l2_reg=self._l2_reg,
        # )
        wide_embedding_dim=int(self.wide_embedding_dim) if hasattr(self, "wide_embedding_dim") else None
        wide_init_fn=self.wide_init_fn if hasattr(self, "wide_init_fn") else None
        feature_groups = list(self._base_model_config.feature_groups)
        return Backbone(
            config=self._base_model_config.rank_backbone.backbone,
            features=self._feature_dict,
            embedding_group=self.embedding_group,
            feature_groups=feature_groups,
            wide_embedding_dim=wide_embedding_dim,
            wide_init_fn=wide_init_fn,
            # input_layer=self._input_layer,
            l2_reg=self._l2_reg,
        )

    def backbone(
        self, 
        # group_features: Dict[str, torch.Tensor], 
        batch: Batch
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
                # group_features=group_features,
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
        # grouped_features = self.build_input(batch)
        # output = self.backbone(group_features=grouped_features, batch=batch)
        output = self.backbone( batch=batch)
        y = self.output_mlp(output)
        return self._output_to_prediction(y)
