# Copyright (c) 2025, Alibaba Group;
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
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.modules.backbone import Backbone
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MultiTaskBackbone(MultiTaskRank):
    """Multi-task backbone model.

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

        # 构建backbone网络
        self._backbone_net = self.build_backbone_network()

        # 构建任务塔
        self._task_towers = self.build_task_towers()

    def build_backbone_network(self):
        """Build backbone network."""
        wide_embedding_dim = (
            int(self.wide_embedding_dim)
            if hasattr(self, "wide_embedding_dim")
            else None
        )
        wide_init_fn = self.wide_init_fn if hasattr(self, "wide_init_fn") else None
        feature_groups = list(self._base_model_config.feature_groups)

        return Backbone(
            config=self._base_model_config.multi_task_backbone.backbone,
            features=self._features,
            embedding_group=None,  # 让Backbone自己创建EmbeddingGroup
            feature_groups=feature_groups,
            wide_embedding_dim=wide_embedding_dim,
            wide_init_fn=wide_init_fn,
        )

    def build_task_towers(self):
        """Build task towers based on backbone output dimension."""
        # 获取backbone的最终输出维度
        backbone_output_dim = self._backbone_net.output_dim()

        task_towers = nn.ModuleDict()
        for task_tower_cfg in self._task_tower_cfgs:
            tower_name = task_tower_cfg.tower_name
            num_class = task_tower_cfg.num_class

            # 检查是否有自定义MLP配置
            if task_tower_cfg.HasField("mlp"):
                from tzrec.modules.mlp import MLP

                mlp_config = config_to_kwargs(task_tower_cfg.mlp)
                task_tower = nn.Sequential(
                    MLP(in_features=backbone_output_dim, **mlp_config),
                    nn.Linear(mlp_config["hidden_units"][-1], num_class),
                )
            else:
                # 直接连接到输出层
                task_tower = nn.Linear(backbone_output_dim, num_class)

            task_towers[tower_name] = task_tower

        return task_towers

    def backbone(self, batch: Batch) -> torch.Tensor:
        """Get backbone output."""
        if self._backbone_net:
            kwargs = {
                "loss_modules": self._loss_modules,
                "metric_modules": self._metric_modules,
                "labels": self._labels,
            }
            return self._backbone_net(
                is_training=self.training,
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
        # 获取backbone输出
        backbone_output = self.backbone(batch)

        # 处理backbone输出：可能是单个tensor或tensor列表
        if isinstance(backbone_output, (list, tuple)):
            # backbone返回列表（如MMoE模块），需要与任务塔一一对应
            if len(backbone_output) != len(self._task_tower_cfgs):
                raise ValueError(
                    f"The number of backbone outputs ({len(backbone_output)}) and "
                    f"task towers ({len(self._task_tower_cfgs)}) must be equal"
                )
            task_input_list = backbone_output
        else:
            # backbone返回单个tensor，复制给所有任务塔
            task_input_list = [backbone_output] * len(self._task_tower_cfgs)

        # 通过各个任务塔生成预测
        tower_outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            tower_name = task_tower_cfg.tower_name
            task_input = task_input_list[i]  # 使用对应的输入
            tower_output = self._task_towers[tower_name](task_input)
            tower_outputs[tower_name] = tower_output

        # 转换为最终预测格式
        return self._multi_task_output_to_prediction(tower_outputs)
