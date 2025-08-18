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

from typing import Any, Dict, List, Optional, Union

import torch
from torch import nn

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.layers.backbone import Backbone
from tzrec.models.match_model import MatchModel
from tzrec.protos import simi_pb2
from tzrec.protos.model_pb2 import ModelConfig


class MatchBackbone(MatchModel):
    """Match backbone model for flexible dual-tower matching with configurable backbone.

    This implementation supports various matching models (DSSM, DAT, etc.) by using
    a flexible backbone network that can output features for different towers.

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

        # 获取match_backbone配置
        self._match_backbone_config = self._base_model_config.match_backbone

        # 从model_params获取基本参数，设置默认值
        model_params = getattr(self._match_backbone_config, "model_params", None)
        self._output_dim = 64  # 默认输出维度
        self._similarity_type = simi_pb2.INNER_PRODUCT  # 默认相似度类型
        self._temperature = 1.0  # 默认温度参数

        # 尝试从不同来源获取参数
        if model_params:
            # 从model_params获取参数（如果有的话）
            self._output_dim = getattr(model_params, "output_dim", self._output_dim)
            if hasattr(model_params, "similarity"):
                self._similarity_type = model_params.similarity
            if hasattr(model_params, "temperature"):
                self._temperature = model_params.temperature

        # 也可以从kwargs中获取参数（运行时传入）
        self._output_dim = kwargs.get("output_dim", self._output_dim)
        self._similarity_type = kwargs.get("similarity", self._similarity_type)
        self._temperature = kwargs.get("temperature", self._temperature)

        # 构建backbone网络
        self._backbone_net = self.build_backbone_network()

        # 获取backbone的输出配置
        self._output_blocks = self._get_output_blocks()

        # 根据输出blocks确定用户塔和物品塔的输入
        self._user_tower_input = self._output_blocks.get("user", None)
        self._item_tower_input = self._output_blocks.get("item", None)

        # 如果没有明确指定用户塔和物品塔输入，使用默认逻辑
        if not self._user_tower_input and not self._item_tower_input:
            self._setup_default_tower_inputs()

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
            config=self._match_backbone_config.backbone,
            features=self._features,
            embedding_group=None,  # 让Backbone自己创建EmbeddingGroup
            feature_groups=feature_groups,
            wide_embedding_dim=wide_embedding_dim,
            wide_init_fn=wide_init_fn
        )

    def _get_output_blocks(self) -> Dict[str, str]:
        """Get output blocks configuration for different towers.

        Returns:
            Dict[str, str]: mapping from tower name to block name.
        """
        output_blocks = {}
        backbone_config = self._match_backbone_config.backbone

        # 检查是否有output_blocks配置
        if hasattr(backbone_config, "output_blocks") and backbone_config.output_blocks:
            output_block_list = list(backbone_config.output_blocks)

            # 尝试根据block名称推断用户塔和物品塔
            for block_name in output_block_list:
                if "user" in block_name.lower():
                    output_blocks["user"] = block_name
                elif "item" in block_name.lower() or "product" in block_name.lower():
                    output_blocks["item"] = block_name

            # 如果有2个输出blocks但没有匹配到用户/物品，按顺序分配
            if len(output_block_list) == 2 and len(output_blocks) == 0:
                output_blocks["user"] = output_block_list[0]
                output_blocks["item"] = output_block_list[1]

        return output_blocks

    def _setup_default_tower_inputs(self):
        """Setup default tower inputs when not explicitly configured."""
        # 默认假设backbone输出单个tensor或两个tensor
        backbone_output_names = self._backbone_net.get_output_block_names()

        if len(backbone_output_names) >= 2:
            self._user_tower_input = backbone_output_names[0]
            self._item_tower_input = backbone_output_names[1]
        else:
            # 单输出情况下，用户塔和物品塔共享同一个输出
            self._user_tower_input = (
                backbone_output_names[0] if backbone_output_names else "shared"
            )
            self._item_tower_input = self._user_tower_input

    def backbone(
        self, batch: Batch
    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
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

    def _extract_tower_feature(
        self,
        backbone_output: Union[
            torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]
        ],
        tower_input: str,
    ) -> torch.Tensor:
        """Extract tower-specific feature from backbone output.

        Args:
            backbone_output: Output from backbone network.
            tower_input: Name of the input for this tower.

        Returns:
            torch.Tensor: Tower-specific feature tensor.
        """
        if isinstance(backbone_output, dict):
            # 如果backbone返回字典，直接按名称获取
            if tower_input in backbone_output:
                return backbone_output[tower_input]
            else:
                # 如果找不到指定的tower_input，尝试一些通用的键名
                for key in backbone_output.keys():
                    if tower_input.lower() in key.lower():
                        return backbone_output[key]
                # 如果都找不到，返回第一个值
                return list(backbone_output.values())[0]
        elif isinstance(backbone_output, (list, tuple)):
            # 如果backbone返回列表，需要根据tower_input确定索引
            if tower_input == self._user_tower_input and len(backbone_output) > 0:
                return backbone_output[0]
            elif tower_input == self._item_tower_input and len(backbone_output) > 1:
                return backbone_output[1]
            else:
                return backbone_output[0]
        else:
            # 如果是单个tensor，直接返回
            return backbone_output

    def user_tower(self, batch: Batch) -> torch.Tensor:
        """Extract user embedding from backbone output.

        Args:
            batch (Batch): input batch data.

        Returns:
            torch.Tensor: user embedding tensor.
        """
        backbone_output = self.backbone(batch)
        user_feature = self._extract_tower_feature(
            backbone_output, self._user_tower_input
        )

        # 如果特征维度与输出维度不匹配，需要投影
        if user_feature.size(-1) != self._output_dim:
            if not hasattr(self, "_user_projection_layer"):
                self._user_projection_layer = nn.Linear(
                    user_feature.size(-1), self._output_dim
                )
                if torch.cuda.is_available() and user_feature.is_cuda:
                    self._user_projection_layer = self._user_projection_layer.cuda()
            user_emb = self._user_projection_layer(user_feature)
        else:
            user_emb = user_feature

        # 根据相似度类型决定是否归一化
        if self._similarity_type == simi_pb2.COSINE:
            user_emb = nn.functional.normalize(user_emb, p=2, dim=-1)

        return user_emb

    def item_tower(self, batch: Batch) -> torch.Tensor:
        """Extract item embedding from backbone output.

        Args:
            batch (Batch): input batch data.

        Returns:
            torch.Tensor: item embedding tensor.
        """
        backbone_output = self.backbone(batch)
        item_feature = self._extract_tower_feature(
            backbone_output, self._item_tower_input
        )

        # 如果特征维度与输出维度不匹配，需要投影
        if item_feature.size(-1) != self._output_dim:
            if not hasattr(self, "_item_projection_layer"):
                self._item_projection_layer = nn.Linear(
                    item_feature.size(-1), self._output_dim
                )
                if torch.cuda.is_available() and item_feature.is_cuda:
                    self._item_projection_layer = self._item_projection_layer.cuda()
            item_emb = self._item_projection_layer(item_feature)
        else:
            item_emb = item_feature

        # 根据相似度类型决定是否归一化
        if self._similarity_type == simi_pb2.COSINE:
            item_emb = nn.functional.normalize(item_emb, p=2, dim=-1)

        return item_emb

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Predict the model.

        Args:
            batch (Batch): input batch data.

        Return:
            predictions (dict): a dict of predicted result.
        """
        # 获取用户和物品的embedding
        user_emb = self.user_tower(batch)
        item_emb = self.item_tower(batch)

        # 计算相似度
        hard_neg_indices = getattr(batch, "hard_neg_indices", None)
        similarity = self.sim(user_emb, item_emb, hard_neg_indices)

        # 应用温度缩放
        if self._temperature != 1.0:
            similarity = similarity / self._temperature

        return {"similarity": similarity}

    def get_user_tower(self) -> nn.Module:
        """Get user tower for inference.

        Returns:
            nn.Module: user tower module for jit scripting.
        """

        class UserTowerInference(nn.Module):
            def __init__(self, match_backbone_model):
                super().__init__()
                self.backbone_net = match_backbone_model._backbone_net
                self._user_tower_input = match_backbone_model._user_tower_input
                self._output_dim = match_backbone_model._output_dim
                self._similarity_type = match_backbone_model._similarity_type

                # 复制投影层如果存在
                if hasattr(match_backbone_model, "_user_projection_layer"):
                    self.user_projection_layer = (
                        match_backbone_model._user_projection_layer
                    )
                else:
                    self.user_projection_layer = None

            def forward(self, batch: Batch) -> torch.Tensor:
                backbone_output = self.backbone_net(is_training=False, batch=batch)

                # 提取用户特征
                if isinstance(backbone_output, dict):
                    if self._user_tower_input in backbone_output:
                        user_feature = backbone_output[self._user_tower_input]
                    else:
                        user_feature = list(backbone_output.values())[0]
                elif isinstance(backbone_output, (list, tuple)):
                    user_feature = backbone_output[0]
                else:
                    user_feature = backbone_output

                # 应用投影层
                if self.user_projection_layer is not None:
                    user_emb = self.user_projection_layer(user_feature)
                else:
                    user_emb = user_feature

                # 归一化
                if self._similarity_type == simi_pb2.COSINE:
                    user_emb = nn.functional.normalize(user_emb, p=2, dim=-1)

                return user_emb

        return UserTowerInference(self)

    def get_item_tower(self) -> nn.Module:
        """Get item tower for inference.

        Returns:
            nn.Module: item tower module for jit scripting.
        """

        class ItemTowerInference(nn.Module):
            def __init__(self, match_backbone_model):
                super().__init__()
                self.backbone_net = match_backbone_model._backbone_net
                self._item_tower_input = match_backbone_model._item_tower_input
                self._output_dim = match_backbone_model._output_dim
                self._similarity_type = match_backbone_model._similarity_type

                # 复制投影层如果存在
                if hasattr(match_backbone_model, "_item_projection_layer"):
                    self.item_projection_layer = (
                        match_backbone_model._item_projection_layer
                    )
                else:
                    self.item_projection_layer = None

            def forward(self, batch: Batch) -> torch.Tensor:
                backbone_output = self.backbone_net(is_training=False, batch=batch)

                # 提取物品特征
                if isinstance(backbone_output, dict):
                    if self._item_tower_input in backbone_output:
                        item_feature = backbone_output[self._item_tower_input]
                    else:
                        item_feature = list(backbone_output.values())[0]
                elif isinstance(backbone_output, (list, tuple)):
                    item_feature = (
                        backbone_output[1]
                        if len(backbone_output) > 1
                        else backbone_output[0]
                    )
                else:
                    item_feature = backbone_output

                # 应用投影层
                if self.item_projection_layer is not None:
                    item_emb = self.item_projection_layer(item_feature)
                else:
                    item_emb = item_feature

                # 归一化
                if self._similarity_type == simi_pb2.COSINE:
                    item_emb = nn.functional.normalize(item_emb, p=2, dim=-1)

                return item_emb

        return ItemTowerInference(self)
