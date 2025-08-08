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

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from tzrec.datasets.utils import Batch
from tzrec.modules.embedding import EmbeddingGroup


class EnhancedEmbeddingGroup(nn.Module):
    """对EmbeddingGroup输出的分组特征做增强处理：归一化、特征Dropout、普通Dropout等."""

    def __init__(
        self,
        embedding_group: EmbeddingGroup,
        group_name: str,
        do_batch_norm: bool = False,
        do_layer_norm: bool = False,
        dropout_rate: float = 0.0,
        feature_dropout_rate: float = 0.0,
        only_output_feature_list: bool = False,
        only_output_3d_tensor: bool = False,
        output_2d_tensor_and_feature_list: bool = False,
        concat_seq_feature: bool = False,
        output_seq_and_normal_feature: bool = False,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.group_name = group_name
        self.embedding_group = embedding_group

        self.do_batch_norm = do_batch_norm
        self.do_layer_norm = do_layer_norm
        self.dropout_rate = dropout_rate
        self.feature_dropout_rate = feature_dropout_rate

        self.only_output_feature_list = only_output_feature_list
        self.only_output_3d_tensor = only_output_3d_tensor
        self.output_2d_tensor_and_feature_list = output_2d_tensor_and_feature_list
        self.concat_seq_feature = concat_seq_feature
        self.output_seq_and_normal_feature = output_seq_and_normal_feature

        # 归一化/Dropout层后面动态创建
        self._built = False

    def output_dim(self) -> int:
        """获取整体拼接后（默认输出）的特征总维度.

        对应 default 返回 torch.cat(processed_features, dim=-1) 的维度.
        """
        # 用 group_total_dim 方法最合理
        return self.group_total_dim()

    def group_feature_dims(self) -> Dict[str, int]:
        """返回该 group 内每个特征的维度，字典格式：特征名 -> 维度."""
        return self.embedding_group.group_feature_dims(self.group_name)

    def group_dims(self) -> List[int]:
        """返回该 group 内每个特征的维度，list形式."""
        dims = self.group_feature_dims()
        return list(dims.values())

    def group_total_dim(self) -> int:
        """该 group 所有特征拼接起来的总维度."""
        # 推荐调用 embedding_group 的 group_total_dim
        return self.embedding_group.group_total_dim(self.group_name)

    # 可选，实现一个能返回3D输出时每个维的size的方法
    def output_3d_shape(self, batch_size: int) -> torch.Size:
        """如果 only_output_3d_tensor 为 True，返回输出tensor的shape."""
        dims = self.group_dims()
        return torch.Size([batch_size, len(dims), max(dims)])

    def build(self, sample_feature: torch.Tensor):
        """Build normalization and dropout layers based on feature dimensions."""
        feature_dim = sample_feature.shape[-1]
        if self.do_batch_norm:
            self.bn = nn.BatchNorm1d(feature_dim)
        else:
            self.bn = None
        if self.do_layer_norm:
            self.ln = nn.LayerNorm(feature_dim)
        else:
            self.ln = None
        if 0.0 < self.dropout_rate < 1.0:
            self.dropout = nn.Dropout(self.dropout_rate)
        else:
            self.dropout = None
        self._built = True

    def forward(
        self, batch: Batch, is_training: bool = True
    ) -> Union[
        torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]
    ]:
        """Forward pass with enhanced feature processing.

        Args:
            batch: Input batch data.
            is_training: Whether in training mode.

        Returns:
            Processed features in various formats based on configuration.
        """
        # Step 1: 调用embedding_group获得特征
        group_features = self.embedding_group.forward(batch)
        # group_features: dict[group_name] -> torch.Tensor or list
        # 兼容你旧用法，这里只取目标group
        features = group_features[self.group_name]

        # for sequence特征你可以自定义适配
        if isinstance(features, (list, tuple)):
            feature_list = list(features)
            features = (
                torch.cat(feature_list, dim=-1)
                if self.concat_seq_feature
                else feature_list
            )
        else:
            feature_list = [features]

        if not self._built:
            if isinstance(features, torch.Tensor):
                self.build(features)
            elif isinstance(feature_list[0], torch.Tensor):
                self.build(feature_list[0])
            else:
                raise RuntimeError("Feature shape error.")

        # Step 2: 归一化/Dropout/特征Dropout处理
        # 特征列表分别处理
        processed_features = []
        for fea in feature_list:
            out = fea
            if self.do_batch_norm:
                # BatchNorm1d要求shape=(N, C)，如果是高维要flatten
                if out.dim() > 2:
                    orig_shape = out.shape
                    out = out.view(-1, out.shape[-1])
                    out = self.bn(out)
                    out = out.view(orig_shape)
                else:
                    out = self.bn(out)
            if self.do_layer_norm:
                out = self.ln(out)
            if is_training and 0.0 < self.feature_dropout_rate < 1.0:
                mask = torch.bernoulli(
                    torch.full(
                        out.shape, 1 - self.feature_dropout_rate, device=out.device
                    )
                )
                out = out * mask / (1 - self.feature_dropout_rate)
            if self.dropout is not None:
                out = self.dropout(out)
            processed_features.append(out)

        # 合并拼接逻辑
        if self.concat_seq_feature:
            features_concat = torch.cat(processed_features, dim=-1)
        else:
            features_concat = processed_features

        # Step 3: 输出内容按配置返回
        if self.only_output_feature_list:
            return processed_features
        if self.only_output_3d_tensor:
            return torch.stack(processed_features, dim=1)
        if self.output_2d_tensor_and_feature_list:
            return features_concat, processed_features
        # 默认：输出拼接后的特征
        return features_concat

    def predict(self, batch: Batch) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Perform prediction with training mode disabled."""
        return self.forward(batch, is_training=False)


# embedding_group = EmbeddingGroup(...)
# enhanced = EnhancedEmbeddingGroup(
#     embedding_group,
#     group_name="wide",
#     do_batch_norm=True,
#     dropout_rate=0.2,
#     only_output_feature_list=False,
#     # 其它配置...
# )
# out = enhanced(batch)
