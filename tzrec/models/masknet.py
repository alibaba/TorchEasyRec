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
from tzrec.models.rank_model import RankModel
from tzrec.modules.mlp import MLP
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.config_util import config_to_kwargs


class MaskBlock(nn.Module):
    """MaskBlock module."""

    def __init__(
        self, input_dim, mask_input_dim, reduction_ratio, aggregation_dim, output_dim
    ):
        super(MaskBlock, self).__init__()
        self.output_dim = output_dim
        self.ln_emb = nn.LayerNorm(input_dim)

        if aggregation_dim:
            self.aggregation_dim = aggregation_dim
        if reduction_ratio:
            self.aggregation_dim = int(mask_input_dim * reduction_ratio)
        self.aggregation_layer = nn.Linear(mask_input_dim, aggregation_dim)
        self.projection_layer = nn.Linear(aggregation_dim, input_dim)
        self.hidden_layer = nn.Linear(input_dim, output_dim)

        self.ln_output = nn.LayerNorm(output_dim)
        self.relu_mask = nn.ReLU()
        self.relu_out = nn.ReLU()

    def forward(self, input_1, input_2):
        """Forward pass of MaskBlock."""
        ln_emb = self.ln_emb(input_1)
        weights = self.projection_layer(self.relu_mask(self.aggregation_layer(input_2)))
        masked_emb = ln_emb * weights
        output = self.ln_output(self.hidden_layer(masked_emb))
        output = self.relu_out(output)

        return output


class MaskNet(RankModel):
    """Masknet model.

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
        self.group_name = self.embedding_group.group_names()[0]
        feature_dim = self.embedding_group.group_total_dim(self.group_name)

        masknet_config = model_config.mask_net
        self.use_parallel = masknet_config.use_parallel
        if self.use_parallel:
            self.mask_blocks = nn.ModuleList(
                [
                    MaskBlock(
                        feature_dim,
                        feature_dim,
                        masknet_config.mask_block.reduction_ratio,
                        masknet_config.mask_block.mask_block_aggregation_dim,
                        masknet_config.mask_block.mask_block_output_dim,
                    )
                    for _ in range(masknet_config.n_mask_blocks)
                ]
            )
            self.top_mlp = MLP(
                in_features=masknet_config.mask_block.mask_block_output_dim
                * masknet_config.n_mask_blocks,
                **config_to_kwargs(masknet_config.top_mlp),
            )
        else:
            self.mask_blocks = nn.ModuleList(
                [
                    MaskBlock(
                        feature_dim,
                        feature_dim,
                        masknet_config.mask_block.reduction_ratio,
                        masknet_config.mask_block.mask_block_aggregation_dim,
                        masknet_config.mask_block.mask_block_output_dim,
                    )
                ]
            )
            for _ in range(1, masknet_config.n_mask_blocks):
                self.mask_blocks.append(
                    MaskBlock(
                        masknet_config.mask_block_output_dim,
                        feature_dim,
                        masknet_config.mask_block.reduction_ratio,
                        masknet_config.mask_block.mask_block_aggregation_dim,
                        masknet_config.mask_block.mask_block_output_dim,
                    )
                )
            self.top_mlp = MLP(
                in_features=masknet_config.mask_block.mask_block_output_dim,
                **config_to_kwargs(masknet_config.top_mlp),
            )

        self.output_linear = nn.Linear(
            masknet_config.top_mlp.hidden_units[-1], self._num_class, bias=False
        )

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward."""
        feature_dict = self.build_input(batch)
        features = feature_dict[self.group_name]

        if self.use_parallel:  # parallel mask blocks
            hidden = torch.concat(
                [
                    self.mask_blocks[i](features, features)
                    for i in range(len(self.mask_blocks))
                ],
                dim=-1,
            )
        else:  # serial mask blocks
            hidden = self.mask_blocks[0](features, features)
            for i in range(1, len(self.mask_blocks)):
                hidden = self.mask_blocks[i](hidden, features)

        output = self.output_linear(self.top_mlp(hidden))
        return self._output_to_prediction(output)
