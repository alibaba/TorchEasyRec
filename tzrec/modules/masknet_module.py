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

from typing import Any

import torch
from torch import nn

from tzrec.modules.mlp import MLP
from tzrec.protos.module_pb2 import MaskNetModule as MaskNetModuleConfig
from tzrec.utils.config_util import config_to_kwargs


class MaskBlock(nn.Module):
    """MaskBlock module.

    Args:
        input_dim (int): Input dimension, either feature embedding dim(parallel mode)
            or hidden state dim(serial mode).
        mask_input_dim (int): Mask input dimension, is always the feature embedding dim
            for both para and serial modes.
        reduction_ratio (float): Reduction ratio, aggregation_dim / mask_input_dim.
        aggregation_dim (int): Aggregation layer dim, mask_input_dim*reduction_ratio.
        hidden_dim (int): Hidden layer dimension for feedforward network.
    """

    def __init__(
        self,
        input_dim: int,
        mask_input_dim: int,
        reduction_ratio: float,
        aggregation_dim: int,
        hidden_dim: int,
    ) -> None:
        super(MaskBlock, self).__init__()
        self.ln_emb = nn.LayerNorm(input_dim)

        if not aggregation_dim and not reduction_ratio:
            raise ValueError(
                "Either aggregation_dim or reduction_ratio must be provided."
            )

        if aggregation_dim:
            self.aggregation_dim = aggregation_dim
        if reduction_ratio:
            self.aggregation_dim = int(mask_input_dim * reduction_ratio)

        assert self.aggregation_dim > 0, (
            "aggregation_dim must be > 0, check your aggregation_dim or "
        )
        "redudction_ratio settings."

        self.mask_generator = nn.Sequential(
            nn.Linear(mask_input_dim, self.aggregation_dim),
            nn.ReLU(),
            nn.Linear(self.aggregation_dim, input_dim),
        )

        assert hidden_dim > 0, "hidden_dim must be > 0."

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, input: torch.Tensor, mask_input: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaskBlock."""
        ln_emb = self.ln_emb(input)
        weights = self.mask_generator(mask_input)
        weighted_emb = ln_emb * weights
        output = self.ffn(weighted_emb)

        return output


class MaskNetModule(nn.Module):
    """Masknet module.

    Args:
        model_config (ModelConfig): an instance of ModelConfig.
        feature_dim (int): input feature dim.
    """

    def __init__(
        self,
        module_config: MaskNetModuleConfig,
        feature_dim: int,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.use_parallel = module_config.use_parallel

        self.mask_blocks = nn.ModuleList(
            [
                MaskBlock(
                    feature_dim,
                    feature_dim,
                    module_config.mask_block.reduction_ratio,
                    module_config.mask_block.aggregation_dim,
                    module_config.mask_block.hidden_dim,
                )
                for _ in range(module_config.n_mask_blocks)
            ]
        )
        if self.use_parallel:
            self.top_mlp = MLP(
                in_features=feature_dim * module_config.n_mask_blocks,
                **config_to_kwargs(module_config.top_mlp),
            )
        else:
            self.top_mlp = MLP(
                in_features=feature_dim,
                **config_to_kwargs(module_config.top_mlp),
            )

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        if self.use_parallel:  # parallel mask blocks
            hidden = torch.concat(
                [
                    self.mask_blocks[i](feature_emb, feature_emb)
                    for i in range(len(self.mask_blocks))
                ],
                dim=-1,
            )
        else:  # serial mask blocks
            hidden = self.mask_blocks[0](feature_emb, feature_emb)
            for i in range(1, len(self.mask_blocks)):
                hidden = self.mask_blocks[i](hidden, feature_emb)

        return self.top_mlp(hidden)
