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

from typing import Any, Dict, Optional

import torch
from torch import nn

from tzrec.modules.mlp import MLP


class MaskBlock(nn.Module):
    """MaskBlock module.

    Args:
        input_dim (int): Input dimension, either feature embedding dim(parallel mode)
            or hidden state dim(serial mode).
        mask_input_dim (int): Mask input dimension, is always the feature embedding dim
            for both para and serial modes.
        hidden_dim (int): Hidden layer dimension for feedforward network.
        reduction_ratio (float): Reduction ratio, aggregation_dim / mask_input_dim.
        aggregation_dim (int): Aggregation layer dim, mask_input_dim*reduction_ratio.
    """

    def __init__(
        self,
        input_dim: int,
        mask_input_dim: int,
        hidden_dim: int,
        reduction_ratio: float = 1.0,
        aggregation_dim: int = 0,
    ) -> None:
        super(MaskBlock, self).__init__()

        if not aggregation_dim and not reduction_ratio:
            raise ValueError(
                "Either aggregation_dim or reduction_ratio must be provided."
            )

        if aggregation_dim:
            self.aggregation_dim = aggregation_dim
        if reduction_ratio:
            self.aggregation_dim = int(input_dim * reduction_ratio)

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
        self._hidden_dim = hidden_dim

        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._hidden_dim

    def forward(
        self, feature_input: torch.Tensor, mask_input: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass of MaskBlock."""
        weights = self.mask_generator(mask_input)
        weighted_emb = feature_input * weights
        output = self.ffn(weighted_emb)

        return output


class MaskNetModule(nn.Module):
    """Masknet module.

    Args:
        feature_dim (int): input feature dim.
        n_mask_blocks (int): number of mask blocks
        mask_block (dict): MaskBlock module parameters.
        top_mlp (dict): top MLP module parameters.
        use_parallel (bool): use parallel or serial mask blocks
    """

    def __init__(
        self,
        feature_dim: int,
        n_mask_blocks: int,
        mask_block: Dict[str, Any],
        top_mlp: Optional[Dict[str, Any]],
        use_parallel: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.ln_emb = nn.LayerNorm(feature_dim)

        self.use_parallel = use_parallel

        if self.use_parallel:
            self.mask_blocks = nn.ModuleList(
                [
                    MaskBlock(feature_dim, feature_dim, **mask_block)
                    for _ in range(n_mask_blocks)
                ]
            )
            self._output_dim = self.mask_blocks[0].output_dim() * n_mask_blocks
        else:
            self.mask_blocks = nn.ModuleList()
            self._output_dim = feature_dim
            for _ in range(n_mask_blocks):
                self.mask_blocks.append(
                    MaskBlock(self._output_dim, feature_dim, **mask_block)
                )
                self._output_dim = self.mask_blocks[0].output_dim()

        self.top_mlp = None
        if top_mlp:
            self.top_mlp = MLP(
                in_features=self._output_dim,
                **top_mlp,
            )
            self._output_dim = self.top_mlp.output_dim()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self._output_dim

    def forward(self, feature_emb: torch.Tensor) -> torch.Tensor:
        """Forward method."""
        ln_emb = self.ln_emb(feature_emb)
        if self.use_parallel:  # parallel mask blocks
            hidden = torch.concat(
                [
                    self.mask_blocks[i](ln_emb, feature_emb)
                    for i in range(len(self.mask_blocks))
                ],
                dim=-1,
            )
        else:  # serial mask blocks
            hidden = self.mask_blocks[0](ln_emb, feature_emb)
            for i in range(1, len(self.mask_blocks)):
                hidden = self.mask_blocks[i](hidden, feature_emb)

        if self.top_mlp is not None:
            hidden = self.top_mlp(hidden)

        return hidden
