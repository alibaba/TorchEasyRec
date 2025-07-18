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

# We use the Contextual MLPs from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import abc
from typing import Any, Dict, Optional, Union

import torch

from tzrec.modules.norm import LayerNorm, SwishLayerNorm
from tzrec.modules.utils import BaseModule, init_linear_xavier_weights_zero_bias
from tzrec.ops.jagged_tensors import jagged_dense_bmm_broadcast_add
from tzrec.protos import module_pb2
from tzrec.utils.config_util import config_to_kwargs


class ContextualizedMLP(BaseModule):
    """An abstract class for contextual mlp for HSTU preprocessors."""

    @abc.abstractmethod
    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            seq_embeddings (torch.Tensor): input sequence embeddings.
            seq_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            contextual_embeddings (torch.Tensor): input contextual embeddings.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        pass


class SimpleContextualizedMLP(ContextualizedMLP):
    """SimpleContextualizedMLP for HSTU preprocessors.

    Args:
        sequential_input_dim (int): sequence input dimension.
        sequential_output_dim (int): sequence output dimension.
        hidden_dim (int): mlp hidden dimension.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        sequential_input_dim: int,
        sequential_output_dim: int,
        hidden_dim: int,
        is_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._mlp: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=sequential_input_dim,
                out_features=hidden_dim,
            ),
            SwishLayerNorm(hidden_dim, is_inference=is_inference),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_output_dim,
            ),
            LayerNorm(sequential_output_dim),
        ).apply(init_linear_xavier_weights_zero_bias)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            seq_embeddings (torch.Tensor): input sequence embeddings.
            seq_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            contextual_embeddings (torch.Tensor): input contextual embeddings.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        return self._mlp(seq_embeddings)


class ParameterizedContextualizedMLP(ContextualizedMLP):
    """ParameterizedContextualizedMLP for HSTU preprocessors.

    Args:
        contextual_embedding_dim (int): contextual feature input dimension.
        sequential_input_dim (int): sequence input dimension.
        sequential_output_dim (int): sequence output dimension.
        hidden_dim (int): mlp hidden dimension.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        contextual_embedding_dim: int,
        sequential_input_dim: int,
        sequential_output_dim: int,
        hidden_dim: int,
        contextual_dropout_ratio: float = 0.3,
        is_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._sequential_input_dim: int = sequential_input_dim
        self._sequential_output_dim: int = sequential_output_dim
        self._contextual_dropout_ratio: float = contextual_dropout_ratio

        self._dense_features_compress: torch.nn.Module = torch.nn.Linear(
            in_features=contextual_embedding_dim,
            out_features=hidden_dim,
        ).apply(init_linear_xavier_weights_zero_bias)

        self._attn_raw_weights: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_input_dim * sequential_output_dim,
            ),
        ).apply(init_linear_xavier_weights_zero_bias)

        self._attn_weights_norm: torch.nn.Module = torch.nn.LayerNorm(
            [sequential_input_dim, sequential_output_dim]
        )

        self._res_weights: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            SwishLayerNorm(hidden_dim),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=sequential_output_dim,
            ),
        ).apply(init_linear_xavier_weights_zero_bias)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_offsets: torch.Tensor,
        max_seq_len: int,
        contextual_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            seq_embeddings (torch.Tensor): input sequence embeddings.
            seq_offsets (torch.Tensor): input sequence offsets.
            max_seq_len (int): maximum sequence length.
            contextual_embeddings (torch.Tensor): input contextual embeddings.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        contextual_embeddings = torch.nn.functional.dropout(
            contextual_embeddings,
            p=self._contextual_dropout_ratio,
            training=self.training,
        )
        shared_input = self._dense_features_compress(contextual_embeddings)
        attn_weights = self._attn_weights_norm(
            self._attn_raw_weights(shared_input).reshape(
                -1, self._sequential_input_dim, self._sequential_output_dim
            )
        )
        return jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=seq_embeddings,
            dense=attn_weights.to(seq_embeddings.dtype),
            bias=self._res_weights(shared_input),
            kernel=self.kernel(),
        )


def create_contextualized_mlp(
    mlp_cfg: Union[module_pb2.GRContextualizedMLP, Dict[str, Any]], **kwargs
) -> ContextualizedMLP:
    """Create ContextualizedMLP."""
    if isinstance(mlp_cfg, module_pb2.GRContextualizedMLP):
        mlp_type = mlp_cfg.WhichOneof("contextualized_mlp")
        config_dict = config_to_kwargs(getattr(mlp_cfg, mlp_type))
    else:
        assert len(mlp_cfg) == 1, (
            f"mlp_cfg should be {{mlp_type: mlp_kwargs}}, but got {mlp_cfg}"
        )
        mlp_type, config_dict = mlp_cfg.popitem()

    config_dict = dict(config_dict, **kwargs)
    if mlp_type == "simple_mlp":
        return SimpleContextualizedMLP(**config_dict)
    elif mlp_type == "parameterized_mlp":
        return ParameterizedContextualizedMLP(**config_dict)
    else:
        raise RuntimeError(f"Unknown contextualized mlp type: {mlp_type}")
