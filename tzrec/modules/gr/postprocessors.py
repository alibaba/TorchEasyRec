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

# We use the OutputPostprocessors from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from abc import abstractmethod
from typing import Any, Dict, List, Union

import torch

from tzrec.modules.utils import BaseModule, init_linear_xavier_weights_zero_bias
from tzrec.protos import module_pb2
from tzrec.utils.config_util import config_to_kwargs


@torch.fx.wrap
def _cast_dtype(t: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if t.dtype != dtype:
        return t.to(dtype)
    return t


class OutputPostprocessor(BaseModule):
    """An abstract class for post-processing user embeddings after HSTU layers."""

    @abstractmethod
    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            seq_embeddings: (L, D)
            seq_timestamps: (L, )
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            postprocessed seq_embeddings, (L, D)
        """
        pass


class L2NormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with l2 norm.

    Args:
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(self, is_inference: bool = False, **kwargs: Any) -> None:
        super().__init__(is_inference=is_inference)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward the postprocessor.

        Args:
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_payloads (Dict[str, torch.Tensor]):input sequence payload tensors.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        return seq_embeddings / torch.linalg.norm(
            seq_embeddings, ord=2, dim=-1, keepdim=True
        ).clamp(min=1e-6)


class LayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with layer norm.

    Args:
        embedding_dim (int): the dimension of the sequence embedding.
        eps (float): layer norm epsilon.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        eps: float = 1e-5,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward the postprocessor.

        Args:
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_payloads (Dict[str, torch.Tensor]):input sequence payload tensors.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        # pyre-fixme[6]: For 1st argument expected `dtype` but got `Union[dtype,
        #  Tensor, Module]`.
        return self._layer_norm(seq_embeddings.to(self._layer_norm.weight.dtype))


@torch.fx.wrap
def _unsqueeze_if_needed(t: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
    if embedding.dim() == 3:
        return t.unsqueeze(0)
    return t


class TimestampLayerNormPostprocessor(OutputPostprocessor):
    """Postprocesses user embeddings with timestamp-based MLP -> layer norm.

    Args:
        embedding_dim (int): the dimension of the sequence embedding.
        time_duration_period_units (List[int]): time duration period units,
            e.g. 60 * 60 for hour of day.
        time_duration_units_per_period (List[int]): time duration units per period,
            e.g. 24 for hour of day.
        eps (float): layer norm epsilon.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        time_duration_period_units: List[int],
        time_duration_units_per_period: List[int],
        eps: float = 1e-5,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference=is_inference)

        self._layer_norm: torch.nn.Module = torch.nn.LayerNorm(
            normalized_shape=[embedding_dim], eps=eps
        )
        assert len(time_duration_period_units) == len(time_duration_units_per_period)
        self.register_buffer(
            "_period_units",
            torch.Tensor(time_duration_period_units).view(1, -1),
        )
        self.register_buffer(
            "_units_per_period",
            torch.Tensor(time_duration_units_per_period).view(1, -1),
        )
        self._time_feature_combiner: torch.nn.Module = torch.nn.Linear(
            embedding_dim + 2 * len(time_duration_period_units),
            embedding_dim,
        ).apply(init_linear_xavier_weights_zero_bias)

    def _concat_time_features(
        self,
        combined_embeddings: torch.Tensor,
        timestamps: torch.Tensor,  # [B] or [B, D]
    ) -> torch.Tensor:
        # concat time representation to combined embeddings
        period_units = self._period_units
        units_per_period = self._units_per_period

        timestamps = timestamps.unsqueeze(-1)
        period_units = _unsqueeze_if_needed(period_units, combined_embeddings)
        units_per_period = _unsqueeze_if_needed(units_per_period, combined_embeddings)
        _units_since_epoch = torch.div(
            timestamps, period_units, rounding_mode="floor"
        )  # [sum(N_i), num_time_features] or [B, N, num_time_features]
        _units_elapsed = (
            (torch.remainder(_units_since_epoch, units_per_period) / units_per_period)
            * 2
            * 3.14
        )
        # Note: `torch.polar` does not support bfloat16 datatype
        _units_elapsed_type: torch.dtype = _units_elapsed.dtype
        _units_elapsed = torch.view_as_real(
            torch.polar(
                _cast_dtype(torch.ones_like(_units_elapsed), torch.float32),
                _cast_dtype(_units_elapsed, torch.float32),
            )
        ).flatten(
            -2, -1
        )  # [sum(N_i), num_time_features * 2] or [B, N, num_time_features * 2]
        _units_elapsed = _cast_dtype(_units_elapsed, _units_elapsed_type)
        combined_embeddings = torch.cat([combined_embeddings, _units_elapsed], dim=-1)
        return combined_embeddings

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Forward the postprocessor.

        Args:
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_payloads (Dict[str, torch.Tensor]):input sequence payload tensors.

        Returns:
            torch.Tensor: output sequence embedding tensor.
        """
        user_embeddings = self._time_feature_combiner(
            self._concat_time_features(seq_embeddings, timestamps=seq_timestamps)
        )
        return self._layer_norm(user_embeddings)


def create_output_postprocessor(
    postprocessor_cfg: Union[module_pb2.GROutputPostprocessor, Dict[str, Any]],
    **kwargs: Any,
) -> OutputPostprocessor:
    """Create OutputPostprocessor."""
    if isinstance(postprocessor_cfg, module_pb2.GROutputPostprocessor):
        postprocessor_type = postprocessor_cfg.WhichOneof("input_preprocessor")
        config_dict = config_to_kwargs(getattr(postprocessor_cfg, postprocessor_type))
    else:
        assert len(postprocessor_cfg) == 1, (
            f"postprocessor_cfg should be {{postprocessor_type: postprocessor_kwargs}},"
            f" but got {postprocessor_cfg}"
        )
        postprocessor_type, config_dict = postprocessor_cfg.popitem()

    config_dict = dict(config_dict, **kwargs)
    if postprocessor_type == "l2norm_postprocessor":
        return L2NormPostprocessor(**config_dict)
    elif postprocessor_type == "layernorm_postprocessor":
        return LayerNormPostprocessor(**config_dict)
    elif postprocessor_type == "timestamp_layernorm_postprocessor":
        return TimestampLayerNormPostprocessor(**config_dict)
    else:
        raise RuntimeError(f"Unknown postprocessor type: {postprocessor_type}")
