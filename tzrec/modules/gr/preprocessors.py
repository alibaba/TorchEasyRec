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

# We use the InputPreprocessors from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import abc
from math import sqrt
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from tzrec.modules.gr.action_encoder import ActionEncoder
from tzrec.modules.gr.content_encoder import ContentEncoder
from tzrec.modules.gr.contextualize_mlps import (
    ContextualizedMLP,
    create_contextualized_mlp,
)
from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import concat_2D_jagged
from tzrec.protos import module_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.fx_util import fx_infer_max_len, fx_unwrap_optional_tensor

torch.fx.wrap(fx_infer_max_len)


class InputPreprocessor(BaseModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """Forward the module.

        Args:
            max_seq_len: int
            seq_lengths: (B,)
            seq_embeddings: (L, D)
            seq_timestamps: (B, N)
            num_targets: (B,) Optional.
            seq_payloads: str-keyed tensors. Implementation specific.

        Returns:
            (max_seq_len, lengths, offsets, timestamps, embeddings,
            num_targets, payloads) updated based on input preprocessor.
        """
        pass

    def interleave_targets(self) -> bool:
        """Interleave targets or not."""
        return False

    def contextual_seq_len(self) -> int:
        """Contextual feature sequence length."""
        return 0


def _get_contextual_input_embeddings(
    seq_lengths: torch.Tensor,
    seq_payloads: Dict[str, torch.Tensor],
    contextual_feature_to_max_length: Dict[str, int],
    contextual_feature_to_min_uih_length: Dict[str, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    padded_values: List[torch.Tensor] = []
    for key, max_len in contextual_feature_to_max_length.items():
        v = torch.flatten(
            torch.ops.fbgemm.jagged_to_padded_dense(
                values=seq_payloads[key].to(dtype),
                offsets=[seq_payloads[key + "_offsets"]],
                max_lengths=[max_len],
                padding_value=0.0,
            ),
            1,
            2,
        )
        min_uih_length = contextual_feature_to_min_uih_length.get(key, 0)
        if min_uih_length > 0:
            v = v * (seq_lengths.view(-1, 1) >= min_uih_length)
        padded_values.append(v)
    return torch.cat(padded_values, dim=1)


class ContextualPreprocessor(InputPreprocessor):
    """Contextual Preprocessor for HSTU.

    Args:
        input_embedding_dim (int): The dimension of the sequence embeddings.
        output_embedding_dim (int): The dimension of the sequence embeddings.
        contextual_feature_to_max_length (Dict[str, int]): A mapping from contextual
            feature to maximum padding length.
        contextual_feature_to_min_uih_length (Dict[str, int]): A mapping from contextual
            feature to uih length, if uih length < min_uih_length, will mask the
            contextual feature.
        content_mlp (Dict[str, Any]): content MLP module params.
        action_encoder (Dict[str, Any]): ActionEncoder module params.
        action_mlp (Dict[str, Any]): action MLP module params.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        content_mlp: Dict[str, Any],
        action_encoder: Optional[Dict[str, Any]] = None,
        action_mlp: Optional[Dict[str, Any]] = None,
        is_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._output_embedding_dim: int = output_embedding_dim
        self._input_embedding_dim: int = input_embedding_dim
        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        if self._max_contextual_seq_len > 0:
            std = 1.0 * sqrt(
                2.0 / float(input_embedding_dim + self._output_embedding_dim)
            )
            self._batched_contextual_linear_weights: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (
                            self._max_contextual_seq_len,
                            input_embedding_dim,
                            self._output_embedding_dim,
                        )
                    ).normal_(0.0, std)
                )
            )
            self._batched_contextual_linear_bias: torch.nn.Parameter = (
                torch.nn.Parameter(
                    torch.empty(
                        (self._max_contextual_seq_len, self._output_embedding_dim)
                    ).fill_(0.0)
                )
            )
        self._content_embedding_mlp: torch.nn.Module = create_contextualized_mlp(
            content_mlp,
            sequential_input_dim=self._input_embedding_dim,
            sequential_output_dim=self._output_embedding_dim,
            is_inference=is_inference,
        )

        self._action_encoder_cfg = action_encoder
        if self._action_encoder_cfg is not None:
            self._action_encoder: ActionEncoder = ActionEncoder(
                **self._action_encoder_cfg,
                is_inference=is_inference,
            )
            self._action_embedding_mlp: torch.nn.Module = create_contextualized_mlp(
                action_mlp,
                sequential_input_dim=self._action_encoder.output_dim,
                sequential_output_dim=self._output_embedding_dim,
                is_inference=is_inference,
            )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward the module.

        Args:
            max_seq_len (int): maximum sequence length.
            seq_lengths (torch.Tensor): input sequence lengths.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            seq_payloads (Dict[str, torch.Tensor]): sequence payload features.

        Returns:
            output_max_seq_len (int): output maximum sequence length.
            output_seq_lengths (torch.Tensor): output sequence lengths.
            output_seq_offsets (torch.Tensor): output sequence lengths.
            output_seq_timestamps (torch.Tensor): output sequence timestamp tensor.
            output_seq_embeddings (torch.Tensor): output sequence embedding tensor.
            output_num_targets (torch.Tensor): output number of targets.
        """
        # get contextual embeddings
        contextual_input_embeddings: Optional[torch.Tensor] = None
        contextual_embeddings: Optional[torch.Tensor] = None
        if self._max_contextual_seq_len > 0:
            contextual_input_embeddings = _get_contextual_input_embeddings(
                seq_lengths=seq_lengths,
                seq_payloads=seq_payloads,
                contextual_feature_to_max_length=self._contextual_feature_to_max_length,
                contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
                dtype=seq_embeddings.dtype,
            )
            contextual_embeddings = torch.baddbmm(
                self._batched_contextual_linear_bias.to(
                    contextual_input_embeddings.dtype
                ).unsqueeze(1),
                contextual_input_embeddings.view(
                    -1, self._max_contextual_seq_len, self._input_embedding_dim
                ).transpose(0, 1),
                self._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            ).transpose(0, 1)

        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        output_seq_embeddings = self._content_embedding_mlp(
            seq_embeddings,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            contextual_embeddings=contextual_input_embeddings,
        )
        if self._action_encoder_cfg is not None:
            action_embeddings = self._action_encoder(
                max_seq_len=max_seq_len,
                seq_lengths=seq_lengths,
                seq_offsets=seq_offsets,
                seq_payloads=seq_payloads,
                num_targets=num_targets,
            )
            output_seq_embeddings = output_seq_embeddings + self._action_embedding_mlp(
                action_embeddings,
                seq_offsets=seq_offsets,
                max_seq_len=max_seq_len,
                contextual_embeddings=contextual_input_embeddings,
            )

        output_max_seq_len = max_seq_len
        output_seq_lengths = seq_lengths
        output_num_targets = num_targets
        output_seq_timestamps = seq_timestamps
        output_seq_offsets = seq_offsets
        # concat contextual embeddings
        if self._max_contextual_seq_len > 0:
            output_seq_embeddings = concat_2D_jagged(
                values_left=fx_unwrap_optional_tensor(contextual_embeddings).reshape(
                    -1, self._output_embedding_dim
                ),
                values_right=output_seq_embeddings,
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.kernel(),
            )
            output_seq_timestamps = concat_2D_jagged(
                values_left=torch.zeros(
                    (output_seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                    dtype=output_seq_timestamps.dtype,
                    device=output_seq_timestamps.device,
                ),
                values_right=output_seq_timestamps.unsqueeze(-1),
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.kernel(),
            ).squeeze(-1)
            output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
            output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                output_seq_lengths
            )

        return (
            output_max_seq_len,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def contextual_seq_len(self) -> int:
        """Contextual feature sequence length."""
        return self._max_contextual_seq_len


class ContextualInterleavePreprocessor(InputPreprocessor):
    """Contextual Interleave Preprocessor for HSTU.

    Args:
        input_embedding_dim (int): The dimension of the sequence embeddings.
        output_embedding_dim (int): The dimension of the sequence embeddings.
        contextual_feature_to_max_length (Dict[str, int]): A mapping from contextual
            feature to maximum padding length.
        contextual_feature_to_min_uih_length (Dict[str, int]): A mapping from contextual
            feature to uih length, if uih length < min_uih_length, will mask the
            contextual feature.
        content_encoder (Dict[str, Any]): ContentEncoder module params.
        content_mlp (Dict[str, Any]): content MLP module params.
        action_encoder (Dict[str, Any]): ActionEncoder module params.
        action_mlp (Dict[str, Any]): action MLP module params.
        enable_interleaving (bool): enable interleaving target or not.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        input_embedding_dim: int,
        output_embedding_dim: int,
        contextual_feature_to_max_length: Dict[str, int],
        contextual_feature_to_min_uih_length: Dict[str, int],
        content_encoder: Dict[str, Any],
        content_mlp: Dict[str, Any],
        action_encoder: Dict[str, Any],
        action_mlp: Dict[str, Any],
        enable_interleaving: bool = True,
        is_inference: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._input_embedding_dim: int = input_embedding_dim
        self._output_embedding_dim: int = output_embedding_dim
        self._contextual_feature_to_max_length: Dict[str, int] = (
            contextual_feature_to_max_length
        )
        self._max_contextual_seq_len: int = sum(
            contextual_feature_to_max_length.values()
        )
        self._contextual_feature_to_min_uih_length: Dict[str, int] = (
            contextual_feature_to_min_uih_length
        )
        std = 1.0 * sqrt(2.0 / float(input_embedding_dim + output_embedding_dim))
        self._batched_contextual_linear_weights = torch.nn.Parameter(
            torch.empty(
                (
                    self._max_contextual_seq_len,
                    input_embedding_dim,
                    output_embedding_dim,
                )
            ).normal_(0.0, std)
        )
        self._batched_contextual_linear_bias = torch.nn.Parameter(
            torch.empty((self._max_contextual_seq_len, 1, output_embedding_dim)).fill_(
                0.0
            )
        )
        contextual_embedding_dim: int = (
            self._max_contextual_seq_len * input_embedding_dim
        )
        self._content_encoder: ContentEncoder = ContentEncoder(
            input_embedding_dim=input_embedding_dim,
            **content_encoder,
            is_inference=is_inference,
        )
        self._content_embedding_mlp: ContextualizedMLP = create_contextualized_mlp(
            content_mlp,
            contextual_embedding_dim=contextual_embedding_dim,
            sequential_input_dim=self._content_encoder.output_dim,
            sequential_output_dim=output_embedding_dim,
            is_inference=is_inference,
        )
        self._action_encoder: ActionEncoder = ActionEncoder(
            **action_encoder,
            is_inference=is_inference,
        )
        self._action_embedding_mlp: ContextualizedMLP = create_contextualized_mlp(
            action_mlp,
            contextual_embedding_dim=contextual_embedding_dim,
            sequential_input_dim=self._action_encoder.output_dim,
            sequential_output_dim=output_embedding_dim,
            is_inference=is_inference,
        )
        self._enable_interleaving: bool = enable_interleaving

    def _combine_embeddings(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        content_embeddings: torch.Tensor,
        action_embeddings: torch.Tensor,
        contextual_embeddings: Optional[torch.Tensor],
        num_targets: torch.Tensor,
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if self._enable_interleaving:
            output_seq_timestamps = seq_timestamps.repeat_interleave(2)
            output_seq_embeddings = torch.stack(
                [content_embeddings, action_embeddings], dim=1
            ).reshape(-1, self._output_embedding_dim)
            if self.interleave_targets():
                output_seq_lengths = seq_lengths * 2
                output_max_seq_len = max_seq_len * 2
                output_num_targets = num_targets * 2
            else:
                seq_lengths_by_2 = seq_lengths * 2
                output_seq_lengths = seq_lengths_by_2 - num_targets
                output_max_seq_len = fx_infer_max_len(output_seq_lengths)
                indices = torch.arange(2 * max_seq_len, device=seq_lengths.device).view(
                    1, -1
                )
                valid_mask = torch.logical_and(
                    indices < seq_lengths_by_2.view(-1, 1),
                    torch.logical_or(
                        indices < (output_seq_lengths - num_targets).view(-1, 1),
                        torch.remainder(indices, 2) == 0,
                    ),
                )
                jagged_valid_mask = (
                    torch.ops.fbgemm.dense_to_jagged(
                        valid_mask.int().unsqueeze(-1),
                        [
                            torch.ops.fbgemm.asynchronous_complete_cumsum(
                                seq_lengths_by_2
                            )
                        ],
                    )[0]
                    .to(torch.bool)
                    .squeeze(1)
                )
                output_seq_embeddings = output_seq_embeddings[jagged_valid_mask]
                output_seq_timestamps = output_seq_timestamps[jagged_valid_mask]
                output_num_targets = num_targets
        else:
            output_max_seq_len = max_seq_len
            output_seq_lengths = seq_lengths
            output_num_targets = num_targets
            output_seq_timestamps = seq_timestamps
            output_seq_embeddings = content_embeddings + action_embeddings

        # concat contextual embeddings
        output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            output_seq_lengths
        )
        if self._max_contextual_seq_len > 0:
            output_seq_embeddings = concat_2D_jagged(
                values_left=fx_unwrap_optional_tensor(contextual_embeddings).reshape(
                    -1, self._output_embedding_dim
                ),
                values_right=output_seq_embeddings,
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.kernel(),
            )
            output_seq_timestamps = concat_2D_jagged(
                values_left=torch.zeros(
                    (output_seq_lengths.size(0) * self._max_contextual_seq_len, 1),
                    dtype=output_seq_timestamps.dtype,
                    device=output_seq_timestamps.device,
                ),
                values_right=output_seq_timestamps.unsqueeze(-1),
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.kernel(),
            ).squeeze(-1)
            output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
            output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                output_seq_lengths
            )

        return (
            output_max_seq_len,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def forward(  # noqa C901
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
    ]:
        """Forward the module.

        Args:
            max_seq_len (int): maximum sequence length.
            seq_lengths (torch.Tensor): input sequence lengths.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            seq_payloads (Dict[str, torch.Tensor]): sequence payload features.

        Returns:
            output_max_seq_len (int): output maximum sequence length.
            output_seq_lengths (torch.Tensor): output sequence lengths.
            output_seq_offsets (torch.Tensor): output sequence lengths.
            output_seq_timestamps (torch.Tensor): output sequence timestamp tensor.
            output_seq_embeddings (torch.Tensor): output sequence embedding tensor.
            output_num_targets (torch.Tensor): output number of targets.
        """
        # get contextual_embeddings
        contextual_input_embeddings: Optional[torch.Tensor] = None
        contextual_embeddings: Optional[torch.Tensor] = None
        if self._max_contextual_seq_len > 0:
            contextual_input_embeddings = _get_contextual_input_embeddings(
                seq_lengths=seq_lengths,
                seq_payloads=seq_payloads,
                contextual_feature_to_max_length=self._contextual_feature_to_max_length,
                contextual_feature_to_min_uih_length=self._contextual_feature_to_min_uih_length,
                dtype=seq_embeddings.dtype,
            )
            contextual_embeddings = torch.baddbmm(
                self._batched_contextual_linear_bias.to(
                    contextual_input_embeddings.dtype
                ),
                contextual_input_embeddings.view(
                    -1, self._max_contextual_seq_len, self._input_embedding_dim
                ).transpose(0, 1),
                self._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            ).transpose(0, 1)

        # content embeddings
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(seq_lengths)
        content_embeddings = self._content_encoder(
            max_seq_len=max_seq_len,
            seq_embeddings=seq_embeddings,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_payloads=seq_payloads,
            num_targets=num_targets,
        )
        content_embeddings = self._content_embedding_mlp(
            seq_embeddings=content_embeddings,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            contextual_embeddings=contextual_input_embeddings,
        )

        # action embeddings
        action_embeddings = self._action_encoder(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_offsets=seq_offsets,
            seq_payloads=seq_payloads,
            num_targets=num_targets,
        ).to(seq_embeddings.dtype)
        action_embeddings = self._action_embedding_mlp(
            seq_embeddings=action_embeddings,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            contextual_embeddings=contextual_input_embeddings,
        )

        (
            output_max_seq_len,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        ) = self._combine_embeddings(
            max_seq_len=max_seq_len,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            content_embeddings=content_embeddings,
            action_embeddings=action_embeddings,
            contextual_embeddings=contextual_embeddings,
            num_targets=num_targets,
        )

        return (
            output_max_seq_len,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
            seq_payloads,
        )

    def interleave_targets(self) -> bool:
        """Interleave targets or not."""
        return self.is_train and self._enable_interleaving

    def contextual_seq_len(self) -> int:
        """Contextual feature sequence length."""
        return self._max_contextual_seq_len


def create_input_preprocessor(
    preprocessor_cfg: Union[module_pb2.GRInputPreprocessor, Dict[str, Any]], **kwargs
) -> InputPreprocessor:
    """Create InputPreprocessor."""
    if isinstance(preprocessor_cfg, module_pb2.GRInputPreprocessor):
        preprocessor_type = preprocessor_cfg.WhichOneof("input_preprocessor")
        config_dict = config_to_kwargs(getattr(preprocessor_cfg, preprocessor_type))
    else:
        assert len(preprocessor_cfg) == 1, (
            f"preprocessor_cfg should be {{preprocessor_type: preprocessor_kwargs}}, "
            f"but got {preprocessor_cfg}"
        )
        preprocessor_type, config_dict = preprocessor_cfg.popitem()

    config_dict = dict(config_dict, **kwargs)
    if preprocessor_type == "contextual_preprocessor":
        return ContextualPreprocessor(**config_dict)
    elif preprocessor_type == "contextual_interleave_preprocessor":
        return ContextualInterleavePreprocessor(**config_dict)
    else:
        raise RuntimeError(f"Unknown preprocessor type: {preprocessor_type}")
