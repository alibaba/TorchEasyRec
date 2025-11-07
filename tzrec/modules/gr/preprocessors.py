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
from typing import Any, Dict, Optional, Tuple, Union

import torch

from tzrec.modules.gr.action_encoder import ActionEncoder, create_action_encoder
from tzrec.modules.gr.content_encoder import ContentEncoder, create_content_encoder
from tzrec.modules.gr.contextualize_mlps import (
    ContextualizedMLP,
    create_contextualized_mlp,
)
from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import concat_2D_jagged
from tzrec.protos import module_pb2
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.fx_util import fx_int_item, fx_numel, fx_unwrap_optional_tensor

torch.fx.wrap(fx_unwrap_optional_tensor)
torch.fx.wrap(fx_int_item)
torch.fx.wrap(fx_numel)


@torch.fx.wrap
def _fx_timestamp_contextual_zeros(
    seq_timestamp: torch.Tensor, seq_lengths: torch.Tensor, max_contextual_seq_len: int
) -> torch.Tensor:
    return torch.zeros(
        (seq_lengths.size(0) * max_contextual_seq_len, 1),
        dtype=seq_timestamp.dtype,
        device=seq_timestamp.device,
    )


class InputPreprocessor(BaseModule):
    """An abstract class for pre-processing sequence embeddings before HSTU layers."""

    @abc.abstractmethod
    def forward(
        self,
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: torch.Tensor,
        seq_payloads: Dict[str, torch.Tensor],
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward the module.

        Args:
            max_uih_len (int): maximum user history sequence length.
            max_targets (int): maximum candidates length.
            total_uih_len (int): total user history sequence length.
            total_targets (int): total candidates length.
            seq_lengths (torch.Tensor): input sequence lengths.
            seq_timestamps (torch.Tensor): input sequence timestamp tensor.
            seq_embeddings (torch.Tensor): input sequence embedding tensor.
            num_targets (torch.Tensor): number of targets.
            seq_payloads (Dict[str, torch.Tensor]): sequence payload features.

        Returns:
            output_max_seq_len (int): output maximum sequence length.
            output_total_uih_len (int): output total user history sequence length.
            output_total_targets (int): output total candidates length.
            output_seq_lengths (torch.Tensor): output sequence lengths.
            output_seq_offsets (torch.Tensor): output sequence lengths.
            output_seq_timestamps (torch.Tensor): output sequence timestamp tensor.
            output_seq_embeddings (torch.Tensor): output sequence embedding tensor.
            output_num_targets (torch.Tensor): output number of targets.
        """
        pass

    def interleave_targets(self) -> bool:
        """Interleave targets or not."""
        return False

    def contextual_seq_len(self) -> int:
        """Contextual feature sequence length."""
        return 0


class ContextualInterleavePreprocessor(InputPreprocessor):
    """Contextual Interleave Preprocessor for HSTU.

    Args:
        uih_embedding_dim (int): The dimension of the uih sequence embeddings.
        target_embedding_dim (int): The dimension of the candidate sequence embeddings.
        output_embedding_dim (int): The dimension of the sequence embeddings.
        content_encoder (Dict[str, Any]): ContentEncoder module params.
        content_mlp (Dict[str, Any]): content MLP module params.
        action_encoder (Dict[str, Any]): ActionEncoder module params.
        action_mlp (Dict[str, Any]): action MLP module params.
        contextual_feature_dim (int): contextual feature dimension.
        max_contextual_seq_len (int): contextual feature num.
        enable_interleaving (bool): enable interleaving target or not.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        uih_embedding_dim: int,
        target_embedding_dim: int,
        output_embedding_dim: int,
        content_encoder: Dict[str, Any],
        content_mlp: Dict[str, Any],
        action_encoder: Optional[Dict[str, Any]] = None,
        action_mlp: Optional[Dict[str, Any]] = None,
        contextual_feature_dim: int = 0,
        max_contextual_seq_len: int = 0,
        enable_interleaving: bool = True,
        is_inference: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._uih_embedding_dim: int = uih_embedding_dim
        self._target_embedding_dim: int = target_embedding_dim
        self._output_embedding_dim: int = output_embedding_dim

        self._contextual_feature_dim: int = contextual_feature_dim
        self._max_contextual_seq_len: int = max_contextual_seq_len
        if max_contextual_seq_len > 0:
            std = 1.0 * sqrt(
                2.0 / float(self._contextual_feature_dim + output_embedding_dim)
            )
            self._batched_contextual_linear_weights = torch.nn.Parameter(
                torch.empty(
                    (
                        self._max_contextual_seq_len,
                        self._contextual_feature_dim,
                        output_embedding_dim,
                    )
                ).normal_(0.0, std)
            )
            self._batched_contextual_linear_bias = torch.nn.Parameter(
                torch.empty(
                    (self._max_contextual_seq_len, 1, output_embedding_dim)
                ).fill_(0.0)
            )

        contextual_embedding_dim: int = (
            self._max_contextual_seq_len * self._contextual_feature_dim
        )
        self._content_encoder: ContentEncoder = create_content_encoder(
            content_encoder,
            uih_embedding_dim=self._uih_embedding_dim,
            target_embedding_dim=self._target_embedding_dim,
            is_inference=is_inference,
        )
        self._content_embedding_mlp: ContextualizedMLP = create_contextualized_mlp(
            content_mlp,
            contextual_embedding_dim=contextual_embedding_dim,
            sequential_input_dim=self._content_encoder.output_dim,
            sequential_output_dim=output_embedding_dim,
            is_inference=is_inference,
        )
        if enable_interleaving:
            assert action_encoder is not None, (
                "when enable interleaving, must set action_encoder."
            )

        self._action_encoder_cfg = action_encoder
        if self._action_encoder_cfg is not None:
            self._action_encoder: ActionEncoder = create_action_encoder(
                action_encoder,
                is_inference=is_inference,
            )
            assert action_mlp is not None, (
                "when enable interleaving, must set action_mlp."
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
        max_uih_len: int,
        max_targets: int,
        total_uih_len: int,
        total_targets: int,
        seq_lengths: torch.Tensor,
        seq_timestamps: torch.Tensor,
        content_embeddings: torch.Tensor,
        action_embeddings: Optional[torch.Tensor],
        contextual_embeddings: Optional[torch.Tensor],
        num_targets: torch.Tensor,
    ) -> Tuple[
        int,
        int,
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
                output_max_seq_len = (max_uih_len + max_targets) * 2
                output_num_targets = num_targets * 2
                output_total_uih_len = total_uih_len * 2
                output_total_targets = total_targets * 2
            else:
                seq_lengths_by_2 = seq_lengths * 2
                output_seq_lengths = seq_lengths_by_2 - num_targets
                output_max_seq_len = 2 * max_uih_len + max_targets
                indices = torch.arange(
                    2 * (max_uih_len + max_targets), device=seq_lengths.device
                ).view(1, -1)
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
                output_total_uih_len = total_uih_len * 2
                output_total_targets = total_targets
        else:
            output_max_seq_len = max_uih_len + max_targets
            output_seq_lengths = seq_lengths
            output_num_targets = num_targets
            output_seq_timestamps = seq_timestamps
            if self._action_encoder_cfg is not None:
                output_seq_embeddings = content_embeddings + fx_unwrap_optional_tensor(
                    action_embeddings
                )
            else:
                output_seq_embeddings = content_embeddings
            output_total_uih_len = total_uih_len
            output_total_targets = total_targets

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
                values_left=_fx_timestamp_contextual_zeros(
                    output_seq_timestamps,
                    output_seq_lengths,
                    self._max_contextual_seq_len,
                ),
                values_right=output_seq_timestamps.unsqueeze(-1),
                max_len_left=self._max_contextual_seq_len,
                max_len_right=output_max_seq_len,
                offsets_left=None,
                offsets_right=output_seq_offsets,
                kernel=self.kernel(),
            ).squeeze(-1)
            output_max_seq_len = output_max_seq_len + self._max_contextual_seq_len
            output_total_uih_len = (
                output_total_uih_len
                + self._max_contextual_seq_len * output_seq_lengths.size(0)
            )
            output_seq_lengths = output_seq_lengths + self._max_contextual_seq_len
            output_seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                output_seq_lengths
            )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def forward(
        self, grouped_features: Dict[str, torch.Tensor]
    ) -> Tuple[
        int,
        int,
        int,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward the module.

        Args:
            grouped_features (Dict[str, torch.Tensor]): embedding group features.

        Returns:
            output_max_seq_len (int): output maximum sequence length.
            output_total_uih_len (int): output total user history sequence length.
            output_total_targets (int): output total candidates length.
            output_seq_lengths (torch.Tensor): output sequence lengths.
            output_seq_offsets (torch.Tensor): output sequence lengths.
            output_seq_timestamps (torch.Tensor): output sequence timestamp tensor.
            output_seq_embeddings (torch.Tensor): output sequence embedding tensor.
            output_num_targets (torch.Tensor): output number of targets.
        """
        uih_embeddings = grouped_features["uih.sequence"]
        uih_seq_lengths = grouped_features["uih.sequence_length"]
        max_uih_len = fx_int_item(uih_seq_lengths.max())
        total_uih_len = fx_int_item(uih_seq_lengths.sum())

        target_embeddings = grouped_features["candidate.sequence"]
        num_targets = grouped_features["candidate.sequence_length"]
        max_targets = fx_int_item(num_targets.max())
        total_targets = fx_int_item(num_targets.sum())

        uih_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(uih_seq_lengths)
        target_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(num_targets)

        # get contextual_embeddings
        contextual_input_embeddings: Optional[torch.Tensor] = None
        contextual_embeddings: Optional[torch.Tensor] = None
        if self._max_contextual_seq_len > 0:
            contextual_input_embeddings = grouped_features["contextual"]
            contextual_embeddings = torch.baddbmm(
                self._batched_contextual_linear_bias.to(
                    contextual_input_embeddings.dtype
                ),
                contextual_input_embeddings.view(
                    -1, self._max_contextual_seq_len, self._contextual_feature_dim
                ).transpose(0, 1),
                self._batched_contextual_linear_weights.to(
                    contextual_input_embeddings.dtype
                ),
            ).transpose(0, 1)

        # combine uih and target embedding
        content_embeddings = self._content_encoder(
            uih_embeddings=uih_embeddings,
            target_embeddings=target_embeddings,
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            uih_offsets=uih_offsets,
            target_offsets=target_offsets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
        )
        seq_offsets = uih_offsets + target_offsets
        max_seq_len = max_uih_len + max_targets
        seq_lengths = uih_seq_lengths + num_targets

        content_embeddings = self._content_embedding_mlp(
            seq_embeddings=content_embeddings,
            seq_offsets=seq_offsets,
            max_seq_len=max_seq_len,
            contextual_embeddings=contextual_input_embeddings,
        )

        # action embeddings
        action_embeddings = None
        if self._action_encoder_cfg is not None:
            action_embeddings = self._action_encoder(
                seq_actions=grouped_features["uih_action.sequence"].to(torch.int64),
                max_uih_len=max_uih_len,
                max_targets=max_targets,
                uih_offsets=uih_offsets,
                target_offsets=target_offsets,
                total_uih_len=total_uih_len,
                total_targets=total_targets,
                seq_watchtimes=grouped_features["uih_watchtime.sequence"]
                if self._action_encoder.need_watchtime
                else None,
            ).to(uih_embeddings.dtype)
            action_embeddings = self._action_embedding_mlp(
                seq_embeddings=action_embeddings,
                seq_offsets=seq_offsets,
                max_seq_len=max_seq_len,
                contextual_embeddings=contextual_input_embeddings,
            )

        seq_timestamps = concat_2D_jagged(
            values_left=grouped_features["uih_timestamp.sequence"],
            values_right=grouped_features["candidate_timestamp.sequence"],
            max_len_left=max_uih_len,
            max_len_right=max_targets,
            offsets_left=uih_offsets,
            offsets_right=target_offsets,
            kernel=self.kernel(),
        ).squeeze(-1)
        (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        ) = self._combine_embeddings(
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            total_uih_len=total_uih_len,
            total_targets=total_targets,
            seq_lengths=seq_lengths,
            seq_timestamps=seq_timestamps,
            content_embeddings=content_embeddings,
            action_embeddings=action_embeddings,
            contextual_embeddings=contextual_embeddings,
            num_targets=num_targets,
        )

        return (
            output_max_seq_len,
            output_total_uih_len,
            output_total_targets,
            output_seq_lengths,
            output_seq_offsets,
            output_seq_timestamps,
            output_seq_embeddings,
            output_num_targets,
        )

    def interleave_targets(self) -> bool:
        """Interleave targets or not."""
        return self.is_train and self._enable_interleaving

    def contextual_seq_len(self) -> int:
        """Contextual feature sequence length."""
        return self._max_contextual_seq_len


def create_input_preprocessor(
    preprocessor_cfg: Union[module_pb2.GRInputPreprocessor, Dict[str, Any]],
    **kwargs: Any,
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
        return ContextualInterleavePreprocessor(
            **config_dict, enable_interleaving=False
        )
    elif preprocessor_type == "contextual_interleave_preprocessor":
        return ContextualInterleavePreprocessor(**config_dict)
    else:
        raise RuntimeError(f"Unknown preprocessor type: {preprocessor_type}")
