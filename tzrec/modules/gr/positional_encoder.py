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

# We use the position encoder from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from math import sqrt
from typing import Optional

import torch

from tzrec.modules.utils import BaseModule
from tzrec.ops.position import (
    add_positional_embeddings,
    add_timestamp_positional_embeddings,
)


class HSTUPositionalEncoder(BaseModule):
    """HSTU Position Encoder.

    Args:
        embedding_dim (int): input embedding dim.
        num_position_buckets (int): position bucket number.
        num_time_buckets (int): time bucket number.
        use_time_encoding (bool): use timestamp encoding or not.
        contextual_seq_len (int): contextual feature sequence length.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_position_buckets: int,
        num_time_buckets: int = 0,
        use_time_encoding: bool = False,
        contextual_seq_len: int = 0,
        is_inference: bool = True,
    ) -> None:
        super().__init__(is_inference=is_inference)
        self._use_time_encoding: bool = use_time_encoding
        self._contextual_seq_len: int = contextual_seq_len
        self._embedding_dim: int = embedding_dim
        self._position_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
            torch.empty(num_position_buckets, embedding_dim).uniform_(
                -sqrt(1.0 / num_position_buckets),
                sqrt(1.0 / num_position_buckets),
            ),
        )
        if self._use_time_encoding:
            self._timestamp_embeddings_weight: torch.nn.Parameter = torch.nn.Parameter(
                torch.empty(num_time_buckets + 1, embedding_dim).uniform_(
                    -sqrt(1.0 / num_time_buckets),
                    sqrt(1.0 / num_time_buckets),
                ),
            )

    def forward(
        self,
        max_seq_len: int,
        seq_lengths: torch.Tensor,
        seq_offsets: torch.Tensor,
        seq_timestamps: torch.Tensor,
        seq_embeddings: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Forward the module.

        Args:
            max_seq_len (int): maximum sequence length.
            seq_lengths (torch.Tensor): input sequence lengths.
            seq_offsets (torch.Tensor): input sequence offsets.
            seq_timestamps (torch.Tensor): input sequence timestamps.
            seq_embeddings (torch.Tensor): input sequence embeddings.
            num_targets (int): number of targets.

        Returns:
            torch.Tensor: output sequence embedding with position embedding.
        """
        if self._use_time_encoding:
            seq_embeddings = add_timestamp_positional_embeddings(
                alpha=self._embedding_dim**0.5,
                max_seq_len=max_seq_len,
                max_contextual_seq_len=self._contextual_seq_len,
                position_embeddings_weight=self._position_embeddings_weight,
                timestamp_embeddings_weight=self._timestamp_embeddings_weight,
                seq_offsets=seq_offsets,
                seq_lengths=seq_lengths,
                seq_embeddings=seq_embeddings,
                timestamps=seq_timestamps,
                num_targets=num_targets,
                interleave_targets=False,
                kernel=self.kernel(),
            )
        else:
            seq_embeddings = add_positional_embeddings(
                alpha=self._embedding_dim**0.5,
                max_seq_len=max_seq_len,
                position_embeddings_weight=self._position_embeddings_weight,
                seq_offsets=seq_offsets,
                seq_lengths=seq_lengths,
                seq_embeddings=seq_embeddings,
                num_targets=num_targets,
                interleave_targets=False,
                kernel=self.kernel(),
            )
        return seq_embeddings
