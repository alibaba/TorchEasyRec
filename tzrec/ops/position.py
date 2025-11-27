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

# We use the position ecnoder ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from typing import Optional

import torch
from torch.fx._symbolic_trace import is_fx_tracing

from tzrec.ops import Kernel
from tzrec.ops._pytorch.pt_position import (
    pytorch_add_position_embeddings,
    pytorch_add_timestamp_positional_embeddings,
)


@torch.fx.wrap
def _get_high_inds(
    high_inds: torch.Tensor,
    position_embeddings_weight: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
) -> torch.Tensor:
    max_pos_ind = position_embeddings_weight.size(0)
    if num_targets is not None:
        if interleave_targets:
            high_inds = high_inds - num_targets * 2
        else:
            high_inds = high_inds - num_targets
    high_inds = torch.clamp(high_inds, max=max_pos_ind - 1)
    return high_inds


def add_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    high_inds = _get_high_inds(
        seq_lengths, position_embeddings_weight, num_targets, interleave_targets
    )
    if not is_fx_tracing():
        _, D = seq_embeddings.shape
        torch._assert(
            seq_offsets.size(0) - 1 == high_inds.size(0),
            "wrong jagged_offsets shape[0]",
        )
        _, D2 = position_embeddings_weight.shape
        torch._assert(D2 == D, "wrong dense shape[1]")

    if kernel == Kernel.TRITON:
        from tzrec.ops._triton.triton_position import triton_add_position_embeddings

        return triton_add_position_embeddings(
            jagged=seq_embeddings,
            jagged_offsets=seq_offsets,
            high_inds=high_inds,
            max_seq_len=max_seq_len,
            dense=position_embeddings_weight,
            scale=alpha,
        )
    else:
        return pytorch_add_position_embeddings(
            jagged=seq_embeddings,
            jagged_offsets=seq_offsets,
            high_inds=high_inds,
            max_seq_len=max_seq_len,
            dense=position_embeddings_weight,
            scale=alpha,
        )


def add_timestamp_positional_embeddings(
    alpha: float,
    max_seq_len: int,
    max_contextual_seq_len: int,
    position_embeddings_weight: torch.Tensor,
    timestamp_embeddings_weight: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    seq_embeddings: torch.Tensor,
    timestamps: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    interleave_targets: bool,
    time_bucket_fn: str = "sqrt",
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    assert time_bucket_fn in ["sqrt", "log"]
    seq_embeddings = seq_embeddings * alpha
    if kernel == Kernel.TRITON:
        from tzrec.ops._triton.triton_position import (
            triton_add_timestamp_positional_embeddings,
        )

        return triton_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
    else:
        return pytorch_add_timestamp_positional_embeddings(
            seq_embeddings=seq_embeddings,
            seq_offsets=seq_offsets,
            pos_embeddings=position_embeddings_weight,
            ts_embeddings=timestamp_embeddings_weight,
            timestamps=timestamps,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            seq_lengths=seq_lengths,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
        )
