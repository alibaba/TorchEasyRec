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

# We use the jagged_tensors ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from typing import Optional, Tuple

import torch
from torch.fx._symbolic_trace import is_fx_tracing
from torch.utils._triton import has_triton

from tzrec.ops import Kernel
from tzrec.ops.pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_jagged_dense_bmm_broadcast_add,
    pytorch_split_2D_jagged,
)

if has_triton():
    from tzrec.ops.triton.triton_jagged_tensors import (
        triton_concat_2D_jagged,
        triton_jagged_dense_bmm_broadcast_add,
        triton_split_2D_jagged,
    )
else:
    triton_concat_2D_jagged = pytorch_concat_2D_jagged
    triton_jagged_dense_bmm_broadcast_add = pytorch_jagged_dense_bmm_broadcast_add
    triton_split_2D_jagged = pytorch_split_2D_jagged


def concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if not is_fx_tracing():
        torch._assert(values_left.dim() == 2, "values_left must be 2D")
        torch._assert(values_right.dim() == 2, "values_right must be 2D")
        torch._assert(
            values_right.shape[1] == values_left.shape[1],
            f"values_left shape[1] must be equal to values_right shape[1] {values_left.shape[1]} vs {values_right.shape[1]}",  # NOQA
        )
    if kernel == Kernel.TRITON:
        return triton_concat_2D_jagged(
            values_left=values_left,
            values_right=values_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )
    else:
        return pytorch_concat_2D_jagged(
            values_left=values_left,
            values_right=values_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )


def split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int] = None,
    total_len_right: Optional[int] = None,
    max_len_left: Optional[int] = None,
    max_len_right: Optional[int] = None,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: Kernel = Kernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not is_fx_tracing():
        torch._assert(values.dim() == 2, "values must be 2D")
        torch._assert(
            offsets_left is not None or offsets_right is not None,
            "offsets_left and offsets_right cannot be None at the same time",
        )
        if offsets_left is None:
            torch._assert(
                max_len_left is not None,
                "max_len_left must be provided when offsets_left is None",
            )
        if offsets_right is None:
            torch._assert(
                max_len_right is not None,
                "max_len_right must be provided when offsets_right is None",
            )
        if offsets_left is not None and offsets_right is not None:
            torch._assert(
                offsets_left.shape[0] == offsets_right.shape[0],
                "offsets_left shape[0] must be equal to offsets_right shape[0]",
            )
    if kernel == Kernel.TRITON:
        return triton_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            total_len_left=total_len_left,
            total_len_right=total_len_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )
    else:
        return pytorch_split_2D_jagged(
            max_seq_len=max_seq_len,
            values=values,
            total_len_left=total_len_left,
            total_len_right=total_len_right,
            max_len_left=max_len_left,
            max_len_right=max_len_right,
            offsets_left=offsets_left,
            offsets_right=offsets_right,
        )


def jagged_dense_bmm_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    """Computing out = jagged x dense + bias.

    jagged has shape (sum_B(M_i), K), dense has shape (B, K, N), and bias has
    shape (B, N), out has shape (sum_B(M_i), N)
    """
    if not is_fx_tracing():
        _, K = jagged.shape
        B, _, N = dense.shape
        torch._assert(dense.shape[1] == K, "wrong dense shape[1]")
        torch._assert(seq_offsets.shape[0] == B + 1, "wrong seq_offsets shape[0]")
        torch._assert(bias.shape[0] == B, "wrong bias shape[0]")
        torch._assert(bias.shape[1] == N, "wrong bias shape[1]")
    if kernel == Kernel.TRITON:
        return triton_jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
        )
    else:
        return pytorch_jagged_dense_bmm_broadcast_add(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
        )
