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

from tzrec.ops import Kernel
from tzrec.ops._pytorch.pt_jagged_tensors import (
    pytorch_concat_2D_jagged,
    pytorch_jagged_dense_bmm_broadcast_add,
    pytorch_split_2D_jagged,
)


def concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: Optional[torch.Tensor] = None,
    offsets_right: Optional[torch.Tensor] = None,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if kernel == Kernel.CUTLASS:
        kernel = Kernel.TRITON
    if not is_fx_tracing():
        torch._assert(values_left.dim() == 2, "values_left must be 2D")
        torch._assert(values_right.dim() == 2, "values_right must be 2D")
        torch._assert(
            values_right.shape[1] == values_left.shape[1],
            f"values_left shape[1] must be equal to values_right shape[1] {values_left.shape[1]} vs {values_right.shape[1]}",  # NOQA
        )
    if kernel == Kernel.TRITON:
        from tzrec.ops._triton.triton_jagged_tensors import triton_concat_2D_jagged

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
    if kernel == Kernel.CUTLASS:
        kernel = Kernel.TRITON
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
        from tzrec.ops._triton.triton_jagged_tensors import (
            triton_split_2D_jagged,
        )

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
    if kernel == Kernel.CUTLASS:
        kernel = Kernel.TRITON
    if not is_fx_tracing():
        _, K = jagged.shape
        B, _, N = dense.shape
        torch._assert(dense.shape[1] == K, "wrong dense shape[1]")
        torch._assert(seq_offsets.shape[0] == B + 1, "wrong seq_offsets shape[0]")
        torch._assert(bias.shape[0] == B, "wrong bias shape[0]")
        torch._assert(bias.shape[1] == N, "wrong bias shape[1]")
    if kernel == Kernel.TRITON:
        from tzrec.ops._triton.triton_jagged_tensors import (
            triton_jagged_dense_bmm_broadcast_add,
        )

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


@torch.fx.wrap
def truncate_jagged_tail(
    values: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    tail_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Truncate each jagged sequence to its last ``tail_len`` tokens.

    For attention-truncation in ULTRA-HSTU: after N1 layers on full sequence L,
    keep only the last L' = tail_len tokens per user for the remaining N2 layers.

    Args:
        values: jagged values tensor of shape (total, D).
        seq_offsets: cumulative offsets of shape (B+1,).
        seq_lengths: per-sequence lengths of shape (B,).
        num_targets: per-sequence target counts of shape (B,) or None.
        tail_len: number of trailing tokens to keep per sequence.

    Returns:
        truncated_values: jagged values of shape (total_trunc, D).
        new_offsets: cumulative offsets of shape (B+1,).
        new_lengths: per-sequence lengths of shape (B,).
        new_num_targets: clamped target counts (B,) or None.
        new_max_seq_len: int, equals tail_len.
    """
    B = seq_lengths.size(0)
    new_lengths = torch.clamp(seq_lengths, max=tail_len)
    # Start position of the tail segment within each original sequence.
    starts = seq_offsets[:-1] + (seq_lengths - new_lengths)
    # Build index tensor to gather the tail tokens.
    arange = torch.arange(tail_len, device=values.device)
    # (B, tail_len): absolute indices into the jagged values tensor.
    gather_idx = starts.unsqueeze(1) + arange.unsqueeze(0)  # (B, tail_len)
    # Clamp to valid range for the gather; padded positions (beyond each
    # sequence's actual length) will gather a valid but irrelevant row and
    # are discarded by the dense_to_jagged repacking below.
    gather_idx = torch.clamp(gather_idx, max=values.size(0) - 1)
    gathered = values[gather_idx.view(-1)]  # (B * tail_len, D)
    # Rebuild as jagged: pack only valid positions.
    gathered = gathered.view(B, tail_len, -1)
    # For simplicity, use dense_to_jagged to repack.
    new_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(new_lengths)
    total_trunc = int(new_offsets[-1].item())
    truncated_values = torch.ops.fbgemm.dense_to_jagged(
        gathered, [new_offsets], total_trunc
    )[0]
    new_num_targets: Optional[torch.Tensor] = None
    if num_targets is not None:
        new_num_targets = torch.clamp(num_targets, max=new_lengths)
    return truncated_values, new_offsets, new_lengths, new_num_targets, tail_len
