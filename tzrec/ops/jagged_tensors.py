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


def jagged_segment_ids(
    lengths: torch.Tensor, output_size: Optional[int] = None
) -> torch.Tensor:
    """Map each jagged row to its segment index.

    ``output_size`` -- the statically known ``sum(lengths)`` -- avoids the
    hidden device->host sync (and data-dependent graph break) that
    ``repeat_interleave`` with tensor repeats otherwise performs.
    """
    return torch.repeat_interleave(
        torch.arange(lengths.size(0), device=lengths.device),
        lengths,
        output_size=output_size,
    )


def jagged_segment_sum(
    values: torch.Tensor,
    lengths: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Sum a jagged ``(total, C)`` tensor within each segment; empties -> 0.

    Reduced-precision inputs (fp16/bf16) accumulate in fp32 and cast back:
    ``index_add_`` is not on autocast's promote list, so bf16 inputs would
    otherwise add through bf16 atomics whose reorder noise sits at bf16
    rounding scale.  ``promote_types`` keeps the dtype choice a graph node
    rather than Python control flow, so fx tracing still inlines this.
    """
    acc_dtype = torch.promote_types(values.dtype, torch.float32)
    sums = torch.zeros(
        (lengths.size(0), values.size(-1)),
        dtype=acc_dtype,
        device=values.device,
    )
    sums.index_add_(0, segment_ids, values.to(acc_dtype))
    return sums.to(values.dtype)


def jagged_segment_max(
    values: torch.Tensor,
    lengths: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Max-reduce a jagged ``(total, C)`` tensor within each segment.

    ``scatter_reduce_`` rather than the still-beta ``index_reduce_`` (which
    warns on every call); it has no amax backward either, so callers using
    the result as a shift constant must detach ``values``.  Empty segments
    keep ``-inf``, which would make downstream broadcasts NaN under
    torch.compile, so they are rewritten to 0 -- nothing reads them back.
    """
    maxes = torch.full(
        (lengths.size(0), values.size(-1)),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    ).scatter_reduce_(
        0,
        segment_ids.unsqueeze(-1).expand_as(values),
        values,
        "amax",
        include_self=False,
    )
    return torch.nan_to_num(maxes, neginf=0.0)
