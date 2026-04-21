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

"""Backend-agnostic helpers for HSTU attention.

Contains mask / func-tensor builders that feed the ``attn_func`` input of
``hstu_mha`` / ``cutlass_hstu_mha`` / ``pytorch_hstu_mha``.  The helpers
themselves have no kernel-specific dependencies: the ``attn_func`` NFUNC=3
layout is shared between the CUTLASS production path and the PyTorch
reference path.
"""

from typing import Optional, Tuple

import torch

from tzrec.ops import Kernel
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged


def build_sla_func_tensor(
    nheads: int,
    sla_k1: int,
    sla_k2: int,
    seq_offsets: torch.Tensor,
    total_q: int,
    num_targets: Optional[torch.Tensor] = None,
    contextual_seq_len: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the NFUNC=3 func tensor for Semi-Local Attention (SLA).

    The HSTU CUTLASS kernel's arbitrary-mask path addresses the func tensor
    via ``func_ptr + cu_seqlens[b]``, requiring a jagged layout of shape
    ``(nheads, 3, total_q)``.

    NFUNC=3 encodes two disjoint column-intervals per query row:
      Interval 0: ``[0, col_max0)``
      Interval 1: ``[col_min0, col_max1)``

    For **history tokens** (position < seq_len - num_targets):
      SLA mask = causal ∩ (local-K1 ∪ global-prefix).
      ``effective_k2 = max(sla_k2, contextual_seq_len)``
      ``col_max0 = min(effective_k2, pos + 1)``
      ``col_min0 = max(effective_k2, pos - K1 + 1)``
      ``col_max1 = pos + 1``

    For **target tokens** (position >= seq_len - num_targets):
      See all history but not other targets.
      ``col_max0 = H_b``, ``col_min0 = H_b``, ``col_max1 = H_b``
      where ``H_b = seq_len - num_targets`` is the history boundary.

    Args:
        nheads: number of attention heads.
        sla_k1: local causal window size.
        sla_k2: global prefix length.
        seq_offsets: int32 cumulative sequence offsets of shape (B+1,).
        total_q: total jagged tokens in the batch (= ``seq_offsets[-1]``).
            Taken from the caller's tensor metadata (e.g. ``q.size(0)``)
            to avoid a D->H ``.item()`` sync per forward.
        num_targets: int32 target counts per batch element (B,), or None.
        contextual_seq_len: number of contextual tokens per sequence.
        device: target device (inferred from seq_offsets if None).

    Returns:
        func tensor of shape (nheads, 3, total_q), dtype int32.
    """
    if sla_k1 < 0 or sla_k2 < 0 or contextual_seq_len < 0:
        raise ValueError(
            f"SLA params must be non-negative; got "
            f"sla_k1={sla_k1}, sla_k2={sla_k2}, "
            f"contextual_seq_len={contextual_seq_len}"
        )
    if device is None:
        device = seq_offsets.device
    # Only cast when needed; callers (STUStack, STULayer) already supply
    # int32 offsets from fbgemm's cumsum.
    if seq_offsets.dtype != torch.int32:
        seq_offsets_i32 = seq_offsets.to(torch.int32)
    else:
        seq_offsets_i32 = seq_offsets
    effective_k2 = max(sla_k2, contextual_seq_len)

    # Map each jagged position to its batch element and local position.
    pos_global = torch.arange(total_q, device=device, dtype=torch.int32)
    batch_ids = torch.searchsorted(seq_offsets_i32[1:], pos_global, right=True)
    pos_local = pos_global - seq_offsets_i32[batch_ids]

    seq_lengths = seq_offsets_i32[1:] - seq_offsets_i32[:-1]  # (B,)
    L = seq_lengths[batch_ids]  # per-position sequence length

    if num_targets is not None:
        T = num_targets.to(torch.int32)[batch_ids]  # per-position target count
    else:
        T = torch.zeros_like(pos_local)
    # Clamp so pathological inputs (num_targets[b] > seq_lengths[b]) can't
    # produce a negative history boundary that would collapse every row's
    # intervals to an empty set and silently yield NaN attention output.
    H_boundary = torch.clamp(L - T, min=0)

    is_history = pos_local < H_boundary

    # History tokens: SLA intervals.  ``torch.clamp(..., max=effective_k2)``
    # avoids the host-tensor + expand_as roundtrip used previously.
    hist_col_max0 = torch.clamp(pos_local + 1, max=effective_k2)
    hist_col_min0 = torch.clamp(pos_local - sla_k1 + 1, min=effective_k2)
    hist_col_max1 = pos_local + 1

    # Target tokens: see all history [0, H_b), nothing else
    col_max0 = torch.where(is_history, hist_col_max0, H_boundary)
    col_min0 = torch.where(is_history, hist_col_min0, H_boundary)
    col_max1 = torch.where(is_history, hist_col_max1, H_boundary)

    # Stack as (3, total_q) then expand to (nheads, 3, total_q)
    func_2d = torch.stack([col_max0, col_min0, col_max1], dim=0)  # (3, total_q)
    return func_2d.unsqueeze(0).expand(nheads, 3, total_q).contiguous()


def apply_truncation(
    x: torch.Tensor,
    x_offsets: torch.Tensor,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len: int,
    *,
    truncate_tail_len: int,
    contextual_seq_len: int = 0,
    kernel: Kernel = Kernel.PYTORCH,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], int]:
    """Drop the UIH head while preserving the contextual prefix.

    Pre-truncation layout per sample is
    ``[contextual(C) | UIH(U) | targets(T)]`` with ``L_b = C + U + T``.
    Returns the post-truncation tuple
    ``(x, x_offsets, seq_lengths, num_targets, max_seq_len)`` keeping
    ``[contextual | last (new_length_b - C) tokens of UIH+targets]`` per
    sample, where ``new_length_b = min(L_b, truncate_tail_len)``.

    The result is a three-part keep-drop-keep slice per sample, so it is
    implemented as ``split_2D_jagged`` (peel off the prefix) +
    ``split_2D_jagged`` (drop the rest's head) + ``concat_2D_jagged``
    (re-join). When ``contextual_seq_len == 0`` the prefix step is
    skipped and a single split suffices.

    Args:
        x: jagged values of shape ``(total, D)``.
        x_offsets: cumulative offsets ``(B + 1,)``.
        seq_lengths: per-sample lengths ``(B,)``.
        num_targets: per-sample target counts ``(B,)`` or ``None``.
        max_seq_len: padded max length of the input.
        truncate_tail_len: keep-cap; must be > ``contextual_seq_len``.
        contextual_seq_len: number of leading contextual tokens to
            preserve (uniform across the batch).
        kernel: backend for the underlying jagged ops.

    Returns:
        ``(x, x_offsets, seq_lengths, num_targets, max_seq_len)`` with
        post-truncation values. ``num_targets`` is clamped to
        ``new_length - contextual_seq_len`` so the reported count never
        exceeds the available post-truncation slots.
    """
    if truncate_tail_len <= contextual_seq_len:
        raise ValueError(
            f"truncate_tail_len ({truncate_tail_len}) must be strictly "
            f"greater than contextual_seq_len ({contextual_seq_len}); the "
            f"contextual prefix is preserved by truncation and the tail "
            f"must leave room for at least one UIH/target token."
        )
    new_lengths = torch.clamp(seq_lengths, max=truncate_tail_len)
    if contextual_seq_len > 0:
        B = seq_lengths.size(0)
        # Step 1: peel off [contextual | rest-of-sequence].
        offsets_prefix = (
            torch.arange(B + 1, device=x.device, dtype=x_offsets.dtype)
            * contextual_seq_len
        )
        offsets_rest = x_offsets - offsets_prefix
        x_prefix, x_rest = split_2D_jagged(
            values=x,
            max_seq_len=max_seq_len,
            offsets_left=offsets_prefix,
            offsets_right=offsets_rest,
            kernel=kernel,
        )
        # Step 2: drop the head of the rest; keep its tail.
        rest_head_lengths = seq_lengths - new_lengths
        rest_tail_lengths = new_lengths - contextual_seq_len
        offsets_rest_head = torch.ops.fbgemm.asynchronous_complete_cumsum(
            rest_head_lengths
        )
        offsets_rest_tail = torch.ops.fbgemm.asynchronous_complete_cumsum(
            rest_tail_lengths
        )
        _, x_rest_kept = split_2D_jagged(
            values=x_rest,
            max_seq_len=max_seq_len - contextual_seq_len,
            offsets_left=offsets_rest_head,
            offsets_right=offsets_rest_tail,
            kernel=kernel,
        )
        # Step 3: re-join [contextual | rest_tail].
        x = concat_2D_jagged(
            values_left=x_prefix,
            values_right=x_rest_kept,
            max_len_left=contextual_seq_len,
            max_len_right=truncate_tail_len - contextual_seq_len,
            offsets_left=offsets_prefix,
            offsets_right=offsets_rest_tail,
            kernel=kernel,
        )
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(new_lengths)
    else:
        head_lengths = seq_lengths - new_lengths
        offsets_head = torch.ops.fbgemm.asynchronous_complete_cumsum(head_lengths)
        offsets_tail = torch.ops.fbgemm.asynchronous_complete_cumsum(new_lengths)
        _, x = split_2D_jagged(
            values=x,
            max_seq_len=max_seq_len,
            offsets_left=offsets_head,
            offsets_right=offsets_tail,
            kernel=kernel,
        )
        x_offsets = offsets_tail
    seq_lengths = new_lengths
    if num_targets is not None:
        num_targets = torch.clamp(num_targets, max=new_lengths - contextual_seq_len)
    return x, x_offsets, seq_lengths, num_targets, truncate_tail_len
