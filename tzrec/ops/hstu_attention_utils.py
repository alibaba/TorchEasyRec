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

"""Backend-agnostic helpers for HSTU attention: SLA mask builder + truncation."""

from dataclasses import dataclass
from typing import Optional

import torch

from tzrec.ops import Kernel
from tzrec.ops.jagged_tensors import concat_2D_jagged, split_2D_jagged
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


# Register a black-box op for the (3, total_q) -> (nheads, 3, total_q)
# broadcast so Inductor doesn't try to lower it as a Pointwise.  Inductor's
# combine_contiguous_dims for that 3D iteration with symbolic last dim
# emits ModularIndexing(idx, total_q, 3) whose sympy simplifier hits a
# ZeroDivisionError during AOT compile.  Use the low-level
# torch.library.define/impl/register_fake API (matching the convention in
# ``cutlass_hstu_attention.py``) rather than ``@torch.library.custom_op``
# to avoid the AOTI multi-thread-predict deadlock the latter triggers.
_SLA_LIB = torch.library.Library("tzrec", "FRAGMENT")
_SLA_LIB.define("_sla_broadcast_func_to_heads(Tensor func_2d, int nheads) -> Tensor")


def _sla_broadcast_func_to_heads_impl(
    func_2d: torch.Tensor, nheads: int
) -> torch.Tensor:
    return func_2d.unsqueeze(0).expand(nheads, *func_2d.shape).contiguous()


def _sla_broadcast_func_to_heads_meta(
    func_2d: torch.Tensor, nheads: int
) -> torch.Tensor:
    return torch.empty(
        nheads, *func_2d.shape, dtype=func_2d.dtype, device=func_2d.device
    )


_SLA_LIB.impl("_sla_broadcast_func_to_heads", _sla_broadcast_func_to_heads_impl, "CUDA")
_SLA_LIB.impl("_sla_broadcast_func_to_heads", _sla_broadcast_func_to_heads_impl, "CPU")
torch.library.register_fake("tzrec::_sla_broadcast_func_to_heads")(
    _sla_broadcast_func_to_heads_meta
)


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
        func tensor of shape (nheads, 3, total_q), dtype int32, contiguous
        (head dim materialized inside ``_sla_broadcast_func_to_heads``).
    """
    if sla_k1 < 0 or sla_k2 < 0 or contextual_seq_len < 0:
        raise ValueError(
            f"SLA params must be non-negative; got "
            f"sla_k1={sla_k1}, sla_k2={sla_k2}, "
            f"contextual_seq_len={contextual_seq_len}"
        )
    if device is None:
        device = seq_offsets.device
    # Unconditional cast: avoids fx tracing `Proxy.dtype != int32` as a
    # Python bool; no-op on the fast path (callers pass int32).
    seq_offsets_i32 = seq_offsets.to(torch.int32)
    effective_k2 = max(sla_k2, contextual_seq_len)

    # diff + repeat_interleave (not searchsorted on a slice): the slice
    # `seq_offsets[1:]` produces a SliceView that crashes Inductor's
    # searchsorted boundaries lowering during AOT compile.
    seq_lengths = torch.diff(seq_offsets_i32)  # (B,)
    B = seq_lengths.size(0)
    pos_global = torch.arange(total_q, device=device, dtype=torch.int32)
    batch_ids = torch.repeat_interleave(
        torch.arange(B, device=device, dtype=torch.int32),
        seq_lengths,
    )
    pos_local = pos_global - seq_offsets_i32[batch_ids]
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

    func_2d = torch.stack([col_max0, col_min0, col_max1], dim=0)  # (3, total_q)
    # Custom op (black-box to Inductor) does the head-dim broadcast +
    # contiguous-ification.  See the op definition above for the AOT
    # codegen reason; cannot use plain ``.unsqueeze(0).expand(...)`` here.
    return torch.ops.tzrec._sla_broadcast_func_to_heads(func_2d, nheads)


@dataclass(frozen=True)
class STUTruncationPlan:
    """Precomputed offsets for a UIH-only truncation, replayable across tensors.

    Fields:
        new_lengths: per-sample lengths after truncation, shape ``(B,)``.
        new_x_offsets: cumulative offsets after truncation, shape ``(B+1,)``.
        new_max_seq_len: tight post-truncation ``max(new_lengths)``;
            only D->H sync on the truncation path.
        pre_max_seq_len: input ``max_seq_len`` before truncation.
        contextual_seq_len: leading contextual tokens preserved uniformly.
        offsets_head: cumulative drop counts, shape ``(B+1,)``.
        offsets_tail: post-truncation offsets into the contextual-stripped
            subsequence; equals ``new_x_offsets`` when
            ``contextual_seq_len == 0``.
        total_dropped: ``offsets_head[-1]`` as a static int; passed
            to ``split_2D_jagged`` as ``total_len_left`` to skip its
            ``.item()`` fallback under fx / AOT export.
        total_kept: ``offsets_tail[-1]`` as a static int.
        offsets_prefix: contextual-prefix cumulative offsets ``(B+1,)``;
            only set when ``contextual_seq_len > 0``.
        offsets_rest: post-prefix cumulative offsets ``(B+1,)``;
            only set when ``contextual_seq_len > 0``.
        total_prefix: ``B * contextual_seq_len`` (0 when no contextual).
        total_rest: ``offsets_rest[-1]`` as a static int (0 when no
            contextual).
    """

    new_lengths: torch.Tensor
    new_x_offsets: torch.Tensor
    new_max_seq_len: int
    pre_max_seq_len: int
    contextual_seq_len: int
    offsets_head: torch.Tensor
    offsets_tail: torch.Tensor
    total_dropped: int
    total_kept: int
    offsets_prefix: Optional[torch.Tensor] = None
    offsets_rest: Optional[torch.Tensor] = None
    total_prefix: int = 0
    total_rest: int = 0


def compute_stu_truncation_plan(
    x_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor],
    max_seq_len: int,
    *,
    truncate_tail_len: int,
    contextual_seq_len: int = 0,
) -> STUTruncationPlan:
    """Compute the offset / length math for a UIH-only truncation.

    No jagged-value moves happen here -- the returned plan can be applied
    to any number of jagged tensors that share the same ``(B,
    seq_lengths)`` layout via :func:`apply_stu_truncation_plan`.

    Sample layout is ``[contextual(C) | UIH(U_b) | targets(T_b)]`` with
    ``L_b = C + U_b + T_b``.  ``truncate_tail_len`` caps the UIH portion
    only; the contextual prefix and all targets are preserved as full
    structural elements.

    Args:
        x_offsets: cumulative offsets ``(B + 1,)``; per-sample lengths
            are derived as ``x_offsets[1:] - x_offsets[:-1]``.
        num_targets: per-sample target counts ``(B,)`` or ``None``
            (listwise mode -- the entire sample counts as UIH).
        max_seq_len: padded max length of the input batch.
        truncate_tail_len: maximum UIH tokens kept per sample.  Must be
            non-negative; ``0`` drops all UIH (only contextual + targets
            survive).
        contextual_seq_len: number of leading contextual tokens preserved
            uniformly across the batch.

    Returns:
        A :class:`STUTruncationPlan`.  Contains one D->H sync to compute
        the tight ``new_max_seq_len``.
    """
    if truncate_tail_len < 0:
        raise ValueError(
            f"truncate_tail_len must be non-negative; got {truncate_tail_len}"
        )
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative; got {contextual_seq_len}"
        )
    seq_lengths = x_offsets[1:] - x_offsets[:-1]
    if num_targets is not None:
        uih_lengths = seq_lengths - contextual_seq_len - num_targets
    else:
        uih_lengths = seq_lengths - contextual_seq_len
    new_uih_lengths = torch.clamp(uih_lengths, max=truncate_tail_len)
    drop_count = uih_lengths - new_uih_lengths
    new_lengths = seq_lengths - drop_count
    # ``rest_tail_lengths`` collapses to ``new_lengths`` when C == 0, so
    # both branches share the same offsets math.
    rest_tail_lengths = new_lengths - contextual_seq_len
    offsets_head = torch.ops.fbgemm.asynchronous_complete_cumsum(drop_count)
    offsets_tail = torch.ops.fbgemm.asynchronous_complete_cumsum(rest_tail_lengths)
    B_static = seq_lengths.size(0)
    # fx_int_item is fx-wrapped to avoid `int(Proxy)` under symbolic trace.
    total_dropped = fx_int_item(offsets_head[-1])
    total_kept = fx_int_item(offsets_tail[-1])
    if contextual_seq_len > 0:
        offsets_prefix = (
            torch.arange(B_static + 1, device=seq_lengths.device, dtype=x_offsets.dtype)
            * contextual_seq_len
        )
        offsets_rest = x_offsets - offsets_prefix
        # ``new_lengths_b = C + rest_tail_lengths_b`` so cumsum(new_lengths)
        # equals offsets_prefix + offsets_tail; saves one cumsum kernel.
        new_x_offsets = offsets_prefix + offsets_tail
        total_prefix = B_static * contextual_seq_len
        total_rest = fx_int_item(offsets_rest[-1])
    else:
        offsets_prefix = None
        offsets_rest = None
        new_x_offsets = offsets_tail
        total_prefix = 0
        total_rest = 0
    return STUTruncationPlan(
        new_lengths=new_lengths,
        new_x_offsets=new_x_offsets,
        new_max_seq_len=fx_int_item(new_lengths.max()),
        pre_max_seq_len=max_seq_len,
        contextual_seq_len=contextual_seq_len,
        offsets_head=offsets_head,
        offsets_tail=offsets_tail,
        total_dropped=total_dropped,
        total_kept=total_kept,
        offsets_prefix=offsets_prefix,
        offsets_rest=offsets_rest,
        total_prefix=total_prefix,
        total_rest=total_rest,
    )


def apply_stu_truncation_plan(
    x: torch.Tensor,
    plan: STUTruncationPlan,
    kernel: Kernel,
) -> torch.Tensor:
    """Apply a precomputed truncation plan to a jagged tensor.

    The plan must come from a call to :func:`compute_stu_truncation_plan`
    on offsets / lengths that match ``x``'s layout.  Performs no D->H syncs.

    Args:
        x: jagged values of shape ``(total, D)``.
        plan: precomputed truncation plan.
        kernel: backend for the underlying jagged ops.

    Returns:
        Post-truncation jagged values of shape
        ``(plan.new_x_offsets[-1], D)``.
    """
    contextual_seq_len = plan.contextual_seq_len
    pre_max = plan.pre_max_seq_len
    if contextual_seq_len > 0:
        offsets_prefix = plan.offsets_prefix
        offsets_rest = plan.offsets_rest
        assert offsets_prefix is not None and offsets_rest is not None
        # total_len_{left,right} from plan: skips split_2D_jagged's
        # triton `.item()` fallback (fails under FakeTensor / AOT).
        x_prefix, x_rest = split_2D_jagged(
            values=x,
            max_seq_len=pre_max,
            total_len_left=plan.total_prefix,
            total_len_right=plan.total_rest,
            offsets_left=offsets_prefix,
            offsets_right=offsets_rest,
            kernel=kernel,
        )
        _, x_rest_kept = split_2D_jagged(
            values=x_rest,
            max_seq_len=pre_max - contextual_seq_len,
            total_len_left=plan.total_dropped,
            total_len_right=plan.total_kept,
            offsets_left=plan.offsets_head,
            offsets_right=plan.offsets_tail,
            kernel=kernel,
        )
        return concat_2D_jagged(
            values_left=x_prefix,
            values_right=x_rest_kept,
            max_len_left=contextual_seq_len,
            max_len_right=pre_max - contextual_seq_len,
            offsets_left=offsets_prefix,
            offsets_right=plan.offsets_tail,
            kernel=kernel,
        )
    _, x_kept = split_2D_jagged(
        values=x,
        max_seq_len=pre_max,
        total_len_left=plan.total_dropped,
        total_len_right=plan.total_kept,
        offsets_left=plan.offsets_head,
        offsets_right=plan.offsets_tail,
        kernel=kernel,
    )
    return x_kept
