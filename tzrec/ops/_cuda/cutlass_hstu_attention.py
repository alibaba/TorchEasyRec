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

# We use the low-level torch.library.define / torch.library.impl /
# register_fake API here instead of @torch.library.custom_op on purpose:
# custom_op wraps the user function in @torch.compiler.disable AND adds an
# autograd_impl / forward_no_grad dispatch layer (even without
# register_autograd). The combination of those wrappers with AOT-Inductor's
# compiled-model dispatch deadlocks when the predict pipeline calls the AOT
# model from multiple worker threads. Confirmed via py-spy dump showing two
# `_forward_loop` threads stuck inside the AOTI `__call__`, one of them
# blocked inside `_cutlass_hstu_mha_fwd` and the other waiting at the entry
# of the AOTI `__call__`. Using the low-level API gives us a single-layer
# dispatch (just our impl) and the deadlock disappears.

from typing import List, Optional

import torch

_LIB = torch.library.Library("tzrec", "FRAGMENT")

_FWD_SCHEMA = (
    "cutlass_hstu_mha_fwd("
    "Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, "
    "SymInt max_seq_len, SymInt scaling_seqlen, "
    "Tensor? num_contexts, Tensor? num_targets, "
    "SymInt target_group_size, SymInt window_size_left, "
    "SymInt window_size_right, float alpha, Tensor? func"
    ") -> Tensor"
)
_BWD_SCHEMA = (
    "cutlass_hstu_mha_bwd("
    "Tensor dout, Tensor q, Tensor k, Tensor v, Tensor cu_seqlens, "
    "Tensor? num_contexts, Tensor? num_targets, "
    "SymInt max_seq_len, SymInt scaling_seqlen, "
    "SymInt target_group_size, SymInt window_size_left, "
    "SymInt window_size_right, float alpha, Tensor? func"
    ") -> Tensor[]"
)

_LIB.define(_FWD_SCHEMA)
_LIB.define(_BWD_SCHEMA)


def _cutlass_hstu_mha_fwd_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    scaling_seqlen: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    import hstu_attn_2_cuda as hstu_attn_cuda

    num_heads = q.size(1)
    head_dim = q.size(2)
    out, _ = hstu_attn_cuda.varlen_fwd(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seq_len,
        max_seq_len,
        scaling_seqlen,
        num_contexts,
        num_targets,
        target_group_size,
        window_size_left,
        window_size_right,
        alpha,
        None,  # rab
        func,
        None,  # kv_cache
        None,  # page_offsets
        None,  # page_ids
        None,  # last_page_lens
        None,  # cu_seqlens_t
    )
    return out[:, :, :head_dim].reshape(-1, num_heads, head_dim)


def _cutlass_hstu_mha_bwd_impl(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    max_seq_len: int,
    scaling_seqlen: int,
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    import hstu_attn_2_cuda as hstu_attn_cuda

    num_heads = q.size(1)
    head_dim = q.size(2)
    dq, dk, dv, _ = hstu_attn_cuda.varlen_bwd(
        dout.view(-1, num_heads, head_dim),
        q,
        k,
        v,
        None,
        None,
        None,
        cu_seqlens,
        cu_seqlens,
        max_seq_len,
        max_seq_len,
        scaling_seqlen,
        num_contexts,
        num_targets,
        target_group_size,
        window_size_left,
        window_size_right,
        alpha,
        None,  # rab_padded
        False,  # has_drab
        func,
        False,  # deterministic
    )
    return [dq, dk, dv]


def _cutlass_hstu_mha_fwd_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seq_len: int,
    scaling_seqlen: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    # Output shape is (total, nheads, hidden_dim); that matches v, not q.
    # Under the current attention_dim == hidden_dim constraint q and v are
    # the same shape, but keying the fake off v makes the meta robust if
    # that constraint is relaxed later.
    return torch.empty_like(v)


def _cutlass_hstu_mha_bwd_meta(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    max_seq_len: int,
    scaling_seqlen: int,
    target_group_size: int,
    window_size_left: int,
    window_size_right: int,
    alpha: float,
    func: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]


_LIB.impl("cutlass_hstu_mha_fwd", _cutlass_hstu_mha_fwd_impl, "CUDA")
_LIB.impl("cutlass_hstu_mha_bwd", _cutlass_hstu_mha_bwd_impl, "CUDA")
torch.library.register_fake("tzrec::cutlass_hstu_mha_fwd")(_cutlass_hstu_mha_fwd_meta)
torch.library.register_fake("tzrec::cutlass_hstu_mha_bwd")(_cutlass_hstu_mha_bwd_meta)


class _CutlassHstuMhaFunction(torch.autograd.Function):
    """Python autograd.Function wrapping the cutlass low-level torch ops.

    Backward is implemented at the Python autograd level (not via
    ``register_autograd`` on the op) so that the inference dispatch goes
    straight to the registered impl, avoiding the autograd_impl /
    forward_no_grad wrapper layer that triggers the multi-threaded AOTI
    deadlock under predict workloads.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seq_len: int,
        scaling_seqlen: int,
        num_contexts: Optional[torch.Tensor],
        num_targets: Optional[torch.Tensor],
        target_group_size: int,
        window_size_left: int,
        window_size_right: int,
        alpha: float,
        func: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        out = torch.ops.tzrec.cutlass_hstu_mha_fwd(
            q,
            k,
            v,
            cu_seqlens,
            max_seq_len,
            scaling_seqlen,
            num_contexts,
            num_targets,
            target_group_size,
            window_size_left,
            window_size_right,
            alpha,
            func,
        )
        ctx.save_for_backward(q, k, v, cu_seqlens, num_contexts, num_targets, func)
        ctx.max_seq_len = max_seq_len
        ctx.scaling_seqlen = scaling_seqlen
        ctx.target_group_size = target_group_size
        ctx.window_size_left = window_size_left
        ctx.window_size_right = window_size_right
        ctx.alpha = alpha
        return out

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        q, k, v, cu_seqlens, num_contexts, num_targets, func = ctx.saved_tensors
        dq, dk, dv = torch.ops.tzrec.cutlass_hstu_mha_bwd(
            grad_output.contiguous(),
            q,
            k,
            v,
            cu_seqlens,
            num_contexts,
            num_targets,
            ctx.max_seq_len,
            ctx.scaling_seqlen,
            ctx.target_group_size,
            ctx.window_size_left,
            ctx.window_size_right,
            ctx.alpha,
            func,
        )
        return (
            dq,
            dk,
            dv,
            None,  # cu_seqlens
            None,  # max_seq_len
            None,  # scaling_seqlen
            None,  # num_contexts
            None,  # num_targets
            None,  # target_group_size
            None,  # window_size_left
            None,  # window_size_right
            None,  # alpha
            None,  # func
        )


def build_sla_func_tensor(
    nheads: int,
    sla_k1: int,
    sla_k2: int,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    contextual_seq_len: int = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the NFUNC=3 func tensor for Semi-Local Attention (SLA).

    The HSTU CUTLASS kernel's arbitrary-mask path addresses the func tensor
    via ``func_ptr + cu_seqlens[b]``, requiring a jagged layout of shape
    ``(nheads, 3, total_q)`` where ``total_q = seq_offsets[-1]``.

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
        num_targets: int32 target counts per batch element (B,), or None.
        contextual_seq_len: number of contextual tokens per sequence.
        device: target device (inferred from seq_offsets if None).

    Returns:
        func tensor of shape (nheads, 3, total_q), dtype int32.
    """
    if device is None:
        device = seq_offsets.device
    seq_offsets_i32 = seq_offsets.to(torch.int32)
    total_q = int(seq_offsets_i32[-1].item())
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
    H_boundary = L - T  # per-position history boundary

    is_history = pos_local < H_boundary

    # History tokens: SLA intervals
    ek2 = torch.tensor(effective_k2, device=device, dtype=torch.int32)
    hist_col_max0 = torch.minimum(pos_local + 1, ek2.expand_as(pos_local))
    hist_col_min0 = torch.clamp(pos_local - sla_k1 + 1, min=effective_k2)
    hist_col_max1 = pos_local + 1

    # Target tokens: see all history [0, H_b), nothing else
    col_max0 = torch.where(is_history, hist_col_max0, H_boundary)
    col_min0 = torch.where(is_history, hist_col_min0, H_boundary)
    col_max1 = torch.where(is_history, hist_col_max1, H_boundary)

    # Stack as (3, total_q) then expand to (nheads, 3, total_q)
    func_2d = torch.stack([col_max0, col_min0, col_max1], dim=0)  # (3, total_q)
    return func_2d.unsqueeze(0).expand(nheads, 3, total_q).contiguous()


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16)


@torch.fx.wrap
def cutlass_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    sla_k1: int = 0,
    sla_k2: int = 0,
) -> torch.Tensor:
    """CUTLASS-based HSTU multi-head attention.

    Supports standard causal/local/context/target masks via the fixed mask
    parameters, and Semi-Local Attention (SLA) via the arbitrary-mask NFUNC
    path when ``sla_k1 > 0 or sla_k2 > 0``.

    The CUTLASS kernel uses int32 cu_seqlens internally, so the cumulative
    sum ``seq_offsets[-1]`` (total token count in the batch) must fit in
    int32 (< 2**31 ≈ 2.1B tokens).

    Args:
        max_seq_len: maximum sequence length in the batch.
        alpha: scaling factor for attention scores.
        q: query tensor of shape (total, nheads, attn_dim).
        k: key tensor of shape (total, nheads, attn_dim).
        v: value tensor of shape (total, nheads, hidden_dim).
        seq_offsets: cumulative sequence offsets of shape (batch_size + 1,).
        causal: whether to apply causal masking.
        num_targets: number of target tokens per batch element.
        max_attn_len: maximum attention window length (0 means unlimited).
        contextual_seq_len: number of contextual tokens per sequence.
        sla_k1: Semi-Local Attention local causal window size (0 = disabled).
        sla_k2: Semi-Local Attention global prefix length (0 = disabled).

    Returns:
        output tensor of shape (total, nheads, hidden_dim).
    """
    if q.shape[2] != v.shape[2]:
        raise ValueError(
            f"CUTLASS hstu_attn requires attention_dim == hidden_dim, "
            f"got q.shape[2]={q.shape[2]} != v.shape[2]={v.shape[2]}"
        )
    if q.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            f"CUTLASS hstu_attn supports fp16, bf16, and fp8_e4m3, got {q.dtype}. "
            f"Set train_config.mixed_precision to 'BF16', 'FP16', or 'FP8'."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = seq_offsets.to(torch.int32)

    # SLA mode: the func tensor fully describes the mask (SLA + contextual
    # prefix + target isolation).  We still pass num_contexts / num_targets
    # to the kernel for potential scaling effects, but masking is entirely
    # driven by the NFUNC intervals.
    func: Optional[torch.Tensor] = None
    if sla_k1 > 0 or sla_k2 > 0:
        batch_size = seq_offsets.size(0) - 1
        nheads = q.size(1)

        num_targets_int32: Optional[torch.Tensor] = None
        if num_targets is not None:
            num_targets_int32 = num_targets.to(torch.int32)

        func = build_sla_func_tensor(
            nheads=nheads,
            sla_k1=sla_k1,
            sla_k2=sla_k2,
            seq_offsets=cu_seqlens,
            num_targets=num_targets_int32,
            contextual_seq_len=contextual_seq_len,
            device=q.device,
        )
        window_size_left, window_size_right = -1, 0

        num_contexts_tensor: Optional[torch.Tensor] = None
        if contextual_seq_len > 0:
            num_contexts_tensor = torch.full(
                (batch_size,),
                contextual_seq_len,
                dtype=torch.int32,
                device=q.device,
            )
    else:
        # Legacy fixed-mask path (unchanged from PR #465).
        if causal:
            if max_attn_len > 0:
                window_size_left, window_size_right = max_attn_len, 0
            else:
                window_size_left, window_size_right = -1, 0
        else:
            window_size_left, window_size_right = -1, -1

        num_contexts_tensor = None
        if contextual_seq_len > 0:
            batch_size = seq_offsets.size(0) - 1
            num_contexts_tensor = torch.full(
                (batch_size,),
                contextual_seq_len,
                dtype=torch.int32,
                device=q.device,
            )

        num_targets_int32 = None
        if num_targets is not None:
            num_targets_int32 = num_targets.to(torch.int32)

    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _CutlassHstuMhaFunction.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seq_len,
            max_seq_len,  # scaling_seqlen
            num_contexts_tensor,
            num_targets_int32,
            1,  # target_group_size
            window_size_left,
            window_size_right,
            alpha,
            func,
        )
    return torch.ops.tzrec.cutlass_hstu_mha_fwd(
        q,
        k,
        v,
        cu_seqlens,
        max_seq_len,
        max_seq_len,  # scaling_seqlen
        num_contexts_tensor,
        num_targets_int32,
        1,  # target_group_size
        window_size_left,
        window_size_right,
        alpha,
        func,
    )
