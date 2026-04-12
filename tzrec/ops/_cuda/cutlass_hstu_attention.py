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
    batch_size: int,
    nheads: int,
    max_seqlen_q: int,
    sla_k1: int,
    sla_k2: int,
    device: torch.device,
) -> torch.Tensor:
    """Build the NFUNC=3 func tensor for Semi-Local Attention (SLA).

    The HSTU CUTLASS kernel uses a table-driven arbitrary-mask API where the
    func tensor has shape (B, H, NFUNC, Lq) with dtype int32.  NFUNC=3 encodes
    two disjoint column intervals per query row i:
      Interval 0 (global prefix): [0, col_max(0))
      Interval 1 (local causal):  [col_min(0), col_max(1))

    For SLA: M[i,j] = 1 iff j <= i AND ((i-j) < K1 OR j < K2).

    Args:
        batch_size: number of sequences in the batch.
        nheads: number of attention heads.
        max_seqlen_q: maximum query sequence length.
        sla_k1: local causal window size.
        sla_k2: global prefix length.
        device: target device.

    Returns:
        func tensor of shape (batch_size, nheads, 3, max_seqlen_q), int32.
    """
    idx = torch.arange(max_seqlen_q, device=device, dtype=torch.int32)
    # Interval 0: global prefix [0, min(K2, i+1))
    col_max0 = torch.minimum(idx + 1, torch.full_like(idx, sla_k2))
    # Interval 1: local causal [max(K2, i-K1+1), i+1)
    col_min0 = torch.clamp(idx - sla_k1 + 1, min=sla_k2)
    col_max1 = idx + 1
    # Kernel reads interleaved: [max(0), min(0), max(1)]
    row = torch.stack([col_max0, col_min0, col_max1], dim=0)  # (3, Lq)
    return (
        row.view(1, 1, 3, max_seqlen_q)
        .expand(batch_size, nheads, 3, max_seqlen_q)
        .contiguous()
    )


_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)


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

    # SLA mode: use the arbitrary-mask NFUNC path; the func tensor supersedes
    # the fixed window_size / num_contexts / num_targets mask parameters.
    func: Optional[torch.Tensor] = None
    if sla_k1 > 0 or sla_k2 > 0:
        batch_size = seq_offsets.size(0) - 1
        nheads = q.size(1)
        func = build_sla_func_tensor(
            batch_size, nheads, max_seq_len, sla_k1, sla_k2, q.device
        )
        # Under SLA the mask is fully described by the func tensor, so we
        # use full causal window and disable context/target fixed masks.
        window_size_left, window_size_right = -1, 0
        num_contexts_tensor: Optional[torch.Tensor] = None
        num_targets_int32: Optional[torch.Tensor] = None
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
