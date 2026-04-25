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


def _assert_attn_func_shape(func: Optional[torch.Tensor], q: torch.Tensor) -> None:
    """Mirror cutlass_hstu_mha's attn_func validation in fake/meta land."""
    if func is None:
        return
    torch._assert(func.dtype == torch.int32, "attn_func must be int32")
    torch._assert(func.dim() == 3, "attn_func must be 3-D")
    torch._assert(
        func.shape[0] == q.shape[1]
        and func.shape[1] == 3
        and func.shape[2] == q.shape[0],
        "attn_func must have shape (nheads, 3, total_q)",
    )
    torch._assert(
        func.device == q.device,
        "attn_func must be on the same device as q",
    )


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
    _assert_attn_func_shape(func, q)
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
    _assert_attn_func_shape(func, q)
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
    attn_func: Optional[torch.Tensor] = None,
    scaling_seqlen: int = -1,
) -> torch.Tensor:
    """CUTLASS-based HSTU multi-head attention.

    Supports two mask modes:

    - Standard fixed-mask path (``attn_func=None``): causal/local/
      context/target masking driven by ``causal``, ``max_attn_len``,
      ``contextual_seq_len`` and ``num_targets``.
    - Arbitrary-mask NFUNC path (``attn_func`` provided): the kernel
      interprets ``attn_func`` as a jagged ``(nheads, 3, total_q)`` int32
      tensor encoding two disjoint column-intervals per query row. The
      caller is responsible for constructing it (see
      ``build_sla_func_tensor`` for the SLA case). In this mode the
      kernel forces ``window_size_left=-1, window_size_right=0`` so
      ``causal`` and ``max_attn_len`` are redundant and must be left at
      their defaults (``causal=True, max_attn_len=0``).

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
        causal: whether to apply causal masking (fixed-mask path only).
        num_targets: number of target tokens per batch element.
        max_attn_len: maximum attention window length (fixed-mask path;
            0 means unlimited).
        contextual_seq_len: number of contextual tokens per sequence.
        attn_func: pre-built arbitrary-mask func tensor of shape
            ``(nheads, 3, total_q)``, int32. When provided, selects the
            NFUNC mask path; ``causal`` and ``max_attn_len`` must be at
            defaults.
        scaling_seqlen: divisor used to scale the attention output inside
            the kernel. ``-1`` (default) falls back to ``max_seq_len`` so
            the behavior matches the legacy code path.

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
            f"CUTLASS hstu_attn supports fp16 and bf16, got {q.dtype}. "
            f"Set train_config.mixed_precision to 'BF16' or 'FP16'."
        )
    if contextual_seq_len < 0:
        raise ValueError(
            f"contextual_seq_len must be non-negative; got {contextual_seq_len}"
        )
    if attn_func is not None:
        if not causal:
            raise ValueError(
                "attn_func requires causal=True; the NFUNC mask path "
                "forces window_size_right=0 so causal=False has no effect."
            )
        if max_attn_len > 0:
            raise ValueError(
                f"attn_func is mutually exclusive with max_attn_len "
                f"(got max_attn_len={max_attn_len}); any local window "
                "must be encoded by the func tensor itself."
            )
        # The CUDA kernel addresses ``attn_func`` via pointer arithmetic
        # (``func_ptr + cu_seqlens[b]``) assuming a contiguous int32
        # ``(nheads, 3, total_q)`` jagged layout on the same device as q.
        # A caller that bypasses ``build_sla_func_tensor`` (mismatched
        # total_q, int64 dtype, a permuted view, or a CPU tensor) would
        # otherwise produce OOB reads or silent NaNs with no upstream
        # signal -- catch it here.
        torch._assert(
            attn_func.dtype == torch.int32,
            "attn_func must be int32",
        )
        torch._assert(attn_func.dim() == 3, "attn_func must be 3-D")
        torch._assert(
            attn_func.shape[0] == q.shape[1]
            and attn_func.shape[1] == 3
            and attn_func.shape[2] == q.shape[0],
            "attn_func must have shape (nheads, 3, total_q)",
        )
        torch._assert(
            attn_func.device == q.device,
            "attn_func must be on the same device as q",
        )
        attn_func = attn_func.contiguous()

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    cu_seqlens = seq_offsets.to(torch.int32)

    num_targets_int32: Optional[torch.Tensor] = None
    if num_targets is not None:
        num_targets_int32 = num_targets.to(torch.int32)

    num_contexts_tensor: Optional[torch.Tensor] = None
    if contextual_seq_len > 0:
        batch_size = seq_offsets.size(0) - 1
        num_contexts_tensor = torch.full(
            (batch_size,),
            contextual_seq_len,
            dtype=torch.int32,
            device=q.device,
        )

    if attn_func is not None:
        # NFUNC mask path: the func tensor fully describes the mask. The
        # kernel forces window_size_left=-1, window_size_right=0; causal /
        # max_attn_len are already validated to be at defaults.
        window_size_left, window_size_right = -1, 0
    else:
        # Fixed-mask path (unchanged from PR #465).
        if causal:
            if max_attn_len > 0:
                window_size_left, window_size_right = max_attn_len, 0
            else:
                window_size_left, window_size_right = -1, 0
        else:
            window_size_left, window_size_right = -1, -1

    if scaling_seqlen == -1:
        scaling_seqlen = max_seq_len

    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        return _CutlassHstuMhaFunction.apply(
            q,
            k,
            v,
            cu_seqlens,
            max_seq_len,
            scaling_seqlen,
            num_contexts_tensor,
            num_targets_int32,
            1,  # target_group_size
            window_size_left,
            window_size_right,
            alpha,
            attn_func,
        )
    return torch.ops.tzrec.cutlass_hstu_mha_fwd(
        q,
        k,
        v,
        cu_seqlens,
        max_seq_len,
        scaling_seqlen,
        num_contexts_tensor,
        num_targets_int32,
        1,  # target_group_size
        window_size_left,
        window_size_right,
        alpha,
        attn_func,
    )
