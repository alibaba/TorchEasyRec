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

from typing import List, Optional

import torch


@torch.library.custom_op("tzrec::cutlass_hstu_mha_fwd", mutates_args=())
def _cutlass_hstu_mha_fwd(
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
        None,  # func
        None,  # kv_cache
        None,  # page_offsets
        None,  # page_ids
        None,  # last_page_lens
        None,  # cu_seqlens_t
    )
    return out[:, :, :head_dim].reshape(-1, num_heads, head_dim).contiguous()


@_cutlass_hstu_mha_fwd.register_fake
def _(
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
) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("tzrec::cutlass_hstu_mha_bwd", mutates_args=())
def _cutlass_hstu_mha_bwd(
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
        None,  # func
        False,  # deterministic
    )
    return [dq, dk, dv]


@_cutlass_hstu_mha_bwd.register_fake
def _(
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
) -> List[torch.Tensor]:
    return [torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)]


def _setup_context(ctx, inputs, output):
    (
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
    ) = inputs
    ctx.save_for_backward(q, k, v, cu_seqlens, num_contexts, num_targets)
    ctx.max_seq_len = max_seq_len
    ctx.scaling_seqlen = scaling_seqlen
    ctx.target_group_size = target_group_size
    ctx.window_size_left = window_size_left
    ctx.window_size_right = window_size_right
    ctx.alpha = alpha


def _backward(ctx, grad_output):
    q, k, v, cu_seqlens, num_contexts, num_targets = ctx.saved_tensors
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
    )
    return (
        dq,  # q
        dk,  # k
        dv,  # v
        None,  # cu_seqlens
        None,  # max_seq_len
        None,  # scaling_seqlen
        None,  # num_contexts
        None,  # num_targets
        None,  # target_group_size
        None,  # window_size_left
        None,  # window_size_right
        None,  # alpha
    )


_cutlass_hstu_mha_fwd.register_autograd(_backward, setup_context=_setup_context)


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
) -> torch.Tensor:
    """CUTLASS-based HSTU multi-head attention.

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

    Returns:
        output tensor of shape (total, nheads, hidden_dim).
    """
    if q.shape[2] != v.shape[2]:
        raise ValueError(
            f"CUTLASS hstu_attn requires attention_dim == hidden_dim, "
            f"got q.shape[2]={q.shape[2]} != v.shape[2]={v.shape[2]}"
        )
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"CUTLASS hstu_attn only supports fp16 and bf16, got {q.dtype}. "
            f"Set train_config.mixed_precision to 'BF16' or 'FP16'."
        )

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    seq_offsets = seq_offsets.contiguous()
    cu_seqlens = seq_offsets.to(torch.int32)

    if causal:
        if max_attn_len > 0:
            window_size_left, window_size_right = max_attn_len, 0
        else:
            window_size_left, window_size_right = -1, 0
    else:
        window_size_left, window_size_right = -1, -1

    num_contexts_tensor: Optional[torch.Tensor] = None
    if contextual_seq_len > 0:
        batch_size = seq_offsets.size(0) - 1
        num_contexts_tensor = torch.full(
            (batch_size,),
            contextual_seq_len,
            dtype=torch.int32,
            device=q.device,
        )

    num_targets_int32: Optional[torch.Tensor] = None
    if num_targets is not None:
        num_targets_int32 = num_targets.to(torch.int32)

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
    )
