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

from typing import Optional

import torch
from hstu_attn import hstu_attn_varlen_func


def _needs_triton_fallback(
    max_attn_len: int,
    contextual_seq_len: int,
    num_targets: Optional[torch.Tensor],
) -> bool:
    """Check if we need to fall back to triton.

    The CUTLASS kernel does not support combining local window attention
    (max_attn_len > 0) with context or target masking.
    """
    has_local_window = max_attn_len > 0
    has_context_or_target = contextual_seq_len > 0 or num_targets is not None
    return has_local_window and has_context_or_target


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
    if _needs_triton_fallback(max_attn_len, contextual_seq_len, num_targets):
        from tzrec.ops._triton.triton_hstu_attention import triton_hstu_mha

        return triton_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )

    assert q.shape[2] == v.shape[2], (
        f"CUTLASS hstu_attn requires attention_dim == hidden_dim, "
        f"got q.shape[2]={q.shape[2]} != v.shape[2]={v.shape[2]}"
    )

    cu_seqlens = seq_offsets.to(torch.int32)

    if causal:
        if max_attn_len > 0:
            window_size = (max_attn_len, 0)
        else:
            window_size = (-1, 0)
    else:
        window_size = (-1, -1)

    num_contexts = None
    if contextual_seq_len > 0:
        batch_size = seq_offsets.size(0) - 1
        num_contexts = torch.full(
            (batch_size,),
            contextual_seq_len,
            dtype=torch.int32,
            device=q.device,
        )

    if num_targets is not None:
        num_targets = num_targets.to(torch.int32)

    return hstu_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        max_seq_len,
        max_seq_len,
        num_contexts=num_contexts,
        num_targets=num_targets,
        window_size=window_size,
        alpha=alpha,
        scaling_seqlen=max_seq_len,
    )


@torch.fx.wrap
def cutlass_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    enable_tma: bool = False,
) -> torch.Tensor:
    """Cached HSTU attention for delta queries.

    Falls back to Triton implementation since the CUTLASS kernel does not
    support the delta-query pattern with separate q/k sequence lengths.
    """
    from tzrec.ops._triton.triton_hstu_attention import triton_cached_hstu_mha

    return triton_cached_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        enable_tma=enable_tma,
    )
