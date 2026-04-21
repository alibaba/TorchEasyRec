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

# We use the hstu_attention ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import functools
from typing import Optional

import torch
from torch.fx._symbolic_trace import is_fx_tracing

from tzrec.ops import Kernel
from tzrec.ops._pytorch.pt_hstu_attention import (
    pytorch_cached_hstu_mha,
    pytorch_hstu_mha,
)
from tzrec.ops.utils import switch_to_contiguous_if_needed
from tzrec.utils.logging_util import logger


@functools.lru_cache(maxsize=1)
def _warn_cutlass_fallback_local_target() -> None:
    logger.warning(
        "hstu_mha: requested kernel=Kernel.CUTLASS with max_attn_len>0 "
        "combined with contextual_seq_len>0 or num_targets!=None -- the "
        "CUTLASS Is_local dispatch does not support context/target masks "
        "in this combination. Falling back to Kernel.TRITON for this run. "
        "Benchmarks / memory reports for CUTLASS will not reflect this "
        "call site."
    )


def hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
    sort_by_length: bool = False,
    kernel: Kernel = Kernel.PYTORCH,
    enable_tma: bool = False,
    attn_func: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """HSTU multi-head attention with kernel backend dispatch.

    Args:
        max_seq_len: maximum sequence length in the batch.
        alpha: scaling factor for attention scores.
        q: query tensor of shape (total, nheads, attn_dim).
        k: key tensor of shape (total, nheads, attn_dim).
        v: value tensor of shape (total, nheads, hidden_dim).
        seq_offsets: cumulative sequence offsets (batch_size + 1,).
        causal: whether to apply causal masking.
        dropout_pr: dropout probability (PYTORCH backend only).
        training: whether in training mode (PYTORCH backend only).
        num_targets: number of target tokens per batch element.
        max_attn_len: max attention window length (0 = unlimited).
        contextual_seq_len: number of contextual tokens per sequence.
        min_full_attn_seq_len: min seq len for full attention (PYTORCH only).
        sort_by_length: sort sequences by length (TRITON only).
        kernel: backend kernel to use (PYTORCH, TRITON, CUTLASS).
        enable_tma: enable TMA (TRITON only).
        attn_func: pre-built arbitrary-mask func tensor of shape
            ``(nheads, 3, total_q)``, int32 — selects the CUTLASS NFUNC
            mask path. Only supported when ``kernel=Kernel.CUTLASS``.

    Returns:
        output tensor of shape (total, nheads, hidden_dim).
    """
    _, H, _ = q.shape
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(q.dim() == 3, "q must be 3-D")
        torch._assert(k.shape == q.shape, "k must be the same shape as q")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[0] == q.shape[0], "wrong v shape[0]")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")
        torch._assert(causal, "only support causal attention")

    if attn_func is not None and kernel == Kernel.TRITON:
        raise ValueError(
            "attn_func (arbitrary-mask NFUNC path) is not supported on "
            "Kernel.TRITON. Use Kernel.CUTLASS for production training or "
            "Kernel.PYTORCH for the reference implementation."
        )

    if kernel == Kernel.CUTLASS and attn_func is None:
        # Without an arbitrary mask, the CUTLASS kernel's local-window path
        # (Is_local) cannot combine with context/target masking — fall back
        # to Triton in that combination.
        _has_local_window = max_attn_len > 0
        _has_ctx_or_tgt = contextual_seq_len > 0 or num_targets is not None
        if _has_local_window and _has_ctx_or_tgt:
            kernel = Kernel.TRITON
            _warn_cutlass_fallback_local_target()

    if kernel == Kernel.CUTLASS:
        # cutlass_hstu_mha is @torch.fx.wrap'd; FX treats it as a leaf so
        # we call it directly without going through the contiguous/assertion
        # preprocessing block below (which has control flow that would
        # break FX symbolic tracing). The CUTLASS kernel requires fp16/bf16
        # inputs; we rely on the CudaAutocastWrapper applied in
        # tzrec/acc/aot_utils.py and trt_utils.py (driven by
        # export_config.mixed_precision / train_config.mixed_precision) to
        # ensure q/k/v are in a supported dtype when reaching this dispatch.
        from tzrec.ops._cuda.cutlass_hstu_attention import cutlass_hstu_mha

        return cutlass_hstu_mha(
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
            attn_func=attn_func,
        )

    if kernel == Kernel.TRITON:
        if not is_fx_tracing():
            torch._assert(q.is_cuda, "q must be CUDA tensor")
            torch._assert(k.is_cuda, "k must be CUDA tensor")
            torch._assert(v.is_cuda, "v must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            torch._assert(dropout_pr < 1e-6, "dropout for triton path not implemented")
            torch._assert(
                min_full_attn_seq_len == 0, "min_full_attn_seq_len not implemented"
            )
        q = switch_to_contiguous_if_needed(q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)
        seq_offsets = seq_offsets.contiguous()

    if kernel == Kernel.TRITON:
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
            sort_by_length=sort_by_length,
            enable_tma=enable_tma,
        )
    else:
        return pytorch_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            dropout_pr=dropout_pr,
            training=training,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
            attn_func=attn_func,
        )


def delta_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    kernel: Kernel = Kernel.PYTORCH,
    enable_tma: bool = False,
) -> torch.Tensor:
    if kernel == Kernel.CUTLASS:
        kernel = Kernel.TRITON
    L, H, D = delta_q.shape
    B = seq_offsets.size(0) - 1
    DeltaSize = L // B  # NOQA
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(delta_q.dim() == 3, "delta_q must be 3-D")
        torch._assert(L % B == 0, "delta_q must be padded")
        torch._assert(k.dim() == 3, "k must be 3-D")
        torch._assert(k.shape[1] == H, "wrong k shape[1]")
        torch._assert(k.shape[2] == D, "wrong k shape[2]")
        torch._assert(v.dim() == 3, "v must be 3-D")
        torch._assert(v.shape[1] == H, "wrong v shape[1]")

    if kernel in [Kernel.TRITON]:
        if not is_fx_tracing():
            torch._assert(delta_q.is_cuda, "q must be CUDA tensor")
            torch._assert(seq_offsets.is_cuda, "seq_offsets must be CUDA tensor")
            if num_targets is not None:
                torch._assert(num_targets.is_cuda, "num_targets must be CUDA tensor")
        seq_offsets = seq_offsets.contiguous()
        delta_q = switch_to_contiguous_if_needed(delta_q)
        k = switch_to_contiguous_if_needed(k)
        v = switch_to_contiguous_if_needed(v)

    if kernel == Kernel.TRITON:
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
    else:
        return pytorch_cached_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
        )
