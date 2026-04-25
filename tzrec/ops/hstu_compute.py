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

# We use the hstu_compute ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.fx._symbolic_trace import is_fx_tracing

from tzrec.ops import Kernel
from tzrec.ops._pytorch.pt_hstu_linear import (
    pytorch_hstu_compute_output,
)
from tzrec.ops.hstu_attention import hstu_mha
from tzrec.ops.layer_norm import layer_norm
from tzrec.ops.mm import addmm
from tzrec.utils.logging_util import logger


@functools.lru_cache(maxsize=1)
def _warn_cutlass_to_triton_uqvk() -> None:
    logger.warning(
        "hstu_compute_uqvk: Kernel.CUTLASS has no CUTLASS implementation "
        "for UQVK projection; silently using Kernel.TRITON for the LN + "
        "addmm primitives. Rematerialization flags "
        "(recompute_uvqk_in_backward / recompute_normed_x_in_backward) "
        "still apply. This warning is emitted once per process."
    )


@functools.lru_cache(maxsize=1)
def _warn_cutlass_to_triton_output() -> None:
    logger.warning(
        "hstu_compute_output: Kernel.CUTLASS has no CUTLASS implementation "
        "for the output projection; silently using Kernel.TRITON. This "
        "warning is emitted once per process."
    )


def _split_silu(
    uvqk: torch.Tensor,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split UVQK into u, v, q, k and apply SiLU on u."""
    u, v, q, k = torch.split(
        uvqk,
        [
            hidden_dim * num_heads,
            hidden_dim * num_heads,
            attn_dim * num_heads,
            attn_dim * num_heads,
        ],
        dim=1,
    )
    u = F.silu(u)
    q = q.view(-1, num_heads, attn_dim)
    k = k.view(-1, num_heads, attn_dim)
    v = v.view(-1, num_heads, hidden_dim)
    return u, q, k, v


def _ln_then_addmm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: Kernel,
) -> torch.Tensor:
    """LN + addmm, returning UVQK only.

    Used inside checkpoint for the recompute_normed_x_only case so normed_x
    isn't saved as addmm's activation.
    """
    normed_x = layer_norm(
        x, weight=norm_weight, bias=norm_bias, eps=norm_eps, kernel=kernel
    )
    # NOTE: for AMD training, we go with torch.addmm instead of the triton
    # version before Triton on AMD achieves on-par perf with NV GPU.
    if torch.version.hip and kernel == Kernel.TRITON:
        return torch.addmm(uvqk_bias, normed_x, uvqk_weight)
    return addmm(uvqk_bias, normed_x, uvqk_weight, kernel)


def _addmm_split_silu(
    normed_x: torch.Tensor,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: Kernel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Addmm + split + SiLU.

    Used inside checkpoint for the recompute_uvqk_only case so UVQK isn't
    saved as a forward activation.
    """
    if torch.version.hip and kernel == Kernel.TRITON:
        uvqk = torch.addmm(uvqk_bias, normed_x, uvqk_weight)
    else:
        uvqk = addmm(uvqk_bias, normed_x, uvqk_weight, kernel)
    return _split_silu(uvqk, num_heads, attn_dim, hidden_dim)


def _hstu_compute_uqvk_impl(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: Kernel,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla LN + addmm + split + silu (no rematerialization)."""
    normed_x = layer_norm(
        x, weight=norm_weight, bias=norm_bias, eps=norm_eps, kernel=kernel
    )
    return _addmm_split_silu(
        normed_x, num_heads, attn_dim, hidden_dim, uvqk_weight, uvqk_bias, kernel
    )


def hstu_compute_uqvk(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    kernel: Kernel = Kernel.PYTORCH,
    recompute_normed_x_in_backward: bool = False,
    recompute_uvqk_in_backward: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """LN + UVQK projection with selective rematerialization.

    Mirrors the Triton fused kernel's per-flag granularity via four
    distinct checkpoint placements:

    - (False, False): vanilla — both normed_x and UVQK are saved.
    - (True,  False): checkpoint LN+addmm → normed_x is recomputed
      (cannot be saved as addmm's activation since addmm is inside the
      checkpoint), UVQK is saved (downstream split/silu/attention
      consumers reference it).
    - (False, True ): LN runs vanilla saving normed_x, then checkpoint
      addmm+split+silu → UVQK is recomputed in backward.
    - (True,  True ): checkpoint the full LN+addmm+split+silu chain →
      both recomputed.

    The flags are no-ops outside ``torch.is_grad_enabled()`` (inference).
    """
    if kernel == Kernel.CUTLASS:
        _warn_cutlass_to_triton_uqvk()
        kernel = Kernel.TRITON
    use_remat = torch.is_grad_enabled()
    rn = recompute_normed_x_in_backward and use_remat
    ru = recompute_uvqk_in_backward and use_remat

    if rn and ru:
        return torch.utils.checkpoint.checkpoint(
            _hstu_compute_uqvk_impl,
            x,
            norm_weight,
            norm_bias,
            norm_eps,
            num_heads,
            attn_dim,
            hidden_dim,
            uvqk_weight,
            uvqk_bias,
            kernel,
            use_reentrant=False,
        )

    if rn:
        uvqk = torch.utils.checkpoint.checkpoint(
            _ln_then_addmm,
            x,
            norm_weight,
            norm_bias,
            norm_eps,
            uvqk_weight,
            uvqk_bias,
            kernel,
            use_reentrant=False,
        )
        return _split_silu(uvqk, num_heads, attn_dim, hidden_dim)

    if ru:
        normed_x = layer_norm(
            x, weight=norm_weight, bias=norm_bias, eps=norm_eps, kernel=kernel
        )
        return torch.utils.checkpoint.checkpoint(
            _addmm_split_silu,
            normed_x,
            num_heads,
            attn_dim,
            hidden_dim,
            uvqk_weight,
            uvqk_bias,
            kernel,
            use_reentrant=False,
        )

    return _hstu_compute_uqvk_impl(
        x,
        norm_weight,
        norm_bias,
        norm_eps,
        num_heads,
        attn_dim,
        hidden_dim,
        uvqk_weight,
        uvqk_bias,
        kernel,
    )


def hstu_compute_output(
    attn: torch.Tensor,
    u: torch.Tensor,
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    output_weight: torch.Tensor,
    num_heads: int,
    linear_dim: int,
    dropout_ratio: float,
    training: bool,
    concat_ux: bool,
    group_norm: bool,
    recompute_y_in_backward: bool,
    kernel: Kernel = Kernel.PYTORCH,
) -> torch.Tensor:
    if kernel == Kernel.CUTLASS:
        _warn_cutlass_to_triton_output()
        kernel = Kernel.TRITON
    if kernel == Kernel.TRITON:
        from tzrec.ops._triton.triton_hstu_linear import (
            triton_hstu_compute_output,
        )

        return triton_hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            output_weight=output_weight,
            eps=norm_eps,
            dropout_ratio=dropout_ratio,
            training=training,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=linear_dim,
            seed=None,
            recompute_y_in_backward=recompute_y_in_backward,
        )
    else:
        return pytorch_hstu_compute_output(
            attn=attn,
            u=u,
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            output_weight=output_weight,
            eps=norm_eps,
            dropout_ratio=dropout_ratio,
            training=training,
            concat_ux=concat_ux,
            group_norm=group_norm,
            num_heads=num_heads,
            linear_dim=linear_dim,
        )


def hstu_preprocess_and_attention(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    norm_eps: float,
    num_heads: int,
    attn_dim: int,
    hidden_dim: int,
    uvqk_weight: torch.Tensor,
    uvqk_bias: torch.Tensor,
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    attn_alpha: float,
    causal: bool,
    num_targets: Optional[torch.Tensor],
    max_attn_len: int,
    contextual_seq_len: int,
    recompute_uvqk_in_backward: bool,
    recompute_normed_x_in_backward: bool,
    sort_by_length: bool,
    prefill: bool = False,
    kernel: Kernel = Kernel.PYTORCH,
    enable_tma: bool = False,
    attn_func: Optional[torch.Tensor] = None,
    scaling_seqlen: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    if not is_fx_tracing():
        torch._assert(max_seq_len > 0, "max_seq_len must be larger than 0")
        torch._assert(x.dim() == 2, "x must be 2-D")
        torch._assert(
            x.shape[1] == uvqk_weight.shape[0],
            "x.shape[1] must equal uvqk_weight.shape[0]",
        )
        torch._assert(
            uvqk_weight.shape[1] == 2 * num_heads * (hidden_dim + attn_dim),
            "uvqk_weight.shape[1] must equal 2 * num_heads * (hidden_dim + attn_dim)",
        )
    if kernel == Kernel.TRITON and prefill is False:
        # The fused Triton preprocess+attention does not implement the
        # NFUNC arbitrary-mask path.  Defense-in-depth: today STUStack
        # only builds attn_func for SLA, which is gated to CUTLASS or
        # PYTORCH (so this branch is unreachable from the stack), but a
        # direct caller could otherwise pass attn_func and silently get
        # the wrong attention.
        if attn_func is not None:
            raise ValueError(
                "attn_func (NFUNC mask path) is not supported on the "
                "fused Triton preprocess+attention path; use "
                "kernel=Kernel.CUTLASS or kernel=Kernel.PYTORCH, or "
                "split the call into separate preprocess and hstu_mha "
                "(prefill=True)."
            )
        from tzrec.ops._triton.triton_hstu_preprocess_and_attention import (
            triton_hstu_preprocess_and_attention,
        )

        u, attn_output = triton_hstu_preprocess_and_attention(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=num_heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            attn_alpha=attn_alpha,
            causal=causal,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            sort_by_length=sort_by_length,
            enable_tma=enable_tma,
            scaling_seqlen=scaling_seqlen,
        )
        attn_output = attn_output.view(-1, hidden_dim * num_heads)
        k = None
        v = None
    else:
        # The Triton fused path honors the recompute flags natively; for the
        # non-fused (CUTLASS, PyTorch) path, hstu_compute_uqvk applies the
        # equivalent rematerialization via torch.utils.checkpoint internally.
        # The output side (Y) is handled separately by
        # recompute_y_in_backward in hstu_compute_output.
        u, q, k, v = hstu_compute_uqvk(
            x=x,
            norm_weight=norm_weight,
            norm_bias=norm_bias,
            norm_eps=norm_eps,
            num_heads=num_heads,
            attn_dim=attn_dim,
            hidden_dim=hidden_dim,
            uvqk_weight=uvqk_weight,
            uvqk_bias=uvqk_bias,
            kernel=kernel,
            recompute_normed_x_in_backward=recompute_normed_x_in_backward,
            recompute_uvqk_in_backward=recompute_uvqk_in_backward,
        )
        attn_output = hstu_mha(
            max_seq_len=max_seq_len,
            alpha=attn_alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=causal,
            dropout_pr=0.0,
            training=False,
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            sort_by_length=sort_by_length,
            kernel=kernel,
            attn_func=attn_func,
            scaling_seqlen=scaling_seqlen,
        ).view(-1, hidden_dim * num_heads)
    return u, attn_output, k, v
