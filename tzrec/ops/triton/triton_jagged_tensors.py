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


from typing import List, Optional, Tuple

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.runtime.autotuner import autotune as triton_autotune

from tzrec.ops.utils import autotune_max_seq_len, switch_to_contiguous_if_needed
from tzrec.utils.fx_util import fx_int_item

torch.fx.wrap(fx_int_item)


def _get_bmm_configs() -> List[triton.Config]:
    configs = []
    for BLOCK_M in [64, 128]:
        for BLOCK_N in [64, 128]:
            for BLOCK_K in [32, 64]:
                for num_stages in [2, 3]:
                    for num_warps in [4, 8]:
                        configs.append(
                            triton.Config(
                                {
                                    "BLOCK_M": BLOCK_M,
                                    "BLOCK_N": BLOCK_N,
                                    "BLOCK_K": BLOCK_K,
                                },
                                num_stages=num_stages,
                                num_warps=num_warps,
                            )
                        )
    return configs


@triton.jit
def _concat_2D_jagged(
    ValuesA,
    ValuesB,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
    Out,
    D,
    stride_ad,
    stride_bd,
    stride_od,
    n_prefix_from_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    offs_d = tl.arange(0, BLOCK_D)
    out_seq_start = seq_start_a + seq_start_b + off_n
    out_ptrs = Out + out_seq_start.to(tl.int64) * stride_od + offs_d
    if off_n < n_prefix_from_B:
        in_ptrs = ValuesB + (off_n + seq_start_b).to(tl.int64) * stride_bd + offs_d
    elif off_n < seq_len_a + n_prefix_from_B:
        in_ptrs = (
            ValuesA
            + (off_n - n_prefix_from_B + seq_start_a).to(tl.int64) * stride_ad
            + offs_d
        )
    else:
        in_ptrs = (
            ValuesB
            + (off_n - seq_len_a + seq_start_b).to(tl.int64) * stride_bd
            + offs_d
        )
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton.jit
def _split_2D_jagged(
    JaggedIn,
    OffsetsA,
    OffsetsB,
    MaxLenA,
    MaxLenB,
    OutA,
    OutB,
    D,
    stride_id,
    stride_ad,
    stride_bd,
    n_prefix_to_B,
    IS_DENSE_A: tl.constexpr,
    IS_DENSE_B: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    off_z = tl.program_id(1)
    off_n = tl.program_id(0)
    if IS_DENSE_A:
        seq_start_a = off_z * MaxLenA
        seq_len_a = MaxLenA
    else:
        seq_start_a = tl.load(OffsetsA + off_z)
        seq_end_a = tl.load(OffsetsA + off_z + 1)
        seq_len_a = seq_end_a - seq_start_a
    if IS_DENSE_B:
        seq_start_b = off_z * MaxLenB
        seq_len_b = MaxLenB
    else:
        seq_start_b = tl.load(OffsetsB + off_z)
        seq_end_b = tl.load(OffsetsB + off_z + 1)
        seq_len_b = seq_end_b - seq_start_b
    seq_len = seq_len_a + seq_len_b
    if off_n >= seq_len:
        return
    seq_start = seq_start_a + seq_start_b
    offs_d = tl.arange(0, BLOCK_D)
    in_ptrs = JaggedIn + (seq_start + off_n).to(tl.int64) * stride_id + offs_d
    if off_n < n_prefix_to_B:
        out_ptrs = OutB + (off_n + seq_start_b).to(tl.int64) * stride_bd + offs_d
    elif off_n < seq_len_a + n_prefix_to_B:
        out_ptrs = (
            OutA
            + (off_n - n_prefix_to_B + seq_start_a).to(tl.int64) * stride_ad
            + offs_d
        )
    else:
        out_ptrs = (
            OutB + (off_n - seq_len_a + seq_start_b).to(tl.int64) * stride_bd + offs_d
        )
    v = tl.load(in_ptrs, mask=offs_d < D)
    tl.store(out_ptrs, v, mask=offs_d < D)


@triton_autotune(
    configs=_get_bmm_configs(),
    key=["AUTOTUNE_MAX_SEQ_LEN", "N", "K"],
)
@triton.jit
def jagged_dense_bmm_broadcast_add_kernel(
    seq_offsets,
    Jagged,
    Dense,
    Bias,
    Out,
    AUTOTUNE_MAX_SEQ_LEN,
    N,
    K,
    stride_jm,
    stride_db,
    stride_dk,
    stride_dn,
    stride_bias_b,
    stride_om,
    HAS_BIAS: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Computing bmm Out = Jagged x Dense + Bias.

    M is the jagged dimension
    Jagged has shape (sum_B(M_i), K), Dense has shape (B, K, N), Bias has shape (B, N),
    and Out has shape (sum_B(M_i), N)
    """
    off_n = tl.program_id(0)
    off_m = tl.program_id(1)
    off_b = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start
    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N
    if start_m >= seq_len:
        return

    Jagged += seq_start * stride_jm
    Dense += off_b.to(tl.int64) * stride_db
    Out += seq_start * stride_om

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    jg_ptrs = Jagged + offs_m[:, None] * stride_jm + offs_k[None, :]
    dn_ptrs = Dense + offs_k[:, None] * stride_dk + offs_n[None, :] * stride_dn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        jg = tl.load(
            jg_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < seq_len) and ((k + offs_k)[None, :] < K),
            other=0.0,
        )
        dn = tl.load(
            dn_ptrs,
            mask=((k + offs_k)[:, None] < K) and (offs_n[None, :] < N),
            other=0.0,
        )
        accumulator += tl.dot(jg, dn, allow_tf32=ALLOW_TF32)
        jg_ptrs += BLOCK_K
        dn_ptrs += BLOCK_K * stride_dk

    if HAS_BIAS:
        bias_ptrs = Bias + off_b * stride_bias_b + offs_n
        bias = tl.load(bias_ptrs, mask=offs_n < N)
        accumulator += bias[None, :].to(tl.float32)

    out = accumulator.to(Out.dtype.element_ty)

    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < seq_len) & (offs_n[None, :] < N))


@triton_autotune(
    configs=_get_bmm_configs(),
    key=["M", "N", "AUTOTUNE_MAX_SEQ_LEN"],
)
@triton.jit
def _jagged_jagged_bmm_reduce_sum(
    seq_offsets,
    JaggedA,
    JaggedB,
    Out,
    ReduceOut,
    M,
    N,
    AUTOTUNE_MAX_SEQ_LEN,
    stride_ak,
    stride_bk,
    stride_ob,
    stride_om,
    stride_on,
    stride_orb,
    stride_orn,
    REDUCE_JAGGEDB: tl.constexpr,
    ALLOW_TF32: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Computing bmm Out = Jagged x Jagged.

    K is the jagged dimension
    JaggedA has shape (sum_B(K_i), M), JaggedB has shape (sum_B(K_i), N),
    and Out has shape (B, M, N)
    """
    off_b = tl.program_id(0)
    off_m = tl.program_id(1)
    off_n = tl.program_id(2)

    seq_start = tl.load(seq_offsets + off_b).to(tl.int64)
    seq_end = tl.load(seq_offsets + off_b + 1)
    seq_len = seq_end - seq_start

    start_m = off_m * BLOCK_M
    start_n = off_n * BLOCK_N

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Out += off_b.to(tl.int64) * stride_ob
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_n = start_n + tl.arange(0, BLOCK_N)
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    if REDUCE_JAGGEDB:
        out_reduce_ptrs = ReduceOut + off_b * stride_orb + offs_n * stride_orn
        acc_reduce = tl.zeros((BLOCK_N,), dtype=tl.float32)
    if seq_len == 0:
        out = accumulator.to(Out.dtype.element_ty)
        tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
        if REDUCE_JAGGEDB:
            if off_m == 0:
                tl.store(
                    out_reduce_ptrs,  # pyre-ignore [61]
                    acc_reduce.to(ReduceOut.dtype.element_ty),
                    mask=(offs_n < N),
                )
        return

    JaggedA += seq_start * stride_ak
    JaggedB += seq_start * stride_bk
    offs_k = tl.arange(0, BLOCK_K)
    jg_a_ptrs = JaggedA + offs_k[None, :] * stride_ak + offs_m[:, None]
    jg_b_ptrs = JaggedB + offs_k[:, None] * stride_bk + offs_n[None, :]

    for k in range(0, seq_len, BLOCK_K):
        jg_a = tl.load(
            jg_a_ptrs,
            # pyre-fixme[16]: `int` has no attribute `__getitem__`.
            mask=(offs_m[:, None] < M) and ((k + offs_k)[None, :] < seq_len),
            other=0.0,
        )
        jg_b = tl.load(
            jg_b_ptrs,
            mask=(offs_n[None, :] < N) and ((k + offs_k)[:, None] < seq_len),
            other=0.0,
        )

        accumulator += tl.dot(jg_a, jg_b, allow_tf32=ALLOW_TF32)
        if REDUCE_JAGGEDB:
            if off_m == 0:
                acc_reduce += tl.sum(jg_b.to(tl.float32), axis=0)

        jg_a_ptrs += BLOCK_K * stride_ak
        jg_b_ptrs += BLOCK_K * stride_bk

    out = accumulator.to(Out.dtype.element_ty)
    tl.store(out_ptrs, out, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    if REDUCE_JAGGEDB:
        if off_m == 0:
            tl.store(
                out_reduce_ptrs,  # pyre-ignore [61]
                acc_reduce.to(ReduceOut.dtype.element_ty),
                mask=(offs_n < N),
            )


@triton_op("tzrec::triton_concat_2d_jagged_fwd", mutates_args={})
def triton_concat_2d_jagged_fwd(
    values_a: torch.Tensor,
    values_b: torch.Tensor,
    max_len_a: int,
    max_len_b: int,
    offsets_a: Optional[torch.Tensor],
    offsets_b: Optional[torch.Tensor],
    n_prefix_from_B: int,
) -> Tuple[torch.Tensor, int, int, int, bool, bool, int]:
    values_a = switch_to_contiguous_if_needed(values_a)
    values_b = switch_to_contiguous_if_needed(values_b)
    is_dense_a = offsets_a is None
    is_dense_b = offsets_b is None
    total_len_a, D = values_a.shape
    total_len_b, _ = values_b.shape
    if is_dense_a:
        B = total_len_a // max_len_a
    else:
        assert offsets_a is not None
        B = offsets_a.shape[0] - 1
    if is_dense_b:
        B = total_len_b // max_len_b
    else:
        assert offsets_b is not None
        B = offsets_b.shape[0] - 1
    total_seq_len = total_len_a + total_len_b
    max_seq_len = max_len_a + max_len_b
    BLOCK_D = triton.next_power_of_2(D)
    values_out = torch.empty(
        (total_seq_len, D), device=values_a.device, dtype=values_a.dtype
    )
    wrap_triton(_concat_2D_jagged)[(max_seq_len, B)](
        ValuesA=values_a,
        ValuesB=values_b,
        OffsetsA=offsets_a,
        OffsetsB=offsets_b,
        MaxLenA=max_len_a,
        MaxLenB=max_len_b,
        Out=values_out,
        D=D,
        stride_ad=values_a.stride(-2),
        stride_bd=values_b.stride(-2),
        stride_od=values_out.stride(-2),
        n_prefix_from_B=n_prefix_from_B,
        # pyre-ignore[6]
        IS_DENSE_A=is_dense_a,
        # pyre-ignore[6]
        IS_DENSE_B=is_dense_b,
        BLOCK_D=BLOCK_D,
    )
    return values_out, max_seq_len, total_len_a, total_len_b, is_dense_a, is_dense_b, B


class _Concat2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        values_a: torch.Tensor,
        values_b: torch.Tensor,
        max_len_a: int,
        max_len_b: int,
        offsets_a: Optional[torch.Tensor],
        offsets_b: Optional[torch.Tensor],
        n_prefix_from_B: int,
    ):
        values_out, max_seq_len, total_len_a, total_len_b, is_dense_a, is_dense_b, B = (
            triton_concat_2d_jagged_fwd(
                values_a=values_a,
                values_b=values_b,
                max_len_a=max_len_a,
                max_len_b=max_len_b,
                offsets_a=offsets_a,
                offsets_b=offsets_b,
                n_prefix_from_B=n_prefix_from_B,
            )
        )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.total_len_a = total_len_a
        ctx.total_len_b = total_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.max_len_a = max_len_a
        ctx.max_len_b = max_len_b
        ctx.B = B
        ctx.n_prefix_from_B = n_prefix_from_B
        return values_out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        _, D = d_out.shape
        BLOCK_D = triton.next_power_of_2(D)
        d_values_a = torch.zeros(
            (ctx.total_len_a, D), device=d_out.device, dtype=d_out.dtype
        )
        d_values_b = torch.empty(
            (ctx.total_len_b, D), device=d_out.device, dtype=d_out.dtype
        )
        _split_2D_jagged[(ctx.max_seq_len, ctx.B)](
            JaggedIn=d_out,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=ctx.max_len_a,
            MaxLenB=ctx.max_len_b,
            OutA=d_values_a,
            OutB=d_values_b,
            D=D,
            stride_id=d_out.stride(-2),
            stride_ad=d_values_a.stride(-2),
            stride_bd=d_values_b.stride(-2),
            n_prefix_to_B=ctx.n_prefix_from_B,
            BLOCK_D=BLOCK_D,
            IS_DENSE_A=ctx.is_dense_a,
            IS_DENSE_B=ctx.is_dense_b,
        )
        return d_values_a, d_values_b, None, None, None, None, None


@triton_op("tzrec::triton_split_2d_jagged_fwd", mutates_args={})
def triton_split_2d_jagged_fwd(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int],
    total_len_right: Optional[int],
    max_len_a: Optional[int],
    max_len_b: Optional[int],
    offsets_a: Optional[torch.Tensor],
    offsets_b: Optional[torch.Tensor],
    n_prefix_to_B: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, bool, bool, int, int]:
    values = switch_to_contiguous_if_needed(values)
    is_dense_a: bool = offsets_a is None
    is_dense_b: bool = offsets_b is None
    total_seq_len, D = values.shape
    if is_dense_a:
        assert is_dense_b is False
        assert offsets_b is not None
        assert max_len_a is not None
        B = offsets_b.shape[0] - 1
        total_len_a = max_len_a * B
        total_len_b = total_seq_len - total_len_a
    elif is_dense_b:
        assert is_dense_a is False
        assert offsets_a is not None
        assert max_len_b is not None
        B = offsets_a.shape[0] - 1
        total_len_b = max_len_b * B
        total_len_a = total_seq_len - total_len_b
    else:
        assert offsets_a is not None and offsets_b is not None
        B = offsets_a.shape[0] - 1
        if total_len_left is not None and total_len_right is not None:
            assert total_len_left + total_len_right == total_seq_len
            total_len_a = total_len_left
            total_len_b = total_len_right
            torch._check_is_size(total_len_a)
            # torch._check(total_len_a > 0)
            # torch._check(total_len_a < 10**9)
        else:
            total_len_a = int(offsets_a[-1].item())
            total_len_b = values.size(0) - total_len_a
    _, D = values.shape
    BLOCK_D = triton.next_power_of_2(D)
    values_a = torch.empty((total_len_a, D), device=values.device, dtype=values.dtype)
    values_b = torch.empty((total_len_b, D), device=values.device, dtype=values.dtype)
    wrap_triton(_split_2D_jagged)[(max_seq_len, B)](
        JaggedIn=values,
        OffsetsA=offsets_a,
        OffsetsB=offsets_b,
        MaxLenA=max_len_a,
        MaxLenB=max_len_b,
        OutA=values_a,
        OutB=values_b,
        D=D,
        stride_id=values.stride(0),
        stride_ad=values_a.stride(0),
        stride_bd=values_b.stride(0),
        n_prefix_to_B=n_prefix_to_B,
        # pyre-ignore[6]
        IS_DENSE_A=is_dense_a,
        # pyre-ignore[6]
        IS_DENSE_B=is_dense_b,
        BLOCK_D=BLOCK_D,
    )
    return values_a, values_b, total_seq_len, is_dense_a, is_dense_b, B, D


class _Split2DJaggedFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        values: torch.Tensor,
        total_len_left: Optional[int],
        total_len_right: Optional[int],
        max_len_a: Optional[int],
        max_len_b: Optional[int],
        offsets_a: Optional[torch.Tensor],
        offsets_b: Optional[torch.Tensor],
        n_prefix_to_B: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values_a, values_b, total_seq_len, is_dense_a, is_dense_b, B, D = (
            triton_split_2d_jagged_fwd(
                max_seq_len=max_seq_len,
                values=values,
                total_len_left=total_len_left,
                total_len_right=total_len_right,
                max_len_a=max_len_a,
                max_len_b=max_len_b,
                offsets_a=offsets_a,
                offsets_b=offsets_b,
                n_prefix_to_B=n_prefix_to_B,
            )
        )
        ctx.save_for_backward(offsets_a, offsets_b)
        ctx.max_seq_len = max_seq_len
        ctx.total_seq_len = total_seq_len
        ctx.max_len_a = max_len_a
        ctx.max_len_b = max_len_b
        ctx.is_dense_a = is_dense_a
        ctx.is_dense_b = is_dense_b
        ctx.B = B
        ctx.D = D
        ctx.n_prefix_to_B = n_prefix_to_B
        return values_a, values_b

    @staticmethod
    def backward(
        ctx, *d_values
    ) -> Tuple[None, torch.Tensor, None, None, None, None, None, None, None]:
        offsets_a, offsets_b = ctx.saved_tensors
        d_values_a, d_values_b = d_values
        BLOCK_D = triton.next_power_of_2(ctx.D)
        d_jagged_in = torch.empty(
            (ctx.total_seq_len, ctx.D),
            device=d_values_a.device,
            dtype=d_values_a.dtype,
        )
        _concat_2D_jagged[(ctx.max_seq_len, ctx.B)](
            ValuesA=d_values_a,
            ValuesB=d_values_b,
            OffsetsA=offsets_a,
            OffsetsB=offsets_b,
            MaxLenA=ctx.max_len_a,
            MaxLenB=ctx.max_len_b,
            Out=d_jagged_in,
            D=ctx.D,
            stride_ad=d_values_a.stride(-2),
            stride_bd=d_values_b.stride(-2),
            stride_od=d_jagged_in.stride(-2),
            n_prefix_from_B=ctx.n_prefix_to_B,
            IS_DENSE_A=ctx.is_dense_a,
            IS_DENSE_B=ctx.is_dense_b,
            BLOCK_D=BLOCK_D,
        )

        return None, d_jagged_in, None, None, None, None, None, None, None


@triton_op("tzrec::triton_jagged_dense_bmm_broadcast_add_fwd", mutates_args={})
def triton_jagged_dense_bmm_broadcast_add_fwd(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
) -> Tuple[torch.Tensor, int, int, int]:
    jagged = switch_to_contiguous_if_needed(jagged)
    bias = switch_to_contiguous_if_needed(bias)
    L, K = jagged.shape
    B, _, N = dense.shape
    out = torch.empty((L, N), dtype=jagged.dtype, device=jagged.device)

    grid = lambda meta: (  # noqa E731
        triton.cdiv(N, meta["BLOCK_N"]),
        triton.cdiv(max_seq_len, meta["BLOCK_M"]),
        B,
    )

    wrap_triton(jagged_dense_bmm_broadcast_add_kernel)[grid](
        seq_offsets=seq_offsets,
        Jagged=jagged,
        Dense=dense,
        Bias=bias,
        Out=out,
        AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(max_seq_len),
        N=N,
        K=K,
        stride_jm=jagged.stride(0),
        stride_db=dense.stride(0),
        stride_dk=dense.stride(1),
        stride_dn=dense.stride(2),
        stride_bias_b=bias.stride(0),
        stride_om=out.stride(0),
        HAS_BIAS=True,
        ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
    )
    return out, B, K, N


class _JaggedDenseBmmBroadcastAddFunction(torch.autograd.Function):
    @staticmethod
    # pyre-ignore[14]
    def forward(
        ctx,
        max_seq_len: int,
        seq_offsets: torch.Tensor,
        jagged: torch.Tensor,
        dense: torch.Tensor,
        bias: torch.Tensor,
    ):
        out, B, K, N = triton_jagged_dense_bmm_broadcast_add_fwd(
            max_seq_len=max_seq_len,
            seq_offsets=seq_offsets,
            jagged=jagged,
            dense=dense,
            bias=bias,
        )

        ctx.save_for_backward(seq_offsets, jagged, dense)
        ctx.B = B
        ctx.max_seq_len = max_seq_len
        ctx.K = K
        ctx.N = N
        return out

    @staticmethod
    # pyre-ignore[14]
    def backward(
        ctx, d_out: torch.Tensor
    ) -> Tuple[None, None, torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_offsets, jagged, dense = ctx.saved_tensors
        d_jagged = torch.empty_like(jagged)
        d_dense = torch.empty_like(dense)
        d_bias = torch.empty((ctx.B, ctx.N), device=d_out.device, dtype=d_out.dtype)

        grid = lambda meta: (  # noqa E731
            triton.cdiv(ctx.K, meta["BLOCK_N"]),
            triton.cdiv(ctx.max_seq_len, meta["BLOCK_M"]),
            ctx.B,
        )
        jagged_dense_bmm_broadcast_add_kernel[grid](
            seq_offsets=seq_offsets,
            Jagged=d_out,
            Dense=dense,
            Bias=None,
            Out=d_jagged,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            N=ctx.K,
            K=ctx.N,
            stride_jm=d_out.stride(0),
            stride_db=dense.stride(0),
            stride_dk=dense.stride(2),
            stride_dn=dense.stride(1),
            stride_bias_b=0,
            stride_om=d_jagged.stride(0),
            HAS_BIAS=False,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        grid = lambda meta: (  # noqa E731
            ctx.B,
            triton.cdiv(ctx.K, meta["BLOCK_M"]),
            triton.cdiv(ctx.N, meta["BLOCK_N"]),
        )
        _jagged_jagged_bmm_reduce_sum[grid](
            seq_offsets=seq_offsets,
            JaggedA=jagged,
            JaggedB=d_out,
            Out=d_dense,
            ReduceOut=d_bias,
            M=ctx.K,
            N=ctx.N,
            AUTOTUNE_MAX_SEQ_LEN=autotune_max_seq_len(ctx.max_seq_len),
            stride_ak=jagged.stride(0),
            stride_bk=d_out.stride(0),
            stride_ob=d_dense.stride(0),
            stride_om=d_dense.stride(1),
            stride_on=d_dense.stride(2),
            stride_orb=d_bias.stride(0),
            stride_orn=d_bias.stride(1),
            REDUCE_JAGGEDB=True,
            ALLOW_TF32=torch.backends.cuda.matmul.allow_tf32,
        )

        return None, None, d_jagged, d_dense, d_bias


def triton_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
    n_prefix_from_right: int = 0,
) -> torch.Tensor:
    return _Concat2DJaggedFunction.apply(
        values_left,
        values_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
        n_prefix_from_right,
    )


def triton_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    total_len_left: Optional[int],
    total_len_right: Optional[int],
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
    n_prefix_to_right: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return _Split2DJaggedFunction.apply(
        max_seq_len,
        values,
        total_len_left,
        total_len_right,
        max_len_left,
        max_len_right,
        offsets_left,
        offsets_right,
        n_prefix_to_right,
    )


def triton_jagged_dense_bmm_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    return _JaggedDenseBmmBroadcastAddFunction.apply(
        max_seq_len, seq_offsets, jagged, dense, bias
    )
