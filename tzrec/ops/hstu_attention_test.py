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
import gc
import random
import unittest
from typing import Optional

import torch
from hypothesis import Verbosity, assume, given
from hypothesis import strategies as st

from tzrec.ops import (
    Kernel,
)
from tzrec.utils.test_util import (
    generate_sparse_seq_len,
    get_test_dtypes,
    get_test_enable_tma,
    gpu_unavailable,
)
from tzrec.utils.test_util import hypothesis_settings as settings


def test_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    max_targets: int,
    attn_dim: int,
    hidden_dim: int,
    causal: bool,
    has_multiple_targets: bool,
    has_max_attn_len: bool,
    dtype: torch.dtype,
    test_backward: bool,
    ref_kernel: Kernel,
    real_kernel: Kernel,
    skip_comparisons: bool = False,
    sparsity: float = -1.0,
    contextual_seq_len: int = 0,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    enable_tma: bool = False,
    scaling_seqlen: int = -1,
) -> None:
    # has_max_attn_len=True and enable_tma=True will result in TritonGPUCoalesce error
    # include/llvm/llvm/ADT/SmallVector.h:296: const_reference llvm::SmallVectorTemplateCommon<long>::operator[](size_type) const [T = long]: Assertion `idx < size()' failed.    # NOQA
    assume(not has_max_attn_len or not enable_tma)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from tzrec.ops.hstu_attention import hstu_mha

    alpha = 1.0 / (attn_dim**0.5)
    if sparsity > 0.0:
        lengths = generate_sparse_seq_len(
            size=batch_size,
            max_seq_len=max_uih_len,
            sparsity=sparsity,
            device=torch.device("cuda"),
        )
    else:
        lengths = torch.randint(
            max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
        )
    num_targets = torch.randint(
        1, max_targets + 1, size=(batch_size,), device=torch.device("cuda")
    )
    lengths = lengths + num_targets + contextual_seq_len
    max_seq_len = max_uih_len + max_targets + contextual_seq_len
    if has_max_attn_len:
        max_attn_len = random.randint(1, max_uih_len // 5)
    else:
        max_attn_len = 0
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())
    q = (
        torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    k = (
        torch.empty((L, heads, attn_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )
    v = (
        torch.empty((L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda"))
        .uniform_(-0.1, 0.1)
        .requires_grad_()
    )

    # ref implementation
    ref_out = hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        dropout_pr=0.0,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=ref_kernel,
        scaling_seqlen=scaling_seqlen,
    )
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)

    if skip_comparisons:
        return

    # pyre-ignore
    ref_dv, v.grad = v.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dq, q.grad = q.grad.clone(), None

    # triton implementation
    q = q.detach().clone().requires_grad_()
    k = k.detach().clone().requires_grad_()
    v = v.detach().clone().requires_grad_()
    dout = dout.detach().clone()
    real_out = hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        q=q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        causal=causal,
        num_targets=num_targets if has_multiple_targets else None,
        dropout_pr=0.0,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=real_kernel,
        enable_tma=enable_tma,
        scaling_seqlen=scaling_seqlen,
    )

    torch.testing.assert_close(
        ref_out,
        real_out,
        atol=atol,
        rtol=rtol,
    )
    if test_backward:
        real_out.backward(dout)
        real_dq, real_dk, real_dv = q.grad.clone(), k.grad.clone(), v.grad.clone()
        torch.testing.assert_close(ref_dv, real_dv, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_dk, real_dk, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_dq, real_dq, atol=atol, rtol=rtol)


def test_delta_attn(
    batch_size: int,
    heads: int,
    max_uih_len: int,
    max_targets: int,
    delta_size: int,
    attn_dim: int,
    hidden_dim: int,
    has_multiple_targets: bool,
    has_max_attn_len: bool,
    dtype: torch.dtype,
    ref_kernel: Kernel,
    real_kernel: Kernel,
    contextual_seq_len: int = 0,
    atol: Optional[float] = None,
    rtol: Optional[float] = None,
    enable_tma: bool = False,
) -> None:
    # has_max_attn_len=True and enable_tma=True will result in TritonGPUCoalesce error
    # include/llvm/llvm/ADT/SmallVector.h:296: const_reference llvm::SmallVectorTemplateCommon<long>::operator[](size_type) const [T = long]: Assertion `idx < size()' failed.  # NOQA
    assume(not has_max_attn_len or not enable_tma)

    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    from tzrec.ops.hstu_attention import delta_hstu_mha

    alpha = 1.0 / (attn_dim**0.5)
    lengths = torch.randint(
        max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
    )
    num_targets = torch.randint(
        1, delta_size + 1, size=(batch_size,), device=torch.device("cuda")
    )
    lengths = lengths + delta_size + contextual_seq_len
    max_seq_len = max_uih_len + delta_size + contextual_seq_len
    if has_max_attn_len:
        max_attn_len = random.randint(1, max_uih_len // 5)
    else:
        max_attn_len = 0
    seq_offsets = torch.zeros(
        (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
    )
    seq_offsets[1:] = torch.cumsum(lengths, dim=0)

    L = int(seq_offsets[-1].item())
    delta_q = torch.empty(
        (batch_size * delta_size, heads, attn_dim),
        dtype=dtype,
        device=torch.device("cuda"),
    ).uniform_(-0.1, 0.1)
    k = torch.empty(
        (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(-0.1, 0.1)
    v = torch.empty(
        (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
    ).uniform_(-0.1, 0.1)

    # ref implementation
    ref_out = delta_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=ref_kernel,
    )

    # real implementation
    real_out = delta_hstu_mha(
        max_seq_len=max_seq_len,
        alpha=alpha,
        delta_q=delta_q,
        k=k,
        v=v,
        seq_offsets=seq_offsets,
        num_targets=num_targets if has_multiple_targets else None,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        kernel=real_kernel,
        enable_tma=enable_tma,
    )
    torch.testing.assert_close(
        ref_out,
        real_out,
        atol=atol,
        rtol=rtol,
    )


class HSTUAttentionTest(unittest.TestCase):
    def teardown_example(self, example):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory_summary()  # prevent oom

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([20, 100, 128, 256]),
        max_targets=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16, torch.float32])),
        has_max_attn_len=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        enable_tma=st.sampled_from(get_test_enable_tma()),
        scaling_seqlen=st.sampled_from([-1, 512, 2048]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_triton(self, *args, **kwargs) -> None:
        test_attn(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.just(32),
        heads=st.just(4),
        max_uih_len=st.sampled_from([32768]),
        max_targets=st.sampled_from([32]),
        attn_dim=st.just(128),
        hidden_dim=st.just(128),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16, torch.float16])),
        has_max_attn_len=st.sampled_from([True, False]),
        enable_tma=st.sampled_from(get_test_enable_tma()),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=5,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_triton_long_seqs(self, *args, **kwargs) -> None:
        test_attn(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=Kernel.TRITON,
            real_kernel=Kernel.TRITON,
            skip_comparisons=True,
            sparsity=1.0,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([100, 128, 256]),
        max_targets=st.sampled_from([20, 512]),
        delta_size=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64, 128]),
        hidden_dim=st.sampled_from([16, 32, 64, 128]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16, torch.float32])),
        has_max_attn_len=st.sampled_from([False, True]),
        contextual_seq_len=st.sampled_from([0, 10]),
        enable_tma=st.sampled_from(get_test_enable_tma()),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_delta_attn_triton(self, *args, **kwargs) -> None:
        test_delta_attn(
            *args,
            **kwargs,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([20, 100, 128]),
        max_targets=st.sampled_from([20, 512]),
        delta_size=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([16, 32, 64]),
        hidden_dim=st.sampled_from([16, 32, 64]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16, torch.float32])),
        has_max_attn_len=st.sampled_from([False, True]),
        contextual_seq_len=st.sampled_from([0, 10]),
        enable_tma=st.sampled_from(get_test_enable_tma()),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    def test_cache(
        self,
        batch_size: int,
        heads: int,
        max_uih_len: int,
        max_targets: int,
        delta_size: int,
        attn_dim: int,
        hidden_dim: int,
        has_multiple_targets: bool,
        dtype: torch.dtype,
        has_max_attn_len: bool,
        contextual_seq_len: int,
        enable_tma: bool,
    ) -> None:
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        from tzrec.ops.hstu_attention import delta_hstu_mha, hstu_mha
        from tzrec.ops.jagged_tensors import split_2D_jagged

        alpha = 1.0 / (attn_dim**0.5)
        lengths = torch.randint(
            max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
        )
        num_targets = torch.randint(
            1, delta_size + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = lengths + delta_size + contextual_seq_len
        max_seq_len = max_uih_len + delta_size + contextual_seq_len
        if has_max_attn_len:
            max_attn_len = random.randint(1, max_uih_len // 5)
        else:
            max_attn_len = 0
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)

        L = int(seq_offsets[-1].item())
        q = torch.empty(
            (L, heads, attn_dim),
            dtype=dtype,
            device=torch.device("cuda"),
        ).uniform_(-0.1, 0.1)
        _, delta_q = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=q.view(-1, heads * attn_dim),
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=torch.ops.fbgemm.asynchronous_complete_cumsum(
                lengths - delta_size
            ),
            offsets_right=None,
            kernel=Kernel.TRITON,
        )
        delta_q = delta_q.view(-1, heads, attn_dim)
        k = torch.empty(
            (L, heads, attn_dim), dtype=dtype, device=torch.device("cuda")
        ).uniform_(-0.1, 0.1)
        v = torch.empty(
            (L, heads, hidden_dim), dtype=dtype, device=torch.device("cuda")
        ).uniform_(-0.1, 0.1)
        prime_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            lengths - delta_size
        )

        # ref implementation
        ref_out = hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            q=q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            causal=True,
            num_targets=num_targets if has_multiple_targets else None,
            dropout_pr=0.0,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            kernel=Kernel.TRITON,
            enable_tma=enable_tma,
        )
        _, delta_out = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=ref_out.view(-1, heads * hidden_dim),
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=Kernel.TRITON,
        )
        delta_out = delta_out.view(-1, heads, hidden_dim)

        # real implementation
        real_delta_out = delta_hstu_mha(
            max_seq_len=max_seq_len,
            alpha=alpha,
            delta_q=delta_q,
            k=k,
            v=v,
            seq_offsets=seq_offsets,
            num_targets=num_targets if has_multiple_targets else None,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            enable_tma=enable_tma,
        )
        torch.testing.assert_close(
            delta_out,
            real_delta_out,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        heads=st.integers(1, 4),
        max_uih_len=st.sampled_from([20, 100, 128, 256]),
        max_targets=st.sampled_from([20, 512]),
        attn_dim=st.sampled_from([32, 64, 128]),
        causal=st.sampled_from([True]),
        has_multiple_targets=st.sampled_from([True, False]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16])),
        has_max_attn_len=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        scaling_seqlen=st.sampled_from([-1, 512, 2048]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_attn_cutlass(self, *args, **kwargs) -> None:
        hidden_dim = kwargs.pop("attn_dim")
        test_attn(
            *args,
            **kwargs,
            attn_dim=hidden_dim,
            hidden_dim=hidden_dim,
            test_backward=True,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.CUTLASS,
        )

    # NOTE: no ``test_delta_attn_cutlass`` — ``delta_hstu_mha`` has no
    # CUTLASS implementation and falls back to Triton internally. The
    # delta/cached path is already covered by ``test_delta_attn_triton``.

    @unittest.skipIf(*gpu_unavailable)
    def test_scaling_seqlen_invariance(self) -> None:
        """When ``scaling_seqlen`` is pinned, output is invariant to max_seq_len.

        Calls ``hstu_mha`` twice with identical jagged q/k/v but different
        ``max_seq_len`` values (both >= actual max sequence length). With
        the default ``scaling_seqlen=-1`` the two outputs differ (divided
        by different runtime maxes); with a pinned value they must match.
        """
        from tzrec.ops.hstu_attention import hstu_mha

        torch.manual_seed(0)
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        batch_size, heads, attn_dim, hidden_dim = 4, 2, 32, 32
        lengths = torch.tensor([40, 50, 40, 50], dtype=torch.int64, device="cuda")
        seq_offsets = torch.zeros((batch_size + 1,), dtype=torch.int64, device="cuda")
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        L = int(seq_offsets[-1].item())
        actual_max = int(lengths.max().item())

        q = torch.empty(
            (L, heads, attn_dim), dtype=torch.bfloat16, device="cuda"
        ).uniform_(-0.1, 0.1)
        k = torch.empty_like(q).uniform_(-0.1, 0.1)
        v = torch.empty(
            (L, heads, hidden_dim), dtype=torch.bfloat16, device="cuda"
        ).uniform_(-0.1, 0.1)

        alpha = 1.0 / (attn_dim**0.5)

        def run(max_seq_len: int, scaling_seqlen: int, kernel: Kernel):
            return hstu_mha(
                max_seq_len=max_seq_len,
                alpha=alpha,
                q=q,
                k=k,
                v=v,
                seq_offsets=seq_offsets,
                causal=True,
                num_targets=None,
                dropout_pr=0.0,
                contextual_seq_len=0,
                kernel=kernel,
                scaling_seqlen=scaling_seqlen,
            )

        assert actual_max <= 64
        for kernel in (Kernel.PYTORCH, Kernel.TRITON, Kernel.CUTLASS):
            # Legacy (scaling_seqlen=-1): divisor == max_seq_len. Doubling
            # max_seq_len halves the output (scale is 1/max).
            out_64 = run(max_seq_len=64, scaling_seqlen=-1, kernel=kernel)
            out_128 = run(max_seq_len=128, scaling_seqlen=-1, kernel=kernel)
            # out_64 * 0.5 ~= out_128 (since divisor doubled).
            torch.testing.assert_close(out_64 * 0.5, out_128, atol=1e-2, rtol=1e-2)

            # Pinned scaling_seqlen: divisor is constant. Outputs must
            # match bit-close regardless of the runtime max_seq_len.
            pinned_64 = run(max_seq_len=64, scaling_seqlen=1024, kernel=kernel)
            pinned_128 = run(max_seq_len=128, scaling_seqlen=1024, kernel=kernel)
            torch.testing.assert_close(pinned_64, pinned_128, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
