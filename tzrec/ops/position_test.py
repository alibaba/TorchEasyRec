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
import unittest

import torch
from hypothesis import Verbosity, given
from hypothesis import strategies as st

from tzrec.ops import Kernel
from tzrec.utils.test_util import (
    generate_sparse_seq_len,
    get_test_dtypes,
    gpu_unavailable,
)
from tzrec.utils.test_util import hypothesis_settings as settings


class PositionEmbeddingsTest(unittest.TestCase):
    def teardown_example(self, example):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.memory_summary()  # prevent oom

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        batch_size=st.integers(4, 8),
        max_uih_len=st.integers(50, 500),
        max_targets=st.sampled_from([10, 20]),
        D=st.integers(20, 200),
        alpha=st.sampled_from([0.5, 1.0]),
        dtype=st.sampled_from([torch.float32]),
        interleave_targets=st.sampled_from([True, False]),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_add_positional_embeddings_triton(self, *args, **kwargs) -> None:
        self._test_add_position_embeddings(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    def _test_add_position_embeddings(
        self,
        batch_size: int,
        max_uih_len: int,
        max_targets: int,
        D: int,
        alpha: float,
        dtype: torch.dtype,
        interleave_targets: bool,
        ref_kernel: Kernel,
        real_kernel: Kernel,
        test_backward: bool,
    ) -> None:
        from tzrec.ops.position import add_positional_embeddings

        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        lengths = torch.randint(
            max_targets * 2,
            max_uih_len + 1,
            size=(batch_size,),
            device=torch.device("cuda"),
        )
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        max_seq_len = max_uih_len + max_targets

        position_embeddings_weight = (
            torch.empty(
                (max_seq_len, D), dtype=torch.float32, device=torch.device("cuda")
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        seq_embeddings = (
            torch.empty(
                (int(seq_offsets[-1].item()), D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )

        # ref implementation
        ref_out = add_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            kernel=ref_kernel,
        )

        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)
        # pyre-ignore
        ref_d_seq_embeddings, seq_embeddings.grad = seq_embeddings.grad.clone(), None
        ref_d_pos_emb_weight, position_embeddings_weight.grad = (
            position_embeddings_weight.grad.clone(),
            None,
        )

        # real implementation
        seq_embeddings = seq_embeddings.detach().clone().requires_grad_()
        position_embeddings_weight = (
            position_embeddings_weight.detach().clone().requires_grad_()
        )
        real_out = add_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            kernel=real_kernel,
        )
        torch.testing.assert_close(ref_out, real_out)
        if test_backward:
            real_out.backward(dout)
            real_d_seq_embeddings = seq_embeddings.grad.clone()
            real_d_pos_emb_weight = position_embeddings_weight.grad.clone()
            torch.testing.assert_close(ref_d_seq_embeddings, real_d_seq_embeddings)
            torch.testing.assert_close(ref_d_pos_emb_weight, real_d_pos_emb_weight)

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        alpha=st.sampled_from([0.5]),
        max_uih_len=st.integers(50, 500),
        max_contextual_seq_len=st.sampled_from([10]),
        interleave_targets=st.sampled_from([True, False]),
        batch_size=st.integers(16, 32),
        D=st.integers(20, 200),
        max_targets=st.sampled_from([10, 20]),
        time_bucket_fn=st.sampled_from(["log"]),
        dtype=st.sampled_from(
            get_test_dtypes([torch.float32, torch.bfloat16, torch.float16])
        ),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=20,
        deadline=None,
    )
    # pyre-ignore[2]
    def test_add_timestamp_positional_embeddings_triton(self, *args, **kwargs) -> None:
        self._test_add_timestamp_positional_embeddings(
            *args,
            **kwargs,
            test_backward=True,
            ref_kernel=Kernel.PYTORCH,
            real_kernel=Kernel.TRITON,
        )

    @unittest.skipIf(*gpu_unavailable)
    # pyre-ignore
    @given(
        alpha=st.sampled_from([0.5]),
        max_uih_len=st.sampled_from([32768]),
        max_contextual_seq_len=st.sampled_from([10]),
        interleave_targets=st.sampled_from([False]),
        batch_size=st.sampled_from([130]),
        D=st.sampled_from([128]),
        max_targets=st.sampled_from([10]),
        time_bucket_fn=st.sampled_from(["log"]),
        dtype=st.sampled_from(get_test_dtypes([torch.bfloat16, torch.float16])),
    )
    @settings(
        verbosity=Verbosity.verbose,
        max_examples=2,
        deadline=None,
    )
    def test_add_timestamp_positional_embeddings_triton_large_tensor(
        self,
        # pyre-fixme[2]: Parameter must be annotated.
        *args,
        # pyre-ignore[2]
        **kwargs,
    ) -> None:
        self._test_add_timestamp_positional_embeddings(
            *args,
            **kwargs,
            test_backward=False,
            ref_kernel=Kernel.TRITON,
            real_kernel=Kernel.TRITON,
            sparsity=1.0,
        )

    def _test_add_timestamp_positional_embeddings(
        self,
        alpha: float,
        max_uih_len: int,
        max_contextual_seq_len: int,
        interleave_targets: bool,
        batch_size: int,
        D: int,
        max_targets: int,
        time_bucket_fn: str,
        dtype: torch.dtype,
        ref_kernel: Kernel,
        real_kernel: Kernel,
        test_backward: bool,
        sparsity: float = -1,
    ) -> None:
        from tzrec.ops.position import (
            add_timestamp_positional_embeddings,
        )

        num_targets = torch.randint(
            max_targets + 1, size=(batch_size,), device=torch.device("cuda")
        )
        if sparsity > 0.0:
            lengths = generate_sparse_seq_len(
                size=batch_size,
                max_seq_len=max_uih_len,
                sparsity=sparsity,
                device=torch.device("cuda"),
            ).to(torch.int64)
        else:
            lengths = torch.randint(
                max_uih_len + 1, size=(batch_size,), device=torch.device("cuda")
            )
        seq_offsets = torch.zeros(
            (batch_size + 1,), dtype=torch.int64, device=torch.device("cuda")
        )
        seq_offsets[1:] = torch.cumsum(lengths, dim=0)
        max_seq_len = max_uih_len + max_targets

        position_embeddings_weight = (
            torch.empty(
                (max_seq_len, D), dtype=torch.float32, device=torch.device("cuda")
            )
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        timestamp_embeddings_weight = (
            torch.empty((2048, D), dtype=torch.float32, device=torch.device("cuda"))
            .uniform_(-1.0, 1.0)
            .requires_grad_()
        )
        seq_embeddings = (
            torch.empty(
                (int(seq_offsets[-1].item()), D),
                dtype=dtype,
                device=torch.device("cuda"),
            )
            .uniform_(-0.1, 0.1)
            .requires_grad_()
        )
        timestamp_deltas: torch.Tensor = torch.randint(
            86400,
            size=(batch_size, max_seq_len),
            device="cuda",
        )
        timestamps = timestamp_deltas.cumsum(dim=1)
        mask = torch.arange(max_seq_len, device=timestamps.device) < lengths.unsqueeze(
            1
        )
        timestamps = timestamps[mask.view(batch_size, -1)]

        ref_out = add_timestamp_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            timestamp_embeddings_weight=timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            timestamps=timestamps,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
            kernel=ref_kernel,
        )
        dout = torch.randn_like(ref_out) * 0.01
        ref_out.backward(dout)
        # pyre-ignore
        ref_d_seq_embeddings, seq_embeddings.grad = seq_embeddings.grad.clone(), None
        ref_d_position_embeddings_weight, position_embeddings_weight.grad = (
            position_embeddings_weight.grad.clone(),
            None,
        )
        ref_d_timestamp_embeddings_weight, timestamp_embeddings_weight.grad = (
            timestamp_embeddings_weight.grad.clone(),
            None,
        )

        real_out = add_timestamp_positional_embeddings(
            alpha=alpha,
            max_seq_len=max_seq_len,
            max_contextual_seq_len=max_contextual_seq_len,
            position_embeddings_weight=position_embeddings_weight,
            timestamp_embeddings_weight=timestamp_embeddings_weight,
            seq_offsets=seq_offsets,
            seq_lengths=lengths,
            seq_embeddings=seq_embeddings,
            timestamps=timestamps,
            num_targets=num_targets,
            interleave_targets=interleave_targets,
            time_bucket_fn=time_bucket_fn,
            kernel=real_kernel,
        )

        torch.testing.assert_close(ref_out, real_out)
        if test_backward:
            real_out.backward(dout)
            real_d_seq_embeddings = seq_embeddings.grad.clone()
            real_d_position_embeddings_weight = position_embeddings_weight.grad.clone()
            real_d_timestamp_embeddings_weight = (
                timestamp_embeddings_weight.grad.clone()
            )
            torch.testing.assert_close(ref_d_seq_embeddings, real_d_seq_embeddings)
            torch.testing.assert_close(
                ref_d_position_embeddings_weight,
                real_d_position_embeddings_weight,
                atol=5e-2 if dtype != torch.float32 else None,
                rtol=2e-2 if dtype != torch.float32 else None,
            )
            torch.testing.assert_close(
                ref_d_timestamp_embeddings_weight,
                real_d_timestamp_embeddings_weight,
                atol=5e-2 if dtype != torch.float32 else None,
                rtol=2e-2 if dtype != torch.float32 else None,
            )


if __name__ == "__main__":
    unittest.main()
