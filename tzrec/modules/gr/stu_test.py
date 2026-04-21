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

import copy
import unittest
from typing import List

import torch
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st

from tzrec.ops import Kernel
from tzrec.utils.test_util import gpu_unavailable


def _inplace_swap(
    batch_size: int,
    x: torch.Tensor,
    swap_from: torch.Tensor,
    swap_to: torch.Tensor,
) -> torch.Tensor:
    for i in range(batch_size):
        tmp = x[i, swap_from[i], :].detach().clone()
        x[i, swap_from[i], :] = x[i, swap_to[i], :]
        x[i, swap_to[i], :] = tmp
    return x


class StuTest(unittest.TestCase):
    # pyre-ignore
    @given(
        causal=st.sampled_from([True]),
        num_layers=st.sampled_from([2]),
        num_heads=st.sampled_from([1, 2]),
        max_uih_len=st.sampled_from([20, 64]),
        batch_size=st.sampled_from([8]),
        embedding_dim=st.sampled_from([16]),
        attention_dim=st.sampled_from([32]),
        linear_hidden_dim=st.sampled_from([64]),
        has_multiple_targets=st.sampled_from([True, False]),
        contextual_seq_len=st.sampled_from([0, 10]),
        use_group_norm=st.sampled_from([True, False]),
        recompute_uvqk_in_backward=st.sampled_from([True, False]),
        recompute_normed_x_in_backward=st.sampled_from([True, False]),
        recompute_y_in_backward=st.sampled_from([True, False]),
        empty_inputs=st.sampled_from([False]),
        dtype=st.sampled_from([torch.float32]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=100, deadline=None)
    def test_triton(
        self,
        causal: bool,
        num_layers: int,
        num_heads: int,
        max_uih_len: int,
        batch_size: int,
        embedding_dim: int,
        attention_dim: int,
        linear_hidden_dim: int,
        has_multiple_targets: bool,
        contextual_seq_len: int,
        use_group_norm: bool,
        recompute_uvqk_in_backward: bool,
        recompute_normed_x_in_backward: bool,
        recompute_y_in_backward: bool,
        empty_inputs: bool,  # test the case where all the seqlen in the batch are 0
        dtype: torch.dtype,
    ) -> None:
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        device = torch.device("cuda")

        stu_layers: List[STU] = [
            STULayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                output_dropout_ratio=0.0,
                causal=causal,
                target_aware=has_multiple_targets,
                max_attn_len=None,
                attn_alpha=None,
                use_group_norm=use_group_norm,
                recompute_normed_x=recompute_normed_x_in_backward,
                recompute_uvqk=recompute_uvqk_in_backward,
                recompute_y=recompute_y_in_backward,
                sort_by_length=True,
                contextual_seq_len=contextual_seq_len,
                is_inference=False,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=False,
        ).to(device)
        stu.set_kernel(Kernel.PYTORCH)
        stu_triton = copy.deepcopy(stu)
        stu_triton.set_kernel(Kernel.TRITON)

        if empty_inputs:
            x_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)
            num_targets = torch.zeros(batch_size, dtype=torch.int32, device=device)
            contextual_seq_len = 0
            max_seq_len = 16
        else:
            x_lengths = torch.randint(max_uih_len + 1, (batch_size,), device=device)
            x_lengths = x_lengths + contextual_seq_len
            max_seq_len = max_uih_len + contextual_seq_len
            max_targets = 20
            num_targets = torch.randint(
                1, max_targets, size=(batch_size,), device=device
            )
            if has_multiple_targets:
                x_lengths = x_lengths + num_targets
                max_seq_len = max_seq_len + max_targets
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu().item())
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        x_triton = x.clone().detach().requires_grad_()
        stu_output, _, _, _ = stu(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        stu_triton_output, _, _, _ = stu_triton(
            x=x_triton,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        atol = 5e-3 if dtype == torch.bfloat16 else None
        rtol = 1e-2 if dtype == torch.bfloat16 else None
        torch.testing.assert_close(stu_triton_output, stu_output, atol=atol, rtol=rtol)
        dout = torch.randn_like(stu_output)
        stu_output.backward(dout)
        dout = dout.detach().clone()
        stu_triton_output.backward(dout)
        torch.testing.assert_close(x.grad, x_triton.grad, atol=atol, rtol=rtol)

    # pyre-ignore
    @given(
        dtype=st.sampled_from([torch.float32]),
    )
    @unittest.skipIf(*gpu_unavailable)
    @settings(verbosity=Verbosity.verbose, max_examples=8, deadline=None)
    def test_target_invariance(
        self,
        dtype: torch.dtype,
    ) -> None:
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        device = torch.device("cuda")
        num_layers = 2
        num_heads = 2
        max_seq_len = 32
        batch_size = 8
        embedding_dim = 16
        attention_dim = 32
        linear_hidden_dim = 32
        causal = True
        use_group_norm = False
        recompute_normed_x_in_backward = False
        recompute_uvqk_in_backward = False
        recompute_y_in_backward = False
        max_attn_len = None
        stu_layers: List[STU] = [
            STULayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                output_dropout_ratio=0.0,
                causal=causal,
                target_aware=True,
                max_attn_len=max_attn_len,
                attn_alpha=None,
                use_group_norm=use_group_norm,
                recompute_normed_x=recompute_normed_x_in_backward,
                recompute_uvqk=recompute_uvqk_in_backward,
                recompute_y=recompute_y_in_backward,
                sort_by_length=True,
                contextual_seq_len=0,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=False,
        ).to(device)

        x_lengths = torch.randint(
            low=2, high=max_seq_len + 1, size=(batch_size,), device=device
        )
        num_targets = torch.randint(low=2, high=10, size=(batch_size,), device=device)
        x_lengths = x_lengths + num_targets
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu())

        swap_from = torch.remainder(
            torch.randint(20, (batch_size,), device=device), num_targets
        )
        swap_to = torch.remainder(
            torch.randint(20, (batch_size,), device=device), num_targets
        )
        swap_from = x_lengths - 1 - swap_from
        swap_to = x_lengths - 1 - swap_to
        max_seq_len = int(x_lengths.max().item())

        # forward()
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
            dtype=dtype,
        ).requires_grad_(True)
        stu_output, _, _, _ = stu(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        stu_output_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=stu_output,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )

        # swapped forward().
        dense_x = torch.ops.fbgemm.jagged_to_padded_dense(
            x.detach(),
            [x_offsets],
            [max_seq_len],
        )
        swapped_dense_x = _inplace_swap(batch_size, dense_x, swap_from, swap_to)
        swapped_x = torch.ops.fbgemm.dense_to_jagged(
            swapped_dense_x,
            [x_offsets],
        )[0].requires_grad_(True)
        swapped_stu_output, _, _, _ = stu(
            x=swapped_x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        swapped_stu_output_dense = torch.ops.fbgemm.jagged_to_padded_dense(
            values=swapped_stu_output,
            offsets=[x_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )

        # backward
        dout = torch.randn_like(stu_output_dense)
        stu_output_dense.backward(dout)
        dout = dout.detach().clone()
        swapped_stu_output_dense.backward(
            _inplace_swap(batch_size, dout, swap_from, swap_to)
        )

        swapped_swapped_stu_output_dense = _inplace_swap(
            batch_size, swapped_stu_output_dense, swap_from, swap_to
        )
        torch.testing.assert_close(stu_output_dense, swapped_swapped_stu_output_dense)

        # backward
        torch.testing.assert_close(
            torch.ops.fbgemm.jagged_to_padded_dense(
                swapped_x.grad,
                [x_offsets],
                [max_seq_len],
            ),
            _inplace_swap(
                batch_size,
                torch.ops.fbgemm.jagged_to_padded_dense(
                    x.grad,
                    [x_offsets],
                    [max_seq_len],
                ),
                swap_from,
                swap_to,
            ),
        )

    # pyre-ignore[56]
    @given(
        num_layers=st.sampled_from([1, 2, 4]),
        num_heads=st.sampled_from([1, 4]),
        max_uih_len=st.sampled_from([20, 128]),
        batch_size=st.sampled_from([4, 8]),
        embedding_dim=st.sampled_from([32]),
        attention_dim=st.sampled_from([16]),
        linear_hidden_dim=st.sampled_from([64]),
        contextual_seq_len=st.sampled_from([0, 10]),
    )
    @settings(verbosity=Verbosity.verbose, max_examples=20, deadline=None)
    @unittest.skipIf(*gpu_unavailable)
    @torch.inference_mode()
    def test_cached_forward(
        self,
        num_layers: int,
        num_heads: int,
        max_uih_len: int,
        batch_size: int,
        embedding_dim: int,
        attention_dim: int,
        linear_hidden_dim: int,
        contextual_seq_len: int,
    ) -> None:
        from tzrec.modules.gr.stu import STU, STULayer, STUStack
        from tzrec.ops.jagged_tensors import split_2D_jagged

        device = torch.device("cuda")

        use_group_norm = False
        recompute_normed_x_in_backward = False
        recompute_uvqk_in_backward = False
        recompute_y_in_backward = False
        max_attn_len = None
        stu_layers: List[STU] = [
            STULayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=linear_hidden_dim,
                attention_dim=attention_dim,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                max_attn_len=max_attn_len,
                attn_alpha=None,
                use_group_norm=use_group_norm,
                recompute_normed_x=recompute_normed_x_in_backward,
                recompute_uvqk=recompute_uvqk_in_backward,
                recompute_y=recompute_y_in_backward,
                sort_by_length=True,
                contextual_seq_len=contextual_seq_len,
                is_inference=True,
            )
            for _ in range(num_layers)
        ]
        stu = STUStack(
            stu_list=stu_layers,
            is_inference=True,
        ).to(device)
        stu.set_kernel(Kernel.TRITON)
        stu.eval()

        x_lengths = torch.randint(
            max_uih_len, max_uih_len + 1, (batch_size,), device=device
        )
        x_lengths = x_lengths + contextual_seq_len
        max_seq_len = max_uih_len + contextual_seq_len
        delta_size = 20
        max_targets = delta_size * 2
        num_targets = torch.randint(
            delta_size, max_targets + 1, size=(batch_size,), device=device
        )
        x_lengths = x_lengths + num_targets + contextual_seq_len
        max_seq_len = max_seq_len + max_targets + contextual_seq_len
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total_seq_len = int(x_offsets[-1].cpu().item())
        x = torch.randn(
            int(total_seq_len),
            embedding_dim,
            device=device,
        ).requires_grad_(True)

        # default forward().
        ref_y, _, _, _ = stu(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        prime_lengths = x_lengths - delta_size
        prime_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(prime_lengths)
        _, ref_delta_y = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=ref_y,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=Kernel.TRITON,
        )

        # cached forward().
        prime_x, delta_x = split_2D_jagged(
            max_seq_len=max_seq_len,
            values=x,
            max_len_left=None,
            max_len_right=delta_size,
            offsets_left=prime_offsets,
            offsets_right=None,
            kernel=Kernel.TRITON,
        )
        _, _, _, _ = stu(
            x=prime_x,
            x_offsets=prime_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets - delta_size,
            max_kv_caching_len=max_seq_len - delta_size,
            kv_caching_lengths=x_lengths - delta_size,
        )
        delta_y = stu.cached_forward(
            delta_x=delta_x,
            num_targets=num_targets,
        )

        torch.testing.assert_close(ref_delta_y, delta_y)


class STUStackTruncationTest(unittest.TestCase):
    """Cover the mid-stack truncation block in STUStack.forward.

    The truncation branch rewrites ``x`` / ``x_offsets`` / ``num_targets``
    / ``max_seq_len`` and has no coverage from ``StuTest`` (which leaves
    ``truncate_*`` at their defaults).
    """

    def _make_stack(
        self,
        num_layers: int,
        truncate_split_layer: int,
        truncate_tail_len: int,
    ):
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        embedding_dim, num_heads, hidden_dim, attn_dim = 16, 2, 32, 32
        stu_layers: List[STU] = [
            STULayer(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                hidden_dim=hidden_dim,
                attention_dim=attn_dim,
                output_dropout_ratio=0.0,
                causal=True,
                target_aware=True,
                max_attn_len=None,
                attn_alpha=None,
                use_group_norm=False,
                recompute_normed_x=False,
                recompute_uvqk=False,
                recompute_y=False,
                sort_by_length=False,
                contextual_seq_len=0,
                is_inference=False,
            )
            for _ in range(num_layers)
        ]
        stack = STUStack(
            stu_list=stu_layers,
            truncate_split_layer=truncate_split_layer,
            truncate_tail_len=truncate_tail_len,
            is_inference=False,
        )
        stack.set_kernel(Kernel.PYTORCH)
        return stack, embedding_dim

    def test_init_validates_split_layer_bounds(self) -> None:
        """C8: truncate_split_layer must be in (0, num_layers)."""
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        layer_kwargs = dict(
            embedding_dim=16,
            num_heads=2,
            hidden_dim=32,
            attention_dim=32,
            output_dropout_ratio=0.0,
            causal=True,
            target_aware=True,
            max_attn_len=None,
            attn_alpha=None,
            use_group_norm=False,
            recompute_normed_x=False,
            recompute_uvqk=False,
            recompute_y=False,
            sort_by_length=False,
            contextual_seq_len=0,
            is_inference=False,
        )
        stu_list: List[STU] = [STULayer(**layer_kwargs) for _ in range(3)]
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=0,  # invalid: must be > 0
                truncate_tail_len=4,
            )
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=3,  # invalid: must be < len(stu_list)
                truncate_tail_len=4,
            )
        # Valid configurations must not raise.
        STUStack(stu_list=stu_list, truncate_split_layer=1, truncate_tail_len=4)
        # truncate_tail_len == 0 disables truncation; split_layer is free.
        STUStack(stu_list=stu_list, truncate_split_layer=99, truncate_tail_len=0)

    def _forward_and_check_truncation(
        self,
        x_lengths: torch.Tensor,
        num_targets_val,
        truncate_tail_len: int,
    ):
        stack, D = self._make_stack(
            num_layers=3,
            truncate_split_layer=1,
            truncate_tail_len=truncate_tail_len,
        )
        B = x_lengths.size(0)
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total = int(x_offsets[-1].item())
        x = torch.randn(total, D)
        max_seq_len = int(x_lengths.max().item())
        num_targets = (
            torch.full((B,), num_targets_val, dtype=torch.int64)
            if num_targets_val is not None
            else None
        )
        out, new_offsets, new_num_targets, new_max_seq_len = stack(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        return (
            out,
            new_offsets,
            new_num_targets,
            new_max_seq_len,
            x_lengths,
            num_targets,
        )

    def test_truncation_applied_when_seq_longer_than_tail(self) -> None:
        """Sequences longer than tail_len are truncated; shorter ones pass."""
        x_lengths = torch.tensor([12, 20, 5, 30], dtype=torch.int64)
        tail_len = 8
        (
            out,
            new_offsets,
            new_num_targets,
            new_max_seq_len,
            orig_lengths,
            _,
        ) = self._forward_and_check_truncation(x_lengths, 2, tail_len)
        expected_lengths = torch.clamp(x_lengths, max=tail_len)
        new_lengths = new_offsets[1:] - new_offsets[:-1]
        self.assertEqual(new_max_seq_len, tail_len)
        torch.testing.assert_close(new_lengths, expected_lengths)
        # Total tokens in x matches sum of new lengths.
        self.assertEqual(out.size(0), int(expected_lengths.sum().item()))
        # num_targets is clamped when it exceeds the truncated length.
        assert new_num_targets is not None
        torch.testing.assert_close(
            new_num_targets,
            torch.clamp(torch.full_like(new_lengths, 2), max=new_lengths),
        )

    def test_truncation_no_op_when_all_shorter(self) -> None:
        """Sequences all shorter than tail_len: x unchanged in shape."""
        x_lengths = torch.tensor([3, 5, 2], dtype=torch.int64)
        tail_len = 16
        out, new_offsets, _, new_max_seq_len, _, _ = self._forward_and_check_truncation(
            x_lengths, 1, tail_len
        )
        orig_total = int(x_lengths.sum().item())
        self.assertEqual(out.size(0), orig_total)
        # max_seq_len is still updated to tail_len (post-truncation cap).
        self.assertEqual(new_max_seq_len, tail_len)

    def test_num_targets_clamped(self) -> None:
        """num_targets > new_length gets clamped to new_length."""
        x_lengths = torch.tensor([12, 20, 6], dtype=torch.int64)
        tail_len = 4
        # num_targets=10 exceeds tail_len=4; should be clamped to 4 for all.
        _, new_offsets, new_num_targets, _, _, _ = self._forward_and_check_truncation(
            x_lengths, 10, tail_len
        )
        assert new_num_targets is not None
        new_lengths = new_offsets[1:] - new_offsets[:-1]
        expected = torch.clamp(torch.full_like(new_lengths, 10), max=new_lengths)
        torch.testing.assert_close(new_num_targets, expected)

    def test_num_targets_none_is_preserved(self) -> None:
        """num_targets=None stays None through truncation."""
        x_lengths = torch.tensor([10, 14, 6], dtype=torch.int64)
        tail_len = 5
        _, _, new_num_targets, _, _, _ = self._forward_and_check_truncation(
            x_lengths, None, tail_len
        )
        self.assertIsNone(new_num_targets)

    def test_return_signature_is_four_tuple(self) -> None:
        """Forward always returns a 4-tuple.

        Even when truncation is disabled; callers must unpack.
        """
        stack, D = self._make_stack(
            num_layers=2,
            truncate_split_layer=0,
            truncate_tail_len=0,
        )
        x_lengths = torch.tensor([4, 6], dtype=torch.int64)
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        x = torch.randn(int(x_offsets[-1].item()), D)
        out = stack(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=6,
            num_targets=torch.tensor([1, 2], dtype=torch.int64),
        )
        self.assertEqual(len(out), 4)
        returned_x, returned_offsets, returned_num_targets, returned_max = out
        # No truncation: offsets/max are the inputs, num_targets preserved.
        torch.testing.assert_close(returned_offsets, x_offsets)
        self.assertEqual(returned_max, 6)
        self.assertEqual(returned_x.size(0), x.size(0))
        assert returned_num_targets is not None

    def test_sla_on_triton_kernel_raises(self) -> None:
        """C4: STUStack surfaces SLA-on-Kernel.TRITON as a loud error.

        CUTLASS and PyTorch both support the NFUNC path; Triton does
        not, so SLA + Kernel.TRITON must raise before any kernel call.
        """
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        stu_list: List[STU] = [
            STULayer(
                embedding_dim=16,
                num_heads=2,
                hidden_dim=32,
                attention_dim=32,
                causal=True,
                target_aware=True,
                sla_k1=8,
                sla_k2=4,
                contextual_seq_len=0,
                is_inference=False,
            )
        ]
        stack = STUStack(stu_list=stu_list)
        stack.set_kernel(Kernel.TRITON)  # not supported for SLA
        x_lengths = torch.tensor([6], dtype=torch.int64)
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        x = torch.randn(int(x_offsets[-1].item()), 16)
        with self.assertRaisesRegex(ValueError, "Kernel.TRITON"):
            stack(
                x=x,
                x_offsets=x_offsets,
                max_seq_len=6,
                num_targets=torch.tensor([1], dtype=torch.int64),
            )


if __name__ == "__main__":
    unittest.main()
