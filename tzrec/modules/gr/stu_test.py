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
from parameterized import parameterized

from tzrec.ops import Kernel
from tzrec.utils.test_util import (
    gpu_unavailable,
    mark_ci_scope,
    reference_stu_truncation,
)


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


@mark_ci_scope("h20")
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
        _ = stu(
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

    def _make_sla_layer(self, sla_k1: int = 0, sla_k2: int = 0):
        from tzrec.modules.gr.stu import STULayer

        layer = STULayer(
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
            sla_k1=sla_k1,
            sla_k2=sla_k2,
            is_inference=False,
        )
        layer.set_kernel(Kernel.PYTORCH)
        return layer

    def _sla_inputs(self):
        x_lengths = torch.tensor([4, 6], dtype=torch.int64)
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        x = torch.randn(int(x_offsets[-1].item()), 16)
        num_targets = torch.tensor([1, 2], dtype=torch.int64)
        return x, x_offsets, int(x_lengths.max().item()), num_targets

    def test_consecutive_same_sig_layers_share_func(self) -> None:
        """A second SLA layer with matching sig reuses the prev tensor."""
        layer1 = self._make_sla_layer(sla_k1=4, sla_k2=2)
        layer2 = self._make_sla_layer(sla_k1=4, sla_k2=2)
        x, x_offsets, max_seq_len, num_targets = self._sla_inputs()

        out1, attn_func1 = layer1(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        _, attn_func2 = layer2(
            x=out1,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
            prev_attn_func=attn_func1,
            prev_attn_func_sig=layer1.attn_func_static_sig,
        )
        self.assertEqual(layer2.attn_func_static_sig, layer1.attn_func_static_sig)
        # Tensor identity confirms layer2 reused rather than rebuilt.
        self.assertIs(attn_func2, attn_func1)

    def test_distinct_sig_rebuilds(self) -> None:
        """A second SLA layer with a different sig builds a fresh tensor."""
        layer1 = self._make_sla_layer(sla_k1=4, sla_k2=2)
        layer2 = self._make_sla_layer(sla_k1=8, sla_k2=2)  # different k1
        x, x_offsets, max_seq_len, num_targets = self._sla_inputs()

        out1, attn_func1 = layer1(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )
        _, attn_func2 = layer2(
            x=out1,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
            prev_attn_func=attn_func1,
            prev_attn_func_sig=layer1.attn_func_static_sig,
        )
        self.assertNotEqual(layer2.attn_func_static_sig, layer1.attn_func_static_sig)
        self.assertIsNot(attn_func2, attn_func1)

    def test_sla_on_triton_kernel_raises(self) -> None:
        """SLA paths require CUTLASS or PYTORCH; Triton has no NFUNC kernel."""
        layer = self._make_sla_layer(sla_k1=4, sla_k2=2)
        layer.set_kernel(Kernel.TRITON)
        x, x_offsets, max_seq_len, num_targets = self._sla_inputs()
        with self.assertRaisesRegex(ValueError, "Kernel.TRITON"):
            layer(
                x=x,
                x_offsets=x_offsets,
                max_seq_len=max_seq_len,
                num_targets=num_targets,
            )


@mark_ci_scope("h20")
class STUStackTruncationTest(unittest.TestCase):
    """Cover the mid-stack truncation block in ``STUStack.forward``.

    The truncation branch rewrites ``x`` / ``x_offsets`` / ``max_seq_len``
    and has no coverage from ``StuTest`` (which leaves ``truncate_*`` at
    their defaults).
    """

    _LAYER_KWARGS = dict(
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

    def _make_stack(
        self,
        num_layers: int,
        truncate_split_layer: int,
        truncate_tail_len: int,
        contextual_seq_len: int = 0,
        sla_k1: int = 0,
        sla_k2: int = 0,
    ):
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        kwargs = dict(self._LAYER_KWARGS)
        kwargs["contextual_seq_len"] = contextual_seq_len
        kwargs["sla_k1"] = sla_k1
        kwargs["sla_k2"] = sla_k2
        embedding_dim = kwargs["embedding_dim"]
        layers: List[STU] = [STULayer(**kwargs) for _ in range(num_layers)]
        stack = STUStack(
            stu_list=layers,
            truncate_split_layer=truncate_split_layer,
            truncate_tail_len=truncate_tail_len,
            is_inference=False,
        )
        stack.set_kernel(Kernel.PYTORCH)
        return stack, embedding_dim

    def test_init_validates_split_layer_bounds(self) -> None:
        """Both bounds + symmetric (only-one-positive rejected)."""
        from tzrec.modules.gr.stu import STU, STULayer, STUStack

        stu_list: List[STU] = [STULayer(**self._LAYER_KWARGS) for _ in range(3)]
        # split_layer must be > 0 when tail_len > 0.
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=0,
                truncate_tail_len=4,
            )
        # split_layer must be < len(stu_list).
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=3,
                truncate_tail_len=4,
            )
        # Asymmetric: setting split_layer without tail_len.
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=1,
                truncate_tail_len=0,
            )
        # Asymmetric: setting tail_len without split_layer.
        with self.assertRaises(ValueError):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=0,
                truncate_tail_len=4,
            )
        # Negative pair: would XOR-equal to False and silently disable.
        with self.assertRaisesRegex(ValueError, "non-negative"):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=-1,
                truncate_tail_len=-1,
            )
        # Single negative: rejected before the XOR check.
        with self.assertRaisesRegex(ValueError, "non-negative"):
            STUStack(
                stu_list=stu_list,
                truncate_split_layer=-1,
                truncate_tail_len=4,
            )
        # Valid: enabled.
        STUStack(stu_list=stu_list, truncate_split_layer=1, truncate_tail_len=4)
        # Valid: disabled (defaults).
        STUStack(stu_list=stu_list)

    def _forward(
        self,
        stack,
        embedding_dim: int,
        x_lengths: torch.Tensor,
        num_targets,
    ):
        x_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
        total = int(x_offsets[-1].item())
        x = torch.randn(total, embedding_dim)
        max_seq_len = int(x_lengths.max().item())
        return stack(
            x=x,
            x_offsets=x_offsets,
            max_seq_len=max_seq_len,
            num_targets=num_targets,
        )

    @parameterized.expand(
        [
            # split, tail, num_targets, lengths, expects_plan
            [0, 0, [1, 2], [4, 6], False],  # disabled
            [1, 8, [2, 2, 2, 2], [12, 20, 5, 30], True],  # truncated
            [1, 16, [1, 1, 1], [3, 5, 2], True],  # no-op (plan still returned)
            [1, 8, None, [12, 20, 5, 30], True],  # listwise (num_targets=None)
        ]
    )
    def test_forward_threads_truncation_metadata(
        self, split, tail, targets, lengths, expects_plan
    ) -> None:
        """Stack returns ``(x, offsets, max, plan)`` and threads num_targets.

        Asserts post-truncation lengths against an independent Python
        reference, so a regression that silently corrupts ``x``'s row
        count fails (the trivial ``out.size(0) == new_offsets[-1]``
        identity would otherwise pass).
        """
        stack, D = self._make_stack(
            num_layers=3, truncate_split_layer=split, truncate_tail_len=tail
        )
        x_lengths = torch.tensor(lengths, dtype=torch.int64)
        num_targets = (
            torch.tensor(targets, dtype=torch.int64) if targets is not None else None
        )
        out, new_offsets, new_max, plan = self._forward(
            stack, D, x_lengths, num_targets
        )
        if split > 0 and tail > 0:
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(x_lengths)
            _, expected_lens = reference_stu_truncation(
                torch.zeros(int(offsets[-1].item()), 1),
                offsets,
                targets,
                truncate_tail_len=tail,
            )
        else:
            expected_lens = list(lengths)
        self.assertEqual(plan is not None, expects_plan)
        self.assertEqual((new_offsets[1:] - new_offsets[:-1]).tolist(), expected_lens)
        self.assertEqual(out.size(0), sum(expected_lens))
        self.assertEqual(new_max, max(expected_lens))

    def test_cached_forward_raises_on_truncation(self) -> None:
        """Train/serve firewall: ``cached_forward`` refuses truncation."""
        stack, D = self._make_stack(
            num_layers=3, truncate_split_layer=1, truncate_tail_len=4
        )
        delta_x = torch.randn(8, D)
        num_targets = torch.tensor([2, 2], dtype=torch.int64)
        with self.assertRaisesRegex(NotImplementedError, "truncation"):
            stack.cached_forward(
                delta_x=delta_x,
                num_targets=num_targets,
            )

    def test_truncation_with_sla_runs(self) -> None:
        """SLA-enabled stack with truncation produces a finite output.

        Ensures that resetting ``prev_attn_func`` across the truncation
        boundary lets the next layer's cache-miss path rebuild the SLA
        func tensor against the truncated offsets without crashing.
        """
        stack, D = self._make_stack(
            num_layers=3,
            truncate_split_layer=1,
            truncate_tail_len=6,
            sla_k1=4,
            sla_k2=2,
        )
        x_lengths = torch.tensor([16, 20, 8], dtype=torch.int64)
        num_targets = torch.tensor([2, 2, 2], dtype=torch.int64)
        out, new_offsets, _, plan = self._forward(stack, D, x_lengths, num_targets)
        self.assertIsNotNone(plan)
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(out.size(0), int(new_offsets[-1].item()))


if __name__ == "__main__":
    unittest.main()
