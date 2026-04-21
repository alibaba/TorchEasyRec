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

"""Unit tests for ``tzrec.ops.hstu_attention_utils``.

These tests exercise the helper functions in isolation (CPU only), so
every CI lane has direct coverage of the NFUNC mask construction and
the truncation slicing logic without depending on the GPU attention
kernels.
"""

import unittest

import torch

from tzrec.ops import Kernel
from tzrec.ops.hstu_attention_utils import apply_truncation, build_sla_func_tensor


class BuildSlaFuncTensorTest(unittest.TestCase):
    """Verify the per-position interval values written to the func tensor."""

    def _build(
        self,
        seq_lengths,
        sla_k1: int,
        sla_k2: int,
        nheads: int = 1,
        num_targets=None,
        contextual_seq_len: int = 0,
    ) -> torch.Tensor:
        seq_offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(seq_lengths, dtype=torch.int32)
        )
        total_q = int(sum(seq_lengths))
        targets_t = (
            torch.tensor(num_targets, dtype=torch.int32)
            if num_targets is not None
            else None
        )
        return build_sla_func_tensor(
            nheads=nheads,
            sla_k1=sla_k1,
            sla_k2=sla_k2,
            seq_offsets=seq_offsets,
            total_q=total_q,
            num_targets=targets_t,
            contextual_seq_len=contextual_seq_len,
        )

    def test_history_intervals_match_spec(self) -> None:
        """For history rows, interval values follow the SLA formulas."""
        L, K1, K2 = 8, 4, 2
        func = self._build([L], sla_k1=K1, sla_k2=K2)
        # All heads share the same row (build_sla_func_tensor expands).
        col_max0 = func[0, 0]  # (L,)
        col_min0 = func[0, 1]
        col_max1 = func[0, 2]
        for q in range(L):
            self.assertEqual(col_max0[q].item(), min(K2, q + 1), f"q={q}")
            self.assertEqual(col_min0[q].item(), max(K2, q - K1 + 1), f"q={q}")
            self.assertEqual(col_max1[q].item(), q + 1, f"q={q}")

    def test_target_rows_collapse_to_history_boundary(self) -> None:
        """Target rows have all three interval bounds == history boundary."""
        L, T = 6, 2
        func = self._build([L], sla_k1=4, sla_k2=2, num_targets=[T])
        H = L - T  # = 4
        # Target row indices: [4, 5]; history rows: [0..3].
        col_max0 = func[0, 0]
        col_min0 = func[0, 1]
        col_max1 = func[0, 2]
        for q in range(H, L):
            self.assertEqual(col_max0[q].item(), H)
            self.assertEqual(col_min0[q].item(), H)
            self.assertEqual(col_max1[q].item(), H)

    def test_contextual_seq_len_overrides_sla_k2(self) -> None:
        """``effective_k2 = max(sla_k2, contextual_seq_len)``."""
        L = 10
        func = self._build([L], sla_k1=2, sla_k2=2, contextual_seq_len=8)
        # effective_k2 = 8 → for q=4, col_max0 = min(8, 5) = 5; for q=9,
        # col_max0 = min(8, 10) = 8.
        self.assertEqual(func[0, 0, 4].item(), 5)
        self.assertEqual(func[0, 0, 9].item(), 8)
        # col_min0 = max(8, q - K1 + 1) -- for K1=2, q=9: max(8, 8) = 8.
        self.assertEqual(func[0, 1, 9].item(), 8)

    def test_clamps_negative_history_boundary(self) -> None:
        """``num_targets > seq_length`` must not yield negative bounds.

        Silent-NaN guard: without the clamp, attention denominators go
        to zero and outputs become NaN with no signal upstream.
        """
        L = 4
        func = self._build([L], sla_k1=4, sla_k2=2, num_targets=[10])
        # H_boundary clamped to 0; every row is a "target" row with bound 0.
        for q in range(L):
            self.assertEqual(func[0, 0, q].item(), 0)
            self.assertEqual(func[0, 1, q].item(), 0)
            self.assertEqual(func[0, 2, q].item(), 0)

    def test_negative_params_raise(self) -> None:
        seq_offsets = torch.tensor([0, 4, 8], dtype=torch.int32)
        for sla_k1, sla_k2, ctx in [(-1, 4, 0), (4, -1, 0), (4, 4, -1)]:
            with self.assertRaisesRegex(ValueError, "non-negative"):
                build_sla_func_tensor(
                    nheads=2,
                    sla_k1=sla_k1,
                    sla_k2=sla_k2,
                    seq_offsets=seq_offsets,
                    total_q=8,
                    contextual_seq_len=ctx,
                )

    def test_int32_offsets_skip_cast(self) -> None:
        """int32 offsets pass through; int64 offsets get cast."""
        L = 4
        out_i32 = self._build([L], sla_k1=2, sla_k2=1)  # already int32
        # Same call with int64 offsets should produce identical content.
        seq_offsets_i64 = torch.tensor([0, L], dtype=torch.int64)
        out_from_i64 = build_sla_func_tensor(
            nheads=1,
            sla_k1=2,
            sla_k2=1,
            seq_offsets=seq_offsets_i64,
            total_q=L,
        )
        torch.testing.assert_close(out_i32, out_from_i64)

    def test_per_head_values_are_identical(self) -> None:
        """The expand(nheads,...) means every head sees the same intervals."""
        func = self._build([6], sla_k1=3, sla_k2=2, nheads=4)
        for h in range(1, 4):
            torch.testing.assert_close(func[0], func[h])

    def test_jagged_layout_across_batch(self) -> None:
        """Multi-sample batch: interval values are computed per local pos."""
        # Two samples, lengths 3 and 5. Total = 8.
        func = self._build([3, 5], sla_k1=3, sla_k2=1)
        # First sample occupies global [0, 3); second [3, 8).
        # Local pos 0 in each sample → col_max1 == 1.
        self.assertEqual(func[0, 2, 0].item(), 1)  # sample 0, local q=0
        self.assertEqual(func[0, 2, 3].item(), 1)  # sample 1, local q=0
        # Local pos 2 in each sample → col_max1 == 3.
        self.assertEqual(func[0, 2, 2].item(), 3)
        self.assertEqual(func[0, 2, 5].item(), 3)


class ApplyTruncationTest(unittest.TestCase):
    """Direct content-level coverage of ``apply_truncation``."""

    def _id_marked_input(self, lengths):
        """Build (x, offsets) where row i carries value float(i)."""
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(lengths, dtype=torch.int64)
        )
        total = int(offsets[-1].item())
        x = torch.arange(total, dtype=torch.float32).view(total, 1).repeat(1, 4)
        return x, offsets

    def test_no_op_when_all_shorter(self) -> None:
        lengths = [3, 5, 2]
        x, offsets = self._id_marked_input(lengths)
        out_x, out_offsets, out_lens, out_num, out_max = apply_truncation(
            x=x,
            x_offsets=offsets,
            seq_lengths=offsets[1:] - offsets[:-1],
            num_targets=None,
            max_seq_len=int(max(lengths)),
            truncate_tail_len=16,
        )
        # No row dropped; positions identical.
        torch.testing.assert_close(out_x, x)
        torch.testing.assert_close(out_offsets, offsets)
        self.assertEqual(out_max, 16)

    def test_simple_head_drop_no_contextual(self) -> None:
        """contextual_seq_len=0: keep last tail_len rows per sample."""
        lengths = [4, 10, 7]
        tail = 5
        x, offsets = self._id_marked_input(lengths)
        out_x, out_offsets, _, _, _ = apply_truncation(
            x=x,
            x_offsets=offsets,
            seq_lengths=offsets[1:] - offsets[:-1],
            num_targets=None,
            max_seq_len=int(max(lengths)),
            truncate_tail_len=tail,
        )
        for b in range(len(lengths)):
            new_len = min(lengths[b], tail)
            keep_start_global = int(offsets[b + 1].item()) - new_len
            sample_start = int(out_offsets[b].item())
            for k in range(new_len):
                self.assertEqual(
                    out_x[sample_start + k, 0].item(),
                    float(keep_start_global + k),
                )

    def test_preserves_contextual_prefix(self) -> None:
        """contextual_seq_len > 0: prefix kept, UIH head dropped."""
        lengths = [6, 12, 20]
        tail = 8
        ctx = 3
        x, offsets = self._id_marked_input(lengths)
        seq_lengths = offsets[1:] - offsets[:-1]
        out_x, out_offsets, _, _, _ = apply_truncation(
            x=x,
            x_offsets=offsets,
            seq_lengths=seq_lengths,
            num_targets=None,
            max_seq_len=int(seq_lengths.max().item()),
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
        )
        for b in range(len(lengths)):
            new_len = min(lengths[b], tail)
            sample_start = int(out_offsets[b].item())
            # Prefix: global [offsets[b], offsets[b]+C).
            for j in range(ctx):
                self.assertEqual(
                    out_x[sample_start + j, 0].item(),
                    float(int(offsets[b].item()) + j),
                )
            # Tail: last (new_len - C) of the original sample.
            tail_count = new_len - ctx
            orig_end = int(offsets[b + 1].item())
            for k in range(tail_count):
                self.assertEqual(
                    out_x[sample_start + ctx + k, 0].item(),
                    float(orig_end - tail_count + k),
                )

    def test_num_targets_clamped_to_rest_tail_capacity(self) -> None:
        lengths = [10, 14]
        tail = 6
        ctx = 2  # rest-tail capacity = 6 - 2 = 4
        x, offsets = self._id_marked_input(lengths)
        num_targets = torch.tensor([8, 1], dtype=torch.int64)
        _, _, _, out_num, _ = apply_truncation(
            x=x,
            x_offsets=offsets,
            seq_lengths=offsets[1:] - offsets[:-1],
            num_targets=num_targets,
            max_seq_len=int(max(lengths)),
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
        )
        # Sample 0: clamp 8 -> 4 (rest-tail capacity). Sample 1: 1 stays.
        torch.testing.assert_close(out_num, torch.tensor([4, 1], dtype=torch.int64))

    def test_num_targets_none_is_preserved(self) -> None:
        lengths = [10, 14]
        x, offsets = self._id_marked_input(lengths)
        _, _, _, out_num, _ = apply_truncation(
            x=x,
            x_offsets=offsets,
            seq_lengths=offsets[1:] - offsets[:-1],
            num_targets=None,
            max_seq_len=int(max(lengths)),
            truncate_tail_len=6,
            contextual_seq_len=2,
        )
        self.assertIsNone(out_num)

    def test_validation_raises_when_tail_le_contextual(self) -> None:
        x, offsets = self._id_marked_input([6])
        with self.assertRaisesRegex(ValueError, "contextual_seq_len"):
            apply_truncation(
                x=x,
                x_offsets=offsets,
                seq_lengths=offsets[1:] - offsets[:-1],
                num_targets=None,
                max_seq_len=6,
                truncate_tail_len=4,
                contextual_seq_len=4,  # equal -> rest tail capacity 0
            )
        with self.assertRaisesRegex(ValueError, "contextual_seq_len"):
            apply_truncation(
                x=x,
                x_offsets=offsets,
                seq_lengths=offsets[1:] - offsets[:-1],
                num_targets=None,
                max_seq_len=6,
                truncate_tail_len=2,
                contextual_seq_len=4,  # tail < contextual
            )

    def test_kernel_param_threaded_through(self) -> None:
        """Smoke-test that kernel kwarg routes through both jagged ops."""
        x, offsets = self._id_marked_input([4, 6])
        for kernel in (Kernel.PYTORCH,):  # CPU lane: only PyTorch backend
            apply_truncation(
                x=x,
                x_offsets=offsets,
                seq_lengths=offsets[1:] - offsets[:-1],
                num_targets=None,
                max_seq_len=6,
                truncate_tail_len=4,
                contextual_seq_len=1,
                kernel=kernel,
            )


if __name__ == "__main__":
    unittest.main()
