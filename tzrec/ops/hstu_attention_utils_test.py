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

Numerical-correctness tests are parametrized over
``TestGraphType.NORMAL`` and ``TestGraphType.FX_TRACE``.
"""

import itertools
import unittest
from typing import Optional

import torch
from parameterized import parameterized
from torch import nn

from tzrec.ops import Kernel
from tzrec.ops.hstu_attention_utils import (
    apply_stu_truncation_plan,
    build_sla_func_tensor,
    compute_stu_truncation_plan,
)
from tzrec.utils.test_util import (
    TestGraphType,
    create_test_module,
    reference_stu_truncation,
)

# Cross-product helper: produce one (graph_type, *case) row per
# combination so a single ``parameterized.expand`` covers both eager
# and fx-trace lanes for every input case.
_GRAPH_TYPES = [TestGraphType.NORMAL, TestGraphType.FX_TRACE]


def _xprod(cases):
    return [(gt, *case) for gt in _GRAPH_TYPES for case in cases]


class _BuildSlaFuncTensorWrapper(nn.Module):
    def __init__(
        self,
        sla_k1: int,
        sla_k2: int,
        contextual_seq_len: int = 0,
        nheads: int = 1,
        target_aware: bool = False,
    ) -> None:
        super().__init__()
        self._sla_k1 = sla_k1
        self._sla_k2 = sla_k2
        self._contextual_seq_len = contextual_seq_len
        self._nheads = nheads
        self._target_aware = target_aware

    def forward(
        self,
        x: torch.Tensor,
        seq_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        return build_sla_func_tensor(
            nheads=self._nheads,
            sla_k1=self._sla_k1,
            sla_k2=self._sla_k2,
            seq_offsets=seq_offsets,
            total_q=x.size(0),
            num_targets=num_targets if self._target_aware else None,
            contextual_seq_len=self._contextual_seq_len,
        )


class _StuTruncationWrapper(nn.Module):
    def __init__(
        self,
        truncate_tail_len: int,
        contextual_seq_len: int = 0,
        max_seq_len: int = 32,
        target_aware: bool = False,
    ) -> None:
        super().__init__()
        self._tail = truncate_tail_len
        self._ctx = contextual_seq_len
        self._max_seq_len = max_seq_len
        self._target_aware = target_aware

    def forward(
        self,
        x: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        plan = compute_stu_truncation_plan(
            x_offsets=x_offsets,
            num_targets=num_targets if self._target_aware else None,
            max_seq_len=self._max_seq_len,
            truncate_tail_len=self._tail,
            contextual_seq_len=self._ctx,
        )
        return apply_stu_truncation_plan(x, plan, kernel=Kernel.PYTORCH)


class _ReplayTruncationWrapper(nn.Module):
    def __init__(
        self,
        truncate_tail_len: int,
        contextual_seq_len: int = 0,
        max_seq_len: int = 32,
        target_aware: bool = False,
    ) -> None:
        super().__init__()
        self._tail = truncate_tail_len
        self._ctx = contextual_seq_len
        self._max_seq_len = max_seq_len
        self._target_aware = target_aware

    def forward(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        x_offsets: torch.Tensor,
        num_targets: Optional[torch.Tensor],
    ) -> torch.Tensor:
        plan = compute_stu_truncation_plan(
            x_offsets=x_offsets,
            num_targets=num_targets if self._target_aware else None,
            max_seq_len=self._max_seq_len,
            truncate_tail_len=self._tail,
            contextual_seq_len=self._ctx,
        )
        out_x = apply_stu_truncation_plan(x, plan, kernel=Kernel.PYTORCH)
        out_ts = apply_stu_truncation_plan(ts, plan, kernel=Kernel.PYTORCH)
        return torch.cat([out_x, out_ts], dim=-1)


class BuildSlaFuncTensorTest(unittest.TestCase):
    """Verify the per-position interval values written to the func tensor.

    Each test runs once eager (NORMAL) and once after fx symbolic-trace
    (FX_TRACE); both must produce the same expected values.
    """

    def _build(
        self,
        seq_lengths,
        sla_k1: int,
        sla_k2: int,
        nheads: int = 1,
        num_targets=None,
        contextual_seq_len: int = 0,
        graph_type: TestGraphType = TestGraphType.NORMAL,
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
        # Dummy x of the right batch size so wrapper.total_q = x.size(0)
        # mirrors STULayer.forward's call pattern (Proxy at fx-trace time).
        x = torch.zeros(total_q, 1)
        wrapper = _BuildSlaFuncTensorWrapper(
            sla_k1=sla_k1,
            sla_k2=sla_k2,
            nheads=nheads,
            contextual_seq_len=contextual_seq_len,
            target_aware=(targets_t is not None),
        )
        module = create_test_module(wrapper, graph_type)
        return module(x, seq_offsets, targets_t)

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_history_intervals_match_spec(self, graph_type: TestGraphType) -> None:
        """For history rows, interval values follow the SLA formulas."""
        L, K1, K2 = 8, 4, 2
        func = self._build([L], sla_k1=K1, sla_k2=K2, graph_type=graph_type)
        # All heads share the same row (build_sla_func_tensor expands).
        col_max0 = func[0, 0]  # (L,)
        col_min0 = func[0, 1]
        col_max1 = func[0, 2]
        for q in range(L):
            self.assertEqual(col_max0[q].item(), min(K2, q + 1), f"q={q}")
            self.assertEqual(col_min0[q].item(), max(K2, q - K1 + 1), f"q={q}")
            self.assertEqual(col_max1[q].item(), q + 1, f"q={q}")

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_target_rows_collapse_to_history_boundary(
        self, graph_type: TestGraphType
    ) -> None:
        """Target rows have all three interval bounds == history boundary."""
        L, T = 6, 2
        func = self._build(
            [L], sla_k1=4, sla_k2=2, num_targets=[T], graph_type=graph_type
        )
        H = L - T  # = 4
        # Target row indices: [4, 5]; history rows: [0..3].
        col_max0 = func[0, 0]
        col_min0 = func[0, 1]
        col_max1 = func[0, 2]
        for q in range(H, L):
            self.assertEqual(col_max0[q].item(), H)
            self.assertEqual(col_min0[q].item(), H)
            self.assertEqual(col_max1[q].item(), H)

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_contextual_seq_len_overrides_sla_k2(
        self, graph_type: TestGraphType
    ) -> None:
        """``effective_k2 = max(sla_k2, contextual_seq_len)``."""
        L = 10
        func = self._build(
            [L], sla_k1=2, sla_k2=2, contextual_seq_len=8, graph_type=graph_type
        )
        # effective_k2 = 8 → for q=4, col_max0 = min(8, 5) = 5; for q=9,
        # col_max0 = min(8, 10) = 8.
        self.assertEqual(func[0, 0, 4].item(), 5)
        self.assertEqual(func[0, 0, 9].item(), 8)
        # col_min0 = max(8, q - K1 + 1) -- for K1=2, q=9: max(8, 8) = 8.
        self.assertEqual(func[0, 1, 9].item(), 8)

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_clamps_negative_history_boundary(self, graph_type: TestGraphType) -> None:
        """``num_targets > seq_length`` must not yield negative bounds.

        Silent-NaN guard: without the clamp, attention denominators go
        to zero and outputs become NaN with no signal upstream.
        """
        L = 4
        func = self._build(
            [L], sla_k1=4, sla_k2=2, num_targets=[10], graph_type=graph_type
        )
        # H_boundary clamped to 0; every row is a "target" row with bound 0.
        for q in range(L):
            self.assertEqual(func[0, 0, q].item(), 0)
            self.assertEqual(func[0, 1, q].item(), 0)
            self.assertEqual(func[0, 2, q].item(), 0)

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_int32_offsets_skip_cast(self, graph_type: TestGraphType) -> None:
        """int32 offsets pass through; int64 offsets get cast.

        Both calls use the same wrapper module, just with different
        offsets dtype -- the unconditional ``seq_offsets.to(int32)``
        in build_sla_func_tensor must produce identical content.
        """
        L = 4
        out_i32 = self._build([L], sla_k1=2, sla_k2=1, graph_type=graph_type)
        # Same call with int64 offsets should produce identical content.
        seq_offsets_i64 = torch.tensor([0, L], dtype=torch.int64)
        x = torch.zeros(L, 1)
        wrapper_i64 = _BuildSlaFuncTensorWrapper(sla_k1=2, sla_k2=1)
        module_i64 = create_test_module(wrapper_i64, graph_type)
        out_from_i64 = module_i64(x, seq_offsets_i64, None)
        torch.testing.assert_close(out_i32, out_from_i64)

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_per_head_values_are_identical(self, graph_type: TestGraphType) -> None:
        """The expand(nheads,...) means every head sees the same intervals."""
        func = self._build([6], sla_k1=3, sla_k2=2, nheads=4, graph_type=graph_type)
        for h in range(1, 4):
            torch.testing.assert_close(func[0], func[h])

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_jagged_layout_across_batch(self, graph_type: TestGraphType) -> None:
        """Multi-sample batch: interval values are computed per local pos."""
        # Two samples, lengths 3 and 5. Total = 8.
        func = self._build([3, 5], sla_k1=3, sla_k2=1, graph_type=graph_type)
        # First sample occupies global [0, 3); second [3, 8).
        # Local pos 0 in each sample → col_max1 == 1.
        self.assertEqual(func[0, 2, 0].item(), 1)  # sample 0, local q=0
        self.assertEqual(func[0, 2, 3].item(), 1)  # sample 1, local q=0
        # Local pos 2 in each sample → col_max1 == 3.
        self.assertEqual(func[0, 2, 2].item(), 3)
        self.assertEqual(func[0, 2, 5].item(), 3)


# Input cases for StuTruncationTest.test_matches_reference; cross-product
# with TestGraphType yields the parametrize table.
_TRUNCATION_CASES = [
    # lengths, targets, contextual_seq_len, truncate_tail_len
    [[3, 5, 2], None, 0, 16],  # no-op (tail >> U)
    [[4, 10, 7], None, 0, 5],  # head drop, no ctx, no targets
    [[8, 12], [3, 5], 0, 2],  # targets-aware, no ctx
    [[6, 12, 20], [1, 2, 4], 3, 4],  # ctx + targets + UIH cap
    [[6, 8], [1, 2], 2, 0],  # tail=0 drops all UIH
]


class StuTruncationTest(unittest.TestCase):
    """``compute_stu_truncation_plan`` + ``apply_stu_truncation_plan``.

    Numerical-correctness tests are parametrized over
    (graph_type, input_case): each input case runs once eagerly and
    once after fx symbolic-trace.  ``test_validation_raises_on_negative_params``
    is eager-only (validation lives at construction time, before any
    forward / trace).
    """

    @parameterized.expand(_xprod(_TRUNCATION_CASES))
    def test_matches_reference(
        self,
        graph_type: TestGraphType,
        lengths,
        targets,
        ctx,
        tail,
    ) -> None:
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor(lengths, dtype=torch.int64)
        )
        x = torch.randn(int(offsets[-1].item()), 4)
        num_targets = (
            torch.tensor(targets, dtype=torch.int64) if targets is not None else None
        )
        ref_x, ref_lens = reference_stu_truncation(x, offsets, targets, tail, ctx)

        wrapper = _StuTruncationWrapper(
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
            max_seq_len=int(max(lengths)),
            target_aware=(num_targets is not None),
        )
        module = create_test_module(wrapper, graph_type)
        out_x = module(x, offsets, num_targets)
        torch.testing.assert_close(out_x, ref_x)

        # Plan-level fields are only directly observable via the eager
        # plan object; checking once per case (regardless of graph_type)
        # is sufficient since plan construction is deterministic.
        plan = compute_stu_truncation_plan(
            x_offsets=offsets,
            num_targets=num_targets,
            max_seq_len=int(max(lengths)),
            truncate_tail_len=tail,
            contextual_seq_len=ctx,
        )
        self.assertEqual(plan.new_lengths.tolist(), ref_lens)
        self.assertEqual(plan.new_max_seq_len, max(ref_lens))
        self.assertEqual(
            plan.new_x_offsets.tolist(),
            list(itertools.accumulate([0] + ref_lens)),
        )

    def test_validation_raises_on_negative_params(self) -> None:
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor([6], dtype=torch.int64)
        )
        with self.assertRaisesRegex(ValueError, "truncate_tail_len"):
            compute_stu_truncation_plan(
                x_offsets=offsets,
                num_targets=None,
                max_seq_len=6,
                truncate_tail_len=-1,
            )
        with self.assertRaisesRegex(ValueError, "contextual_seq_len"):
            compute_stu_truncation_plan(
                x_offsets=offsets,
                num_targets=None,
                max_seq_len=6,
                truncate_tail_len=4,
                contextual_seq_len=-1,
            )

    @parameterized.expand([(gt,) for gt in _GRAPH_TYPES])
    def test_replay_on_parallel_jagged(self, graph_type: TestGraphType) -> None:
        """A single plan can be applied to multiple parallel jagged tensors.

        Use case: ``HSTUTransducer.forward`` reuses the plan from
        ``STUStack.forward`` to truncate ``seq_timestamps`` with the
        same offsets used on the embeddings.
        """
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
            torch.tensor([8, 12], dtype=torch.int64)
        )
        total = int(offsets[-1].item())
        x = torch.randn(total, 4)
        # Parallel tensor (D=1): row i holds 7.0 * i, so positional alignment
        # after truncation is checkable with a simple scalar multiply.
        ts = torch.arange(total, dtype=torch.float32).mul(7.0).unsqueeze(-1)
        num_targets = torch.tensor([2, 3], dtype=torch.int64)

        wrapper = _ReplayTruncationWrapper(
            truncate_tail_len=3,
            contextual_seq_len=0,
            max_seq_len=12,
            target_aware=True,  # this test always passes num_targets
        )
        module = create_test_module(wrapper, graph_type)
        # Wrapper concatenates the two outputs on dim=-1; split here.
        cat = module(x, ts, offsets, num_targets)
        out_x = cat[:, :4]
        out_ts = cat[:, 4:]

        ref_x, _ = reference_stu_truncation(x, offsets, [2, 3], truncate_tail_len=3)
        ref_ts, _ = reference_stu_truncation(ts, offsets, [2, 3], truncate_tail_len=3)
        torch.testing.assert_close(out_x, ref_x)
        torch.testing.assert_close(out_ts, ref_ts)


if __name__ == "__main__":
    unittest.main()
