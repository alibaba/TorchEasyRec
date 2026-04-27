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
every CI lane has direct coverage of the NFUNC mask construction without
depending on the GPU attention kernels.
"""

import unittest

import torch

from tzrec.ops.hstu_attention_utils import build_sla_func_tensor


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


if __name__ == "__main__":
    unittest.main()
