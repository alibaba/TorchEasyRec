# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import torch

from tzrec.ops.scatter_ops import (
    lengths_to_index,
    scatter_logsumexp,
    scatter_max,
    scatter_sum,
)


class ScatterOpsTest(unittest.TestCase):
    """Contracts the consumers (losses, SD, task tower) rely on.

    Groups without a row sum to 0, max rewrites them to 0 (not -inf),
    reduced-precision inputs round-trip through fp32 accumulation, and the
    index may arrive in any order.
    """

    def test_scatter_sum_empty_group_is_zero(self) -> None:
        values = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
        index = torch.tensor([0, 0, 2, 2, 2])
        self.assertEqual(scatter_sum(values, index, 3).tolist(), [[3.0], [0.0], [12.0]])

    def test_scatter_sum_round_trips_reduced_precision(self) -> None:
        values = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.bfloat16)
        sums = scatter_sum(values, torch.tensor([0, 0, 0]), 1)
        self.assertEqual(sums.dtype, torch.bfloat16)
        self.assertEqual(sums.tolist(), [[6.0]])

    def test_scatter_max_rewrites_empty_group_to_zero(self) -> None:
        values = torch.tensor([[1.0], [5.0], [2.0], [7.0], [3.0]])
        index = torch.tensor([2, 0, 2, 0, 2])
        self.assertEqual(scatter_max(values, index, 3).tolist(), [[7.0], [0.0], [3.0]])

    def test_scatter_logsumexp_matches_torch_per_group(self) -> None:
        torch.manual_seed(0)
        values = torch.randn(7, 2, requires_grad=True)
        index = torch.tensor([1, 0, 1, 1, 0, 3, 3])
        out = scatter_logsumexp(values, index, 4)
        for group in (0, 1, 3):
            torch.testing.assert_close(
                out[group], torch.logsumexp(values[index == group], dim=0)
            )
        self.assertTrue(torch.isinf(out[2]).all() and (out[2] < 0).all())
        (grad,) = torch.autograd.grad(out[[0, 1, 3]].sum(), values)
        expected = torch.zeros_like(grad)
        for group in (0, 1, 3):
            m = index == group
            expected[m] = torch.softmax(values.detach()[m], dim=0)
        torch.testing.assert_close(grad, expected)


class LengthsToIndexTest(unittest.TestCase):
    def test_index_maps_rows_and_skips_empties(self) -> None:
        lengths = torch.tensor([2, 0, 3])
        ids = lengths_to_index(lengths)
        self.assertEqual(ids.tolist(), [0, 0, 2, 2, 2])

    def test_index_honors_output_size(self) -> None:
        lengths = torch.tensor([2, 0, 3])
        ids = lengths_to_index(lengths, output_size=5)
        self.assertEqual(ids.tolist(), [0, 0, 2, 2, 2])

    def test_scatter_logsumexp_bf16_tracks_fp32(self) -> None:
        torch.manual_seed(0)
        values = torch.randn(9, 3) * 4
        index = torch.tensor([0, 2, 2, 0, 1, 2, 0, 1, 1])
        out = scatter_logsumexp(values.bfloat16(), index, 3)
        self.assertEqual(out.dtype, torch.bfloat16)
        torch.testing.assert_close(
            out.float(), scatter_logsumexp(values, index, 3), atol=5e-2, rtol=2e-2
        )


if __name__ == "__main__":
    unittest.main()
