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

"""Unit tests for ``tzrec.utils.fx_util`` (CPU only).

``fx_avg_counts`` carries the local/global rescaling factors behind
``enable_global_average_loss``, and ``DlrmHSTU.loss`` is its only caller.
Its distributed branch (``dist.all_reduce``) is unreachable from
single-process unittests, so it is pinned through the same
``mock.patch.object`` pattern as ``tzrec.utils.predict_util_test``.
"""

import unittest
from unittest import mock

import torch
import torch.distributed as dist

from tzrec.utils import fx_util
from tzrec.utils.fx_util import fx_avg_counts


class FxAvgCountsTest(unittest.TestCase):
    """The local/global rescaling factors of ``enable_global_average_loss``."""

    def test_single_process_returns_local_counts(self) -> None:
        """Without an initialized process group the local counts are the answer."""
        if dist.is_initialized():
            self.skipTest("dist already initialized in this process")
        lengths = torch.tensor([3, 0, 4], dtype=torch.int64)
        out = fx_avg_counts(lengths)
        self.assertEqual(out.tolist(), [3.0, 7.0])
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, lengths.device)

    def test_empty_shard_is_reported_verbatim(self) -> None:
        """A rank with an empty shard contributes 0 to both averages."""
        out = fx_avg_counts(torch.zeros(0, dtype=torch.int64))
        self.assertEqual(out.tolist(), [0.0, 0.0])

    def test_dist_branch_averages_both_axes_in_one_reduction(self) -> None:
        """The distributed branch must reduce both counts with AVG semantics.

        Two ranks holding (3 requests, 10 candidates) and (5, 30) must read
        (4.0, 20.0). A SUM reduction would read double and halve the
        ``local / global`` loss rescaling factors -- silently, because the
        loss stays finite and the sign stays correct. The two axes share one
        collective, so a per-axis reduction would show up as a second call.
        """
        lengths = torch.tensor([4, 6], dtype=torch.int64)
        peer = torch.tensor([5.0, 30.0])
        with mock.patch.object(fx_util, "dist") as dist_mock:
            dist_mock.is_initialized.return_value = True
            # Hand the mock the real enum so the recorded call is assertable.
            dist_mock.ReduceOp.AVG = dist.ReduceOp.AVG
            dist_mock.all_reduce.side_effect = lambda outcome, op: outcome.copy_(
                (outcome + peer) / 2
            )
            out = fx_avg_counts(lengths)

        self.assertEqual(out.tolist(), [3.5, 20.0])
        dist_mock.all_reduce.assert_called_once()
        self.assertIs(dist_mock.all_reduce.call_args.kwargs["op"], dist.ReduceOp.AVG)
        # The reduced buffer is the returned tensor itself (in place).
        self.assertIs(out, dist_mock.all_reduce.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
