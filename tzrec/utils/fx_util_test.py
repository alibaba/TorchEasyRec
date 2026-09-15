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

"""Unit tests for ``tzrec.utils.fx_util`` (CPU only).

``fx_avg_batch_size`` was promoted here from a ``DlrmHSTU`` private helper
when ``listwise_rank_loss`` was generalized to ``RankModel``, so every rank
model now consumes it.  Its distributed branch (``dist.all_reduce``) is
unreachable from single-process unittests, so it is pinned through the same
``mock.patch.object`` pattern as ``tzrec.utils.predict_util_test``.
"""

import unittest
from unittest import mock

import torch
import torch.distributed as dist

from tzrec.utils import fx_util
from tzrec.utils.fx_util import fx_avg_batch_size


class FxAvgBatchSizeTest(unittest.TestCase):
    """The local/global rescaling factor of ``enable_global_average_loss``."""

    def test_single_process_returns_local_size(self) -> None:
        """Without an initialized process group the local size is the answer."""
        if dist.is_initialized():
            self.skipTest("dist already initialized in this process")
        x = torch.zeros(7)
        out = fx_avg_batch_size(x)
        self.assertEqual(out.item(), 7.0)
        self.assertEqual(out.dtype, torch.float32)
        self.assertEqual(out.device, x.device)

    def test_empty_shard_is_reported_verbatim(self) -> None:
        """A rank with an empty shard contributes 0 to the average."""
        out = fx_avg_batch_size(torch.zeros(0))
        self.assertEqual(out.item(), 0.0)

    def test_dist_branch_averages_across_ranks(self) -> None:
        """The distributed branch must reduce with AVG semantics.

        Two ranks with ragged shard sizes 3 and 5 must both read 4.0 as
        the global average; a SUM reduction would read 8.0 and double the
        ``local / global`` loss rescaling factor -- silently, because the
        loss stays finite and the sign stays correct.
        """
        x = torch.zeros(3)
        with mock.patch.object(fx_util, "dist") as dist_mock:
            dist_mock.is_initialized.return_value = True
            # Hand the mock the real enum so the recorded call is assertable.
            dist_mock.ReduceOp.AVG = dist.ReduceOp.AVG
            # Emulate ReduceOp.AVG against a peer that owns 5 rows.
            dist_mock.all_reduce.side_effect = lambda outcome, op: outcome.fill_(
                (outcome.item() + 5) / 2
            )
            out = fx_avg_batch_size(x)

        self.assertEqual(out.item(), 4.0)
        dist_mock.all_reduce.assert_called_once()
        self.assertIs(dist_mock.all_reduce.call_args.kwargs["op"], dist.ReduceOp.AVG)
        # The reduced buffer is the returned tensor itself (in place).
        self.assertIs(out, dist_mock.all_reduce.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
