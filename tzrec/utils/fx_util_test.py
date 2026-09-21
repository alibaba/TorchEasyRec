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

"""Unit tests for ``tzrec.utils.fx_util``.

``fx_avg_counts`` carries the local/global rescaling factors behind
``enable_global_average_loss``, and ``DlrmHSTU.loss`` is its only caller.
Its distributed branch (``dist.all_reduce``) is unreachable from
single-process unittests, so it is pinned through the same
``mock.patch.object`` pattern as ``tzrec.utils.predict_util_test``.
"""

import os
import shutil
import subprocess
import sys
import unittest
from typing import Optional
from unittest import mock

import torch
import torch.distributed as dist
from parameterized import parameterized
from torchrec import KeyedJaggedTensor
from torchrec.modules.mc_modules import _mcc_lazy_init_inplace
from torchrec.quant.embedding_modules import _permute_kjt

from tzrec.utils import fx_util
from tzrec.utils.fx_util import fx_avg_counts
from tzrec.utils.test_util import (
    gpu_unavailable,
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)

# FX wrap registration is local to the caller's globals.
torch.fx.wrap(_mcc_lazy_init_inplace)
torch.fx.wrap(_permute_kjt)


class _PermuteKJT(torch.nn.Module):
    def __init__(self, cached_order: bool = False):
        super().__init__()
        self.register_buffer(
            "order",
            torch.tensor([0, 4, 5, 1, 2, 3], dtype=torch.int32)
            if cached_order
            else None,
        )

    def forward(
        self,
        values: torch.Tensor,
        lengths: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ):
        features = KeyedJaggedTensor(
            keys=["a", "b", "c", "d", "e", "f"],
            values=values,
            lengths=lengths,
            weights=weights,
            stride=1,
        )
        features = _mcc_lazy_init_inplace(
            features=features,
            feature_names=["a", "b", "c", "f", "d", "e"],
            features_order=[0, 1, 2, 5, 3, 4],
            created_feature_order=[True],
        )
        features = _permute_kjt(features, [0, 4, 5, 1, 2, 3], self.order)
        return (
            features.keys(),
            features.values(),
            features.lengths(),
            features.weights_or_none(),
        )


_NATIVE_SCRIPT_RUNNER = """
import importlib.util
from pathlib import Path
import sys
import torch

torch.set_num_threads(1)
root = Path(importlib.util.find_spec("fbgemm_gpu").origin).parent
print("Loading FBGEMM native operators", flush=True)
torch.ops.load_library(str(root / "fbgemm_gpu_py.so"))
assert "fbgemm_gpu" not in sys.modules
assert "tzrec" not in sys.modules
device = sys.argv[3]
print(f"Loading TorchScript on {device}", flush=True)
model = torch.jit.load(sys.argv[1], map_location=device)
cases = torch.load(sys.argv[2], weights_only=True)
with torch.no_grad():
    for index, (inputs, expected) in enumerate(cases):
        print(f"Running native case {index + 1}/{len(cases)}", flush=True)
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        actual = model(*inputs)
        assert len(actual) == len(expected)
        for result, reference in zip(actual, expected):
            if isinstance(reference, torch.Tensor):
                torch.testing.assert_close(result.cpu(), reference)
            else:
                assert result == reference, (result, reference)
print(f"{len(cases)} native TorchScript cases passed on {device}")
"""


@mark_ci_scope("gpu", "h20")
class KJTPermutationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir("fx_kjt_")
        self.addCleanup(shutil.rmtree, self.test_dir)

    @parameterized.expand(
        [("cpu", False), ("cpu", True), ("cuda:0", False), ("cuda:0", True)],
        name_func=parameterized_name_func,
    )
    def test_native_scripted_permutation(self, device, cached_order) -> None:
        if device.startswith("cuda") and gpu_unavailable[0]:
            self.skipTest(gpu_unavailable[1])
        gm = fx_util.symbolic_trace(_PermuteKJT(cached_order))
        model_path = os.path.join(self.test_dir, "model.pt")
        torch.jit.script(gm).save(model_path)

        cases = []
        order = [0, 3, 4, 1, 2, 5]
        for sizes in ([1] * 6, [2, 0, 1, 3, 0, 1], [0] * 6):
            for length_dtype in (torch.int32, torch.int64):
                lengths = torch.tensor(sizes, dtype=length_dtype)
                values = torch.arange(sum(sizes), dtype=torch.int64)
                segments = torch.split(values, sizes)
                expected_values = torch.cat([segments[i] for i in order])
                for weighted in (False, True):
                    weights = values.float() + 0.25 if weighted else None
                    cases.append(
                        (
                            [values, lengths, weights],
                            (
                                ["a", "d", "e", "b", "c", "f"],
                                expected_values,
                                lengths[order],
                                expected_values.float() + 0.25 if weighted else None,
                            ),
                        )
                    )
        cases_path = os.path.join(self.test_dir, "cases.pt")
        torch.save(cases, cases_path)
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    _NATIVE_SCRIPT_RUNNER,
                    model_path,
                    cases_path,
                    device,
                ],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except subprocess.TimeoutExpired as error:
            diagnostics = []
            for name, output in (("stdout", error.stdout), ("stderr", error.stderr)):
                if isinstance(output, bytes):
                    output = output.decode(errors="replace")
                diagnostics.append(f"{name}:\n{output or ''}")
            self.fail(
                f"Native TorchScript timed out after {error.timeout}s\n"
                + "\n".join(diagnostics)
            )
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)

    def test_retrace_preserves_weights_guards(self) -> None:
        gm = fx_util.symbolic_trace(_PermuteKJT())
        retraced = fx_util.symbolic_trace(gm)
        for graph in (gm.graph, retraced.graph):
            guards = [
                node
                for node in graph.nodes
                if node.target == fx_util._restore_unweighted_kjt
            ]
            self.assertEqual(len(guards), 2)
            graph.lint()
        model = torch.jit.script(retraced)
        actual = model(torch.arange(6), torch.ones(6, dtype=torch.int64))
        torch.testing.assert_close(actual[1], torch.tensor([0, 3, 4, 1, 2, 5]))
        self.assertIsNone(actual[3])

    @parameterized.expand([(False,), (True,)], name_func=parameterized_name_func)
    def test_torch_export_preserves_optional_weights(self, weighted) -> None:
        values = torch.arange(6)
        lengths = torch.ones(6, dtype=torch.int64)
        weights = values.float() + 0.25 if weighted else None
        args = (values, lengths, weights)
        model = _PermuteKJT(cached_order=True)
        expected = model(*args)
        gm = fx_util.symbolic_trace(model)
        exported = torch.export.export(gm, args)
        actual = exported.module()(*args)
        self.assertEqual(actual[0], expected[0])
        for result, reference in zip(actual[1:], expected[1:]):
            torch.testing.assert_close(result, reference)


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
