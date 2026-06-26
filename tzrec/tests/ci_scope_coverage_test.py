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
"""Lint: every GPU-only test must carry a CI scope marker.

The per-PR GPU lane runs ``run.py --scope gpu`` and the H20 lane runs
``--scope h20``; the CPU lane runs the full suite but has neither a GPU nor
the ``requirements/extra.txt`` wheels (dynamicemb / fbgemm_gpu_hstu /
torch_fx_tool / tensorrt). So any test that *skips on the CPU lane* runs
**only** on a GPU lane -- and if it carries no ``@mark_ci_scope("gpu")`` /
``("h20")`` it is filtered out there too and silently runs on **no** per-PR
lane. This test statically asserts that can't happen.
"""

import ast
import glob
import os
import unittest

# Tokens whose presence in a skip decorator means the test needs a GPU lane
# (the resource -- a GPU, or an extra.txt-only wheel -- is absent on the CPU lane).
_GPU_SKIP_TOKENS = (
    "gpu_unavailable",
    "cutlass_hstu_unavailable",
    "has_dynamicemb",
    "has_tensorrt",
    "torch_fx_tool_unavailable",
    "device_count",
)
_TZREC_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _src(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return ""


def _is_gpu_required(decorators) -> bool:
    """True if any decorator skips the test when run on the CPU lane."""
    for dec in decorators:
        s = _src(dec)
        if "skipIf" not in s and "skipUnless" not in s:
            continue
        if any(tok in s for tok in _GPU_SKIP_TOKENS):
            return True
        # skipUnless(cuda.is_available()) / skipIf(not cuda.is_available()) skip on
        # CPU; skipIf(cuda.is_available()) (the FAISS CPU-only pattern) does NOT.
        if "cuda.is_available()" in s and (
            "skipUnless" in s or "not torch.cuda.is_available()" in s
        ):
            return True
    return False


def _has_scope(decorators) -> bool:
    for dec in decorators:
        # ast.unparse normalizes string quotes to single, so normalize before matching.
        s = _src(dec).replace("'", '"')
        if "mark_ci_scope" in s and ('"gpu"' in s or '"h20"' in s):
            return True
    return False


class CIScopeCoverageTest(unittest.TestCase):
    def test_gpu_only_tests_carry_a_scope_marker(self) -> None:
        violations = []
        for fp in glob.glob(
            os.path.join(_TZREC_ROOT, "**", "*_test.py"), recursive=True
        ):
            rel = os.path.relpath(fp, os.path.dirname(_TZREC_ROOT))
            tree = ast.parse(open(fp).read(), filename=fp)
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                cls_scoped = _has_scope(cls.decorator_list)
                if _is_gpu_required(cls.decorator_list) and not cls_scoped:
                    violations.append(f"{rel}::{cls.name} (class)")
                for item in cls.body:
                    if not (
                        isinstance(item, ast.FunctionDef)
                        and item.name.startswith("test")
                    ):
                        continue
                    if _is_gpu_required(item.decorator_list) and not (
                        cls_scoped or _has_scope(item.decorator_list)
                    ):
                        violations.append(f"{rel}::{cls.name}.{item.name}")
        self.assertEqual(
            violations,
            [],
            'GPU-only tests missing @mark_ci_scope("gpu"/"h20") -- they skip on '
            "the CPU lane and are filtered out of the scoped GPU lanes, so they run "
            "on NO per-PR lane:\n  " + "\n  ".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
