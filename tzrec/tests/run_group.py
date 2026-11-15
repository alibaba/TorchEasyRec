# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Internal unittest executor for one assigned fixture group."""

import json
import sys
import unittest
from pathlib import Path


def main(request: Path) -> int:
    """Run exact IDs and return native failure blocks and group-level counts."""
    group = json.loads(request.read_text())
    sys.path.insert(0, str(Path(group["test_dir"]).resolve()))
    if group["device_count"]:
        import torch

        if torch.cuda.device_count() != group["device_count"]:
            raise RuntimeError("wrong device view for test group")

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for test_id in group["ids"]:
        loaded = list(loader.loadTestsFromName(test_id))
        if loader.errors:
            raise ImportError("\n".join(map(str, loader.errors)))
        if len(loaded) != 1 or loaded[0].id() != test_id:
            raise ValueError(f"could not load assigned test: {test_id}")
        suite.addTest(loaded[0])

    with request.with_suffix(".errors").open("w") as stream:
        result = unittest.TextTestResult(
            unittest.TextTestRunner(stream=stream).stream,
            descriptions=True,
            verbosity=0,
        )
        result.startTestRun()
        try:
            suite.run(result)
        finally:
            result.stopTestRun()
        result.printErrors()

    counts = {
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "skipped": len(result.skipped),
        "expected failures": len(result.expectedFailures),
        "unexpected successes": len(result.unexpectedSuccesses),
    }
    request.with_suffix(".counts").write_text(json.dumps(counts))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
