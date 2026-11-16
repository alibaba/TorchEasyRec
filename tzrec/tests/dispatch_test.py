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

import io
import json
import os
import shutil
import signal
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from tzrec.tests import dispatch, run
from tzrec.utils.test_util import ci_resources, make_test_dir, mark_ci_scope


class DispatchTest(unittest.TestCase):
    def test_resources_scope_and_boundaries(self):
        @mark_ci_scope("gpu")
        class Example(unittest.TestCase):
            def test_default(self):
                pass

            @ci_resources(slots=2)
            def test_two(self):
                pass

            @ci_resources(split="case", timeout=600)
            def test_separate(self):
                pass

        class Cpu(unittest.TestCase):
            def test_local(self):
                pass

        loader = unittest.TestLoader()
        suite = unittest.TestSuite(
            loader.loadTestsFromTestCase(cls) for cls in (Example, Cpu)
        )
        with mock.patch.object(unittest.TestLoader, "discover", return_value=suite):
            gpu = run.discover("tzrec", "*", "gpu")
            self.assertEqual(len(run.discover("tzrec", "*", None)), 4)
        groups = dispatch.group_cases(gpu)
        self.assertEqual(
            [(g["slots"], len(g["ids"]), g["timeout"]) for g in groups],
            [(2, 2, None), (1, 1, 600)],
        )

    def test_split_cannot_cross_class_fixture(self):
        @ci_resources(split="case")
        class Shared(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                pass

            def test_one(self):
                pass

        with mock.patch.object(
            unittest.TestLoader,
            "discover",
            return_value=unittest.TestSuite([Shared("test_one")]),
        ):
            with self.assertRaisesRegex(ValueError, "class fixture"):
                run.discover("tzrec", "*", None)

    def _deadline_run(self, job_timeout, group_timeout, names):
        root = Path(make_test_dir()).resolve()
        self.addCleanup(shutil.rmtree, root)
        self.addCleanup(os.chdir, Path.cwd())
        os.chdir(root)
        Path("tmp").mkdir()
        cases = [
            dict(
                id=f"{name}.test_{j}",
                module=name,
                split="module",
                slots=1,
                timeout=group_timeout,
                env={},
            )
            for name in names
            for j in range(3 if name == "A" else 1)
        ]
        processes = []
        clock = mock.Mock()
        clock.monotonic.return_value = 0

        def spawn(command, **kwargs):
            request = Path(command[-1])
            group = json.loads(request.read_text())
            proc = mock.Mock(pid=1000 + len(processes), returncode=None)
            proc.poll.side_effect = lambda: proc.returncode
            processes.append(proc)
            # Even the group killed at the job deadline has a valid-looking file.
            request.with_suffix(".counts").write_text(
                json.dumps(
                    {
                        "tests": len(group["ids"]),
                        "failures": 0,
                        "errors": 0,
                        "skipped": 0,
                        "expected failures": 0,
                        "unexpected successes": 0,
                    }
                )
            )
            return proc

        def advance(_):
            clock.monotonic.return_value += 1
            if len(names) > 1:
                processes[0].returncode = 0  # A finished between scheduler polls.

        def reap(proc):
            # In the single-group case A exits naturally after poll returned None.
            proc.returncode = 0 if proc is processes[0] else -signal.SIGTERM

        clock.sleep.side_effect = advance
        output = io.StringIO()
        with (
            mock.patch.object(dispatch, "time", clock),
            mock.patch.object(dispatch.subprocess, "Popen", side_effect=spawn),
            mock.patch.object(dispatch.os, "killpg"),
            mock.patch.object(dispatch, "terminate", side_effect=reap),
            redirect_stderr(output),
        ):
            rc = dispatch.schedule(cases, 2, [], "tzrec", 0, job_timeout)
        return rc, output.getvalue(), processes

    def test_group_finishes_between_poll_and_timeout_reap(self):
        rc, summary, processes = self._deadline_run(60, 1, ["A"])
        self.assertEqual(len(processes), 1)
        self.assertTrue(processes[0].poll.called)
        self.assertEqual(rc, 0, summary)
        self.assertIn("Ran 3 tests", summary)
        self.assertTrue(summary.rstrip().endswith("OK"))
        self.assertNotIn("GROUP ERROR", summary)

    def test_job_deadline_keeps_completed_results_and_does_not_launch_pending(self):
        rc, summary, processes = self._deadline_run(1, None, ["A", "B", "C"])
        self.assertEqual(len(processes), 2)
        self.assertEqual(rc, 1, summary)
        self.assertIn("Ran 3 tests", summary)
        self.assertNotIn("GROUP ERROR: A", summary)
        self.assertIn("B: interrupted at job deadline", summary)
        self.assertIn("1 pending", summary)
        self.assertIn("FAILED (group errors=2)", summary)
