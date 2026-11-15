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

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class TestRunnerTest(unittest.TestCase):
    def setUp(self):
        parent = Path(os.environ.get("TZREC_TEST_TMPDIR", "tmp"))
        parent.mkdir(exist_ok=True)
        tmp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.tests = self.root / "suite"
        self.tests.mkdir()

    def cli(self, *args):
        return subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("run.py").resolve()),
                "--hardware",
                "cpu",
                "--test-dir",
                "suite",
                *args,
            ],
            cwd=self.root,
            env=dict(os.environ, PYTHONPATH="."),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def test_native_failure_summary_across_groups(self):
        (self.tests / "one_test.py").write_text(
            "import unittest\n"
            "class One(unittest.TestCase):\n"
            " def test_fail(self):\n"
            "  print('first group output')\n"
            "  with self.subTest(a=1): self.fail('first failure')\n"
            " def test_skipped(self): self.skipTest('optional')\n"
            " @unittest.expectedFailure\n"
            " def test_unexpected(self): pass\n"
        )
        (self.tests / "two_test.py").write_text(
            "import unittest\n"
            "class Broken(unittest.TestCase):\n"
            " @classmethod\n"
            " def setUpClass(cls): raise RuntimeError('fixture failed')\n"
            " def test_never_started(self): pass\n"
            "class Two(unittest.TestCase):\n"
            " def test_error(self): raise ValueError('second error')\n"
        )
        (self.tests / "pass_test.py").write_text(
            "import unittest\nclass Passing(unittest.TestCase):\n"
            " def test_ok(self): print('successful group output')\n"
        )
        r = self.cli()
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(r.stderr.count("Ran "), 1)
        self.assertIn("Ran 5 tests", r.stderr)
        for message in (
            "AssertionError: first failure",
            "RuntimeError: fixture failed",
            "ValueError: second error",
        ):
            self.assertEqual(r.stderr.count(message), 1, r.stderr)
        self.assertLess(
            r.stderr.index("first failure"), r.stderr.index("fixture failed")
        )
        self.assertIn(
            "FAILED (failures=1, errors=2, skipped=1, unexpected successes=1)", r.stderr
        )

        logs = sorted((self.root / "tmp").rglob("*.log"))
        self.assertEqual(len(logs), 2)
        text = "".join(log.read_text() for log in logs)
        for fragment in (
            "first group output",
            "AssertionError: first failure",
            "RuntimeError: fixture failed",
            "ValueError: second error",
        ):
            self.assertIn(fragment, text)
        self.assertNotIn("successful group output", text)
        self.assertFalse(list(self.root.rglob("summary.txt")))

    def test_case_isolation_and_pre_import_environment(self):
        (self.tests / "case_test.py").write_text(
            "import os, subprocess, sys, tempfile, time, unittest\n"
            "from pathlib import Path\n"
            "value=os.getenv('CASE_BOOT_ENV')\nprint('import diagnostic')\n"
            "class Isolated(unittest.TestCase):\n"
            " _ci_resources=(1,'case',15)\n"
            " def _run_env(self): return {'CASE_BOOT_ENV':'ready'}\n"
            " def record(self):\n"
            "  self.assertEqual(value,'ready')\n"
            f"  root = Path({str(self.root)!r})\n"
            "  self.assertEqual(Path.cwd(),root)\n"
            "  private=Path(os.environ['TZREC_TEST_TMPDIR'])\n"
            "  self.assertTrue(Path(tempfile.gettempdir()).samefile(private))\n"
            "  from multiprocessing.connection import Listener\n"
            "  with Listener(family='AF_UNIX'): pass\n"
            '  child="import tempfile;from pathlib import Path;'
            "Path(tempfile.gettempdir(), 'same-name').touch(exist_ok=False)\"\n"
            "  subprocess.run([sys.executable,'-c',child],check=True)\n"
            "  self.assertTrue((private/'same-name').is_file())\n"
            "  (root/str(os.getpid())).write_text(str(private))\n"
            "  until=time.monotonic()+5\n"
            "  while len(list(root.glob('[0-9]*')))<2:\n"
            "   self.assertLess(time.monotonic(),until);time.sleep(.02)\n"
            " def test_a(self): self.record()\n def test_b(self): self.record()\n"
        )
        listing = self.cli("--list_tests")
        self.assertEqual(listing.returncode, 0, listing.stderr)
        self.assertEqual(len(listing.stdout.splitlines()), 2)
        self.assertFalse((self.root / "tmp").exists())
        r = self.cli()
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Ran 2 tests", r.stderr)
        private_dirs = {
            p.read_text() for p in self.root.iterdir() if p.name.isdecimal()
        }
        self.assertEqual(len(private_dirs), 2)
        self.assertTrue(all(not Path(p).exists() for p in private_dirs))
        self.assertEqual(list((self.root / "tmp").iterdir()), [])

    def test_infrastructure_failure_does_not_hide_other_groups(self):
        (self.tests / "crash_test.py").write_text(
            "import os, unittest\nclass Crash(unittest.TestCase):\n"
            " def test_exit(self):\n  print('last diagnostic',flush=True)\n"
            "  os._exit(0)\n"
        )
        (self.tests / "timeout_test.py").write_text(
            "import time,unittest\nclass Slow(unittest.TestCase):\n"
            " _ci_resources=(1,'case',1)\n def test_wait(self): time.sleep(45)\n"
        )
        r = self.cli()
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("last diagnostic", r.stderr)
        self.assertIn("timed out after 1s", r.stderr)
        self.assertIn("Ran 0 tests", r.stderr)
        self.assertIn("FAILED (group errors=2)", r.stderr)
        logs = list((self.root / "tmp/ci-failures").rglob("*.log"))
        self.assertEqual(len(logs), 2)
        self.assertTrue(all("GROUP ERROR:" in log.read_text() for log in logs))

    def test_import_error_not_filtered_out_by_scope(self):
        (self.tests / "bad_test.py").write_text("raise ImportError('bad import')\n")
        r = self.cli("--list_tests", "--scope", "gpu")
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("bad import", r.stderr)
