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

import importlib
import io
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

from parameterized import parameterized

from tzrec.tests import ci_execution as execution
from tzrec.tests import ci_protocol as wire
from tzrec.tests import dispatch
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.path = Path(make_test_dir("dispatch_"))
        journal = mock.patch.dict(os.environ, {"TZREC_CI_EVENT_LOG": ""})
        journal.start()
        self.addCleanup(journal.stop)

    def tearDown(self):
        shutil.rmtree(self.path)

    def test_empty_serial_report_cannot_pass(self):
        from tzrec.tests import run

        with (
            mock.patch.object(sys, "argv", ["run.py", "--report-dir", str(self.path)]),
            mock.patch.object(
                run,
                "_gather_test_cases",
                return_value=(unittest.TestSuite(), unittest.TestSuite()),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "empty or duplicate"):
                run.main()

    def test_result_subtests_and_teardown_are_one_failed_case(self):
        class Example(unittest.TestCase):
            def test_example(self):
                with self.subTest(part=1):
                    self.fail("subtest failure")

            def tearDown(self):
                raise RuntimeError("teardown failed")

        case = Example("test_example")
        result = unittest.TextTestRunner(
            stream=io.StringIO(), verbosity=2, resultclass=execution.RecordingResult
        ).run(unittest.TestSuite([case]))
        report = result.report([{"id": case.id()}])
        self.assertEqual(report["code"], 1)
        self.assertEqual(len(report["cases"]), 1)
        self.assertEqual(report["cases"][0]["outcome"], "error")
        self.assertEqual(len(report["cases"][0]["details"]), 2)

    def test_skipped_subtest_does_not_become_an_extra_case(self):
        class Example(unittest.TestCase):
            def test_example(self):
                with self.subTest(part=0):
                    self.assertTrue(True)
                with self.subTest(part=1):
                    self.skipTest("optional subtest")

        case = Example("test_example")
        result = unittest.TextTestRunner(
            stream=io.StringIO(), resultclass=execution.RecordingResult
        ).run(unittest.TestSuite([case]))
        report = result.report([{"id": case.id()}])
        self.assertEqual(report["code"], 0)
        self.assertEqual([c["id"] for c in report["cases"]], [case.id()])
        self.assertEqual(len(result.skipped), 1)
        self.assertEqual(
            report["cases"][0]["subtest_skips"][0]["reason"], "optional subtest"
        )

    @parameterized.expand(
        [(False, "blocked_by_fixture", 1), (True, "skip", 0)],
        name_func=parameterized_name_func,
    )
    def test_class_fixture_is_accounted(self, skip, expected, code):
        class Example(unittest.TestCase):
            @classmethod
            def setUpClass(cls):
                if skip:
                    raise unittest.SkipTest("fixture unavailable")
                raise RuntimeError("fixture broken")

            def test_one(self):
                pass

            def test_two(self):
                pass

        cases = [Example("test_one"), Example("test_two")]
        result = unittest.TextTestRunner(
            stream=io.StringIO(), verbosity=2, resultclass=execution.RecordingResult
        ).run(unittest.TestSuite(cases))
        report = result.report([{"id": c.id()} for c in cases])
        self.assertEqual(report["code"], code)
        self.assertEqual([c["outcome"] for c in report["cases"]], [expected, expected])

    def test_subprocess_preserves_run_env_and_true_skip(self):
        from tzrec.tests.run import TZRecTestRunner

        package = "dispatch_fixture_" + self.path.name
        directory = self.path / package
        directory.mkdir()
        (directory / "__init__.py").write_text("")
        (directory / "dataset_test.py").write_text(
            "import os,unittest\n"
            "class Example(unittest.TestCase):\n"
            " def _run_env(self): return {'TZCI_CHILD_CHECK':'yes'}\n"
            " def test_skip(self):\n"
            "  self.assertEqual(os.environ.get('TZCI_CHILD_CHECK'),'yes')\n"
            "  self.skipTest('real child skip')\n"
        )
        sys.path.insert(0, str(self.path.resolve()))
        try:
            module = importlib.import_module(package + ".dataset_test")
            case = module.Example("test_skip")
            runner = TZRecTestRunner(
                stream=io.StringIO(),
                resultclass=partial(
                    execution.RecordingResult, test_dir=str(self.path.resolve())
                ),
            )
            result = runner.run(
                unittest.TestSuite(), unittest.TestSuite([unittest.TestSuite([case])])
            )
            self.assertTrue(result.wasSuccessful())
            self.assertEqual(result.testsRun, 1)
            self.assertEqual(result.skipped, [(case, "real child skip")])
            self.assertEqual(
                result.report([{"id": case.id()}])["cases"][0]["outcome"], "skip"
            )
        finally:
            sys.path.remove(str(self.path.resolve()))
            for key in list(sys.modules):
                if key == package or key.startswith(package + "."):
                    sys.modules.pop(key)

    def test_duplicate_missing_wrong_digest_and_invalid_outcomes_fail(self):
        manifest = {
            "session_id": "test",
            "digest": "sha",
            "groups": [{"key": "a", "cases": [{"id": "a.one"}, {"id": "a.two"}]}],
        }
        report = {
            "version": wire.VERSION,
            "digest": "sha",
            "group": "a",
            "code": 0,
            "fixtures": [],
            "cases": [
                {"id": "a.one", "outcome": "success"},
                {"id": "a.two", "outcome": "success"},
            ],
        }
        self.assertEqual(execution.verify(manifest, [report])["code"], 0)
        for broken in (
            dict(report, cases=report["cases"][:1]),
            dict(report, cases=report["cases"] + report["cases"][:1]),
            dict(report, digest="wrong"),
            dict(report, code=True),
            dict(report, cases=[dict(c, outcome="invented") for c in report["cases"]]),
        ):
            self.assertEqual(execution.verify(manifest, [broken])["code"], 2)
        self.assertEqual(execution.verify(manifest, [report, report])["code"], 2)

    def test_grouping_keeps_unknown_modules_exclusive(self):
        cases = [
            dict(
                id="pkg.First.test_a",
                module="pkg",
                split="module",
                devices=2,
                route="main",
            ),
            dict(
                id="pkg.Second.test_b",
                module="pkg",
                split="module",
                devices="baseline",
                route="main",
            ),
        ]
        groups = execution.group_cases(cases, {})
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["devices"], "baseline")
        self.assertEqual(groups[0]["cases"], cases)
        cases[0]["split"] = "case"
        self.assertEqual(len(execution.group_cases(cases, {})), 2)

    def test_discovery_reuses_scope_and_rejects_fixture_splitting(self):
        from tzrec.tests import run

        class Example(unittest.TestCase):
            _ci_scopes = {"gpu"}
            _ci_resources = (2, "case")

            @classmethod
            def setUpClass(cls):
                pass

            def test_a(self):
                pass

        suite = unittest.TestSuite([Example("test_a")])
        with mock.patch.object(
            run, "_gather_test_cases", return_value=(suite, unittest.TestSuite())
        ):
            with self.assertRaisesRegex(ValueError, "class fixture"):
                execution.discover("tzrec", "*_test.py", "gpu")

    @parameterized.expand(
        [
            ("0,1;2,3", [["GPU-a", "GPU-b"], ["GPU-c", "GPU-d"]]),
            ("2,3;0,1", [["GPU-c", "GPU-d"], ["GPU-a", "GPU-b"]]),
        ],
        name_func=parameterized_name_func,
    )
    def test_device_mapping(self, value, expected):
        self.assertEqual(
            dispatch.device_groups(value, ["GPU-a", "GPU-b", "GPU-c", "GPU-d"]),
            expected,
        )

    @parameterized.expand(
        [("0,1;1,2",), ("0,4",), ("-1,0",), ("0",)], name_func=parameterized_name_func
    )
    def test_bad_device_mapping(self, value):
        with self.assertRaises(ValueError):
            dispatch.device_groups(value, ["0", "1", "2", "3"])

    def test_framing_rejects_eof_and_oversize(self):
        for raw in (b"", b"{}", b"[]\n", b"x" * (wire.MAX_MESSAGE + 1)):
            with self.assertRaises(ValueError):
                wire.receive(io.BytesIO(raw))

    def test_coordinator_cli_needs_only_the_standard_library(self):
        manifest = dict(
            version=wire.VERSION,
            session_id="stdlib",
            deadline=time.time() + 30,
            groups=[dict(key="baseline", devices="baseline", cases=[dict(id="one")])],
        )
        manifest["digest"] = wire.digest(manifest)
        wire.write_json(self.path / "manifest.json", manifest)
        wire.write_json(
            self.path / "results/baseline.json",
            dict(
                version=wire.VERSION,
                digest=manifest["digest"],
                group="baseline",
                code=0,
                cases=[dict(id="one", outcome="success", seconds=0)],
                fixtures=[],
            ),
        )
        process = subprocess.run(
            [
                sys.executable,
                "-S",
                str(Path(dispatch.__file__).resolve()),
                "serve",
                "--session",
                str(self.path),
                "--session-id",
                "stdlib",
                "--workers",
                "0",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(wire.read_json(self.path / "summary.json")["reported"], 1)

    def test_process_timeout_kills_group(self):
        pidfile = self.path / "child.pid"
        command = [
            sys.executable,
            "-c",
            (
                "import subprocess,time,pathlib,sys; "
                "p=subprocess.Popen([sys.executable,'-c',"
                "'import time; time.sleep(120)']);"
                "pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(120)"
            ),
            str(pidfile),
        ]
        with self.assertRaises(subprocess.TimeoutExpired):
            wire.run_process(command, timeout=1)
        pid = int(pidfile.read_text())
        status = Path(f"/proc/{pid}/status")
        self.assertTrue(not status.exists() or "State:\tZ" in status.read_text())

    @unittest.skipUnless(
        sys.platform == "linux", "Unix socket integration uses Linux dirfds"
    )
    def test_coordinator_handles_workers_finishing_out_of_order(self):
        groups = [
            dict(key=str(i), devices=2, estimate=i, cases=[{"id": f"case.{i}"}])
            for i in range(4)
        ]
        manifest = dict(
            version=wire.VERSION,
            session_id="session",
            deadline=time.time() + 10,
            groups=groups,
        )
        manifest["digest"] = wire.digest(manifest)
        sha = manifest["digest"]
        wire.write_json(self.path / "manifest.json", manifest)
        outcome = []

        def coordinate():
            outcome.append(dispatch.serve(self.path, 2, "session"))

        server = threading.Thread(target=coordinate)
        server.start()
        for _ in range(100):
            if (self.path / "dispatch.sock").exists():
                break
            time.sleep(0.01)

        def worker(slot):
            # A dirfd keeps the Unix socket address below the kernel path limit.
            sock = socket.socket(socket.AF_UNIX)
            directory = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                sock.connect(f"/proc/self/fd/{directory}/dispatch.sock")
            finally:
                os.close(directory)
            with sock, sock.makefile("rwb") as stream:
                wire.send(
                    stream,
                    dict(
                        kind="ready",
                        version=wire.VERSION,
                        session_id="session",
                        digest=sha,
                        slot=slot,
                        devices=2,
                    ),
                )
                while True:
                    work = wire.receive(stream)
                    if work["kind"] == "stop":
                        return
                    key = work["group"]
                    time.sleep(0.02 * (slot + 1))
                    wire.write_json(
                        self.path / "results" / f"{key}.json",
                        dict(
                            version=wire.VERSION,
                            digest=sha,
                            group=key,
                            code=0,
                            fixtures=[],
                            cases=[
                                dict(id=f"case.{key}", outcome="success", seconds=1)
                            ],
                        ),
                    )
                    wire.send(stream, dict(kind="done", group=key, digest=sha))

        workers = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
        for thread in workers:
            thread.start()
        for thread in workers + [server]:
            thread.join(timeout=12)
            self.assertFalse(thread.is_alive())
        self.assertEqual(outcome, [0])
        self.assertEqual(wire.read_json(self.path / "summary.json")["reported"], 4)


if __name__ == "__main__":
    unittest.main()
