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
import subprocess
import sys
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

from parameterized import parameterized

from tzrec.tests import dispatch, run
from tzrec.utils.test_util import ci_resources, make_test_dir, parameterized_name_func


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
            stream=io.StringIO(), verbosity=2, resultclass=run.RecordingResult
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
            stream=io.StringIO(), resultclass=run.RecordingResult
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
            stream=io.StringIO(), verbosity=2, resultclass=run.RecordingResult
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
                    run.RecordingResult, test_dir=str(self.path.resolve())
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
        groups = dispatch.group_cases(cases)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["devices"], "baseline")
        self.assertEqual(groups[0]["cases"], cases)
        cases[0]["split"] = "case"
        self.assertEqual(len(dispatch.group_cases(cases)), 2)

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
                run.discover("tzrec", "*_test.py", "gpu")

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
            run.run_process(command, timeout=1)
        pid = int(pidfile.read_text())
        status = Path(f"/proc/{pid}/status")
        self.assertTrue(not status.exists() or "State:\tZ" in status.read_text())

    def test_missing_duplicate_stale_or_false_green_result_is_rejected(self):
        group = dict(
            session_id="session", key="group", cases=[dict(id="a"), dict(id="b")]
        )
        cases = [dict(id="a", outcome="success"), dict(id="b", outcome="skip")]
        report = dict(
            session_id="session", group="group", cases=cases, fixtures=[], code=0
        )
        dispatch.check_result(group, report, 0)
        for broken in (
            dict(report, session_id="stale"),
            dict(report, cases=cases[:1]),
            dict(report, cases=[cases[0], cases[0]]),
            dict(report, cases=[cases[0], dict(id="b", outcome="failure")]),
            dict(report, code=True),
            dict(report, group="other"),
        ):
            with self.assertRaises(RuntimeError):
                dispatch.check_result(group, broken, 0)

    def test_cpu_and_gpu_share_split_but_not_device_requirements(self):
        class Example(unittest.TestCase):
            def test_default(self):
                pass

            @ci_resources(split="class")
            def test_class(self):
                pass

            @ci_resources(devices=2, split="case")
            def test_pair(self):
                pass

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Example)
        with mock.patch.object(
            run, "_gather_test_cases", return_value=(suite, unittest.TestSuite())
        ):
            for scope, devices in [
                (None, [0, 0, 0]),
                ("gpu", ["baseline", "baseline", 2]),
            ]:
                cases = run.discover("tzrec", "*", scope)
                self.assertEqual(
                    [c["split"] for c in cases], ["class", "module", "case"]
                )
                self.assertEqual([c["devices"] for c in cases], devices)

    def test_cpu_module_stays_on_one_slot_across_execution_routes(self):
        cases = [
            dict(id=name, module="pkg", split="module", devices=0, route=route)
            for name, route in (("pkg.A.test_a", "main"), ("pkg.B.test_b", "subproc"))
        ]
        self.assertEqual(dispatch.group_cases(cases), [dict(devices=0, cases=cases)])

    def test_four_cpu_shards_keep_each_fixture_boundary_intact(self):
        groups = [
            dict(cases=[dict(id=f"{i}.one"), dict(id=f"{i}.two")]) for i in range(9)
        ]
        shards = dispatch.cpu_shards(groups, 4)
        self.assertEqual(len(shards), 4)
        for i, group in enumerate(groups):
            self.assertTrue(all(c in shards[i % 4]["cases"] for c in group["cases"]))
        self.assertEqual(sum(len(s["cases"]) for s in shards), 18)


if __name__ == "__main__":
    unittest.main()
