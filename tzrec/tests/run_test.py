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
import unittest
from types import SimpleNamespace
from unittest import mock

from tzrec.tests import run


class TestRunnerTest(unittest.TestCase):
    def test_failed_subprocess_module_reports_import_error(self):
        failed = unittest.loader._FailedTest(
            "tools.tdm.gen_tree.missing_test", ImportError("synthetic discovery error")
        )
        discovered = unittest.TestSuite([unittest.TestSuite([failed])])
        args = SimpleNamespace(
            test_dir="tzrec", pattern="*_test.py", scope=None, list_tests=False
        )
        with mock.patch.object(
            unittest.defaultTestLoader, "discover", return_value=discovered
        ):
            main, subproc = run._gather_test_cases(args)
        self.assertEqual(main.countTestCases(), 1)
        self.assertEqual(subproc.countTestCases(), 0)
        stream = io.StringIO()
        with mock.patch.object(run.subprocess, "run") as launch:
            result = run.TZRecTestRunner(stream=stream).run(main, subproc)
        launch.assert_not_called()
        self.assertFalse(result.wasSuccessful())
        self.assertEqual(len(result.errors), 1)
        self.assertIn("synthetic discovery error", stream.getvalue())

    def test_subprocess_suite_keeps_its_routing(self):
        class DatasetTest(unittest.TestCase):
            def test_example(self):
                pass

        DatasetTest.__module__ = "tests.dataset_test"
        discovered = unittest.TestSuite(
            [unittest.TestSuite([unittest.TestSuite([DatasetTest("test_example")])])]
        )
        args = SimpleNamespace(
            test_dir="tzrec", pattern="*_test.py", scope=None, list_tests=False
        )
        with mock.patch.object(
            unittest.defaultTestLoader, "discover", return_value=discovered
        ):
            main, subproc = run._gather_test_cases(args)
        self.assertEqual(main.countTestCases(), 0)
        self.assertEqual(subproc.countTestCases(), 1)


if __name__ == "__main__":
    unittest.main()
