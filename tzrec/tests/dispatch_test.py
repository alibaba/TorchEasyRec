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

import unittest
from unittest import mock

from tzrec.tests import dispatch, run
from tzrec.utils.test_util import ci_resources, mark_ci_scope


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
