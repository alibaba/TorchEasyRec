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
"""Run all unit tests."""

import argparse
import os
import unittest


def _gather_test_cases(args):
    test_dir = os.path.abspath(args.test_dir)
    test_suite = unittest.TestSuite()
    discover = unittest.defaultTestLoader.discover(
        test_dir, pattern=args.pattern, top_level_dir=None
    )
    for suite_discovered in discover:
        for test_case in suite_discovered:
            test_suite.addTest(test_case)
            if hasattr(test_case, "__iter__"):
                for subcase in test_case:
                    if args.list_tests:
                        print(subcase)
            else:
                if args.list_tests:
                    print(test_case)
    return test_suite


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TorchEasyRec")
    parser.add_argument("--list_tests", action="store_true", help="list all tests")
    parser.add_argument(
        "--pattern", type=str, default="*_test.py", help="test file pattern"
    )
    parser.add_argument(
        "--test_dir", type=str, default="tzrec", help="directory to be tested"
    )
    args = parser.parse_args()

    runner = unittest.TextTestRunner()
    test_suite = _gather_test_cases(args)
    if not args.list_tests:
        runner.run(test_suite)
