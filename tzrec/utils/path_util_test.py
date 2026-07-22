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
import shutil
import unittest
from pathlib import Path

from parameterized import parameterized

from tzrec.utils.path_util import check_path_conflict
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class PathUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _assert_no_path_conflict(self, paths) -> None:
        has_conflict, message = check_path_conflict(paths)
        self.assertFalse(has_conflict)
        self.assertIsNone(message)

    def _assert_path_conflict(self, paths) -> str:
        has_conflict, message = check_path_conflict(paths)
        self.assertTrue(has_conflict)
        self.assertTrue(message)
        return message

    def test_empty_single_and_mixed_type_groups_do_not_conflict(self) -> None:
        local_path = os.path.join(self.test_dir, "output")

        self._assert_no_path_conflict([])
        self._assert_no_path_conflict([local_path])
        self._assert_no_path_conflict([local_path, f"{local_path}_other"])
        self._assert_no_path_conflict([local_path, "odps://project/tables/output"])

    def test_local_aliases_conflict(self) -> None:
        output_path = os.path.join(self.test_dir, "output")

        alias_path = os.path.join(output_path, "..", "output")

        message = self._assert_path_conflict([output_path, alias_path])

        self.assertEqual(
            message,
            f"path conflict: {output_path!r} and {alias_path!r} resolve to the "
            f"same local directory {os.path.realpath(output_path)!r}.",
        )

    def test_aliases_within_one_local_group_do_not_conflict(self) -> None:
        input_dir = os.path.join(self.test_dir, "input")
        os.makedirs(input_dir)
        input_file = os.path.join(input_dir, "part.parquet")
        Path(input_file).touch()

        self._assert_no_path_conflict([f"{input_dir}/*.parquet,{input_file}"])

    def test_empty_local_path_tokens_are_ignored(self) -> None:
        input_file = os.path.join(self.test_dir, "part.parquet")
        Path(input_file).touch()

        self._assert_no_path_conflict([f"{input_file},", os.getcwd()])

    def test_second_local_glob_conflicts_with_output(self) -> None:
        first_dir = os.path.join(self.test_dir, "first")
        second_dir = os.path.join(self.test_dir, "second")
        os.makedirs(first_dir)
        os.makedirs(second_dir)
        Path(os.path.join(first_dir, "part.parquet")).touch()
        Path(os.path.join(second_dir, "part.parquet")).touch()

        message = self._assert_path_conflict(
            [f"{first_dir}/*.parquet,{second_dir}/*.parquet", second_dir]
        )

        self.assertIn(os.path.realpath(second_dir), message)

    def test_symlinked_input_directory_conflicts_with_output(self) -> None:
        input_dir = os.path.join(self.test_dir, "input")
        alias_dir = os.path.join(self.test_dir, "input_alias")
        os.makedirs(input_dir)
        Path(os.path.join(input_dir, "part.parquet")).touch()
        os.symlink(os.path.abspath(input_dir), alias_dir)

        self._assert_path_conflict([f"{alias_dir}/*.parquet", input_dir])

    def test_file_symlink_conflicts_with_target_directory(self) -> None:
        input_dir = os.path.join(self.test_dir, "input")
        alias_dir = os.path.join(self.test_dir, "input_alias")
        os.makedirs(input_dir)
        os.makedirs(alias_dir)
        input_file = os.path.join(input_dir, "part.parquet")
        alias_file = os.path.join(alias_dir, "source.parquet")
        Path(input_file).touch()
        os.symlink(os.path.abspath(input_file), alias_file)

        self._assert_path_conflict([alias_file, input_dir])

    @parameterized.expand(
        [
            (
                "same_partition",
                [
                    "odps://project/tables/input/dt=1",
                    "odps://project/tables/input/dt=1",
                ],
                True,
            ),
            (
                "second_table",
                [
                    "odps://project/tables/first/dt=1,odps://project/tables/second/dt=2",
                    "odps://project/tables/second/dt=2",
                ],
                True,
            ),
            (
                "multi_partition_group",
                [
                    "odps://project/tables/input/dt=1&dt=2",
                    "odps://project/tables/input/dt=2",
                ],
                True,
            ),
            (
                "partition_format_alias",
                [
                    "odps://project/tables/input/dt=1/region=x",
                    "odps://project/tables/input/dt = '1'/region = 'x'",
                ],
                True,
            ),
            (
                "whole_table",
                [
                    "odps://project/tables/input",
                    "odps://project/tables/input/dt=1",
                ],
                True,
            ),
            (
                "trailing_slash",
                [
                    "odps://project/tables/input",
                    "odps://project/tables/input/",
                ],
                True,
            ),
            (
                "different_partitions",
                [
                    "odps://project/tables/input/dt=1",
                    "odps://project/tables/input/dt=2",
                ],
                False,
            ),
            (
                "different_schemas",
                [
                    "odps://project/tables/first.input/dt=1",
                    "odps://project/tables/second.input/dt=1",
                ],
                False,
            ),
            (
                "same_group",
                ["odps://project/tables/input/dt=1&dt=2"],
                False,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_odps_path_groups(self, _name, paths, expected) -> None:
        has_conflict, message = check_path_conflict(paths)

        self.assertEqual(has_conflict, expected)
        self.assertEqual(bool(message), expected)
        if expected:
            self.assertIn(paths[0], message)
            self.assertIn(paths[1], message)


if __name__ == "__main__":
    unittest.main()
