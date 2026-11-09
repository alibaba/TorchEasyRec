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
from types import SimpleNamespace
from unittest import mock

from tzrec.utils import misc_util


class HostMemoryTest(unittest.TestCase):
    """The planner's host memory respects a container memory limit."""

    def _run(self, limit_text):
        files = {"/sys/fs/cgroup/memory.max": limit_text}

        def fake_open(path, *args, **kwargs):
            if path in files:
                return mock.mock_open(read_data=files[path])()
            raise OSError(path)

        with (
            mock.patch.object(
                misc_util.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(total=1 << 40),
            ),
            mock.patch("builtins.open", side_effect=fake_open),
        ):
            return misc_util.host_memory_bytes()

    def test_cgroup_limit_caps_host_memory(self):
        self.assertEqual(self._run(str(170 << 30)), 170 << 30)

    def test_unlimited_cgroup_keeps_ram(self):
        self.assertEqual(self._run("max"), 1 << 40)


if __name__ == "__main__":
    unittest.main()
