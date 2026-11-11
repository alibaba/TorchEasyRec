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

from parameterized import param, parameterized

from tzrec.utils import resource_util
from tzrec.utils.test_util import parameterized_name_func

GIB = 1 << 30


def _container(total, files):
    def fake_open(path, *args, **kwargs):
        if path in files:
            return mock.mock_open(read_data=files[path])()
        raise OSError(path)

    return (
        mock.patch.object(
            resource_util.psutil,
            "virtual_memory",
            return_value=SimpleNamespace(total=total),
        ),
        mock.patch("builtins.open", side_effect=fake_open),
    )


class HostMemoryCapacityTest(unittest.TestCase):
    """Effective per-rank host capacity: container limit, reserve, sharing."""

    @parameterized.expand(
        [
            param("positive_reserve", reserve=0.15, lws=8, files={}, total=1 << 40),
            param("explicit_reserve", reserve=0.5, lws=8, files={}, total=1 << 40),
            param("zero_reserve", reserve=0.0, lws=8, files={}, total=1 << 40),
            param(
                "container_capped",
                reserve=0.15,
                lws=8,
                files={"/sys/fs/cgroup/memory.max": str(128 * GIB)},
                total=128 * GIB,
            ),
            param(
                "unlimited_cgroup",
                reserve=0.15,
                lws=8,
                files={"/sys/fs/cgroup/memory.max": "max"},
                total=1 << 40,
            ),
            param("single_rank", reserve=0.15, lws=1, files={}, total=1 << 40),
            param(
                "v1_container_capped",
                reserve=0.0,
                lws=8,
                files={"/sys/fs/cgroup/memory/memory.limit_in_bytes": str(128 * GIB)},
                total=128 * GIB,
            ),
            param(
                "v1_unlimited_sentinel",
                reserve=0.0,
                lws=8,
                files={
                    "/sys/fs/cgroup/memory/memory.limit_in_bytes": str((1 << 63) - 1)
                },
                total=1 << 40,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_effective_capacity(self, _, reserve, lws, files, total):
        patches = _container(1 << 40, files)
        with patches[0], patches[1]:
            got = resource_util.host_memory_capacity_per_rank(
                local_world_size=lws, reserve_percent=reserve
            )
        self.assertEqual(got, int(total * (1 - reserve) / lws))

    def test_zero_reserve_respects_container_limit(self):
        # A zero reserve drops only the margin: the 128 GiB container cap,
        # not the 1 TiB of node RAM, still bounds the budget.
        patches = _container(1 << 40, {"/sys/fs/cgroup/memory.max": str(128 * GIB)})
        with patches[0], patches[1]:
            got = resource_util.host_memory_capacity_per_rank(
                local_world_size=8, reserve_percent=0.0
            )
        self.assertEqual(got, 16 * GIB)


if __name__ == "__main__":
    unittest.main()
