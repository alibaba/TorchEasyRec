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


def _container(total, cgroup_text=None):
    files = {} if cgroup_text is None else {"/sys/fs/cgroup/memory.max": cgroup_text}

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
            param("default_reserve", reserve=0.15, lws=8, limit=None),
            param("explicit_reserve", reserve=0.5, lws=8, limit=None),
            param("zero_reserve", reserve=0.0, lws=8, limit=None),
            param("container_capped", reserve=0.15, lws=8, limit=str(128 * GIB)),
            param("unlimited_cgroup", reserve=0.15, lws=8, limit="max"),
            param("single_rank", reserve=0.15, lws=1, limit=None),
        ],
        name_func=parameterized_name_func,
    )
    def test_effective_capacity(self, _, reserve, lws, limit):
        patches = _container(1 << 40, limit)
        with patches[0], patches[1]:
            got = resource_util.host_memory_capacity_per_rank(
                local_world_size=lws, reserve_percent=reserve
            )
        total = (1 << 40) if limit in (None, "max") else 128 * GIB
        self.assertEqual(got, int(total * (1 - reserve) / lws))

    def test_zero_reserve_respects_container_limit(self):
        # A zero reserve drops only the margin: the 128 GiB container cap,
        # not the 1 TiB of node RAM, still bounds the budget.
        patches = _container(1 << 40, str(128 * GIB))
        with patches[0], patches[1]:
            got = resource_util.host_memory_capacity_per_rank(
                local_world_size=8, reserve_percent=0.0
            )
        self.assertEqual(got, 16 * GIB)


if __name__ == "__main__":
    unittest.main()
