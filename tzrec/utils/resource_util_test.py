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
import unittest
from types import SimpleNamespace
from unittest import mock

from parameterized import param, parameterized

from tzrec.utils import resource_util
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

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
        total = (1 << 40) if limit is None else 128 * GIB
        self.assertEqual(got, int(total * (1 - reserve) / lws))

    def test_128gib_container_8_ranks(self):
        # 128 GiB container on a 1 TiB host, 8 ranks: 13.6 GiB per rank.
        patches = _container(1 << 40, str(128 * GIB))
        with patches[0], patches[1]:
            got = resource_util.host_memory_capacity_per_rank(
                local_world_size=8, reserve_percent=0.15
            )
        self.assertEqual(got, int(128 * GIB * 0.85 / 8))


class SSDCapacityTest(unittest.TestCase):
    """Per-rank secondary-storage quota under a configured sharing scope."""

    def _statvfs_patch(self, avail):
        return mock.patch.object(
            os,
            "statvfs",
            return_value=SimpleNamespace(f_bavail=avail, f_frsize=1),
        )

    def test_node_local_versus_all_node_sharing(self):
        # 800 GiB free: 4 ranks on one node get 200 GiB each, a shared
        # cross-node volume with 8 ranks gets 100 GiB each.
        root = os.path.realpath(make_test_dir())
        with self._statvfs_patch(800 * GIB):
            self.assertEqual(
                resource_util.ssd_capacity_per_rank(
                    root, sharing_ranks=4, reserve_bytes=0
                ),
                200 * GIB,
            )
            self.assertEqual(
                resource_util.ssd_capacity_per_rank(
                    root, sharing_ranks=8, reserve_bytes=0
                ),
                100 * GIB,
            )

    def test_budget_caps_free_space_and_reserve(self):
        root = os.path.realpath(make_test_dir())
        with self._statvfs_patch(1900 * GIB):
            self.assertEqual(
                resource_util.ssd_capacity_per_rank(
                    root,
                    sharing_ranks=8,
                    budget_bytes=800 * GIB,
                    reserve_bytes=16 * GIB,
                ),
                (800 * GIB - 16 * GIB) // 8,
            )

    def test_zero_remaining_quota(self):
        # Free space below the reserve leaves zero; memory-only remains.
        root = os.path.realpath(make_test_dir())
        with self._statvfs_patch(10 * GIB):
            self.assertEqual(
                resource_util.ssd_capacity_per_rank(
                    root, sharing_ranks=1, reserve_bytes=16 * GIB
                ),
                0,
            )

    def test_missing_root_uses_nearest_ancestor(self):
        root = os.path.join(os.path.realpath(make_test_dir()), "not", "yet")
        with self._statvfs_patch(800 * GIB):
            self.assertEqual(
                resource_util.ssd_capacity_per_rank(
                    root, sharing_ranks=1, reserve_bytes=0
                ),
                800 * GIB,
            )
        self.assertFalse(os.path.exists(root))


if __name__ == "__main__":
    unittest.main()
