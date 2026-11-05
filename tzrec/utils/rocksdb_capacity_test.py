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
from unittest import mock

from parameterized import param, parameterized

from tzrec.protos import train_pb2
from tzrec.utils import rocksdb_capacity
from tzrec.utils.rocksdb_capacity import DISK_RESERVE_BYTES, DiskView, TableDisk
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

GIB = 1 << 30
OVERHEADS = {"compaction_headroom": 10 * GIB, "other_files": GIB}


def _view(rank=0, pool="net:nas.a", free=1 << 50, job=0):
    return DiskView(rank=rank, pool=pool, free_bytes=free, job_bytes=job)


def _table(rows=10**8, row_bytes=100, index=20, cache_rows=0, overshoot=0):
    return TableDisk(
        name="t",
        rows=rows,
        row_bytes=row_bytes,
        index_bytes=index,
        bound="test",
        cache_rows=cache_rows,
        overshoot_rows=overshoot,
    )


class DiskPoolTest(unittest.TestCase):
    """Which ranks share the free space of one disk pool."""

    @parameterized.expand(
        [
            param("nfs_same_server", a=("nfs", "nas.a:/x"), b=("nfs4", "nas.a:/y")),
            param("nfs_other_server", a=("nfs", "nas.a:/x"), b=("nfs", "nas.b:/x")),
            param("local_same_host", a=("ext4", "/dev/vdb1"), b=("ext4", "/dev/vdb1")),
            param(
                "local_other_host",
                a=("ext4", "/dev/vdb1"),
                b=("ext4", "/dev/vdb1"),
                host="h1",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_pool_key(self, name, a, b, host="h0"):
        ka = rocksdb_capacity.pool_key("/mnt/db", a[0], a[1], "h0", 7)
        kb = rocksdb_capacity.pool_key("/mnt/db", b[0], b[1], host, 7)
        self.assertEqual(ka == kb, name in ("nfs_same_server", "local_same_host"))

    def test_fuse_is_rejected(self):
        with self.assertRaisesRegex(rocksdb_capacity.DiskCapacityError, "local disk"):
            rocksdb_capacity.pool_key("/mnt/oss", "fuse.ossfs", "ossfs", "h0", 7)

    def test_allocated_bytes_counts_hard_links_once(self):
        root = make_test_dir()
        with open(os.path.join(root, "a.sst"), "wb") as f:
            f.write(b"x" * (1 << 20))
        os.link(os.path.join(root, "a.sst"), os.path.join(root, "b.sst"))
        once = rocksdb_capacity.allocated_bytes(root)
        self.assertGreaterEqual(once, 1 << 20)
        self.assertLess(once, 2 << 20)

    def test_observe_real_path(self):
        root = os.path.realpath(make_test_dir())
        view = rocksdb_capacity.observe(os.path.join(root, "not", "yet"), root, 3)
        self.assertEqual(view.rank, 3)
        self.assertGreater(view.free_bytes, 0)


class DiskPlanTest(unittest.TestCase):
    """Projected peak per disk pool against the budget and the free space."""

    def _plan(self, tables, views, budget=None):
        return rocksdb_capacity.plan_disk_pools(tables, views, budget, OVERHEADS)

    def test_components(self):
        (p,) = self._plan([_table(cache_rows=10**6, overshoot=1000)], [_view()])
        logical = 10**8 * 120
        self.assertEqual(p.logical_bytes, logical)
        self.assertEqual(p.sst_bytes, int(logical * rocksdb_capacity.SST_FACTOR))
        self.assertEqual(p.save_extra, 10**6 * 100)
        self.assertEqual(
            p.training_peak,
            p.sst_bytes
            + 10 * GIB
            + GIB
            + int(1000 * 120 * rocksdb_capacity.SST_FACTOR),
        )
        self.assertEqual(p.restore_peak, p.sst_bytes + 10 * GIB + GIB + 10**8 * 100)
        # Scenarios are alternatives: the peak is their max plus the reserve.
        self.assertEqual(
            p.projected_peak,
            max(p.training_peak, p.save_peak, p.restore_peak) + DISK_RESERVE_BYTES,
        )
        self.assertTrue(p.feasible)

    def test_budget_and_free_space_are_separate_limits(self):
        (fits,) = self._plan([_table()], [_view()])
        peak = fits.projected_peak
        (over_budget,) = self._plan([_table()], [_view()], budget=peak - 1)
        self.assertIn("budget", over_budget.problems[0])
        # No budget: the free space alone limits the plan.
        (no_space,) = self._plan([_table()], [_view(free=peak - 1)])
        self.assertIn("free", no_space.problems[0])

    def test_existing_job_allocation_is_not_double_counted(self):
        (fresh,) = self._plan([_table()], [_view()])
        peak = fresh.projected_peak
        # The job already holds 100 GiB of the peak; free space already lacks it.
        (resumed,) = self._plan(
            [_table()], [_view(free=peak - 100 * GIB, job=100 * GIB)]
        )
        self.assertTrue(resumed.feasible, resumed.problems)
        (short,) = self._plan(
            [_table()], [_view(free=peak - 100 * GIB - 1, job=100 * GIB)]
        )
        self.assertFalse(short.feasible)

    def test_shared_nas_vs_independent_disks(self):
        shared = self._plan([_table()], [_view(0), _view(1)])
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].ranks, [0, 1])
        (one,) = self._plan([_table()], [_view()])
        self.assertEqual(shared[0].sst_bytes, 2 * one.sst_bytes)
        # The smallest free space any rank sees counts for the shared pool.
        (p,) = self._plan([_table()], [_view(0, free=10), _view(1)])
        self.assertEqual(p.free_bytes, 10)
        local = self._plan(
            [_table()], [_view(0, pool="local:h0:7"), _view(1, pool="local:h1:7")]
        )
        self.assertEqual(len(local), 2)

    def test_pool_capacity(self):
        fixed = rocksdb_capacity.pool_fixed_bytes(OVERHEADS, 5 * GIB)
        self.assertEqual(fixed, 16 * GIB)
        # Two ranks on one NAS with 1 TiB free split it after the reserve.
        views = [_view(0, free=1 << 40), _view(1, free=1 << 40)]
        room = rocksdb_capacity.pool_capacity(views, None, fixed)
        self.assertEqual(room, ((1 << 40) - DISK_RESERVE_BYTES - 2 * fixed) // 2)
        # A budget below the free space governs; the tightest pool counts.
        self.assertEqual(
            rocksdb_capacity.pool_capacity(views, 100 * GIB, fixed),
            (100 * GIB - DISK_RESERVE_BYTES - 2 * fixed) // 2,
        )
        tight = views + [_view(2, pool="local:h2:7", free=50 * GIB)]
        self.assertEqual(
            rocksdb_capacity.pool_capacity(tight, None, fixed),
            50 * GIB - DISK_RESERVE_BYTES - fixed,
        )
        self.assertEqual(
            rocksdb_capacity.pool_capacity([_view(free=GIB)], None, fixed), 0
        )

    def test_growth_is_sst_overshoot_and_flush(self):
        t = _table(cache_rows=10**6, overshoot=1000)
        self.assertEqual(
            t.growth_bytes,
            int((10**8 + 1000) * 120 * rocksdb_capacity.SST_FACTOR) + 10**6 * 100,
        )


class _Session:
    opened = False


class CapacityMonitorTest(unittest.TestCase):
    """Runtime headroom: stop with a save, or without one."""

    def _monitor(self, budget=None):
        cfg = train_pb2.RocksDBConfig(root_dir="/mnt/db")
        if budget is not None:
            cfg.disk_budget_bytes = budget
        return rocksdb_capacity.CapacityMonitor(cfg, "/unused", _Session())

    def _check(self, monitor, free, job, save_need=0, step=100):
        with mock.patch.object(
            rocksdb_capacity, "observe", return_value=_view(free=free, job=job)
        ):
            return monitor.check(step, save_need)

    def test_cadence(self):
        self.assertIsNone(self._check(self._monitor(), 0, 0, step=101))

    @parameterized.expand(
        [
            param("healthy", free=500 * GIB, job=100 * GIB, need=0, action=None),
            # External consumers shrank the free space below 5% of the pool.
            param(
                "critical_save",
                free=20 * GIB,
                job=100 * GIB,
                need=GIB,
                action="save_and_stop",
            ),
            param(
                "critical_no_room",
                free=18 * GIB,
                job=100 * GIB,
                need=20 * GIB,
                action="stop",
            ),
            # This job's own growth approaches its budget.
            param(
                "budget",
                free=900 * GIB,
                job=990 * GIB,
                need=0,
                action="save_and_stop",
                budget=1000 * GIB,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_actions(self, _name, free, job, need, action, budget=None):
        m = self._monitor(budget)
        self.assertEqual(self._check(m, free, job, need), action)
        self.assertEqual(m.last["action"], action)


if __name__ == "__main__":
    unittest.main()
