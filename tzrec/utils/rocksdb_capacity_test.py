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
from tzrec.utils.rocksdb_capacity import DomainObservation, TableDisk
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

GIB = 1 << 30
OVERHEADS = {"compaction_headroom": 10 * GIB, "other_files": GIB}


def _obs(
    rank=0,
    host="h0",
    fstype="nfs",
    source="nas.a:/share",
    free=1 << 50,
    job=0,
    st_dev=7,
):
    return DomainObservation(
        rank=rank,
        host=host,
        path="/mnt/db",
        fstype=fstype,
        source=source,
        mount_point="/mnt",
        st_dev=st_dev,
        free_bytes=free,
        total_bytes=1 << 52,
        job_bytes=job,
    )


def _table(rows=10**8, row_bytes=100, index=20, cache_rows=0, overshoot=0):
    return TableDisk(
        name="t",
        rows=rows,
        row_bytes=row_bytes,
        index_bytes=index,
        bound="test",
        cache_rows=cache_rows,
        overshoot_rows=overshoot,
        value_file_bytes=0,
    )


def _cfg(**kwargs):
    kwargs.setdefault("disk_budget_bytes", 1 << 50)
    kwargs.setdefault("disk_reserve_bytes", GIB)
    return train_pb2.RocksDBConfig(local_path="/mnt/db", **kwargs)


class StorageDomainTest(unittest.TestCase):
    """Which ranks share one capacity pool."""

    @parameterized.expand(
        [
            param("nfs_same_server", a={}, b={"source": "nas.a:/other"}, same=True),
            param("nfs_other_server", a={}, b={"source": "nas.b:/share"}, same=False),
            param(
                "local_same_host",
                a={"fstype": "ext4", "source": "/dev/vdb1"},
                b={"fstype": "ext4", "source": "/dev/vdb1"},
                same=True,
            ),
            param(
                "local_other_host",
                a={"fstype": "ext4", "source": "/dev/vdb1"},
                b={"fstype": "ext4", "source": "/dev/vdb1", "host": "h1"},
                same=False,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_domain_key(self, _name, a, b, same):
        ka = rocksdb_capacity.domain_key(_obs(**a))
        kb = rocksdb_capacity.domain_key(_obs(rank=1, **b))
        self.assertEqual(ka == kb, same)

    def test_fuse_needs_explicit_domain(self):
        obs = _obs(fstype="fuse.ossfs", source="ossfs")
        with self.assertRaisesRegex(
            rocksdb_capacity.DiskCapacityError, "storage_domain"
        ):
            rocksdb_capacity.domain_key(obs)
        self.assertEqual(rocksdb_capacity.domain_key(obs, "pool"), "explicit:pool")

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
        obs = rocksdb_capacity.observe(os.path.join(root, "not", "yet"), root, 3)
        self.assertEqual(obs.rank, 3)
        self.assertTrue(root.startswith(obs.path) or obs.path.startswith(root))
        self.assertGreater(obs.free_bytes, 0)


class DiskPlanTest(unittest.TestCase):
    """Projected peak per domain against the budget and the free space."""

    def _plan(self, tables, observations, **cfg):
        return rocksdb_capacity.plan_domains(
            tables, observations, _cfg(**cfg), OVERHEADS
        )

    def test_components(self):
        (p,) = self._plan([_table(cache_rows=10**6, overshoot=1000)], [_obs()])
        logical = 10**8 * 120
        self.assertEqual(p.logical_bytes, logical)
        self.assertEqual(p.sst_bytes, int(logical * 1.5))
        self.assertEqual(p.sst_factor_source, "provisional, not measured")
        self.assertEqual(p.save_extra, 10**6 * 100)
        self.assertEqual(
            p.training_peak,
            p.sst_bytes + 10 * GIB + GIB + int(1000 * 120 * 1.5),
        )
        # Scenarios are alternatives: the peak is their max plus the reserve.
        self.assertEqual(
            p.projected_peak, max(p.training_peak, p.save_peak, p.restore_peak) + GIB
        )
        self.assertTrue(p.feasible)
        self.assertFalse(p.quota_observable)

    def test_configured_factor(self):
        (p,) = self._plan([_table()], [_obs()], sst_space_factor=1.2)
        self.assertEqual(p.sst_factor_source, "configured")
        # sst_space_factor is a float32 field.
        self.assertAlmostEqual(p.sst_bytes / (10**8 * 120), 1.2, places=6)

    def test_budget_and_free_space_are_separate_limits(self):
        (fits,) = self._plan([_table()], [_obs()])
        peak = fits.projected_peak
        (over_budget,) = self._plan([_table()], [_obs()], disk_budget_bytes=peak - 1)
        self.assertIn("budget", over_budget.problems[0])
        # Quota/budget allows it, but the filesystem has less free space.
        (no_space,) = self._plan([_table()], [_obs(free=peak - 1)])
        self.assertIn("free", no_space.problems[0])

    def test_existing_job_allocation_is_not_double_counted(self):
        (fresh,) = self._plan([_table()], [_obs()])
        peak = fresh.projected_peak
        # The job already holds 100 GiB of the peak; free space already lacks it.
        (resumed,) = self._plan(
            [_table()], [_obs(free=peak - 100 * GIB, job=100 * GIB)]
        )
        self.assertTrue(resumed.feasible, resumed.problems)
        (short,) = self._plan(
            [_table()], [_obs(free=peak - 100 * GIB - 1, job=100 * GIB)]
        )
        self.assertFalse(short.feasible)

    def test_zero_new_allocation(self):
        (p,) = self._plan([], [_obs(free=0, job=0)])
        self.assertEqual(p.logical_bytes, 0)
        self.assertEqual(p.projected_peak, 10 * GIB + GIB + GIB)

    def test_shared_nas_vs_independent_disks(self):
        shared = self._plan([_table()], [_obs(0), _obs(1, host="h1")])
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0].ranks, [0, 1])
        (one,) = self._plan([_table()], [_obs()])
        self.assertEqual(shared[0].sst_bytes, 2 * one.sst_bytes)
        # The smallest free space any rank sees counts for the shared pool.
        (p,) = self._plan([_table()], [_obs(0, free=10), _obs(1, free=1 << 50)])
        self.assertEqual(p.free_bytes, 10)
        local = self._plan(
            [_table()],
            [
                _obs(0, fstype="ext4", source="/dev/nvme0n1"),
                _obs(1, host="h1", fstype="ext4", source="/dev/nvme0n1"),
            ],
        )
        self.assertEqual(len(local), 2)
        self.assertTrue(all(d.quota_observable for d in local))

    def test_missing_budget_is_a_problem(self):
        (p,) = rocksdb_capacity.plan_domains(
            [_table()],
            [_obs()],
            train_pb2.RocksDBConfig(local_path="/mnt/db"),
            OVERHEADS,
        )
        self.assertIn("disk_budget_bytes is not set", p.problems[0])


class _Session:
    opened = False


class CapacityMonitorTest(unittest.TestCase):
    """Runtime headroom: warn, stop with a save, stop without one."""

    def _monitor(self, **cfg):
        cfg.setdefault("capacity_check_interval_steps", 10)
        cfg.setdefault("disk_budget_bytes", 1000 * GIB)
        return rocksdb_capacity.CapacityMonitor(_cfg(**cfg), "/unused", _Session())

    def _check(self, monitor, free, job, save_need=0, step=10):
        with mock.patch.object(
            rocksdb_capacity, "observe", return_value=_obs(free=free, job=job)
        ):
            return monitor.check(step, save_need)

    def test_cadence(self):
        m = self._monitor()
        self.assertIsNone(self._check(m, 0, 0, step=11))

    def test_defaults_are_budget_shares(self):
        m = self._monitor()
        self.assertEqual(m.warn, 100 * GIB)
        self.assertEqual(m.critical, 50 * GIB)

    @parameterized.expand(
        [
            param("healthy", free=500 * GIB, job=100 * GIB, need=0, action=None),
            # External consumers shrank the free space.
            param("warn_free", free=81 * GIB, job=100 * GIB, need=0, action="warn"),
            # This job's own growth approaches its budget.
            param("warn_budget", free=900 * GIB, job=910 * GIB, need=0, action="warn"),
            param(
                "critical_save",
                free=40 * GIB,
                job=100 * GIB,
                need=GIB,
                action="save_and_stop",
            ),
            param(
                "critical_no_room",
                free=10 * GIB,
                job=100 * GIB,
                need=20 * GIB,
                action="stop",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_actions(self, _name, free, job, need, action):
        m = self._monitor()
        self.assertEqual(self._check(m, free, job, need), action)
        self.assertEqual(m.last["action"], action)


if __name__ == "__main__":
    unittest.main()
