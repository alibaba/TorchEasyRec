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

import builtins
import dataclasses
import math
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from tzrec.protos import train_pb2
from tzrec.utils import rocksdb_util
from tzrec.utils.test_util import make_test_dir

try:
    import tzrec_ext.rocksdb  # noqa: F401

    has_rocksdb = True
except ImportError:
    has_rocksdb = False

GIB = 1 << 30
OVERHEADS = {"compaction_headroom": 10 * GIB, "other_files": GIB}


def _backend_config():
    from tzrec_ext.rocksdb import BackendConfig

    return BackendConfig(32 << 20, 32 << 20, 1 << 20, 64 << 20, io_threads=2)


class RocksDBRuntimeTest(unittest.TestCase):
    """Lifecycle of the per-process RocksDB runtime."""

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")
        self.test_dir = make_test_dir()
        self.train_config = train_pb2.TrainConfig(
            rocksdb_config=train_pb2.RocksDBConfig(
                root_dir=os.path.join(self.test_dir, "db")
            )
        )

    def test_factory_returns_a_runtime_for_a_selected_config(self):
        rt = rocksdb_util.create_rocksdb_runtime(
            self.train_config.rocksdb_config, "model", "eval"
        )
        self.assertIsNotNone(rt)
        rt.close()

    def test_missing_wheel_warns_and_stays_in_memory(self):
        # The factory owns the availability policy: a missing top-level
        # tzrec-ext wheel leaves every table in GPU/host memory.
        real_import = builtins.__import__

        def _no_tzrec_ext(name, *args, **kwargs):
            if name == "tzrec_ext":
                raise ModuleNotFoundError(name=name)
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=_no_tzrec_ext):
            self.assertIsNone(
                rocksdb_util.create_rocksdb_runtime(
                    self.train_config.rocksdb_config, "model", "eval"
                )
            )

    def test_cache_rows(self):
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.0), 0)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.001), 1024)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.01), 10240)

    def test_bind_opens_one_session_per_rank(self):
        try:
            from tzrec_ext.rocksdb.storage import RocksDBStorageFactory
        except ModuleNotFoundError as e:
            # ``storage`` imports dynamicemb at module level, and only the
            # GPU requirement variants install it; the facade the class
            # guard checks stays importable without it.
            self.skipTest(f"tzrec_ext.rocksdb.storage needs {e.name}")

        config = _backend_config()
        options = [
            SimpleNamespace(external_storage=RocksDBStorageFactory(config, f"m.{n}"))
            for n in "ab"
        ]
        plan = SimpleNamespace(
            plan={
                "m": {
                    n: SimpleNamespace(dynamicemb_options=o)
                    for n, o in zip("ab", options)
                }
            }
        )
        rt = rocksdb_util.create_rocksdb_runtime(
            self.train_config.rocksdb_config, "model", "eval"
        )
        try:
            rt.bind(plan)
            self.assertEqual(rt.session.config, config)
            self.assertTrue(
                rt.session.work_dir.startswith(os.path.realpath(rt.rank_dir))
            )
            for o in options:
                self.assertIs(o.external_storage.session, rt.session)
            rt.bind(plan)  # the same plan again keeps the session
            other = dataclasses.replace(config, block_cache_bytes=1)
            options[0].external_storage = RocksDBStorageFactory(other, "m.a")
            with self.assertRaisesRegex(RuntimeError, "config"):
                rt.bind(plan)
        finally:
            rt.close()

    def test_close_releases_the_working_db_on_errors(self):
        from tzrec_ext.rocksdb import RocksDBSession

        rt = rocksdb_util.create_rocksdb_runtime(
            self.train_config.rocksdb_config, "model", "eval"
        )
        with self.assertRaisesRegex(ValueError, "boom"):
            try:
                rt.session = RocksDBSession(_backend_config(), rt._new_work_dir())
                self.assertTrue(rt.session.get_or_create_namespace_ids(["t"]))
                work = rt.session.work_dir
                self.assertTrue(os.path.isdir(work))
                raise ValueError("boom")
            finally:
                rt.close()
        self.assertIsNone(rt.session)
        self.assertFalse(os.path.exists(work))

    def test_train_removes_stale_working_dbs(self):
        rt = rocksdb_util.RocksDBRuntime(
            self.train_config.rocksdb_config, "model", "eval"
        )
        stale = [os.path.join(rt.rank_dir, d) for d in ("train-1-2", "eval-1-2")]
        for d in stale:
            os.makedirs(d)
        rocksdb_util.RocksDBRuntime(self.train_config.rocksdb_config, "model", "train")
        self.assertEqual(os.listdir(rt.rank_dir), ["eval-1-2"])


class ActivationCostTest(unittest.TestCase):
    """The shared per-rank backend activation and the phase-bound arithmetic."""

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")

    def test_activation_charges_fixed_costs_once(self):
        # DDR carries the planned fixed buffers and headroom; SSD carries
        # compaction, other DB files and the restore-spill envelope; the
        # SSD reserve is already gone from the topology capacity.
        activation = rocksdb_util.activation_storage(
            local_world_size=1, restore_spill_bytes=5 * GIB
        )
        self.assertEqual(activation.hbm, 0)
        self.assertEqual(activation.ddr, rocksdb_util.fixed_host_bytes())
        from tzrec_ext.rocksdb import disk_overheads

        # The overheads come from the same backend configuration the runtime
        # derives its budgets from.
        overheads = disk_overheads(
            rocksdb_util.backend_config(1, block_cache_bytes=0, import_budget_bytes=0)
        )
        self.assertEqual(
            activation.ssd,
            overheads["compaction_headroom"] + overheads["other_files"] + 5 * GIB,
        )

    def test_charged_budget_covers_training_save_and_restore(self):
        # The additive cost model bounds each phase: charged_ssd = sum of
        # per-shard ceil(SST*(L+O)) + W plus the shared activation, and that
        # covers training (SST + compaction + other), saving (training plus
        # writeback) and restoring (SST + compaction + other + max spill).
        f = rocksdb_util.SST_FACTOR
        # One shard: 1M rows, 100 B live, 20 B index, 1k overshoot, 10k cache.
        live = (1 << 20) * (100 + 20)
        overshoot = 1000 * (100 + 20)
        writeback = 10_000 * 100
        table_cost = math.ceil(f * (live + overshoot)) + writeback
        activation = rocksdb_util.activation_storage(
            local_world_size=1, restore_spill_bytes=(1 << 20) * 100
        )
        charged = table_cost + activation.ssd
        training = math.ceil(f * live)  # per-shard tables sum equivalently
        self.assertGreaterEqual(charged, training)
        self.assertGreaterEqual(charged, training + writeback)
        self.assertGreaterEqual(charged, math.ceil(f * live) + (1 << 20) * 100)
        # Upward rounding never undershoots: ceil of the sum <= sum of ceils.
        self.assertGreaterEqual(table_cost, int(f * (live + overshoot)) + writeback)


if __name__ == "__main__":
    unittest.main()
