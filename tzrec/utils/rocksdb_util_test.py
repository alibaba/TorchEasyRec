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
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from tzrec.protos import train_pb2
from tzrec.utils import rocksdb_util
from tzrec.utils.test_util import make_test_dir


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

    def test_train_preserves_other_working_dbs(self):
        from tzrec_ext.rocksdb import RocksDBSession

        rt = rocksdb_util.RocksDBRuntime(
            self.train_config.rocksdb_config, "model", "eval"
        )
        existing = ["eval-1-2", "train-1-2"]
        for name in existing:
            os.makedirs(os.path.join(rt.rank_dir, name))
        rt = rocksdb_util.RocksDBRuntime(
            self.train_config.rocksdb_config, "model", "train"
        )
        try:
            self.assertEqual(sorted(os.listdir(rt.rank_dir)), existing)
            rt.session = RocksDBSession(_backend_config(), rt._new_work_dir())
            self.assertTrue(rt.session.get_or_create_namespace_ids(["t"]))
            work = rt.session.work_dir
            self.assertTrue(os.path.isdir(work))
        finally:
            rt.close()
        self.assertFalse(os.path.exists(work))
        self.assertEqual(sorted(os.listdir(rt.rank_dir)), existing)


class SharedStorageTest(unittest.TestCase):
    """Shared resources reserved per rank for RocksDB planning."""

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")

    def test_shared_storage(self):
        from tzrec_ext.rocksdb import disk_overheads

        shared = rocksdb_util.shared_storage(local_world_size=1)
        self.assertEqual(shared.hbm, 0)
        self.assertEqual(shared.ddr, int(4.25 * (1 << 30)))
        overheads = disk_overheads(
            rocksdb_util.backend_config(1, block_cache_bytes=0, import_budget_bytes=0)
        )
        self.assertEqual(
            shared.ssd, overheads["compaction_headroom"] + overheads["other_files"]
        )

    def test_configure_backend_budgets(self):
        try:
            from tzrec_ext.rocksdb.storage import RocksDBStorageFactory
        except ModuleNotFoundError as e:
            self.skipTest(f"tzrec_ext.rocksdb.storage needs {e.name}")

        gib = 1 << 30
        selected = [
            SimpleNamespace(
                fqn=name,
                cache_load_factor=0.1,
                dynamicemb_options=SimpleNamespace(
                    external_storage=(
                        RocksDBStorageFactory(_backend_config(), name)
                        if name != "plain"
                        else None
                    )
                ),
                shards=[
                    SimpleNamespace(rank=rank, storage=SimpleNamespace(ddr=ddr))
                    for rank, ddr in enumerate(ddr_bytes)
                ],
            )
            for name, ddr_bytes in (
                ("plain", [3 * gib, 7 * gib]),
                ("sparse.a", [gib // 4, gib // 2]),
                ("sparse.b", [3 * gib // 4, gib]),
            )
        ]
        configured = rocksdb_util.configure_backend(
            selected, ddr_caps=(16 * gib, 16 * gib), local_world_size=2
        )
        self.assertIsNone(configured[0].dynamicemb_options.external_storage)
        for option in configured[1:]:
            factory = option.dynamicemb_options.external_storage
            self.assertEqual(factory.table_namespace, option.fqn)
            self.assertEqual(factory.backend_config.block_cache_bytes, 5 * gib // 2)
            self.assertEqual(factory.backend_config.import_budget_bytes, 27 * gib // 4)


if __name__ == "__main__":
    unittest.main()
