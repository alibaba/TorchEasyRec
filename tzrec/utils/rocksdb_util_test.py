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

from parameterized import parameterized
from torch import nn
from torch.distributed.checkpoint import save

from tzrec.protos import train_pb2
from tzrec.utils import checkpoint_util, rocksdb_util
from tzrec.utils.test_util import make_test_dir, mark_ci_scope, parameterized_name_func


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

    def test_checkpoint_save_uses_the_bound_background_io_rate(self):
        from tzrec_ext.rocksdb import RocksDBSession

        rt = rocksdb_util.RocksDBRuntime(
            self.train_config.rocksdb_config, self.test_dir, "train"
        )
        self.addCleanup(rt.close)
        manager = checkpoint_util.CheckpointManager(self.test_dir, rocksdb=rt)
        self.addCleanup(manager.close)
        model = nn.Linear(1, 1)

        with mock.patch("tzrec.utils.checkpoint_util.save_model"):
            unbound_checkpoint = manager.save(1, model)
        self.assertTrue(os.path.isfile(os.path.join(unbound_checkpoint, "_SUCCESS")))

        config = dataclasses.replace(
            _backend_config(),
            background_io_base_rate_mib_per_sec=96,
            background_io_max_rate_mib_per_sec=512,
        )
        session = RocksDBSession(config, os.path.join(self.test_dir, "working"))
        rt.session = session
        db = session.db
        self.assertEqual(db.background_io_rate, 96 << 20)

        def save_model(*_):
            self.assertEqual(db.background_io_rate, 512 << 20)
            self.assertEqual(session.writeback_source, "checkpoint")

        with mock.patch(
            "tzrec.utils.checkpoint_util.save_model", side_effect=save_model
        ):
            bound_checkpoint = manager.save(2, model)
        self.assertTrue(os.path.isfile(os.path.join(bound_checkpoint, "_SUCCESS")))
        self.assertEqual(db.background_io_rate, 96 << 20)
        self.assertEqual(session.writeback_source, "train")

    @mark_ci_scope("gpu")
    @unittest.skipUnless(checkpoint_util.has_dynamicemb, "DynamicEmb is required.")
    def test_checkpoint_restore_replaces_the_db_only_for_dynamicemb(self):
        from tzrec_ext.rocksdb import RocksDBSession

        rt = rocksdb_util.RocksDBRuntime(
            self.train_config.rocksdb_config, self.test_dir, "train"
        )
        self.addCleanup(rt.close)
        session = RocksDBSession(
            _backend_config(), os.path.join(self.test_dir, "warmup")
        )
        rt.session = session
        warmup_namespaces = session.get_or_create_namespace_ids(["warmup"])
        old_db = session.db
        manager = checkpoint_util.CheckpointManager(self.test_dir, rocksdb=rt)
        self.addCleanup(manager.close)

        checkpoint_dir = os.path.join(self.test_dir, "model.ckpt-1")
        model = nn.Linear(1, 1)
        save(model.state_dict(), checkpoint_id=os.path.join(checkpoint_dir, "model"))

        def load_dynamicemb(*_, **__):
            self.assertEqual(session.all_namespaces(), {})
            self.assertTrue(old_db.closed)

        with (
            mock.patch.object(
                rocksdb_util, "rocksdb_modules", return_value=[mock.Mock()]
            ),
            mock.patch(
                "dynamicemb.dump_load.DynamicEmbLoad", side_effect=load_dynamicemb
            ) as load_embeddings,
        ):
            manager.restore(checkpoint_dir, model)
            self.assertEqual(session.all_namespaces(), warmup_namespaces)
            load_embeddings.assert_not_called()

            os.makedirs(os.path.join(checkpoint_dir, "dynamicemb"))
            manager.restore(checkpoint_dir, model)
            load_embeddings.assert_called_once()


class SharedStorageTest(unittest.TestCase):
    """Shared resources reserved per rank for RocksDB planning."""

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")

    def test_shared_storage(self):
        from tzrec_ext.rocksdb import disk_overheads

        config = rocksdb_util.backend_config(local_world_size=1)
        shared = rocksdb_util.shared_storage(config)
        self.assertEqual(shared.hbm, 0)
        self.assertEqual(shared.ddr, 9 * (1 << 28))
        overheads = disk_overheads(config)
        if isinstance(overheads, dict):
            overheads = sum(overheads.values())
        self.assertEqual(shared.ssd, overheads)
        if hasattr(config, "score_cache_bytes"):
            with_scores = dataclasses.replace(config, score_cache_bytes=960)
            self.assertEqual(
                rocksdb_util.shared_storage(with_scores).ddr, shared.ddr + 8 * 4 * 24
            )

    @parameterized.expand(
        [("master_default", 32, None, 16), ("shard_floor", 8, 8, 8)],
        name_func=parameterized_name_func,
    )
    def test_default_backend_io_threads(self, _, cpus, num_shards, expected):
        config = (
            train_pb2.RocksDBConfig(root_dir="/tmp", num_db_shards_per_rank=num_shards)
            if num_shards is not None
            else None
        )
        with mock.patch("os.sched_getaffinity", return_value=set(range(cpus))):
            backend = rocksdb_util.backend_config(
                2,
                block_cache_bytes=32 << 20,
                import_budget_bytes=64 << 20,
                config=config,
            )
        self.assertEqual(backend.io_threads, expected)

    def test_io_config_sets_rank_budget_and_increases_per_database_ssd_reserve(self):
        config = train_pb2.RocksDBConfig(
            root_dir="/tmp",
            num_db_shards_per_rank=4,
            io_threads_per_rank=24,
            sst_block_size_bytes=2048,
            block_align=True,
            use_direct_reads=True,
            direct_read_alignment_bytes=512,
            background_io_base_rate_per_rank_mib_per_sec=96,
            background_io_max_rate_per_rank_mib_per_sec=512,
        )
        with mock.patch("os.sched_getaffinity", return_value=set(range(16))):
            backend = rocksdb_util.backend_config(1, config=config)
            default_backend = rocksdb_util.backend_config(1)
        self.assertEqual(backend.max_background_jobs, 8)
        self.assertEqual(backend.io_threads, 24)
        self.assertEqual(backend.num_db_shards, 4)
        self.assertEqual(backend.sst_block_size_bytes, 2048)
        self.assertTrue(backend.block_align)
        self.assertTrue(backend.use_direct_reads)
        self.assertEqual(backend.direct_read_alignment_bytes, 512)
        self.assertEqual(backend.background_io_base_rate_mib_per_sec, 96)
        self.assertEqual(backend.background_io_max_rate_mib_per_sec, 512)
        sharded = rocksdb_util.shared_storage(backend)
        baseline = rocksdb_util.shared_storage(default_backend)
        self.assertEqual(sharded.ddr, baseline.ddr)
        self.assertGreater(sharded.ssd, baseline.ssd)

    def test_backend_config_rejects_zero_sst_block_size(self):
        config = train_pb2.RocksDBConfig(root_dir="/tmp", sst_block_size_bytes=0)
        with self.assertRaisesRegex(ValueError, "SST block size"):
            rocksdb_util.backend_config(1, config=config)

    @parameterized.expand(
        [
            ("uneven", [3, 7], 42),
            ("borrowed_rank", [20, 0], 22),
            ("two_nodes", [20, 0, 1, 1], 22),
        ],
        name_func=parameterized_name_func,
    )
    def test_configure_backend_budgets(self, name, builtin_gib, import_quarters):
        try:
            from tzrec_ext.rocksdb.storage import RocksDBStorageFactory
        except ModuleNotFoundError as e:
            self.skipTest(f"tzrec_ext.rocksdb.storage needs {e.name}")

        gib = 1 << 30
        base_config = dataclasses.replace(
            rocksdb_util.backend_config(
                2,
                block_cache_bytes=2 * gib,
                config=train_pb2.RocksDBConfig(
                    root_dir="/tmp", num_db_shards_per_rank=2
                ),
            ),
            staging_buffer_bytes=gib // 2,
        )
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
                ("plain", [value * gib for value in builtin_gib]),
                ("sparse.a", [gib // 4, gib // 2] * (len(builtin_gib) // 2)),
                ("sparse.b", [3 * gib // 4, gib] * (len(builtin_gib) // 2)),
            )
        ]
        configured = rocksdb_util.configure_backend(
            selected,
            base_config=base_config,
            ddr_caps=(16 * gib,) * len(builtin_gib),
            local_world_size=2,
        )
        self.assertIsNone(configured[0].dynamicemb_options.external_storage)
        for option in configured[1:]:
            factory = option.dynamicemb_options.external_storage
            self.assertEqual(factory.table_namespace, option.fqn)
            self.assertEqual(
                factory.backend_config,
                dataclasses.replace(
                    base_config,
                    block_cache_bytes=7 * gib // 2,
                    import_budget_bytes=import_quarters * gib // 4,
                ),
            )


if __name__ == "__main__":
    unittest.main()
