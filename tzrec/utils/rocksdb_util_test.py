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

import dataclasses
import glob
import os
import re
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from tzrec.protos import train_pb2
from tzrec.utils import checkpoint_util, rocksdb_util
from tzrec.utils.logging_util import logger
from tzrec.utils.test_util import make_test_dir, mark_ci_scope

try:
    import tzrec_rocksdb  # noqa: F401
    from dynamicemb import (
        DynamicEmbInitializerArgs,
        DynamicEmbInitializerMode,
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
        EmbOptimType,
    )
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    has_rocksdb = torch.cuda.is_available()
except ImportError:
    has_rocksdb = False


def _backend_config():
    from tzrec_rocksdb import BackendConfig

    return BackendConfig(32 << 20, 32 << 20, 1 << 20, 64 << 20, io_threads=2)


class RocksDBRuntimeTest(unittest.TestCase):
    """Lifecycle of the per-process RocksDB runtime."""

    def setUp(self):
        try:
            import tzrec_rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_rocksdb is not installed")
        self.test_dir = make_test_dir()
        self.train_config = train_pb2.TrainConfig(
            rocksdb_config=train_pb2.RocksDBConfig(
                root_dir=os.path.join(self.test_dir, "db")
            )
        )

    def test_no_config_no_runtime(self):
        with rocksdb_util.open_runtime(train_pb2.TrainConfig(), "m", "eval") as rt:
            self.assertIsNone(rt)

    def test_cache_rows(self):
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.0), 0)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.001), 1024)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.01), 10240)

    def test_bind_opens_one_session_per_rank(self):
        from tzrec_rocksdb.storage import RocksDBStorageFactory

        config = _backend_config()
        options = [
            SimpleNamespace(
                external_storage=RocksDBStorageFactory(config, f"m.{n}", 0.1)
            )
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
        with rocksdb_util.open_runtime(self.train_config, "model", "eval") as rt:
            rt.bind(plan)
            self.assertEqual(rt.session.config, config)
            self.assertTrue(
                rt.session.work_dir.startswith(os.path.realpath(rt.rank_dir))
            )
            for o in options:
                self.assertIs(o.external_storage.session, rt.session)
            rt.bind(plan)  # the same plan again keeps the session
            other = dataclasses.replace(config, block_cache_bytes=1)
            options[0].external_storage = RocksDBStorageFactory(other, "m.a", 0.1)
            with self.assertRaisesRegex(RuntimeError, "config"):
                rt.bind(plan)

    def test_close_releases_the_working_db_on_errors(self):
        from tzrec_rocksdb import RocksDBSession

        with self.assertRaisesRegex(ValueError, "boom"):
            with rocksdb_util.open_runtime(self.train_config, "model", "eval") as rt:
                rt.session = RocksDBSession(_backend_config(), rt._new_work_dir())
                self.assertTrue(rt.session.namespace_ids(["t"]))
                work = rt.session.work_dir
                self.assertTrue(os.path.isdir(work))
                raise ValueError("boom")
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

    def test_incomplete_checkpoints_are_never_selected(self):
        model_dir = make_test_dir()
        for step, done in ((10, True), (20, True), (30, False)):
            ckpt = os.path.join(model_dir, f"model.ckpt-{step}")
            os.makedirs(os.path.join(ckpt, "model"))
            if done:
                open(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w")
        self.assertTrue(
            checkpoint_util.latest_checkpoint(model_dir)[0].endswith("model.ckpt-30")
        )
        _, step = checkpoint_util.latest_checkpoint(model_dir, require_success=True)
        self.assertEqual(step, 20)


class _Model(nn.Module):
    def __init__(self, emb: nn.Module) -> None:
        super().__init__()
        self.emb = emb


@unittest.skipUnless(has_rocksdb, "tzrec_rocksdb and CUDA are required")
@mark_ci_scope("gpu")
class RocksDBCheckpointTest(unittest.TestCase):
    """Logical save/restore of a fused module."""

    def setUp(self):
        from tzrec_rocksdb import RocksDBSession

        self.test_dir = make_test_dir()
        self.model_dir = os.path.join(self.test_dir, "model")
        self.env = mock.patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
        self.env.start()
        self.rt = rocksdb_util.RocksDBRuntime(
            train_pb2.RocksDBConfig(root_dir=os.path.join(self.test_dir, "db")),
            self.model_dir,
            "train",
        )
        self.rt.session = RocksDBSession(_backend_config(), self.rt._new_work_dir())

    def tearDown(self):
        self.rt.close()
        self.env.stop()

    def _model(self, names, strategy="NO_EVICTION", capacity=1 << 16, hbm=1 << 18):
        from tzrec_rocksdb.storage import RocksDBStorageFactory

        session = self.rt.session
        options = [
            DynamicEmbTableOptions(
                dim=16,
                max_capacity=capacity,
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                device_id=0,
                caching=True,
                local_hbm_for_values=hbm,
                score_strategy=DynamicEmbScoreStrategy[strategy],
                initializer_args=DynamicEmbInitializerArgs(
                    mode=DynamicEmbInitializerMode.DEBUG
                ),
                external_storage=RocksDBStorageFactory(
                    session.config, f"m.{n}", session=session
                ),
            )
            for n in names
        ]
        emb = BatchedDynamicEmbeddingTablesV2(
            table_options=options,
            table_names=list(names),
            feature_table_map=list(range(len(names))),
            pooling_mode=DynamicEmbPoolingMode.SUM,
            optimizer=EmbOptimType.ADAM,
            learning_rate=0.01,
        )
        return _Model(emb)

    def _train(self, model, seed, steps=5, batch=64):
        g = torch.Generator().manual_seed(seed)
        n = len(model.emb.table_names)
        for step in range(1, steps + 1):
            keys = torch.randint(0, 20000, (batch * n,), generator=g).cuda()
            offsets = torch.arange(0, batch * n + 1, 1).cuda()
            model.emb(keys, offsets).sum().backward()
            torch.cuda.synchronize()
            self.assertIsNone(self.rt.step_boundary(step))

    def _rows(self, model):
        with self.rt.writeback_scope("checkpoint"):
            model.emb.flush()
        out = {}
        storage = model.emb.tables
        for t, ns in enumerate(storage.namespaces):
            for ids, _, emb, opt in storage.iter_table(t):
                for i, k in enumerate(ids.tolist()):
                    out[(ns, k)] = torch.cat([emb[i], opt[i]])
        return out

    def _logical_save(self, model, step):
        """What save_model does for one module: DynamicEmbDump's module.dump."""
        ckpt = os.path.join(self.model_dir, f"model.ckpt-{step}", "dynamicemb", "m")
        os.makedirs(ckpt)
        with self.rt.writeback_scope("checkpoint"):
            model.emb.dump(ckpt, optim=True, counter=True)
        return ckpt

    def test_logical_restore_rebuilds_a_fresh_db(self):
        model = self._model(["a", "b"], strategy="LFU", capacity=3000)
        self.rt.attach(model)
        self._train(model, 0)
        before = self._rows(model)
        opt_args = model.emb.optimizer.get_opt_args()
        old_work = self.rt.session.work_dir
        ckpt = self._logical_save(model, 5)
        self._train(model, 1)  # state after the checkpoint must be discarded
        self.assertNotEqual(self._rows(model), before)
        self.rt.reset_for_import(model)
        self.assertFalse(os.path.exists(old_work))
        model.emb.load(ckpt, optim=True, counter=True)
        after = self._rows(model)
        self.assertEqual(sorted(after), sorted(before))
        for k in before:
            torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
        self.assertEqual(model.emb.optimizer.get_opt_args(), opt_args)
        storage = model.emb.tables
        for t in range(2):
            self.assertEqual(
                storage.logical_rows(t, None),
                len([k for k in after if k[0] == storage.namespaces[t]]),
            )
        db = self.rt.session.db
        self.assertEqual(db.get_int_property("rocksdb.num-snapshots"), 0)

    def test_save_stats_logs_each_save_alone(self):
        model = self._model(["a"])
        self.rt.attach(model)
        exported = []
        for step in (5, 10):
            self._train(model, step)
            ckpt = self._logical_save(model, step)
            with self.assertLogs(logger, "INFO") as cm:
                self.rt.save_stats(1.0)
            line = [x for x in cm.output if "RocksDB checkpoint" in x][0]
            exported.append(int(re.search(r"logical export \S+ \((\d+) rows", line)[1]))
            keys = glob.glob(os.path.join(ckpt, "a_emb_keys*"))[0]
            self.assertEqual(exported[-1], os.path.getsize(keys) // 8)
            # The save's cache write-back is read from the unflushed memtables.
            held = re.search(r"read from (\S+)MiB unflushed memtables", line)[1]
            self.assertGreater(float(held), 0)
        self.assertGreater(exported[1], exported[0])

    def test_step_boundary_evicts_to_capacity(self):
        # ~1024 cached rows, capacity 4000, ~8000 distinct ids.
        model = self._model(["a"], strategy="STEP", capacity=4000, hbm=1 << 16)
        self.rt.attach(model)
        self._train(model, 0, steps=20, batch=512)
        storage = model.emb.tables
        self.assertLessEqual(storage.logical_rows(0, model.emb.cache), 4000)
        self.assertGreater(self.rt.session.metrics.counters.get("evict_rows", 0), 0)


if __name__ == "__main__":
    unittest.main()
