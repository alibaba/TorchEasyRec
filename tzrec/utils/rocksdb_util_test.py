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

import hashlib
import json
import os
import unittest
from unittest import mock

import torch
from torch import nn

from tzrec.protos import train_pb2
from tzrec.utils import checkpoint_util, rocksdb_util
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


def _tree_hash(path):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as fh:
                h.update(f.encode() + fh.read())
    return h.hexdigest()


class RocksDBBudgetTest(unittest.TestCase):
    """Rank-wide budget split and checkpoint selection helpers."""

    def tearDown(self):
        rocksdb_util.set_rocksdb_config(None)

    def test_cache_ratio(self):
        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(local_path="x", gpu_cache_value_bytes_per_rank=300)
        )
        self.assertAlmostEqual(rocksdb_util.cache_ratio([600, 400]), 0.3)
        with self.assertRaises(ValueError):
            rocksdb_util.cache_ratio([100, 200])
        rocksdb_util.set_rocksdb_config(train_pb2.RocksDBConfig(local_path="x"))
        with self.assertRaisesRegex(ValueError, "GPU cache"):
            rocksdb_util.cache_ratio([600])

    def test_cache_rows_and_host_reserve(self):
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.0), 0)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.001), 1024)
        self.assertEqual(rocksdb_util.cache_rows(10**6, 0.01), 10240)
        self.assertEqual(rocksdb_util.host_reserve_bytes(), 0)
        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(
                local_path="x",
                block_cache_bytes_per_rank=1 << 30,
                write_buffer_bytes_per_rank=1 << 29,
                staging_buffer_bytes_per_rank=1 << 28,
            )
        )
        self.assertEqual(
            rocksdb_util.host_reserve_bytes(),
            (1 << 30) + (1 << 29) + (1 << 28) + rocksdb_util._HOST_OVERHEAD_BYTES,
        )

    def test_incomplete_checkpoints(self):
        model_dir = make_test_dir()
        for step, done in ((10, True), (20, True), (30, False)):
            ckpt = os.path.join(model_dir, f"model.ckpt-{step}")
            os.makedirs(os.path.join(ckpt, "model"))
            if done:
                open(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w")
        self.assertTrue(
            checkpoint_util.latest_checkpoint(model_dir)[0].endswith("model.ckpt-30")
        )
        path, step = checkpoint_util.latest_checkpoint(model_dir, require_success=True)
        self.assertEqual(step, 20)
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            rocksdb_util.remove_incomplete_checkpoints(model_dir, step)
        self.assertEqual(
            sorted(os.listdir(model_dir)), ["model.ckpt-10", "model.ckpt-20"]
        )
        self.assertEqual(checkpoint_util.latest_checkpoint(model_dir)[1], 20)


class _Model(nn.Module):
    def __init__(self, emb: nn.Module) -> None:
        super().__init__()
        self.emb = emb


@unittest.skipUnless(has_rocksdb, "tzrec_rocksdb and CUDA are required")
@mark_ci_scope("gpu")
class RocksDBCheckpointTest(unittest.TestCase):
    """Logical save/restore and legacy native restore of a fused module."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.model_dir = os.path.join(self.test_dir, "model")
        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(
                local_path=os.path.join(self.test_dir, "db"),
                gpu_cache_value_bytes_per_rank=1 << 18,
                disk_budget_bytes=1 << 40,
                disk_reserve_bytes=0,
                block_cache_bytes_per_rank=32 << 20,
                write_buffer_bytes_per_rank=32 << 20,
                staging_buffer_bytes_per_rank=1 << 20,
                io_threads=2,
            )
        )
        self.env = mock.patch.dict(os.environ, {"RANK": "0", "WORLD_SIZE": "1"})
        self.env.start()
        rocksdb_util.open_session(self.model_dir)

    def tearDown(self):
        rocksdb_util.close_session()
        rocksdb_util.set_rocksdb_config(None)
        self.env.stop()

    def _model(self, names, strategy="NO_EVICTION", capacity=1 << 16, hbm=1 << 18):
        from tzrec_rocksdb.storage import RocksDBStorageFactory

        factory = rocksdb_util.backend_config()
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
                external_storage=RocksDBStorageFactory(factory, f"m.{n}"),
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
            self.assertIsNone(rocksdb_util.step_boundary(step))

    def _rows(self, model):
        with rocksdb_util.writeback_scope("checkpoint"):
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
        with rocksdb_util.writeback_scope("checkpoint"):
            model.emb.dump(ckpt, optim=True, counter=True)
        return ckpt

    def test_logical_restore_rebuilds_a_fresh_db(self):
        model = self._model(["a", "b"], strategy="LFU", capacity=3000)
        rocksdb_util.validate_model(model)
        self._train(model, 0)
        before = self._rows(model)
        opt_args = model.emb.optimizer.get_opt_args()
        old_work = rocksdb_util.session().work_dir
        ckpt = self._logical_save(model, 5)
        self._train(model, 1)  # state after the checkpoint must be discarded
        self.assertNotEqual(self._rows(model), before)
        rocksdb_util.reset_for_import(model)
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
        db = rocksdb_util.session().db
        self.assertEqual(db.get_int_property("rocksdb.num-snapshots"), 0)

    def test_step_boundary_evicts_to_capacity(self):
        # ~1024 cached rows, capacity 4000, ~8000 distinct ids.
        model = self._model(["a"], strategy="STEP", capacity=4000, hbm=1 << 16)
        rocksdb_util.validate_model(model)
        self._train(model, 0, steps=20, batch=512)
        storage = model.emb.tables
        self.assertLessEqual(storage.logical_rows(0, model.emb.cache), 4000)
        self.assertGreater(
            rocksdb_util.session().metrics.counters.get("evict_rows", 0), 0
        )

    def test_fused_opt_args_must_agree(self):
        a, b = self._model(["a"]), self._model(["b"])
        self._train(a, 0, steps=3)
        self._train(b, 1, steps=5)
        root = os.path.join(self.test_dir, "relocated")
        for m in (a, b):
            path = os.path.join(root, "m")
            os.makedirs(path, exist_ok=True)
            with rocksdb_util.writeback_scope("checkpoint"):
                m.emb.dump(path, optim=True)
        fused = self._model(["a", "b"])
        with (
            mock.patch(
                "dynamicemb.dump_load.find_sharded_modules",
                return_value=[("m", None, fused)],
            ),
            mock.patch(
                "dynamicemb.dump_load.get_dynamic_emb_module",
                return_value=[fused.emb],
            ),
        ):
            with self.assertRaisesRegex(ValueError, "different optimizer"):
                rocksdb_util.check_fused_opt_args(root, fused)

    def _legacy_save(self, model, step):
        """A checkpoint in the native format written before logical saves."""
        from tzrec_rocksdb.checkpoint import create_checkpoint

        with rocksdb_util.writeback_scope("checkpoint"):
            model.emb.flush()
        ckpt = os.path.join(self.model_dir, f"model.ckpt-{step}")
        os.makedirs(os.path.join(ckpt, "rocksdb"))
        native = os.path.join(rocksdb_util.rank_dir(self.model_dir, 0), f"ckpt-{step}")
        create_checkpoint(rocksdb_util.session(), native, {"step": step})
        storage = model.emb.tables
        layouts = {
            ns: storage.table_layout(t) for t, ns in enumerate(storage.namespaces)
        }
        manifest = {
            "format_version": rocksdb_util.FORMAT_VERSION,
            "step": step,
            "rank": 0,
            "world_size": 1,
            "native_checkpoint": native,
            "groups": [
                {
                    "tables": list(model.emb.table_names),
                    "namespaces": list(storage.namespaces),
                    "layouts": layouts,
                    "dist_types": [o.dist_type for o in model.emb._dynamicemb_options],
                    "num_shards": 1,
                    "shard_index": 0,
                    "opt_args": model.emb.optimizer.get_opt_args(),
                    "scores": dict(model.emb._scores),
                }
            ],
        }
        with open(os.path.join(ckpt, "rocksdb", "rank_0.json"), "w") as f:
            json.dump(manifest, f)
        open(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w")
        return ckpt, native

    def test_legacy_native_checkpoint_restores_and_stays_immutable(self):
        model = self._model(["a", "b"])
        self._train(model, 0)
        before = self._rows(model)
        ckpt, native = self._legacy_save(model, 5)
        native_hash = _tree_hash(native)
        self._train(model, 1)
        reordered = self._model(["b", "a"])
        rocksdb_util.restore_checkpoint(ckpt, reordered, load_optim=True)
        after = self._rows(reordered)
        self.assertEqual(sorted(after), sorted(before))
        for k in before:
            torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
        self._train(reordered, 2)
        self.assertEqual(_tree_hash(native), native_hash)
        os.rename(
            os.path.join(self.model_dir, "model.ckpt-5"),
            os.path.join(self.model_dir, "pruned"),
        )
        rocksdb_util.prune_native_checkpoints()
        self.assertFalse(os.path.exists(native))

    def test_legacy_native_checkpoint_rejects_evicting_tables(self):
        model = self._model(["a"])
        self._train(model, 0)
        ckpt, _ = self._legacy_save(model, 5)
        with self.assertRaisesRegex(ValueError, "eviction index"):
            rocksdb_util.restore_checkpoint(ckpt, self._model(["a"], "STEP"), True)
        os.remove(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            rocksdb_util.restore_checkpoint(ckpt, model, True)


if __name__ == "__main__":
    unittest.main()
