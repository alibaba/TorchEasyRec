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
        self.assertEqual(rocksdb_util.cache_ratio([600]), 0.0)

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
            rocksdb_util.quarantine_incomplete_checkpoints(model_dir, step)
        self.assertFalse(os.path.exists(os.path.join(model_dir, "model.ckpt-30")))
        names = os.listdir(model_dir)
        self.assertTrue(any(n.startswith("incomplete.model.ckpt-30.") for n in names))
        self.assertEqual(checkpoint_util.latest_checkpoint(model_dir)[1], 20)


class _Model(nn.Module):
    def __init__(self, emb: nn.Module) -> None:
        super().__init__()
        self.emb = emb


@unittest.skipUnless(has_rocksdb, "tzrec_rocksdb and CUDA are required")
@mark_ci_scope("gpu")
class RocksDBCheckpointTest(unittest.TestCase):
    """Native checkpoint save/restore around a real fused DynamicEmb module."""

    def setUp(self):
        self.test_dir = make_test_dir()
        self.model_dir = os.path.join(self.test_dir, "model")
        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(
                local_path=os.path.join(self.test_dir, "db"),
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

    def _model(self, names, caching=False):
        factory = rocksdb_util.backend_config()
        from tzrec_rocksdb.storage import RocksDBStorageFactory

        options = [
            DynamicEmbTableOptions(
                dim=16,
                max_capacity=1 << 16,
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                device_id=0,
                caching=caching,
                local_hbm_for_values=(1 << 18) if caching else 0,
                score_strategy=DynamicEmbScoreStrategy.NO_EVICTION,
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

    def _train(self, model, seed, steps=5):
        g = torch.Generator().manual_seed(seed)
        n = len(model.emb.table_names)
        for _ in range(steps):
            keys = torch.randint(0, 20000, (64 * n,), generator=g).cuda()
            offsets = torch.arange(0, 64 * n + 1, 1).cuda()
            model.emb(keys, offsets).sum().backward()

    def _rows(self, model):
        model.emb.flush()
        out = {}
        storage = model.emb.tables
        for t, ns in enumerate(storage.namespaces):
            for ids, _, emb, opt in storage.iter_table(t):
                for i, k in enumerate(ids.tolist()):
                    out[(ns, k)] = torch.cat([emb[i], opt[i]])
        return out

    def _save(self, model, step):
        ckpt = os.path.join(self.model_dir, f"model.ckpt-{step}")
        os.makedirs(ckpt, exist_ok=True)
        rocksdb_util.save_checkpoint(ckpt, step, model)
        open(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w")
        return ckpt

    def test_restore_rolls_back_and_keeps_checkpoint_immutable(self):
        model = self._model(["a", "b"], caching=True)
        self._train(model, 0)
        before = self._rows(model)
        opt_args = model.emb.optimizer.get_opt_args()
        ckpt = self._save(model, 5)
        with open(os.path.join(ckpt, "rocksdb", "rank_0.json")) as f:
            native = json.load(f)["native_checkpoint"]
        native_hash = _tree_hash(native)
        self._train(model, 1)  # state after the checkpoint must be discarded
        self.assertNotEqual(self._rows(model), before)
        rocksdb_util.restore_checkpoint(ckpt, model, load_optim=True)
        after = self._rows(model)
        self.assertEqual(sorted(after), sorted(before))
        for k in before:
            torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)
        self.assertEqual(model.emb.optimizer.get_opt_args(), opt_args)
        self._train(model, 2)
        self._rows(model)
        self.assertEqual(_tree_hash(native), native_hash)

    def test_reordered_group_restores_by_namespace(self):
        model = self._model(["a", "b", "c"])
        self._train(model, 0)
        before = self._rows(model)
        ckpt = self._save(model, 5)
        reordered = self._model(["c", "a", "b"])
        rocksdb_util.restore_checkpoint(ckpt, reordered, load_optim=True)
        after = self._rows(reordered)
        self.assertEqual(sorted(after), sorted(before))
        for k in before:
            torch.testing.assert_close(after[k], before[k], rtol=0, atol=0)

    def test_restore_rejects_incompatible_checkpoints(self):
        model = self._model(["a", "b"])
        self._train(model, 0)
        ckpt = self._save(model, 5)
        with self.assertRaisesRegex(ValueError, "fusion"):
            rocksdb_util.restore_checkpoint(ckpt, self._model(["a"]), True)
        os.remove(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME))
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            rocksdb_util.restore_checkpoint(ckpt, model, True)
        open(os.path.join(ckpt, checkpoint_util.CKPT_SUCCESS_FILENAME), "w")
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(ValueError, "resharding"):
                rocksdb_util.restore_checkpoint(ckpt, model, True)

    def test_prune_follows_model_checkpoints(self):
        model = self._model(["a"])
        self._train(model, 0)
        self._save(model, 1)
        self._save(model, 2)
        rank_dir = rocksdb_util.rank_dir(self.model_dir, 0)
        self.assertTrue(os.path.isdir(os.path.join(rank_dir, "ckpt-1")))
        os.rename(
            os.path.join(self.model_dir, "model.ckpt-1"),
            os.path.join(self.model_dir, "pruned"),
        )
        rocksdb_util.prune_native_checkpoints()
        self.assertFalse(os.path.exists(os.path.join(rank_dir, "ckpt-1")))
        self.assertTrue(os.path.isdir(os.path.join(rank_dir, "ckpt-2")))


if __name__ == "__main__":
    unittest.main()
