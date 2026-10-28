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

import json
import os
import unittest

import torch
from parameterized import parameterized

from tzrec.protos import pipeline_pb2
from tzrec.utils import config_util, rocksdb_util
from tzrec.utils.test_util import make_test_dir, parameterized_name_func

try:
    import tzrec_rocksdb  # noqa: F401
    from dynamicemb import (
        DynamicEmbPoolingMode,
        DynamicEmbScoreStrategy,
        DynamicEmbTableOptions,
        EmbOptimType,
    )
    from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2

    from tzrec.tools.dynamicemb.import_rocksdb_checkpoint import import_checkpoint

    has_rocksdb = torch.cuda.is_available()
except ImportError:
    has_rocksdb = False


@unittest.skipIf(not has_rocksdb, "tzrec_rocksdb and CUDA are required")
class ImportRocksDBCheckpointTest(unittest.TestCase):
    """Legacy dynamicemb dump -> RocksDB native checkpoint."""

    def setUp(self):
        self.test_dir = make_test_dir()

    def tearDown(self):
        rocksdb_util.set_rocksdb_config(None)

    @parameterized.expand(
        [
            ("adam", "ADAM", True),
            ("rowwise_adagrad", "EXACT_ROWWISE_ADAGRAD", True),
            ("no_optim", "ADAM", False),
        ],
        name_func=parameterized_name_func,
    )
    def test_import(self, _, opt_name, include_optim):
        opt_type = getattr(EmbOptimType, opt_name)
        dims = [16, 8]
        options = [
            DynamicEmbTableOptions(
                dim=d,
                max_capacity=1 << 16,
                index_type=torch.int64,
                embedding_dtype=torch.float32,
                device_id=0,
                score_strategy=DynamicEmbScoreStrategy.TIMESTAMP,
            )
            for d in dims
        ]
        module = BatchedDynamicEmbeddingTablesV2(
            table_options=options,
            table_names=["t0", "t1"],
            feature_table_map=[0, 1],
            pooling_mode=DynamicEmbPoolingMode.SUM,
            optimizer=opt_type,
            learning_rate=0.1,
        )
        g = torch.Generator().manual_seed(0)
        for _ in range(5):
            keys = torch.randint(0, 5000, (512,), generator=g).cuda()
            offsets = torch.arange(0, 513, 2).cuda()
            module(keys, offsets).sum().backward()
        ckpt = os.path.join(self.test_dir, "src", "model.ckpt-7")
        collection = os.path.join(ckpt, "dynamicemb", "model.model.ebc")
        os.makedirs(collection)
        module.dump(collection, optim=True)
        os.makedirs(os.path.join(ckpt, "model"))
        with open(os.path.join(ckpt, "meta"), "w") as f:
            json.dump({"step": 7}, f)

        config = pipeline_pb2.EasyRecConfig()
        config.train_config.rocksdb_config.local_path = os.path.join(
            self.test_dir, "db"
        )
        config.train_config.sparse_optimizer.sgd_optimizer.lr = 0.1
        config.train_config.dense_optimizer.adam_optimizer.lr = 0.1
        config_path = os.path.join(self.test_dir, "pipeline.config")
        config_util.save_message(config, config_path)
        out_model_dir = os.path.join(self.test_dir, "out")
        out = import_checkpoint(
            config_path, ckpt, out_model_dir, include_optim=include_optim
        )

        self.assertTrue(os.path.exists(os.path.join(out, "_SUCCESS")))
        self.assertTrue(os.path.isdir(os.path.join(out, "model")))
        with open(os.path.join(out, "rocksdb", "rank_0.json")) as f:
            manifest = json.load(f)
        self.assertTrue(manifest["imported"])
        self.assertEqual(manifest["exact_resume"], include_optim)
        from tzrec_rocksdb.checkpoint import iter_checkpoint_table, open_read_only

        db, native = open_read_only(manifest["native_checkpoint"])
        for t, name in enumerate(["t0", "t1"]):
            expect = {}
            for k, e, o, _ in module.tables.export_keys_values(
                torch.device("cpu"), 4096, t
            ):
                for i, key in enumerate(k.tolist()):
                    expect[key] = (e[i], o[i] if o is not None else None)
            got = {}
            for ids, _, emb, opt in iter_checkpoint_table(
                db, native, f"model.ebc.{name}", 1000
            ):
                for i, key in enumerate(ids.view("int64").tolist()):
                    got[key] = (emb[i], opt[i])
            self.assertEqual(sorted(got), sorted(expect))
            reset = torch.zeros_like(next(iter(got.values()))[1])
            module.optimizer.reset_optimizer_states(reset.view(1, -1), emb_dims=dims[t])
            for key, (e, o) in expect.items():
                torch.testing.assert_close(got[key][0], e, rtol=0, atol=0)
                want = o if include_optim else reset
                torch.testing.assert_close(got[key][1], want, rtol=0, atol=0)
        db.close()


if __name__ == "__main__":
    unittest.main()
