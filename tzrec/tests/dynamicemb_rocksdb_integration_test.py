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

import glob
import json
import os
import shutil
import unittest

import numpy as np
import torch

from tzrec.tests import utils
from tzrec.utils import checkpoint_util, config_util, dynamicemb_util, misc_util
from tzrec.utils.test_util import make_test_dir, mark_ci_scope

try:
    import tzrec_rocksdb  # noqa: F401

    has_rocksdb = True
except ImportError:
    has_rocksdb = False


def _logical_rows(ckpt, collection="*"):
    """Rows per dynamicemb table in a checkpoint's logical files."""
    rows = {}
    pattern = os.path.join(ckpt, "dynamicemb", f"*{collection}", "*_emb_keys.rank_*")
    for path in glob.glob(pattern):
        table = os.path.basename(path).split("_emb_keys.")[0]
        rows[table] = rows.get(table, 0) + os.path.getsize(path) // 8
    return rows


# Mock ids of a dynamicemb feature are drawn below its max_capacity; training
# id_3 with a smaller capacity than the data was generated for forces eviction.
# The larger GPU cache keeps one batch's id_3 ids within its cache share, so
# none sit in the cache's overflow region (which cannot be evicted).
_SMALL_ID3_CAPACITY = {
    "feature_configs[2].id_feature.dynamicemb.max_capacity": 2000,
    "train_config.rocksdb_config.gpu_cache_value_bytes_per_rank": 157286400,
}


def _logs(log_dir):
    text = ""
    for path in glob.glob(os.path.join(log_dir, "**", "*.log"), recursive=True):
        with open(path) as f:
            text += f.read()
    return text


@unittest.skipIf(
    not torch.cuda.is_available()
    or not dynamicemb_util.has_dynamicemb
    or not has_rocksdb,
    "CUDA, dynamicemb and tzrec_rocksdb are required.",
)
@mark_ci_scope("gpu")
class DynamicEmbRocksDBIntegrationTest(unittest.TestCase):
    """Training, recovery, relocation and export of RocksDB dynamicemb tables."""

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir()

    def tearDown(self):
        if self.success:
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _resume(self, config_path, name, edit=None):
        edit = edit or {"train_config.num_epochs": 2}
        cmd = (
            f"PYTHONPATH=. torchrun {utils._standalone()} --nnodes=1 "
            f"--nproc-per-node={utils._get_nproc_per_node()} "
            f"--log_dir {os.path.join(self.test_dir, name)} -r 3 -t 3 "
            f"tzrec/train_eval.py --pipeline_config_path {config_path} "
            f"--continue_train --edit_config_json '{json.dumps(edit)}'"
        )
        ok = misc_util.run_cmd(cmd, os.path.join(self.test_dir, f"{name}.txt"))
        return ok, _logs(os.path.join(self.test_dir, name))

    def test_train_eval_export(self):
        # AUTO placement with one rocksdb {} table and one table left in memory.
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_fg_dynamicemb_rocksdb_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"), self.test_dir
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                self.test_dir,
                env_str="USE_DISTRIBUTED_EMBEDDING=1",
            )
        self.assertTrue(self.success)
        self.success = False
        ckpt, _ = checkpoint_util.latest_checkpoint(
            os.path.join(self.test_dir, "train"), require_success=True
        )
        assert ckpt is not None
        # RocksDB tables are saved in DynamicEmb's logical files like the
        # in-memory ones; no native checkpoint manifests any more.
        self.assertFalse(os.path.exists(os.path.join(ckpt, "rocksdb")))
        rows = _logical_rows(ckpt)
        self.assertGreater(rows.get("id_8_emb", 0), 0)
        self.assertIn("user_id_emb", rows)
        log = _logs(os.path.join(self.test_dir, "log_train_eval"))
        self.assertRegex(log, r"id_8_emb: rocksdb: .*rocksdb \{\} \(explicit\)")
        self.assertRegex(log, r"user_id_emb: built-in: .*fits GPU/host memory")
        sparse_dir = os.path.join(self.test_dir, "export", "sparse")
        with open(os.path.join(sparse_dir, "sparse_embedding.json")) as f:
            metadata = json.load(f)
        dynamic = [info for info in metadata.values() if info["is_dynamic"]]
        self.assertEqual(sum(info["shape"][0] for info in dynamic), sum(rows.values()))
        with np.load(
            os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz")
        ) as arrays:
            for info in dynamic:
                values = arrays[info["value_name"]]
                self.assertEqual(values.shape, tuple(info["shape"]))
                self.assertTrue(np.isfinite(values).all())
        self.success = True

    def test_resume_only_from_checkpoint_files(self):
        edit = json.dumps(
            {"train_config.save_checkpoints_steps": 10, **_SMALL_ID3_CAPACITY}
        )
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_dynamicemb_rocksdb_mock.config",
            self.test_dir,
            args_str=f"--edit_config_json '{edit}'",
        )
        self.assertTrue(self.success)
        self.success = False
        model_dir = os.path.join(self.test_dir, "train")
        ckpt, step = checkpoint_util.latest_checkpoint(model_dir, require_success=True)
        assert ckpt is not None
        # The STEP table id_3 is evicted to its capacity (2000 over 2 shards).
        rows = _logical_rows(ckpt, ".ebc")
        self.assertLessEqual(rows["id_3_emb"], 2000)
        log = _logs(os.path.join(self.test_dir, "log_train_eval"))
        self.assertRegex(log, r"evicted [1-9]\d* rows")
        # A save that died before _SUCCESS, and no local RocksDB state at all.
        crashed = os.path.join(model_dir, f"model.ckpt-{step + 5}")
        shutil.copytree(ckpt, crashed)
        os.remove(os.path.join(crashed, checkpoint_util.CKPT_SUCCESS_FILENAME))
        shutil.rmtree("tmp/tzrec_rocksdb_test", ignore_errors=True)
        ok, log = self._resume(
            os.path.join(self.test_dir, "pipeline.config"),
            "log_resume",
            {"train_config.num_epochs": 2, "train_config.save_checkpoints_steps": 10},
        )
        self.assertTrue(ok)
        self.assertIn("deleting incomplete checkpoint", log)
        self.assertIn("loading RocksDB tables into fresh working DB", log)
        if os.path.exists(crashed):
            self.assertTrue(
                os.path.exists(
                    os.path.join(crashed, checkpoint_util.CKPT_SUCCESS_FILENAME)
                )
            )
        _, final_step = checkpoint_util.latest_checkpoint(
            model_dir, require_success=True
        )
        self.assertGreater(final_step, step)
        self.success = True

    def test_delta_publication_tombstones(self):
        dump = "train_config.delta_embedding_dump_config"
        edit = json.dumps(
            {
                f"{dump}.paired_dump_interval_steps": 3,
                f"{dump}.dump_evicted_tombstones": True,
                **_SMALL_ID3_CAPACITY,
            }
        )
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_dynamicemb_rocksdb_mock.config",
            self.test_dir,
            args_str=f"--edit_config_json '{edit}'",
        )
        self.assertTrue(self.success)
        self.success = False
        import pyarrow.parquet as pq

        files = glob.glob(
            os.path.join(
                self.test_dir, "train", "delta_embedding_dump", "**", "*.parquet"
            ),
            recursive=True,
        )
        self.assertTrue(files)
        table = pq.read_table(files).to_pandas()
        tomb = table[table["source"] == "dynamicemb_evicted"]
        # The STEP table id_3 at capacity 2000 logically evicts; deletions publish.
        self.assertTrue(tomb["table_fqn"].str.contains("ebc.embedding_bags.id_3").any())
        self.assertTrue(all(not any(e) for e in tomb["embedding"]))
        # The LFU table never reaches its capacity: GPU-cache replacement only
        # writes rows back and publishes no deletion.
        self.assertFalse(tomb["table_fqn"].str.contains("id_2").any())
        self.success = True

    def test_relocate_tables_between_backends(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_dynamicemb_rocksdb_mock.config",
            self.test_dir,
        )
        self.assertTrue(self.success)
        self.success = False
        config_path = os.path.join(self.test_dir, "pipeline.config")
        config = config_util.load_pipeline_config(config_path)
        # Resume with every table in GPU/host memory (no RocksDB at all).
        builtin = os.path.join(self.test_dir, "builtin.config")
        config.train_config.ClearField("rocksdb_config")
        for fc in config.feature_configs:
            for f in (
                [fc.id_feature]
                if fc.HasField("id_feature")
                else list(fc.sequence_feature.features)
                if fc.HasField("sequence_feature")
                else []
            ):
                inner = f.id_feature if hasattr(f, "id_feature") else f
                if inner.HasField("dynamicemb"):
                    inner.dynamicemb.ClearField("rocksdb")
        config_util.save_message(config, builtin)
        ok, log = self._resume(builtin, "log_builtin")
        self.assertTrue(ok)
        self.assertNotIn("RocksDB dynamicemb groups", log)
        # And back into RocksDB from the in-memory run's checkpoint.
        ok, log = self._resume(
            config_path, "log_rocksdb", {"train_config.num_epochs": 3}
        )
        self.assertTrue(ok)
        self.assertIn("loading RocksDB tables into fresh working DB", log)
        self.success = True


if __name__ == "__main__":
    unittest.main()
