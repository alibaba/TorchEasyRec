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
import hashlib
import json
import os
import shutil
import unittest

import numpy as np
import torch

from tzrec.tests import utils
from tzrec.utils import checkpoint_util, dynamicemb_util, misc_util
from tzrec.utils.test_util import make_test_dir, mark_ci_scope

try:
    import tzrec_rocksdb  # noqa: F401

    has_rocksdb = True
except ImportError:
    has_rocksdb = False


def _tree_hash(path):
    h = hashlib.sha256()
    for root, _, files in sorted(os.walk(path)):
        for f in sorted(files):
            with open(os.path.join(root, f), "rb") as fh:
                h.update(f.encode() + fh.read())
    return h.hexdigest()


@unittest.skipIf(
    not torch.cuda.is_available()
    or not dynamicemb_util.has_dynamicemb
    or not has_rocksdb,
    "CUDA, dynamicemb and tzrec_rocksdb are required.",
)
@mark_ci_scope("gpu")
class DynamicEmbRocksDBIntegrationTest(unittest.TestCase):
    """Training, recovery and export of RocksDB-backed dynamicemb tables."""

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir()

    def tearDown(self):
        if self.success:
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def _native_dirs(self, ckpt):
        dirs = []
        for path in sorted(glob.glob(os.path.join(ckpt, "rocksdb", "rank_*.json"))):
            with open(path) as f:
                dirs.append(json.load(f)["native_checkpoint"])
        return dirs

    def test_train_eval_export(self):
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
        # RocksDB tables are checkpointed natively, never fully dumped.
        self.assertFalse(
            glob.glob(os.path.join(ckpt, "dynamicemb", "*", "*_emb_keys.rank_*"))
        )
        native_rows = 0
        from tzrec_rocksdb.checkpoint import open_read_only

        for native in self._native_dirs(ckpt):
            db, manifest = open_read_only(native)
            native_rows += sum(db.count(ns) for ns in manifest["namespaces"].values())
            db.close()
        self.assertGreater(native_rows, 0)
        sparse_dir = os.path.join(self.test_dir, "export", "sparse")
        with open(os.path.join(sparse_dir, "sparse_embedding.json")) as f:
            metadata = json.load(f)
        dynamic = [info for info in metadata.values() if info["is_dynamic"]]
        self.assertEqual(sum(info["shape"][0] for info in dynamic), native_rows)
        with np.load(
            os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz")
        ) as arrays:
            for info in dynamic:
                values = arrays[info["value_name"]]
                self.assertEqual(values.shape, tuple(info["shape"]))
                self.assertTrue(np.isfinite(values).all())
        self.success = True

    def test_crash_left_checkpoint_is_not_resumed(self):
        edit = '{"train_config.save_checkpoints_steps":10}'
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
        native_hashes = [_tree_hash(d) for d in self._native_dirs(ckpt)]
        # A save that died after its native checkpoint but before _SUCCESS.
        crashed = os.path.join(model_dir, f"model.ckpt-{step + 5}")
        shutil.copytree(ckpt, crashed)
        os.remove(os.path.join(crashed, checkpoint_util.CKPT_SUCCESS_FILENAME))

        config_path = os.path.join(self.test_dir, "pipeline.config")
        cmd = (
            f"PYTHONPATH=. torchrun {utils._standalone()} --nnodes=1 "
            f"--nproc-per-node={utils._get_nproc_per_node()} "
            f"--log_dir {os.path.join(self.test_dir, 'log_resume')} -r 3 -t 3 "
            f"tzrec/train_eval.py --pipeline_config_path {config_path} "
            "--continue_train --edit_config_json "
            '\'{"train_config.num_epochs":2,"train_config.save_checkpoints_steps":10}\''
        )
        self.assertTrue(
            misc_util.run_cmd(cmd, os.path.join(self.test_dir, "log_resume.txt"))
        )
        log = ""
        for path in glob.glob(
            os.path.join(self.test_dir, "log_resume", "**", "*.log"), recursive=True
        ):
            with open(path) as f:
                log += f.read()
        self.assertIn("restored RocksDB tables from", log)
        self.assertTrue(
            glob.glob(os.path.join(model_dir, f"incomplete.model.ckpt-{step + 5}.*"))
        )
        # The resumed run may legitimately save this step again, completely.
        if os.path.exists(crashed):
            self.assertTrue(
                os.path.exists(
                    os.path.join(crashed, checkpoint_util.CKPT_SUCCESS_FILENAME)
                )
            )
        # The resumed checkpoint is immutable under continued training.
        self.assertEqual(
            [_tree_hash(d) for d in self._native_dirs(ckpt)], native_hashes
        )
        final, final_step = checkpoint_util.latest_checkpoint(
            model_dir, require_success=True
        )
        self.assertGreater(final_step, step)
        for native in self._native_dirs(final):
            from tzrec_rocksdb.checkpoint import read_manifest

            read_manifest(native)
        self.success = True


if __name__ == "__main__":
    unittest.main()
