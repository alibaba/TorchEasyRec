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
# id_3 with a smaller capacity than the data was generated for forces eviction;
# one batch still fits its GPU cache share.
_SMALL_ID3_CAPACITY = {"feature_configs[2].id_feature.dynamicemb.max_capacity": 4000}


def _dcp_state(path):
    """A DCP directory's tensors as a plain dict."""
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    out = path.rstrip("/") + ".pt"
    dcp_to_torch_save(path, out)
    state = torch.load(out, weights_only=False)
    os.remove(out)
    return state


def _dynamicemb_state(ckpt):
    """Per table file: rows ordered by key, scores, optimizer args and counters."""
    tables = {}
    for key_path in glob.glob(os.path.join(ckpt, "dynamicemb", "*", "*_emb_keys.*")):
        d = os.path.dirname(key_path)
        table, suffix = os.path.basename(key_path).split("_emb_keys")
        keys = np.fromfile(key_path, dtype=np.int64)
        order = np.argsort(keys, kind="stable")
        rows = {}
        for kind, dtype in (
            ("emb_values", np.uint8),
            ("emb_opt_values", np.uint8),
            ("emb_scores", np.int64),
        ):
            path = os.path.join(d, f"{table}_{kind}{suffix}")
            if os.path.exists(path):
                data = np.fromfile(path, dtype=dtype)
                rows[kind] = data.reshape(len(keys), -1)[order] if len(keys) else data
        counter_keys = os.path.join(d, f"{table}_counter_keys{suffix}")
        counters = None
        if os.path.exists(counter_keys):
            ck = np.fromfile(counter_keys, dtype=np.int64)
            freq = np.fromfile(
                os.path.join(d, f"{table}_counter_frequencies{suffix}"), np.uint8
            )
            freq = freq.reshape(len(ck), -1) if len(ck) else freq
            counters = {int(k): f.tobytes() for k, f in zip(ck, freq)}
        with open(os.path.join(d, f"{table}_opt_args.json")) as f:
            meta = json.load(f)
        tables[f"{os.path.basename(d)}/{table}{suffix}"] = {
            "keys": keys[order],
            "values": rows.get("emb_values"),
            "opt": rows.get("emb_opt_values"),
            "scores": rows.get("emb_scores"),
            "meta": meta,
            "counters": counters,
        }
    return tables


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
        self.assertNotIn("total_loss: nan", log)
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
        # The STEP table id_3 is evicted to its capacity (4000 over all shards).
        rows = _logical_rows(ckpt, ".ebc")
        self.assertLessEqual(rows["id_3_emb"], 4000)
        log = _logs(os.path.join(self.test_dir, "log_train_eval"))
        self.assertNotIn("total_loss: nan", log)
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
        self.assertNotIn("total_loss: nan", log)
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

    def _assert_same_state(self, got, ref, path):
        if isinstance(ref, torch.Tensor):
            torch.testing.assert_close(got, ref, rtol=0, atol=0, msg=path)
        elif isinstance(ref, dict):
            self.assertEqual(sorted(got), sorted(ref), path)
            for k in ref:
                self._assert_same_state(got[k], ref[k], f"{path}.{k}")
        elif isinstance(ref, (list, tuple)):
            self.assertEqual(len(got), len(ref), path)
            for i, (g, r) in enumerate(zip(got, ref)):
                self._assert_same_state(g, r, f"{path}[{i}]")
        else:
            self.assertEqual(got, ref, path)

    def test_save_restore_round_trip_and_input_position(self):
        # Fused GPU-cache groups with STEP/LFU/TIMESTAMP tables and admission
        # counters, saving every 3 steps.
        edit = {
            "train_config.num_epochs": 2,
            "train_config.save_checkpoints_steps": 3,
            **_SMALL_ID3_CAPACITY,
        }
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_dynamicemb_rocksdb_mock.config",
            self.test_dir,
            args_str=f"--edit_config_json '{json.dumps(edit)}'",
        )
        self.assertTrue(self.success)
        self.success = False
        model_dir = os.path.join(self.test_dir, "train")
        final, final_step = checkpoint_util.latest_checkpoint(
            model_dir, require_success=True
        )
        assert final is not None
        step = (final_step - 1) // 3 * 3
        saved = os.path.join(model_dir, f"model.ckpt-{step}")
        # Resume from step `step` only: no later checkpoint, no working DB.
        resume_dir = os.path.join(self.test_dir, "resume")
        os.makedirs(resume_dir)
        shutil.copytree(saved, os.path.join(resume_dir, os.path.basename(saved)))
        config = config_util.load_pipeline_config(
            os.path.join(self.test_dir, "pipeline.config")
        )
        config.model_dir = resume_dir
        config_path = os.path.join(self.test_dir, "resume.config")
        config_util.save_message(config, config_path)
        shutil.rmtree("tmp/tzrec_rocksdb_test", ignore_errors=True)

        # Restore into a fresh DB, then save again before any training step.
        resave_dir = os.path.join(self.test_dir, "resave")
        cmd = (
            f"PYTHONPATH=. RESAVE_MODEL_DIR={resave_dir} torchrun "
            f"{utils._standalone()} --nnodes=1 "
            f"--nproc-per-node={utils._get_nproc_per_node()} "
            f"--log_dir {os.path.join(self.test_dir, 'log_resave')} -r 3 -t 3 "
            f"tzrec/tests/resave_after_restore.py --pipeline_config_path "
            f"{config_path} --continue_train --edit_config_json '{json.dumps(edit)}'"
        )
        self.assertTrue(
            misc_util.run_cmd(cmd, os.path.join(self.test_dir, "log_resave.txt"))
        )
        resaved = os.path.join(resave_dir, f"model.ckpt-{step}")
        self.assertTrue(
            os.path.exists(os.path.join(resaved, checkpoint_util.CKPT_SUCCESS_FILENAME))
        )
        for part in ("model", "optimizer"):
            self._assert_same_state(
                _dcp_state(os.path.join(resaved, part)),
                _dcp_state(os.path.join(saved, part)),
                part,
            )
        for name in (
            checkpoint_util.DATALOADER_CKPT_FILENAME,
            checkpoint_util.LR_SCHEDULER_CKPT_FILENAME,
        ):
            with open(os.path.join(saved, name)) as fa:
                with open(os.path.join(resaved, name)) as fb:
                    self.assertEqual(json.load(fb), json.load(fa), name)
        a, b = _dynamicemb_state(saved), _dynamicemb_state(resaved)
        self.assertEqual(sorted(a), sorted(b))
        self.assertTrue(any(t["counters"] for t in a.values()))
        for name, ta in a.items():
            tb = b[name]
            for k in ("keys", "values", "opt"):
                np.testing.assert_array_equal(tb[k], ta[k], err_msg=f"{name} {k}")
            self.assertEqual(tb["meta"], ta["meta"], name)
            if ta["counters"] is not None:
                # NOTE: the restore's optimizer warm-up step also counts its
                # batch's ids; counters are not cleared before DynamicEmbLoad.
                restored = {k: tb["counters"].get(k) for k in ta["counters"]}
                self.assertEqual(restored, ta["counters"], name)
            if "click_50_seq__id_3" in name:
                # Built-in NO_EVICTION: scores are row indices, not access scores.
                continue
            if "Lru" in ta["meta"]["evict_strategy"]:
                # TIMESTAMP scores are saved as ages: rebased by one constant.
                shift = tb["scores"] - ta["scores"]
                self.assertTrue((shift == shift.flat[0]).all() if shift.size else True)
            else:
                np.testing.assert_array_equal(tb["scores"], ta["scores"], name)

        # The input position: resuming from `step` with no local state ends
        # where the uninterrupted run ended.
        ok, log = self._resume(config_path, "log_resume", {**edit})
        self.assertTrue(ok)
        self.assertIn("loading RocksDB tables into fresh working DB", log)
        resumed, resumed_step = checkpoint_util.latest_checkpoint(
            resume_dir, require_success=True
        )
        assert resumed is not None
        self.assertEqual(resumed_step, final_step)
        for name in (
            checkpoint_util.DATALOADER_CKPT_FILENAME,
            checkpoint_util.CKPT_META_FILENAME,
        ):
            with open(os.path.join(final, name)) as fa:
                with open(os.path.join(resumed, name)) as fb:
                    self.assertEqual(fb.read(), fa.read(), name)
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
        # The STEP table id_3 at capacity 4000 logically evicts; deletions publish.
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
