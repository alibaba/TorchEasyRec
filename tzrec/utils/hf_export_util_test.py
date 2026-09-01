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
import shutil
import threading
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file
from torch import nn

from tzrec.constant import HF_EXPORT_META_FILENAME
from tzrec.utils.checkpoint_util import save_model, unwrap_to
from tzrec.utils.hf_export_util import (
    dcp_to_hf,
    write_hf_assets,
)
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir


def _tied_lm():
    """The tied-head backbone every case here needs; dcp_to_hf must drop the tie."""
    return create_tiny_causal_lm(64, tie_word_embeddings=True)


class _FakeTokenizer:
    """Writes the two tokenizer asset files `write_hf_assets` copies."""

    def save_pretrained(self, save_dir):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            with open(os.path.join(save_dir, name), "w") as f:
                f.write("{}")


class _GenRec(nn.Module):
    """Stand-in for an HF-backed model exposing the optional tokenizer protocol."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.other = nn.Linear(4, 4)

    def hf_backbone(self):
        return self.lm

    def hf_tokenizer(self):
        return _FakeTokenizer()


class _TrainWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class _DmpLike(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class HfExportUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        # the asset writers are rank-0-gated; pin it without leaking the value.
        patcher = mock.patch.dict(os.environ, {"RANK": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unwrap_terminates_on_a_wrapper_cycle(self) -> None:
        """A .model/.module cycle must return None, not spin.

        Every checkpoint save of every model walks this, so an unbounded loop
        here would hang save() and strand the peers waiting on the collective
        that follows. Run on a thread so a regression fails the test instead of
        hanging the suite.
        """
        a, b = nn.Linear(4, 4), nn.Linear(4, 4)
        object.__setattr__(a, "model", b)
        object.__setattr__(b, "model", a)
        out = []
        t = threading.Thread(target=lambda: out.append(unwrap_to(a, "hf_backbone")))
        t.daemon = True
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "unwrap_to did not terminate")
        self.assertEqual(out, [None])

    def test_write_hf_assets_noop_for_non_hf_model(self) -> None:
        save_dir = os.path.join(self.test_dir, "plain")
        write_hf_assets(_TrainWrapper(nn.Linear(4, 4)), save_dir)
        self.assertFalse(os.path.exists(save_dir))

    def _save_ckpt(self, wrapped):
        ckpt_dir = os.path.join(self.test_dir, "model.ckpt-1")
        save_model(ckpt_dir, wrapped)
        write_hf_assets(wrapped, ckpt_dir)
        return ckpt_dir

    def test_write_hf_assets_records_state_dict_prefix(self) -> None:
        lm = _tied_lm()
        wrapped = _TrainWrapper(_GenRec(lm))
        ckpt_dir = self._save_ckpt(wrapped)
        for name in ("config.json", "tokenizer.json", HF_EXPORT_META_FILENAME):
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, name)), name)
        with open(os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME)) as f:
            prefix = json.load(f)["backbone_state_dict_prefix"]
        self.assertEqual(prefix, "model.lm.")
        # the prefix must reconstruct the exact FQNs save_model wrote
        saved = set(wrapped.state_dict())
        self.assertTrue(all(prefix + k in saved for k in lm.state_dict()))

    def test_dcp_to_hf_round_trip_drops_tied_head(self) -> None:
        from transformers import AutoModelForCausalLM

        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_DmpLike(_TrainWrapper(_GenRec(lm))))
        out_dir = os.path.join(self.test_dir, "hf_out")
        dcp_to_hf(ckpt_dir, out_dir)

        st = load_file(os.path.join(out_dir, "model.safetensors"))
        self.assertNotIn("lm_head.weight", st)
        self.assertIn("model.embed_tokens.weight", st)
        back = AutoModelForCausalLM.from_pretrained(out_dir)
        self.assertEqual(
            back.lm_head.weight.data_ptr(), back.model.embed_tokens.weight.data_ptr()
        )
        for k, v in lm.state_dict().items():
            self.assertTrue(torch.equal(back.state_dict()[k], v), k)

    def test_dcp_to_hf_refuses_a_mismatched_architecture(self) -> None:
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(_tied_lm())))
        # widen the recorded architecture so the checkpoint can no longer fill it
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["num_hidden_layers"] = 4
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with self.assertRaisesRegex(RuntimeError, "Refusing to write"):
            dcp_to_hf(ckpt_dir, os.path.join(self.test_dir, "hf_out_bad"))

    def test_dcp_to_hf_missing_dcp_dir(self) -> None:
        empty = os.path.join(self.test_dir, "no_dcp")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaisesRegex(RuntimeError, "not exists"):
            dcp_to_hf(empty, os.path.join(self.test_dir, "hf_out_missing"))


if __name__ == "__main__":
    unittest.main()
