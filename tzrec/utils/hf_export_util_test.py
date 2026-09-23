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
import unittest
from unittest import mock

import torch
from safetensors.torch import load_file
from torch import nn

from tzrec.utils.checkpoint_util import save_model
from tzrec.utils.hf_export_util import dcp_to_hf
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir


def _tied_lm():
    """The tied-head backbone every case here needs; dcp_to_hf must drop the tie."""
    return create_tiny_causal_lm(64, tie_word_embeddings=True)


class _GenRec(nn.Module):
    """Stand-in for an HF-backed model, with non-backbone params beside the LM."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        self.other = nn.Linear(4, 4)

    def hf_backbone(self):
        return self.lm


class _TrainWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class HfExportUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _save_ckpt(self, wrapped):
        ckpt_dir = os.path.join(self.test_dir, "model.ckpt-1")
        with mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False):
            save_model(ckpt_dir, wrapped)
        return ckpt_dir

    def _convert(self, out_name):
        """Save a checkpoint, convert it, and leave a config.json for the reload."""
        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(lm)))
        out_dir = os.path.join(self.test_dir, out_name)
        dcp_to_hf(ckpt_dir, out_dir, lm.config)
        # the caller composes config.json; here the backbone's own is enough
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(json.loads(lm.config.to_json_string()), f)
        return lm, out_dir

    def test_dcp_to_hf_round_trip_drops_tied_head(self) -> None:
        from transformers import AutoModelForCausalLM

        lm, out_dir = self._convert("hf_out")

        st = load_file(os.path.join(out_dir, "model.safetensors"))
        self.assertNotIn("lm_head.weight", st)
        self.assertIn("model.embed_tokens.weight", st)
        back = AutoModelForCausalLM.from_pretrained(out_dir)
        self.assertEqual(
            back.lm_head.weight.data_ptr(), back.model.embed_tokens.weight.data_ptr()
        )
        for k, v in lm.state_dict().items():
            self.assertTrue(torch.equal(back.state_dict()[k], v), k)

    def test_dcp_to_hf_refuses_keys_from_two_prefixes(self) -> None:
        """One backbone lives under one prefix; a look-alike key cannot stand in."""
        from torch.distributed.checkpoint import state_dict_loader

        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(lm)))
        keys = ["model.lm." + k for k in lm.state_dict()]
        keys[0] = "other.lm." + keys[0][len("model.lm.") :]
        reader = mock.MagicMock()
        reader.read_metadata.return_value.state_dict_metadata = dict.fromkeys(keys)
        patched = mock.patch.object(
            state_dict_loader, "_storage_setup", return_value=reader
        )
        out_dir = os.path.join(self.test_dir, "hf_out_mixed")
        with patched, self.assertRaisesRegex(RuntimeError, "Refusing to write"):
            dcp_to_hf(ckpt_dir, out_dir, lm.config)

    def test_dcp_to_hf_loads_only_the_backbone_keys(self) -> None:
        """The rest of a genrec checkpoint is the sparse tables; never read them."""
        from torch.distributed.checkpoint import state_dict_loader

        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(lm)))
        original = state_dict_loader._load_state_dict_from_keys
        requested = []

        def _spy(keys=None, **kwargs):
            requested.append(keys)
            return original(keys, **kwargs)

        with mock.patch.object(state_dict_loader, "_load_state_dict_from_keys", _spy):
            out_dir = os.path.join(self.test_dir, "hf_out_keys")
            dcp_to_hf(ckpt_dir, out_dir, lm.config)
        self.assertEqual(len(requested), 1)
        self.assertIsNotNone(requested[0])
        self.assertFalse([k for k in requested[0] if ".other." in k])

    def test_dcp_to_hf_refuses_a_mismatched_architecture(self) -> None:
        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(lm)))
        # deepen the architecture so the checkpoint can no longer fill it; a
        # width change would leave the key names intact and convert happily
        cfg = lm.config.to_dict()
        cfg["num_hidden_layers"] = 4
        cfg.pop("layer_types", None)
        with self.assertRaisesRegex(RuntimeError, "Refusing to write"):
            dcp_to_hf(
                ckpt_dir,
                os.path.join(self.test_dir, "hf_out_bad"),
                type(lm.config)(**cfg),
            )

    def test_dcp_to_hf_refuses_a_shape_that_drifted(self) -> None:
        """Same key names, a resized vocabulary: the config no longer fits."""
        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(lm)))
        cfg = lm.config.to_dict()
        cfg["vocab_size"] += 64
        cfg.pop("layer_types", None)
        with self.assertRaisesRegex(RuntimeError, "do not fit"):
            dcp_to_hf(
                ckpt_dir,
                os.path.join(self.test_dir, "hf_out_grown"),
                type(lm.config)(**cfg),
            )

    def test_dcp_to_hf_missing_dcp_dir(self) -> None:
        empty = os.path.join(self.test_dir, "no_dcp")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaisesRegex(RuntimeError, "not exists"):
            dcp_to_hf(
                empty, os.path.join(self.test_dir, "hf_out_missing"), _tied_lm().config
            )


if __name__ == "__main__":
    unittest.main()
