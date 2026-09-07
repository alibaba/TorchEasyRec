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

"""Reads a genrec export the way the SGLang genrec stack does.

No sglang import: these tests pin the contract from the consumer's side --
the composite config its model wrapper resolves, the front-end outputs its
multimodal processor cuts into items, the host-side item hash it re-folds
from ``hole_keys``, and the ``prompt.json`` fields its constraint-index
builder reads -- so a change on this side that would break serving fails here.
"""

import io
import json
import os
import unittest

import numpy as np
import torch
from safetensors.torch import load_file

from tzrec.prompt.frontend import mix64
from tzrec.prompt.types import FoldConstants
from tzrec.tests.prompt_test_util import export_tiny_genrec, offset_sid_codes
from tzrec.utils.test_util import make_test_dir

# sglang's multimodal/processors/prompt_genrec.py folds a slot's per-hole keys
# into one item hash with this mixer and this position constant
_C_POSITION = -6752110988234923001
_MASK64 = (1 << 64) - 1


def _mix64_host(value: int) -> int:
    z = (value * 0x9E3779B97F4A7C15) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _item_hash(keys) -> int:
    total = 0
    for index, key in enumerate(keys.tolist()):
        total = (total + _mix64_host((key + _C_POSITION * index) & _MASK64)) & _MASK64
    return total


class GenrecServingContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.exported = export_tiny_genrec(make_test_dir(), bundle_uuid="bundle-test")
        with open(os.path.join(cls.exported.export_dir, "config.json"), "r") as f:
            cls.config = json.load(f)
        with open(
            os.path.join(cls.exported.export_dir, "prompt", "prompt.json"), "r"
        ) as f:
            cls.contract = json.load(f)
        cls.front_end = torch.jit.load(
            os.path.join(cls.exported.export_dir, "frontend", "scripted_model.pt")
        )

    def _payload(self):
        """A request as the in-process processor receives it: an npz blob."""
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            **{
                "hist.values": offset_sid_codes([0, 1, 2, 3, 0, 1], [4, 4, 4]),
                "hist.lengths": np.array([6], dtype=np.int64),
                "beh.values": np.array([3, 9], dtype=np.int64),
                "beh.lengths": np.array([2], dtype=np.int64),
            },
        )
        buffer.seek(0)
        with np.load(buffer, allow_pickle=False) as data:
            return {key: torch.from_numpy(np.asarray(data[key])) for key in data.files}

    def test_config_is_composite_and_names_the_backbone(self) -> None:
        self.assertEqual(self.config["architectures"], ["PromptGenRecForCausalLM"])
        self.assertEqual(self.config["model_type"], "prompt_genrec")
        text_config = self.config["text_config"]
        self.assertEqual(text_config["model_type"], "qwen2")
        self.assertEqual(
            text_config["vocab_size"],
            self.exported.compiled_prompt.sid_space.target_vocab_size,
        )

    def test_weights_keep_the_backbones_own_names(self) -> None:
        """The wrapper delegates load_weights wholesale, so no remapping exists."""
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.for_model(**self.config["text_config"])
        with torch.device("meta"):
            backbone = AutoModelForCausalLM.from_config(config)
        expected = set(backbone.state_dict().keys())
        exported = set(
            load_file(os.path.join(self.exported.export_dir, "model.safetensors"))
        )
        self.assertTrue(exported <= expected, exported - expected)
        self.assertIn("model.embed_tokens.weight", exported)

    def test_front_end_output_feeds_the_processor(self) -> None:
        out = self.front_end(self._payload())
        for key in (
            "input_ids",
            "hole_positions",
            "slot_embeds",
            "hole_keys",
            "hole_slot_counts",
        ):
            self.assertIn(key, out)
        positions = out["hole_positions"]
        self.assertEqual(positions.dtype, torch.int64)
        self.assertTrue(bool(torch.all(positions[1:] > positions[:-1])))
        self.assertEqual(int(out["hole_slot_counts"].sum()), int(positions.numel()))
        self.assertEqual(
            tuple(out["slot_embeds"].shape),
            (int(positions.numel()), self.config["text_config"]["hidden_size"]),
        )
        self.assertEqual(out["slot_embeds"].dtype, torch.float32)
        sentinel = self.contract["sid_space"]["sentinel_token_id"]
        self.assertTrue(bool(torch.all(out["input_ids"][positions] == sentinel)))
        self.assertEqual(out["hole_keys"].dtype, torch.int64)

    def test_host_item_hash_refolds_hole_keys(self) -> None:
        """The per-item outer fold on the host uses the plan's position constant."""
        self.assertEqual(FoldConstants().position, _C_POSITION)
        for value in (0, 1, -1, 123456789, -(2**40)):
            signed = _mix64_host(value & _MASK64)
            signed = signed - (1 << 64) if signed >= 1 << 63 else signed
            self.assertEqual(int(mix64(torch.tensor([value]))[0]), signed)
        keys = self.front_end(self._payload())["hole_keys"]
        self.assertEqual(_item_hash(keys), _item_hash(keys))
        self.assertNotEqual(_item_hash(keys), _item_hash(keys.flip(0)))

    def test_prompt_json_carries_the_index_builder_inputs(self) -> None:
        space = self.contract["sid_space"]
        compiled = self.exported.compiled_prompt.sid_space
        self.assertEqual(space["base_vocab_size"], compiled.base_vocab_size)
        self.assertEqual(space["num_levels"], 3)
        self.assertEqual(space["band_lo"], list(compiled.band_lo))
        self.assertEqual(space["band_hi"], list(compiled.band_hi))
        self.assertEqual(space["bundle_uuid"], "bundle-test")
        self.assertEqual(len(self.contract["decode"]), 3)
        self.assertEqual(
            self.contract["vocab_hash"], self.exported.compiled_prompt.vocab_hash
        )
        self.assertEqual(self.contract["frontend"]["dir"], "frontend")
        self.assertEqual(self.contract["frontend"]["model"], "scripted_model.pt")
        # the artifact pairs with the contract by identity, not by path
        self.assertEqual(self.front_end.vocab_hash, self.contract["vocab_hash"])
        self.assertEqual(self.front_end.plan_hash, self.contract["plan_hash"])
        self.assertEqual(self.front_end.bundle_uuid, "bundle-test")

    def test_tokenizer_dir_decodes_a_sid_atom(self) -> None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(self.exported.export_dir, "prompt", "tokenizer")
        )
        space = self.contract["sid_space"]
        self.assertEqual(tokenizer.decode([space["band_lo"][0]]), "<|sid_0|>")
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(space["sentinel_token_id"]), "<|pg_hole|>"
        )
        self.assertEqual(tokenizer.eos_token_id, space["eos_token_id"])
        self.assertEqual(tokenizer.pad_token_id, space["pad_token_id"])


if __name__ == "__main__":
    unittest.main()
