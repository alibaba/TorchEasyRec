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
import json
import os
import unittest

from google.protobuf import text_format
from tokenizers import Tokenizer, models, pre_tokenizers

from tzrec.features.feature import FgMode, create_features
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.persist import (
    PROMPT_DIR,
    check_prompt_assets,
    read_prompt_hashes,
    save_prompt_assets,
)
from tzrec.protos import feature_pb2
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import make_test_dir

_WORDS = ["History", "Predict", ":", "<unk>", "<|im_end|>"]


class PromptPersistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        tok_path = os.path.join(self.test_dir, "tok.json")
        tok = Tokenizer(
            models.WordLevel(
                vocab={w: i for i, w in enumerate(_WORDS)}, unk_token="<unk>"
            )
        )
        tok.pre_tokenizer = pre_tokenizers.Whitespace()
        tok.save(tok_path)
        self.tok_path = tok_path

        fc = feature_pb2.FeatureConfig()
        text_format.Merge(
            'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }', fc
        )
        self.features = create_features([fc], fg_mode=FgMode.FG_NONE)

    def _compile(self, codebook=(4, 4, 4), prompt="History : {{hist}}"):
        cfg = PromptConfig(tokenizer=self.tok_path, prompt=prompt)
        cfg.sid_space.codebook.extend(codebook)
        return compile_prompt(cfg, self.features, model_dir=self.test_dir)

    def test_writes_a_self_describing_directory(self) -> None:
        prompt = self._compile()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(prompt, ckpt)

        out = os.path.join(ckpt, PROMPT_DIR)
        for name in ("sid_space.json", "prompt_plan.json", "prompt_hashes.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        # serving reloads the extended tokenizer from the checkpoint
        self.assertTrue(
            os.path.exists(os.path.join(out, "tokenizer", "tokenizer.json"))
        )

    def test_sid_space_round_trips_as_plain_json(self) -> None:
        prompt = self._compile()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(prompt, ckpt)

        with open(os.path.join(ckpt, PROMPT_DIR, "sid_space.json")) as f:
            space = json.load(f)
        self.assertEqual(space["codebook"], [4, 4, 4])
        self.assertEqual(space["level_offsets"], [0, 4, 8])
        self.assertEqual(space["band_lo"][0], prompt.sid_space.base_vocab)
        # every declared field survives, so serving needs no tzrec code
        self.assertEqual(
            set(space), {f.name for f in dataclasses.fields(prompt.sid_space)}
        )

    def test_matching_prompt_passes(self) -> None:
        prompt = self._compile()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(prompt, ckpt)
        check_prompt_assets(self._compile(), ckpt)

    def test_a_changed_codebook_is_fatal(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(self._compile(codebook=(4, 4, 4)), ckpt)
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            check_prompt_assets(self._compile(codebook=(8, 8, 8)), ckpt)

    def test_a_changed_template_only_warns(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(self._compile(), ckpt)
        moved = self._compile(prompt="Predict : {{hist}}")
        # the vocabulary is untouched, so the weights are still usable
        self.assertEqual(moved.vocab_hash, read_prompt_hashes(ckpt)["vocab_hash"])
        self.assertNotEqual(moved.plan_hash, read_prompt_hashes(ckpt)["plan_hash"])
        check_prompt_assets(moved, ckpt)

    def test_a_checkpoint_without_assets_only_warns(self) -> None:
        bare = os.path.join(self.test_dir, "model.ckpt-bare")
        os.makedirs(bare, exist_ok=True)
        self.assertIsNone(read_prompt_hashes(bare))
        check_prompt_assets(self._compile(), bare)

    def test_no_prompt_config_is_a_no_op(self) -> None:
        check_prompt_assets(None, os.path.join(self.test_dir, "nowhere"))


if __name__ == "__main__":
    unittest.main()
