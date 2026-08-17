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
from unittest import mock

from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.persist import (
    PROMPT_DIR,
    check_prompt_assets,
    copy_prompt_assets,
    read_prompt_hashes,
    save_prompt_assets,
)
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import (
    create_prompt_feature,
    create_prompt_tokenizer,
)
from tzrec.utils.test_util import make_test_dir

_WORDS = ["History", "Predict", ":", "<unk>", "<|im_end|>"]


class PromptPersistTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.tok_path = create_prompt_tokenizer(
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )
        self.features = [
            create_prompt_feature(
                'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'
            ),
            create_prompt_feature(
                'sequence_raw_feature { feature_name: "answer" '
                'expression: "item:answer" }'
            ),
        ]

    def _compile(self, codebook=(4, 4, 4), prompt="History : {{hist}}"):
        cfg = PromptConfig(
            tokenizer_path=self.tok_path, prompt=prompt, response="{{answer}}"
        )
        cfg.sid_space.codebook.extend(codebook)
        return compile_prompt(cfg, self.features, model_dir=self.test_dir)

    def test_writes_a_self_describing_directory(self) -> None:
        compiled_prompt = self._compile()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(compiled_prompt, ckpt)

        out = os.path.join(ckpt, PROMPT_DIR)
        for name in ("sid_space.json", "prompt_plan.json", "prompt_hashes.json"):
            self.assertTrue(os.path.exists(os.path.join(out, name)), name)
        # serving reloads the extended tokenizer from the checkpoint
        self.assertTrue(
            os.path.exists(os.path.join(out, "tokenizer", "tokenizer.json"))
        )

    def test_sid_space_round_trips_as_plain_json(self) -> None:
        compiled_prompt = self._compile()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(compiled_prompt, ckpt)

        with open(os.path.join(ckpt, PROMPT_DIR, "sid_space.json")) as f:
            space = json.load(f)
        self.assertEqual(space["codebook"], [4, 4, 4])
        self.assertEqual(space["level_offsets"], [0, 4, 8])
        self.assertEqual(space["band_lo"][0], compiled_prompt.sid_space.base_vocab_size)
        # every declared field survives, so serving needs no tzrec code
        self.assertEqual(
            set(space),
            {f.name for f in dataclasses.fields(compiled_prompt.sid_space)},
        )

    def test_a_changed_codebook_is_fatal(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(self._compile(codebook=(4, 4, 4)), ckpt)
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            check_prompt_assets(self._compile(codebook=(8, 8, 8)), ckpt)

    def test_a_changed_projection_body_warns(self) -> None:
        # iterating a dict yields only its keys, so a body change must not be
        # able to hide behind an unchanged projection name
        def compile_with(hidden_units):
            cfg = PromptConfig(
                tokenizer_path=self.tok_path,
                prompt="History : {{hist}} {{prof}}",
                response="{{answer}}",
            )
            cfg.sid_space.codebook.extend((4, 4, 4))
            slot = cfg.slots.add(name="prof")
            slot.feature_names.append("prof")
            slot.projection.mlp.hidden_units.extend(hidden_units)
            features = self.features + [
                create_prompt_feature(
                    'sequence_id_feature { feature_name: "prof" '
                    'expression: "user:prof" num_buckets: 16 embedding_dim: 8 '
                    "sequence_length: 2 }"
                )
            ]
            return compile_prompt(cfg, features, model_dir=self.test_dir)

        ckpt = os.path.join(self.test_dir, "model.ckpt-proj")
        save_prompt_assets(compile_with([16]), ckpt)
        widened = compile_with([256, 128])
        # only the projection changed, so the vocabulary is still usable
        self.assertEqual(widened.vocab_hash, read_prompt_hashes(ckpt)["vocab_hash"])
        self.assertNotEqual(widened.plan_hash, read_prompt_hashes(ckpt)["plan_hash"])
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(widened, ckpt)
        warning.assert_called_once()

    def test_swapped_projection_routing_warns(self) -> None:
        # identical bodies, so only slot_to_module differs -- without it each
        # slot would silently load the other's learned weights
        def compile_with(pa_module, pb_module):
            cfg = PromptConfig(
                tokenizer_path=self.tok_path,
                prompt="History : {{hist}} {{pa}} {{pb}}",
                response="{{answer}}",
            )
            cfg.sid_space.codebook.extend((4, 4, 4))
            features = list(self.features)
            for name, module_id in (("pa", pa_module), ("pb", pb_module)):
                slot = cfg.slots.add(name=name, projection_name=module_id)
                slot.feature_names.append(name)
                slot.projection.mlp.hidden_units.extend([16])
                features.append(
                    create_prompt_feature(
                        f'sequence_id_feature {{ feature_name: "{name}" '
                        f'expression: "user:{name}" num_buckets: 16 '
                        "embedding_dim: 8 sequence_length: 2 }"
                    )
                )
            return compile_prompt(cfg, features, model_dir=self.test_dir)

        ckpt = os.path.join(self.test_dir, "model.ckpt-route")
        save_prompt_assets(compile_with("X", "Y"), ckpt)
        swapped = compile_with("Y", "X")
        self.assertEqual(swapped.vocab_hash, read_prompt_hashes(ckpt)["vocab_hash"])
        self.assertNotEqual(swapped.plan_hash, read_prompt_hashes(ckpt)["plan_hash"])
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(swapped, ckpt)
        warning.assert_called_once()

    def test_a_changed_template_only_warns(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(self._compile(), ckpt)
        changed_compiled_prompt = self._compile(prompt="Predict : {{hist}}")
        # the vocabulary is untouched, so the weights are still usable
        self.assertEqual(
            changed_compiled_prompt.vocab_hash,
            read_prompt_hashes(ckpt)["vocab_hash"],
        )
        self.assertNotEqual(
            changed_compiled_prompt.plan_hash,
            read_prompt_hashes(ckpt)["plan_hash"],
        )
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(changed_compiled_prompt, ckpt)
        warning.assert_called_once()

    def test_a_checkpoint_without_assets_is_fatal(self) -> None:
        # CheckpointManager.save swallows asset-write failures, so a bare
        # checkpoint must not restore with the vocabulary guard disabled
        bare = os.path.join(self.test_dir, "model.ckpt-bare")
        os.makedirs(bare, exist_ok=True)
        self.assertIsNone(read_prompt_hashes(bare))
        with self.assertRaisesRegex(ValueError, "records no prompt assets"):
            check_prompt_assets(self._compile(), bare)

    def test_no_prompt_config_is_a_no_op(self) -> None:
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(None, os.path.join(self.test_dir, "nowhere"))
        warning.assert_not_called()

    def test_export_carries_the_contract_forward(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        save_prompt_assets(self._compile(), ckpt)
        export = os.path.join(self.test_dir, "export")
        copy_prompt_assets(ckpt, export)

        # the HF branch never builds a model, so save_assets cannot run there
        self.assertEqual(read_prompt_hashes(export), read_prompt_hashes(ckpt))
        self.assertTrue(
            os.path.exists(
                os.path.join(export, PROMPT_DIR, "tokenizer", "tokenizer.json")
            )
        )

    def test_copying_from_a_bare_checkpoint_only_warns(self) -> None:
        bare = os.path.join(self.test_dir, "bare")
        os.makedirs(bare, exist_ok=True)
        export = os.path.join(self.test_dir, "export_bare")
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            copy_prompt_assets(bare, export)
        warning.assert_called_once()
        self.assertIsNone(read_prompt_hashes(export))

    def test_only_rank_zero_writes(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-rank")
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            save_prompt_assets(self._compile(), ckpt)
        # a non-zero rank must not race rank 0's json.dump and copytree
        self.assertFalse(os.path.exists(os.path.join(ckpt, PROMPT_DIR)))

        with mock.patch.dict(os.environ, {"RANK": "0"}):
            save_prompt_assets(self._compile(), ckpt)
        self.assertIsNotNone(read_prompt_hashes(ckpt))


if __name__ == "__main__":
    unittest.main()
