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
from unittest import mock

from tzrec.constant import HF_EXPORT_META_FILENAME
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.persist import (
    check_prompt_assets,
    read_prompt_digests,
)
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import (
    create_prompt_feature,
    create_prompt_tokenizer,
)
from tzrec.utils.test_util import make_test_dir

_WORDS = ["History", "Predict", ":", "<unk>", "<|im_end|>"]


def _record(compiled_prompt, ckpt_dir: str) -> None:
    """Write the digests where write_hf_assets puts them."""
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME), "w") as f:
        json.dump(
            {
                "backbone_state_dict_prefix": "lm.",
                "vocab_hash": compiled_prompt.vocab_hash,
                "plan_hash": compiled_prompt.plan_hash,
            },
            f,
        )


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
        ]

    def _compile(self, codebook=(4, 4, 4), prompt="History : {{hist}}"):
        cfg = PromptConfig(
            tokenizer_path=self.tok_path, prompt=prompt, response="{{answer}}"
        )
        cfg.sid_space.codebook.extend(codebook)
        return compile_prompt(cfg, self.features, ["answer"])

    def test_a_changed_codebook_is_fatal(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        _record(self._compile(codebook=(4, 4, 4)), ckpt)
        with self.assertRaisesRegex(ValueError, "does not match checkpoint"):
            check_prompt_assets(self._compile(codebook=(8, 8, 8)), ckpt)

    def test_a_changed_projection_body_warns(self) -> None:
        # a body change must not hide behind an unchanged projection name
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
            return compile_prompt(cfg, features, ["answer"])

        ckpt = os.path.join(self.test_dir, "model.ckpt-proj")
        _record(compile_with([16]), ckpt)
        widened = compile_with([256, 128])
        # only the projection changed, so the vocabulary is still usable
        self.assertEqual(widened.vocab_hash, read_prompt_digests(ckpt)["vocab_hash"])
        self.assertNotEqual(widened.plan_hash, read_prompt_digests(ckpt)["plan_hash"])
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(widened, ckpt)
        warning.assert_called_once()

    def test_swapped_projection_routing_warns(self) -> None:
        # identical bodies, so only slot_to_module differs
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
            return compile_prompt(cfg, features, ["answer"])

        ckpt = os.path.join(self.test_dir, "model.ckpt-route")
        _record(compile_with("X", "Y"), ckpt)
        swapped = compile_with("Y", "X")
        self.assertEqual(swapped.vocab_hash, read_prompt_digests(ckpt)["vocab_hash"])
        self.assertNotEqual(swapped.plan_hash, read_prompt_digests(ckpt)["plan_hash"])
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(swapped, ckpt)
        warning.assert_called_once()

    def test_a_changed_template_only_warns(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        _record(self._compile(), ckpt)
        changed_compiled_prompt = self._compile(prompt="Predict : {{hist}}")
        # the vocabulary is untouched, so the weights are still usable
        self.assertEqual(
            changed_compiled_prompt.vocab_hash,
            read_prompt_digests(ckpt)["vocab_hash"],
        )
        self.assertNotEqual(
            changed_compiled_prompt.plan_hash,
            read_prompt_digests(ckpt)["plan_hash"],
        )
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(changed_compiled_prompt, ckpt)
        warning.assert_called_once()

    def test_a_checkpoint_without_assets_is_fatal(self) -> None:
        # save() swallows asset-write failures, so a bare checkpoint must fail
        bare = os.path.join(self.test_dir, "model.ckpt-bare")
        os.makedirs(bare, exist_ok=True)
        self.assertIsNone(read_prompt_digests(bare))
        with self.assertRaisesRegex(ValueError, "records no prompt digests"):
            check_prompt_assets(self._compile(), bare)

    def test_hf_metadata_without_digests_is_fatal(self) -> None:
        ckpt = os.path.join(self.test_dir, "model.ckpt-nodigest")
        os.makedirs(ckpt, exist_ok=True)
        with open(os.path.join(ckpt, HF_EXPORT_META_FILENAME), "w") as f:
            json.dump({"backbone_state_dict_prefix": "lm."}, f)

        self.assertIsNone(read_prompt_digests(ckpt))
        with self.assertRaisesRegex(ValueError, "records no prompt digests"):
            check_prompt_assets(self._compile(), ckpt)

    def test_no_prompt_config_is_a_no_op(self) -> None:
        with mock.patch("tzrec.prompt.persist.logger.warning") as warning:
            check_prompt_assets(None, os.path.join(self.test_dir, "nowhere"))
        warning.assert_not_called()


if __name__ == "__main__":
    unittest.main()
