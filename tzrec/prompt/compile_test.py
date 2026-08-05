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

from google.protobuf import text_format
from tokenizers import Tokenizer, models, pre_tokenizers

from tzrec.features.feature import FgMode, create_features
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.plan import FillMode, SlotSeg, Static, WidthKind
from tzrec.protos import feature_pb2
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import make_test_dir

_WORDS = ["History", "Profile", "Predict", ":", ".", "<unk>", "<|im_end|>"]


def _tokenizer(path: str) -> str:
    """Write a minimal word-level tokenizer, so no download is needed."""
    vocab = {w: i for i, w in enumerate(_WORDS)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(path)
    return path


def _feature(text: str):
    fc = feature_pb2.FeatureConfig()
    text_format.Merge(text, fc)
    return create_features([fc], fg_mode=FgMode.FG_NONE)[0]


_HIST = 'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'
_PROF = (
    'sequence_id_feature { feature_name: "prof" expression: "user:prof" '
    "num_buckets: 768 embedding_dim: 16 sequence_length: 4 }"
)
_AGE = 'id_feature { feature_name: "age" expression: "user:age" num_buckets: 8 }'


class CompilePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.tok_path = _tokenizer(os.path.join(self.test_dir, "tok.json"))

    def _config(self, **kwargs) -> PromptConfig:
        cfg = PromptConfig(tokenizer=self.tok_path, **kwargs)
        return cfg

    def _compile(self, cfg, features):
        return compile_prompt(cfg, features, model_dir=self.test_dir)

    def test_sid_space_resolves_offsets_and_bands(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, [_feature(_HIST)])
        space = compiled.sid_space

        base = space.base_vocab
        self.assertEqual(space.num_levels, 3)
        self.assertEqual(space.sid_vocab_size, 12)
        self.assertEqual(space.level_offsets, (0, 4, 8))
        self.assertEqual(space.band_lo, (base, base + 4, base + 8))
        self.assertEqual(space.band_hi, (base + 3, base + 7, base + 11))
        # no slot projects, so no sentinel is materialized
        self.assertIsNone(space.sentinel_token_id)
        self.assertEqual(space.target_vocab % 128, 0)

    def test_inline_needs_no_group_projected_gets_one(self) -> None:
        cfg = self._config(prompt="History : {{hist}} . Profile : {{prof}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, [_feature(_HIST), _feature(_PROF)])

        by_name = {
            s.name: s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg)
        }
        self.assertIs(by_name["hist"].fill, FillMode.INLINE)
        self.assertIs(by_name["prof"].fill, FillMode.PROJECTED)
        # only the projected slot produces a group, and so a hole
        self.assertEqual(
            [s.name for s in compiled.prompt_plan.projected_slots], ["prof"]
        )
        self.assertEqual(compiled.prompt_plan.max_holes, 4)
        self.assertIsNotNone(compiled.sid_space.sentinel_token_id)

    def test_static_runs_are_woven_between_slots(self) -> None:
        cfg = self._config(prompt="History : {{hist}} . Predict :")
        cfg.sid_space.codebook.extend([4])
        compiled = self._compile(cfg, [_feature(_HIST)])
        kinds = [
            "static" if isinstance(s, Static) else s.name
            for s in compiled.prompt_plan.segments
        ]
        self.assertEqual(kinds, ["static", "hist", "static"])
        # the leading run is request-invariant; "History :" is two tokens
        self.assertEqual(compiled.prompt_plan.static_prefix_len, 2)

    def test_scalar_slot_is_one_deep_position(self) -> None:
        cfg = self._config(prompt="Profile : {{age}}")
        cfg.sid_space.codebook.extend([4])
        compiled = self._compile(cfg, [_feature(_AGE)])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.PROJECTED)
        self.assertEqual(seg.output_key, "")
        self.assertIs(seg.width.kind, WidthKind.STATIC)
        self.assertEqual(seg.width.n, 1)

    def test_manifest_mismatch_is_fatal(self) -> None:
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": [8, 8, 8]}, f)
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        cfg.sid_space.manifest_path = manifest
        with self.assertRaisesRegex(ValueError, "does not match the manifest"):
            self._compile(cfg, [_feature(_HIST)])

    def test_manifest_match_compiles(self) -> None:
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": [4, 4, 4]}, f)
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        cfg.sid_space.manifest_path = manifest
        self.assertEqual(self._compile(cfg, [_feature(_HIST)]).sid_space.num_levels, 3)

    def test_rejects_a_mixed_kind_slot(self) -> None:
        cfg = self._config(prompt="X : {{both}}")
        cfg.sid_space.codebook.extend([4])
        slot = cfg.slots.add(name="both")
        slot.feature_names.extend(["hist", "age"])
        with self.assertRaisesRegex(ValueError, "mixes sequence and scalar"):
            self._compile(cfg, [_feature(_HIST), _feature(_AGE)])

    def test_rejects_unknown_feature_and_unreferenced_slot(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        cfg.sid_space.codebook.extend([4])
        slot = cfg.slots.add(name="hist")
        slot.feature_names.append("nope")
        with self.assertRaisesRegex(ValueError, "not in\n?\\s*feature_configs"):
            self._compile(cfg, [_feature(_HIST)])

        cfg2 = self._config(prompt="X : {{hist}}")
        cfg2.sid_space.codebook.extend([4])
        cfg2.slots.add(name="ghost").feature_names.append("hist")
        with self.assertRaisesRegex(ValueError, "never referenced"):
            self._compile(cfg2, [_feature(_HIST)])

    def test_rejects_a_projection_on_an_inline_slot(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        cfg.sid_space.codebook.extend([4])
        slot = cfg.slots.add(name="hist")
        slot.feature_names.append("hist")
        slot.projection.bias = True
        with self.assertRaisesRegex(ValueError, "is INLINE"):
            self._compile(cfg, [_feature(_HIST)])

    def test_atoms_absent_from_the_base_tokenizer(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        cfg.sid_space.codebook.extend([4])
        cfg.sid_space.atom_token_format = "History"
        with self.assertRaisesRegex(ValueError, "already in the base tokenizer"):
            self._compile(cfg, [_feature(_HIST)])

    def test_vocab_hash_tracks_the_codebook(self) -> None:
        def compile_with(sizes):
            cfg = self._config(prompt="History : {{hist}}")
            cfg.sid_space.codebook.extend(sizes)
            return self._compile(cfg, [_feature(_HIST)])

        self.assertEqual(
            compile_with([4, 4]).vocab_hash, compile_with([4, 4]).vocab_hash
        )
        self.assertNotEqual(
            compile_with([4, 4]).vocab_hash, compile_with([4, 8]).vocab_hash
        )

    def test_extended_tokenizer_is_written(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4])
        compiled = self._compile(cfg, [_feature(_HIST)])
        written = os.path.join(compiled.tokenizer_dir, "tokenizer.json")
        self.assertTrue(os.path.exists(written))
        # the atoms round-trip, which is what serving reloads
        reloaded = Tokenizer.from_file(written)
        self.assertIsNotNone(reloaded.token_to_id("<|sid_0|>"))
        self.assertIsNotNone(reloaded.token_to_id("<|sid_7|>"))

    def test_answer_width_comes_from_the_codebook(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        answer = _feature(
            'sequence_raw_feature { feature_name: "answer" expression: "item:answer" }'
        )
        compiled = self._compile(cfg, [_feature(_HIST), answer])

        seg = next(
            s for s in compiled.prompt_plan.response_segments if isinstance(s, SlotSeg)
        )
        # the answer is one SID item, so its width needs no sequence_length
        self.assertIs(seg.width.kind, WidthKind.STATIC)
        self.assertEqual(seg.width.n, 3)
        # +1 because HF shifts logits: the window opens one column before the
        # first supervised label
        self.assertEqual(compiled.prompt_plan.suffix_keep, 4)

    def test_unbounded_response_is_rejected(self) -> None:
        # with no sid_space the response has no codebook-derived width, so the
        # supervised window is unbounded and the logits would cover every
        # position
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        answer = _feature(
            'sequence_raw_feature { feature_name: "answer" expression: "item:answer" }'
        )
        with self.assertRaisesRegex(ValueError, "window cannot be bounded"):
            self._compile(cfg, [_feature(_HIST), answer])


if __name__ == "__main__":
    unittest.main()
