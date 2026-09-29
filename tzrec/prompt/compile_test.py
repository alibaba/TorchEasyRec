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

import pyarrow as pa
import torch
from google.protobuf import text_format
from parameterized import parameterized
from tokenizers import Tokenizer

from tzrec.datasets.data_parser import DataParser
from tzrec.features.feature import FgMode, create_features
from tzrec.prompt.compile import compile_prompt, save_tokenizer_dir
from tzrec.prompt.types import FillMode, SlotSeg, Static, WidthKind
from tzrec.protos import feature_pb2
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import (
    create_genrec_test_tokenizer,
    make_test_dir,
    parameterized_name_func,
)

_WORDS = ["History", "Profile", "Predict", ":", ".", "Histor0", "<unk>", "<|im_end|>"]


_HIST = 'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'
_PROF = (
    'sequence_id_feature { feature_name: "prof" expression: "user:prof" '
    "num_buckets: 768 embedding_dim: 16 sequence_length: 4 }"
)
_AGE = 'id_feature { feature_name: "age" expression: "user:age" num_buckets: 8 }'
_SID = (
    'sequence_id_feature { feature_name: "sid" expression: "user:sid" '
    "value_dim: 3 sequence_length: 4 }"
)
_GROUPED_SID = """sequence_feature {
     sequence_name: "clk" sequence_length: 16 sequence_delim: ";"
     features { id_feature { feature_name: "sid" expression: "item:sid"
                value_dim: 3 } }
   }"""


def _tokenize(vocab_file: str = "", embedding_dim: int = 0) -> str:
    text = (
        'tokenize_feature { feature_name: "title" expression: "user:title" '
        "tokens_as_sequence: true sequence_length: 8"
    )
    if vocab_file:
        text += f' vocab_file: "{vocab_file}"'
    if embedding_dim:
        text += f" embedding_dim: {embedding_dim}"
    return text + " }"


def _feature(text: str):
    config = feature_pb2.FeatureConfig()
    text_format.Merge(text, config)
    return create_features([config], fg_mode=FgMode.FG_NONE)[0]


class CompilePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.tok_path = create_genrec_test_tokenizer(
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )

    def _config(self, **kwargs) -> PromptConfig:
        kwargs.setdefault("response", "{{answer}}")
        cfg = PromptConfig(tokenizer_path=self.tok_path, **kwargs)
        return cfg

    def _compile(self, cfg, features):
        return compile_prompt(cfg, features, ["answer"])

    def test_sid_space_resolves_offsets_and_bands(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, [_feature(_HIST)])
        space = compiled.sid_space

        base_vocab_size = space.base_vocab_size
        self.assertEqual(space.num_levels, 3)
        self.assertEqual(sum(space.codebook), 12)
        self.assertEqual(space.level_offsets, (0, 4, 8))
        self.assertEqual(
            space.band_lo,
            (base_vocab_size, base_vocab_size + 4, base_vocab_size + 8),
        )
        self.assertEqual(
            space.band_hi,
            (base_vocab_size + 3, base_vocab_size + 7, base_vocab_size + 11),
        )
        # no slot projects, so no sentinel is materialized
        self.assertIsNone(space.sentinel_token_id)
        self.assertEqual(space.target_vocab_size % 128, 0)

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
        groups = compiled.projection_plan.feature_groups
        self.assertEqual([g.group_name for g in groups], ["prof"])
        self.assertEqual(list(groups[0].feature_names), ["prof"])
        self.assertEqual(compiled.prompt_plan.max_holes, 4)
        self.assertIsNotNone(compiled.sid_space.sentinel_token_id)
        self.assertEqual(by_name["hist"].id_shift, compiled.sid_space.base_vocab_size)
        self.assertEqual(by_name["prof"].id_shift, 0)

    def test_tokenize_without_embedding_is_inline_with_no_shift(self) -> None:
        cfg = self._config(prompt="Title : {{title}} History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(
            cfg, [_feature(_tokenize(self.tok_path)), _feature(_HIST)]
        )

        by_name = {
            s.name: s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg)
        }
        self.assertIs(by_name["title"].fill, FillMode.INLINE)
        # word ids of the prompt tokenizer are LM ids already
        self.assertEqual(by_name["title"].id_shift, 0)
        self.assertEqual(by_name["hist"].id_shift, compiled.sid_space.base_vocab_size)
        self.assertEqual(len(compiled.projection_plan.feature_groups), 0)
        self.assertIsNone(compiled.sid_space.sentinel_token_id)

    def test_sequence_tokenize_without_embedding_is_inline_with_no_shift(self) -> None:
        """The standalone sequence entry carries the same TokenizeFeature payload."""
        text = (
            'sequence_tokenize_feature { feature_name: "title" '
            'expression: "user:title" sequence_length: 2 '
            f'vocab_file: "{self.tok_path}" }}'
        )
        cfg = self._config(prompt="Title : {{title}}")
        cfg.sid_space.codebook.extend([4])
        compiled = self._compile(cfg, [_feature(text)])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.INLINE)
        self.assertEqual(seg.id_shift, 0)
        # each text is any number of tokens, so no cap on items bounds the slot
        self.assertIs(seg.width.kind, WidthKind.UNBOUNDED)

    def test_tokenize_with_embedding_is_projected(self) -> None:
        cfg = self._config(prompt="Title : {{title}}")
        cfg.sid_space.codebook.extend([4])
        compiled = self._compile(cfg, [_feature(_tokenize(self.tok_path, 8))])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.PROJECTED)
        self.assertEqual(seg.id_shift, 0)

    def test_sid_id_feature_without_embedding_is_inline(self) -> None:
        cfg = self._config(prompt="History : {{sid}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, [_feature(_SID)])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.INLINE)
        self.assertEqual(seg.id_shift, compiled.sid_space.base_vocab_size)
        # width counts positions: sequence_length items of one code per level
        self.assertIs(seg.width.kind, WidthKind.BOUNDED)
        self.assertEqual(seg.width.num_positions, 12)

    def test_grouped_sid_id_feature_is_inline(self) -> None:
        fc = feature_pb2.FeatureConfig()
        text_format.Merge(_GROUPED_SID, fc)
        grouped = create_features([fc], fg_mode=FgMode.FG_NONE)
        cfg = self._config(prompt="History : {{clk__sid}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, grouped)
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.INLINE)
        self.assertEqual(seg.id_shift, compiled.sid_space.base_vocab_size)
        self.assertEqual(seg.width.num_positions, 48)

    def test_inline_id_feature_needs_one_code_per_level(self) -> None:
        cfg = self._config(prompt="History : {{sid}}")
        cfg.sid_space.codebook.extend([4, 4])
        with self.assertRaisesRegex(ValueError, "value_dim: 2"):
            self._compile(cfg, [_feature(_SID)])

    def test_inline_id_feature_may_not_declare_an_id_space(self) -> None:
        text = _SID.replace("value_dim: 3", "value_dim: 3 num_buckets: 32")
        cfg = self._config(prompt="History : {{sid}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        with self.assertRaisesRegex(ValueError, "num_buckets"):
            self._compile(cfg, [_feature(text)])

    @parameterized.expand(
        [[FgMode.FG_NORMAL], [FgMode.FG_DAG]], name_func=parameterized_name_func
    )
    def test_inline_id_feature_may_bucketize_over_the_code_space(self, fg_mode) -> None:
        # FG needs a bucketize config to run the feature at all
        fc = feature_pb2.FeatureConfig()
        text_format.Merge(
            _GROUPED_SID.replace("value_dim: 3", "value_dim: 3 num_buckets: 12"), fc
        )
        grouped = create_features([fc], fg_mode=fg_mode)
        cfg = self._config(prompt="History : {{clk__sid}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, grouped)
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.INLINE)

        data = DataParser(features=grouped).parse(
            input_data={
                "clk__sid": pa.array(
                    [[[0, 5, 11], [3, 4, 8]], [[1, 7, 9]]],
                    type=pa.list_(pa.list_(pa.int64())),
                )
            }
        )
        # every offset code comes out as it went in
        torch.testing.assert_close(
            data["clk__sid.values"],
            torch.tensor([0, 5, 11, 3, 4, 8, 1, 7, 9], dtype=torch.int64),
        )
        torch.testing.assert_close(
            data["clk__sid.lengths"], torch.tensor([2, 1], dtype=torch.int32)
        )
        torch.testing.assert_close(
            data["clk__sid.key_lengths"], torch.tensor([3, 3, 3], dtype=torch.int32)
        )

    def test_unreadable_vocab_file_is_a_config_error(self) -> None:
        cfg = self._config(prompt="Title : {{title}}")
        cfg.sid_space.codebook.extend([4])
        missing = os.path.join(self.test_dir, "missing.json")
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            self._compile(cfg, [_feature(_tokenize(missing))])

    def test_tokenize_vocab_must_match_the_prompt_tokenizer(self) -> None:
        other = create_genrec_test_tokenizer(
            os.path.join(self.test_dir, "other.json"), ["a", "b"]
        )
        cfg = self._config(prompt="Title : {{title}}")
        cfg.sid_space.codebook.extend([4])
        with self.assertRaisesRegex(ValueError, "differs from prompt_config"):
            self._compile(cfg, [_feature(_tokenize(other))])
        # a byte-identical copy elsewhere is the same vocabulary
        copy = os.path.join(self.test_dir, "copy.json")
        shutil.copy(self.tok_path, copy)
        compiled = self._compile(cfg, [_feature(_tokenize(copy))])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.INLINE)

    def test_tokenize_without_vocab_does_not_get_created(self) -> None:
        with self.assertRaisesRegex(ValueError, "load_pipeline_config"):
            _feature(_tokenize())

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
        self.assertEqual(seg.width.num_positions, 1)

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
        self.assertEqual(
            self._compile(cfg, [_feature(_HIST)]).sid_space.num_levels,
            3,
        )

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

    def test_sid_tokens_absent_from_the_base_tokenizer(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        cfg.sid_space.codebook.extend([4])
        # renders Histor0..Histor3, and Histor0 is already in the base vocab
        cfg.sid_space.token_format = "Histor{i}"
        with self.assertRaisesRegex(ValueError, "already in the base tokenizer"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_training_compile_persists_nothing(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4])
        before = sorted(os.listdir(self.test_dir))

        compile_prompt(cfg, [_feature(_HIST)], ["answer"])

        self.assertEqual(sorted(os.listdir(self.test_dir)), before)

    def test_extended_tokenizer_is_written(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        cfg.sid_space.codebook.extend([4, 4])
        out = os.path.join(self.test_dir, "export")
        save_tokenizer_dir(compile_prompt(cfg, [_feature(_HIST)], ["answer"]), out)
        written = os.path.join(out, "tokenizer.json")
        self.assertTrue(os.path.exists(written))
        # the SID tokens round-trip, which is what serving reloads
        reloaded = Tokenizer.from_file(written)
        self.assertIsNotNone(reloaded.token_to_id("<|sid_0|>"))
        self.assertIsNotNone(reloaded.token_to_id("<|sid_7|>"))

    def test_answer_width_comes_from_the_codebook(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        compiled = self._compile(cfg, [_feature(_HIST)])

        seg = next(
            s for s in compiled.prompt_plan.response_segments if isinstance(s, SlotSeg)
        )
        # the answer is one SID item, so its width needs no sequence_length
        self.assertIs(seg.width.kind, WidthKind.STATIC)
        self.assertEqual(seg.width.num_positions, 3)
        # +1 because HF shifts logits: the window opens one column before the
        # first supervised label
        self.assertEqual(compiled.prompt_plan.logits_suffix_len, 4)
        self.assertEqual(seg.id_shift, compiled.sid_space.base_vocab_size)

    def test_response_must_name_a_label_field(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{prof}}")
        cfg.sid_space.codebook.extend([4, 4, 4])

        with self.assertRaisesRegex(
            ValueError,
            r"\[prof\] names \['prof'\], which are not in "
            r"data_config.label_fields",
        ):
            self._compile(cfg, [_feature(_HIST), _feature(_PROF)])

    def test_response_slot_takes_exactly_one_label_field(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        slot = cfg.slots.add(name="answer")
        slot.feature_names.extend(["sid_a", "sid_b"])

        with self.assertRaisesRegex(ValueError, "is one label field"):
            compile_prompt(cfg, [_feature(_HIST)], ["sid_a", "sid_b"])

    def test_response_slot_may_not_declare_a_projection(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        slot = cfg.slots.add(name="answer")
        slot.feature_names.append("answer")
        slot.projection.SetInParent()

        with self.assertRaisesRegex(ValueError, "drop its projection"):
            self._compile(cfg, [_feature(_HIST)])

    def test_missing_sid_space_is_rejected(self) -> None:
        # the response width is codebook-derived, so sid_space must exist
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.ClearField("sid_space")
        with self.assertRaisesRegex(ValueError, "sid_space is required"):
            self._compile(cfg, [_feature(_HIST)])

    def test_token_format_without_a_placeholder_is_rejected(self) -> None:
        # without {i} every token renders alike: one row, not sum(codebook)
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        cfg.sid_space.token_format = "<|sid|>"
        with self.assertRaisesRegex(ValueError, "has no '{i}' placeholder"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_custom_token_format_with_a_placeholder_compiles(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        cfg.sid_space.codebook.extend([4, 4, 4])
        cfg.sid_space.token_format = "C{i}"
        compiled = self._compile(cfg, [_feature(_HIST)])

        space = compiled.sid_space
        self.assertEqual(space.band_hi[-1] - space.band_lo[0] + 1, 12)

    def test_missing_response_is_rejected(self) -> None:
        # no response collapses the window to one ignored position: nan loss
        cfg = self._config(prompt="History : {{hist}}", response="")
        cfg.sid_space.codebook.extend([4, 4, 4])
        cfg.ClearField("response")
        with self.assertRaisesRegex(ValueError, "response is required"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_grouped_feature_inherits_the_group_cap(self) -> None:
        # a SequenceFeature member never sets its own sequence_length; the cap
        # comes from the group, so reading .config here would say UNBOUNDED
        fc = feature_pb2.FeatureConfig()
        text_format.Merge(
            """sequence_feature {
                 sequence_name: "clk" sequence_length: 16 sequence_delim: ";"
                 features { id_feature { feature_name: "h" expression: "item:h"
                            num_buckets: 8 embedding_dim: 4 } }
               }""",
            fc,
        )
        grouped = create_features([fc], fg_mode=FgMode.FG_NONE)
        self.assertFalse(grouped[0].config.HasField("sequence_length"))

        cfg = self._config(prompt="History : {{clk__h}}")
        cfg.sid_space.codebook.extend([4])
        compiled = self._compile(cfg, grouped)

        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.width.kind, WidthKind.BOUNDED)
        self.assertEqual(seg.width.num_positions, 16)


if __name__ == "__main__":
    unittest.main()
