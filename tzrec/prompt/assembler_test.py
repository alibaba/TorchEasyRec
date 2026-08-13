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

import unittest

import numpy as np

from tzrec.prompt.assembler import PromptAssembler
from tzrec.prompt.plan import (
    FillMode,
    PromptPlan,
    ResolvedSidSpace,
    SlotSeg,
    Static,
    Width,
    WidthKind,
)
from tzrec.protos.model_pb2 import FeatureGroupType
from tzrec.tests.prompt_test_util import assemble_into

_BASE = 1000
_SENTINEL = 1099


def _sid_space(codebook=(4, 4, 4)) -> ResolvedSidSpace:
    offsets, running = [], 0
    for size in codebook:
        offsets.append(running)
        running += size
    return ResolvedSidSpace(
        codebook=tuple(codebook),
        num_levels=len(codebook),
        base_vocab=_BASE,
        level_offsets=tuple(offsets),
        band_lo=tuple(_BASE + o for o in offsets),
        band_hi=tuple(_BASE + o + s - 1 for o, s in zip(offsets, codebook)),
        target_vocab=1152,
        sentinel_token_id=_SENTINEL,
        eos_token_id=2,
        pad_token_id=3,
    )


def _slot(
    name,
    fill,
    width_n=None,
    feature_names=None,
    group_type=FeatureGroupType.JAGGED_SEQUENCE,
) -> SlotSeg:
    return SlotSeg(
        slot_id=0,
        name=name,
        feature_names=tuple(feature_names) if feature_names is not None else (name,),
        group_type=group_type,
        output_key=".sequence"
        if group_type == FeatureGroupType.JAGGED_SEQUENCE
        else "",
        fill=fill,
        width=Width(WidthKind.BOUNDED, width_n)
        if width_n
        else Width(WidthKind.STATIC, 1),
    )


def _plan(segments, response=(), max_length=0) -> PromptPlan:
    projected = tuple(
        s
        for s in segments + tuple(response)
        if isinstance(s, SlotSeg) and s.fill is FillMode.PROJECTED
    )
    return PromptPlan(
        segments=tuple(segments),
        response_segments=tuple(response),
        max_length=max_length,
        max_total_length=None,
        max_holes=0,
        logits_suffix_len=None,
        static_prefix_len=0,
        projected_slots=projected,
    )


class PromptAssemblerTest(unittest.TestCase):
    def test_inline_sid_gets_the_base_vocab_shift(self) -> None:
        plan = _plan((Static((7, 8)), _slot("hist", FillMode.INLINE)))
        asm = PromptAssembler(plan, _sid_space())
        # offset codes for one item: level 0 -> 1, level 1 -> 4+2, level 2 -> 8+3
        out = asm.assemble({"hist": [np.array([1, 6, 11])]})

        self.assertEqual(
            out.input_ids.tolist(), [7, 8, _BASE + 1, _BASE + 6, _BASE + 11]
        )
        self.assertEqual(out.cu_seqlens.tolist(), [0, 5])
        self.assertEqual(out.hole_positions.size, 0)

    def test_projected_emits_sentinels_and_records_holes(self) -> None:
        plan = _plan((Static((7,)), _slot("prof", FillMode.PROJECTED, 4)))
        asm = PromptAssembler(plan, _sid_space())
        out = asm.assemble({}, {"prof": np.array([2, 3])}, batch_size=2)

        # sample 0: [7, S, S]   sample 1: [7, S, S, S]
        self.assertEqual(
            out.input_ids.tolist(),
            [7, _SENTINEL, _SENTINEL, 7, _SENTINEL, _SENTINEL, _SENTINEL],
        )
        self.assertEqual(out.cu_seqlens.tolist(), [0, 3, 7])
        # absolute indices into the flat buffer, which is what index_copy needs
        self.assertEqual(out.hole_positions.tolist(), [1, 2, 4, 5, 6])

    def test_holes_are_grouped_by_projected_occurrence_then_sample(self) -> None:
        plan = _plan(
            (
                _slot("a", FillMode.PROJECTED, 2),
                Static((7,)),
                _slot("b", FillMode.PROJECTED, 2),
                _slot("a", FillMode.PROJECTED, 2),
            )
        )
        asm = PromptAssembler(plan, _sid_space())
        out = asm.assemble(
            {},
            {"a": np.array([1, 2]), "b": np.array([2, 1])},
            batch_size=2,
        )

        self.assertEqual(out.cu_seqlens.tolist(), [0, 5, 11])
        self.assertEqual(out.hole_positions.tolist(), [0, 5, 6, 2, 3, 8, 4, 9, 10])

    def test_response_is_optional_and_its_length_is_recorded(self) -> None:
        plan = _plan(
            (Static((7, 8)),),
            response=(Static((9,)), _slot("answer", FillMode.INLINE)),
        )
        asm = PromptAssembler(plan, _sid_space())
        out = asm.assemble({"answer": [np.array([0, 4, 8])]})

        self.assertEqual(out.input_ids.tolist(), [7, 8, 9, _BASE, _BASE + 4, _BASE + 8])
        self.assertEqual(out.response_lengths.tolist(), [4])

        prompt_only = PromptAssembler(
            plan, _sid_space(), include_response=False
        ).assemble({}, batch_size=1)
        self.assertEqual(prompt_only.input_ids.tolist(), [7, 8])
        self.assertEqual(prompt_only.response_lengths.tolist(), [0])

    def test_rejects_raw_codes_that_carry_no_offset(self) -> None:
        plan = _plan((_slot("hist", FillMode.INLINE),))
        asm = PromptAssembler(plan, _sid_space())
        # [1, 2, 3] is a valid raw SID but level 1 and 2 are below their bands
        with self.assertRaisesRegex(ValueError, "offset_codebook column"):
            asm.assemble({"hist": [np.array([1, 2, 3])]})

    def test_rejects_a_partial_item(self) -> None:
        plan = _plan((_slot("hist", FillMode.INLINE),))
        asm = PromptAssembler(plan, _sid_space())
        with self.assertRaisesRegex(ValueError, "whole number of 3-level items"):
            asm.assemble({"hist": [np.array([1, 6])]})

    def test_rejects_an_out_of_band_code(self) -> None:
        plan = _plan((_slot("hist", FillMode.INLINE),))
        asm = PromptAssembler(plan, _sid_space())
        # level 2 admits [8, 12); 12 is the first value past it
        with self.assertRaisesRegex(ValueError, "offset_codebook column"):
            asm.assemble({"hist": [np.array([1, 6, 12])]})

    def test_over_long_row_is_an_error_not_a_truncation(self) -> None:
        plan = _plan((Static((7, 8, 9)), _slot("hist", FillMode.INLINE)), max_length=4)
        asm = PromptAssembler(plan, _sid_space())
        with self.assertRaisesRegex(ValueError, "never truncated"):
            asm.assemble({"hist": [np.array([1, 6, 11])]})

    def test_inline_without_a_sid_space_is_rejected_at_construction(self) -> None:
        plan = _plan((_slot("hist", FillMode.INLINE),))
        with self.assertRaisesRegex(ValueError, "no sid_space was compiled"):
            PromptAssembler(plan, None)

    def test_column_shaped_values_are_flattened(self) -> None:
        # the data parser emits (total, value_dim) for a dense sequence feature
        from tzrec.prompt.plan import CompiledPrompt, ProjectionPlan

        plan = _plan((_slot("hist", FillMode.INLINE),))
        prompt = CompiledPrompt(
            sid_space=_sid_space(),
            prompt_plan=plan,
            projection_plan=ProjectionPlan(projections={}, slot_to_module={}),
            tokenizer_dir="",
            vocab_hash="v",
            plan_hash="p",
        )
        parsed = {
            "hist.values": np.array([[1], [6], [11], [0], [4], [8]]),
            "hist.lengths": np.array([3, 3]),
        }
        out = assemble_into(prompt, parsed)
        self.assertEqual(out["prompt_cu_seqlens"].tolist(), [0, 3, 6])
        self.assertEqual(out["prompt_input_ids"].tolist()[0], _BASE + 1)

    def test_rejects_inconsistent_slot_batch_sizes(self) -> None:
        plan = _plan(
            (
                _slot("hist", FillMode.INLINE),
                _slot("answer", FillMode.INLINE),
            )
        )
        asm = PromptAssembler(plan, _sid_space())
        parsed = {
            "hist.values": np.array([1, 6, 11]),
            "hist.lengths": np.array([3]),
            "answer.values": np.array([0, 4, 8, 1, 6, 11]),
            "answer.lengths": np.array([3, 3]),
        }
        with self.assertRaisesRegex(
            ValueError, r"prompt slot \[answer\] has 2 samples, expected 1"
        ):
            asm.assemble_batch(parsed)

    def test_rejects_mismatched_projected_member_lengths(self) -> None:
        plan = _plan(
            (
                _slot(
                    "profile",
                    FillMode.PROJECTED,
                    4,
                    feature_names=("age", "country"),
                ),
            )
        )
        asm = PromptAssembler(plan, _sid_space())
        parsed = {
            "age.lengths": np.array([2, 1]),
            "country.lengths": np.array([2, 2]),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"PROJECTED features \[age\] and \[country\] have different",
        ):
            asm.assemble_batch(parsed)

    def test_deep_projected_members_emit_one_hole_per_sample(self) -> None:
        plan = _plan(
            (
                _slot(
                    "profile",
                    FillMode.PROJECTED,
                    feature_names=("dense", "sparse"),
                    group_type=FeatureGroupType.DEEP,
                ),
            )
        )
        asm = PromptAssembler(plan, _sid_space())
        out = asm.assemble_batch(
            {
                "dense.values": np.array([[1.0, 2.0], [3.0, 4.0]]),
                "sparse.values": np.array([5, 6, 7]),
                "sparse.lengths": np.array([2, 1]),
            }
        )

        self.assertEqual(out["prompt_input_ids"].tolist(), [_SENTINEL, _SENTINEL])
        self.assertEqual(out["prompt_cu_seqlens"].tolist(), [0, 1, 2])
        self.assertEqual(out["prompt_hole_positions"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
