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
import torch
from parameterized import parameterized

from tzrec.prompt import assembler
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
    PromptAssembler,
)
from tzrec.prompt.types import (
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
from tzrec.utils.test_util import parameterized_name_func

_BASE_VOCAB_SIZE = 1000
_SENTINEL = 1099


_CODEBOOK = (4, 4, 4)
_NUM_LEVELS = len(_CODEBOOK)


def _sid_space(codebook=_CODEBOOK) -> ResolvedSidSpace:
    offsets, running = [], 0
    for size in codebook:
        offsets.append(running)
        running += size
    return ResolvedSidSpace(
        codebook=tuple(codebook),
        num_levels=len(codebook),
        base_vocab_size=_BASE_VOCAB_SIZE,
        level_offsets=tuple(offsets),
        band_lo=tuple(_BASE_VOCAB_SIZE + o for o in offsets),
        band_hi=tuple(_BASE_VOCAB_SIZE + o + s - 1 for o, s in zip(offsets, codebook)),
        target_vocab_size=1152,
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
    slot_id=0,
) -> SlotSeg:
    return SlotSeg(
        slot_id=slot_id,
        name=name,
        feature_names=tuple(feature_names) if feature_names is not None else (name,),
        group_type=group_type,
        output_key=".sequence"
        if group_type == FeatureGroupType.JAGGED_SEQUENCE
        else "",
        fill=fill,
        width=Width(WidthKind.BOUNDED, width_n)
        if width_n
        else Width(WidthKind.STATIC, _NUM_LEVELS),
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


def _asm(segments, response=(), max_length=0, sid_space=None) -> PromptAssembler:
    """An assembler over one ad-hoc plan."""
    return PromptAssembler(
        _plan(segments, response=response, max_length=max_length),
        _sid_space() if sid_space is None else sid_space,
    )


def _parsed(inline=None, projected=None) -> dict:
    """Build parsed features the way DataParser emits them.

    Args:
        inline: INLINE slot name to its per-sample value arrays.
        projected: PROJECTED slot name to its per-sample position counts.
    """
    out = {}
    for name, rows in (inline or {}).items():
        out[f"{name}.values"] = torch.as_tensor(
            np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        )
        out[f"{name}.lengths"] = torch.tensor([len(row) for row in rows])
    for name, lengths in (projected or {}).items():
        out[f"{name}.lengths"] = torch.tensor(lengths)
    return out


class PromptAssemblerTest(unittest.TestCase):
    def test_inline_sid_gets_the_base_vocab_shift(self) -> None:
        asm = _asm((Static((7, 8)), _slot("hist", FillMode.INLINE)))
        # offset codes for one item: level 0 -> 1, level 1 -> 4+2, level 2 -> 8+3
        out = asm(_parsed({"hist": [np.array([1, 6, 11])]}))

        self.assertEqual(
            out[INPUT_IDS].tolist(),
            [
                7,
                8,
                _BASE_VOCAB_SIZE + 1,
                _BASE_VOCAB_SIZE + 6,
                _BASE_VOCAB_SIZE + 11,
            ],
        )
        self.assertEqual(out[CU_SEQLENS].tolist(), [0, 5])
        self.assertEqual(out[HOLE_POSITIONS].numel(), 0)
        self.assertEqual(int(out[MAX_SEQLEN]), 5)

    def test_projected_emits_sentinels_and_records_holes(self) -> None:
        asm = _asm((Static((7,)), _slot("prof", FillMode.PROJECTED, 4)))
        out = asm(
            _parsed(projected={"prof": [2, 3]})
            | {"prof.values": torch.tensor([1, 2, 3, 4, 5])}
        )

        # sample 0: [7, S, S]   sample 1: [7, S, S, S]
        self.assertEqual(
            out[INPUT_IDS].tolist(),
            [7, _SENTINEL, _SENTINEL, 7, _SENTINEL, _SENTINEL, _SENTINEL],
        )
        self.assertEqual(out[CU_SEQLENS].tolist(), [0, 3, 7])
        # absolute indices into the flat buffer, which is what index_copy needs
        self.assertEqual(out[HOLE_POSITIONS].tolist(), [1, 2, 4, 5, 6])
        self.assertEqual(out[HOLE_SLOT_COUNTS].tolist(), [5])
        holes = out[HOLE_POSITIONS]
        self.assertTrue(bool(torch.all(out[INPUT_IDS][holes] == _SENTINEL)))
        self.assertEqual(int(torch.sum(out[INPUT_IDS] == _SENTINEL)), 5)
        self.assertEqual(int(out[MAX_SEQLEN]), 4)

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
        out = asm(
            _parsed(projected={"a": [1, 2], "b": [2, 1]})
            | {"a.values": torch.tensor([1, 2, 3]), "b.values": torch.tensor([4, 5, 6])}
        )

        self.assertEqual(out[CU_SEQLENS].tolist(), [0, 5, 11])
        self.assertEqual(out[HOLE_POSITIONS].tolist(), [0, 5, 6, 2, 3, 8, 4, 9, 10])
        self.assertEqual(out[HOLE_SLOT_COUNTS].tolist(), [3, 3, 3])

    def test_response_is_optional_and_its_length_is_recorded(self) -> None:
        plan = _plan(
            (Static((7,)), _slot("hist", FillMode.INLINE)),
            response=(Static((9,)), _slot("answer", FillMode.INLINE)),
        )
        parsed = _parsed(
            {"hist": [np.array([1, 6, 11])], "answer": [np.array([0, 4, 8])]}
        )
        out = PromptAssembler(plan, _sid_space())(parsed)

        self.assertEqual(
            out[INPUT_IDS].tolist(),
            [
                7,
                _BASE_VOCAB_SIZE + 1,
                _BASE_VOCAB_SIZE + 6,
                _BASE_VOCAB_SIZE + 11,
                9,
                _BASE_VOCAB_SIZE,
                _BASE_VOCAB_SIZE + 4,
                _BASE_VOCAB_SIZE + 8,
            ],
        )
        self.assertEqual(out[RESPONSE_LENGTHS].tolist(), [4])

        prompt_only = PromptAssembler(plan, _sid_space(), include_response=False)(
            _parsed({"hist": [np.array([1, 6, 11])]})
        )
        self.assertEqual(
            prompt_only[INPUT_IDS].tolist(),
            [7, _BASE_VOCAB_SIZE + 1, _BASE_VOCAB_SIZE + 6, _BASE_VOCAB_SIZE + 11],
        )
        self.assertEqual(prompt_only[RESPONSE_LENGTHS].tolist(), [0])

    def test_rejects_a_response_of_the_wrong_width(self) -> None:
        plan = _plan(
            (_slot("hist", FillMode.INLINE),),
            response=(_slot("answer", FillMode.INLINE),),
        )
        asm = PromptAssembler(plan, _sid_space())
        with self.assertRaisesRegex(ValueError, "compiled width is 3"):
            asm(
                _parsed(
                    {
                        "hist": [np.array([1, 6, 11])],
                        "answer": [np.array([0, 4, 8, 1, 5, 9])],
                    }
                )
            )

    def test_a_multi_value_history_walks_like_the_flat_layout(self) -> None:
        """Items with key_lengths and one code per position are one stream."""
        asm = _asm((Static((7,)), _slot("hist", FillMode.INLINE)))
        flat = asm(
            _parsed({"hist": [np.array([1, 6, 11, 2, 7, 10]), np.array([0, 4, 8])]})
        )
        items = asm(
            {
                "hist.values": torch.tensor([1, 6, 11, 2, 7, 10, 0, 4, 8]),
                "hist.lengths": torch.tensor([2, 1]),
                "hist.key_lengths": torch.tensor([3, 3, 3]),
            }
        )
        for key in (INPUT_IDS, CU_SEQLENS, HOLE_POSITIONS, MAX_SEQLEN):
            self.assertTrue(torch.equal(flat[key], items[key]), key)

    def test_scalar_label_is_named_not_a_key_error(self) -> None:
        asm = _asm((_slot("hist", FillMode.INLINE),))

        with self.assertRaisesRegex(ValueError, "must be\\s+list<int64>"):
            asm({"hist.values": torch.tensor([1, 6, 11])})

    @parameterized.expand(
        [
            [[1, 2, 3]],
            [[1, 6, 12]],
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_a_code_outside_its_band(self, values) -> None:
        asm = _asm((_slot("hist", FillMode.INLINE),))
        with self.assertRaisesRegex(ValueError, "offset_codebook column"):
            asm(_parsed({"hist": [np.array(values)]}))

    def test_rejects_a_partial_item(self) -> None:
        asm = _asm((_slot("hist", FillMode.INLINE),))
        with self.assertRaisesRegex(ValueError, "whole number of 3-level items"):
            asm(_parsed({"hist": [np.array([1, 6])]}))

    def test_over_long_row_is_an_error_not_a_truncation(self) -> None:
        asm = _asm((Static((7, 8, 9)), _slot("hist", FillMode.INLINE)), max_length=4)
        with self.assertRaisesRegex(ValueError, "never truncated"):
            asm(_parsed({"hist": [np.array([1, 6, 11])]}))

    def test_column_shaped_values_are_flattened(self) -> None:
        # the data parser emits (total, value_dim) for a dense sequence feature
        from tzrec.prompt.types import CompiledPrompt, ProjectionPlan

        plan = _plan((_slot("hist", FillMode.INLINE),))
        compiled_prompt = CompiledPrompt(
            sid_space=_sid_space(),
            prompt_plan=plan,
            projection_plan=ProjectionPlan(projections={}, slot_to_module={}),
        )
        parsed = {
            "hist.values": np.array([[1], [6], [11], [0], [4], [8]]),
            "hist.lengths": np.array([3, 3]),
        }
        out = assemble_into(compiled_prompt, parsed)
        self.assertEqual(out["prompt_cu_seqlens"].tolist(), [0, 3, 6])
        self.assertEqual(out["prompt_input_ids"].tolist()[0], _BASE_VOCAB_SIZE + 1)
        self.assertEqual(int(out["prompt_max_seqlen"]), 3)

    def test_rejects_inconsistent_slot_batch_sizes(self) -> None:
        plan = _plan(
            (
                _slot("hist", FillMode.INLINE),
                _slot("answer", FillMode.INLINE),
            )
        )
        asm = PromptAssembler(plan, _sid_space())
        parsed = {
            "hist.values": torch.tensor([1, 6, 11]),
            "hist.lengths": torch.tensor([3]),
            "answer.values": torch.tensor([0, 4, 8, 1, 6, 11]),
            "answer.lengths": torch.tensor([3, 3]),
        }
        with self.assertRaisesRegex(
            ValueError, r"prompt slot \[answer\] has 2 samples, expected 1"
        ):
            asm(parsed)

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
            "age.lengths": torch.tensor([2, 1]),
            "country.lengths": torch.tensor([2, 2]),
        }

        with self.assertRaisesRegex(
            ValueError,
            r"PROJECTED features \[age\] and \[country\] have different",
        ):
            asm(parsed)

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
        out = asm(
            {
                "dense.values": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "sparse.values": torch.tensor([5, 6, 7]),
                "sparse.lengths": torch.tensor([2, 1]),
            }
        )

        self.assertEqual(out[INPUT_IDS].tolist(), [_SENTINEL, _SENTINEL])
        self.assertEqual(out[CU_SEQLENS].tolist(), [0, 1, 2])
        self.assertEqual(out[HOLE_POSITIONS].tolist(), [0, 1])

    def test_output_keys_match_the_module_constants(self) -> None:
        """``forward`` writes literals; they must equal the exported names."""
        module = PromptAssembler(_plan((Static((7,)),)), _sid_space())
        out = module({"batch_size": torch.tensor(2)})
        self.assertEqual(
            set(out.keys()),
            {
                INPUT_IDS,
                CU_SEQLENS,
                HOLE_POSITIONS,
                HOLE_SLOT_COUNTS,
                RESPONSE_LENGTHS,
                MAX_SEQLEN,
            },
        )
        self.assertEqual(out[INPUT_IDS].tolist(), [7, 7])
        self.assertEqual(assembler.PROMPT_INPUT_IDS, "prompt_" + INPUT_IDS)

    def test_scripting_preserves_every_output(self) -> None:
        """The artifact and the collator's module are the same function."""
        plan = _plan(
            (
                Static((7,)),
                _slot("hist", FillMode.INLINE),
                _slot("beh", FillMode.PROJECTED, 4),
            ),
            response=(_slot("answer", FillMode.INLINE),),
        )
        module = PromptAssembler(plan, _sid_space())
        batch = {
            "hist.values": torch.tensor([1, 6, 11, 2, 7, 10]),
            "hist.lengths": torch.tensor([1, 1]),
            "hist.key_lengths": torch.tensor([3, 3]),
            "beh.values": torch.tensor([7, 8, 9, 21, 22]),
            "beh.lengths": torch.tensor([3, 2]),
            "answer.values": torch.tensor([3, 7, 11, 0, 4, 8]),
            "answer.lengths": torch.tensor([3, 3]),
        }
        eager = module(batch)
        scripted = torch.jit.script(module)
        for key, value in scripted(batch).items():
            self.assertTrue(torch.equal(eager[key], value), key)

    def test_a_scripted_walk_reports_its_validation(self) -> None:
        """The exported artifact refuses a bad request with the same message."""
        scripted = torch.jit.script(
            _asm((Static((7, 8, 9)), _slot("hist", FillMode.INLINE)), max_length=4)
        )
        with self.assertRaisesRegex(torch.jit.Error, "never truncated"):
            scripted(_parsed({"hist": [np.array([1, 6, 11])]}))


if __name__ == "__main__":
    unittest.main()
