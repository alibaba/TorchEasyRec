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

import os
import unittest

import numpy as np
import torch

from tzrec.prompt import frontend
from tzrec.prompt.assembler import PromptAssembler as ReferenceAssembler
from tzrec.prompt.frontend import PromptAssembler, PromptFrontEnd, SlotTable, mix64
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
from tzrec.utils.test_util import make_test_dir

_BASE_VOCAB = 1000
_CODEBOOK = (4, 4, 4)
_LEVEL_OFFSETS = (0, 4, 8)


def _sid_space() -> ResolvedSidSpace:
    return ResolvedSidSpace(
        codebook=_CODEBOOK,
        num_levels=3,
        base_vocab_size=_BASE_VOCAB,
        level_offsets=_LEVEL_OFFSETS,
        band_lo=tuple(_BASE_VOCAB + o for o in _LEVEL_OFFSETS),
        band_hi=tuple(
            _BASE_VOCAB + o + c - 1 for o, c in zip(_LEVEL_OFFSETS, _CODEBOOK)
        ),
        target_vocab_size=_BASE_VOCAB + 13,
        sentinel_token_id=_BASE_VOCAB + 12,
        eos_token_id=2,
        pad_token_id=3,
        bundle_uuid="test-bundle",
    )


def _slot(slot_id, name, fill, sequence=True, members=None, width=None):
    return SlotSeg(
        slot_id=slot_id,
        name=name,
        feature_names=tuple(members or (name,)),
        group_type=(
            FeatureGroupType.JAGGED_SEQUENCE if sequence else FeatureGroupType.DEEP
        ),
        output_key=".sequence" if sequence else "",
        fill=fill,
        width=width
        or (Width(WidthKind.BOUNDED, 30) if sequence else Width(WidthKind.STATIC, 1)),
    )


def _plan(segments, response_segments=(), projected=()):
    return PromptPlan(
        segments=segments,
        response_segments=response_segments,
        max_length=256,
        max_total_length=None,
        max_holes=30,
        logits_suffix_len=4,
        static_prefix_len=2,
        projected_slots=projected,
        slot_index={seg.name: i for i, seg in enumerate(projected)},
    )


def _tensors(raw):
    return {key: torch.from_numpy(np.asarray(value)) for key, value in raw.items()}


class MixTest(unittest.TestCase):
    def test_mix64_matches_a_host_reference(self):
        """The mixer's masked shifts must reproduce SplitMix64 on negatives."""

        def reference(value: int) -> int:
            mask = (1 << 64) - 1
            z = (value * 0x9E3779B97F4A7C15) & mask
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z = z ^ (z >> 31)
            return z - (1 << 64) if z >= 1 << 63 else z

        values = [0, 1, -1, 2**31, -(2**31), 123456789, -987654321]
        got = mix64(torch.tensor(values, dtype=torch.int64)).tolist()
        self.assertEqual(got, [reference(v) for v in values])

    def test_output_keys_match_the_module_constants(self):
        """``forward`` writes literals; they must equal the exported names."""
        module = PromptAssembler(_plan((Static((7,)),)))
        keys = set(module({"batch_size": torch.tensor(2)}).keys())
        self.assertEqual(
            keys,
            {
                frontend.OUT_INPUT_IDS,
                frontend.OUT_CU_SEQLENS,
                frontend.OUT_HOLE_POSITIONS,
                frontend.OUT_HOLE_KEYS,
                frontend.OUT_HOLE_SLOT_COUNTS,
                frontend.OUT_RESPONSE_LENGTHS,
            },
        )


class WalkTest(unittest.TestCase):
    def setUp(self):
        self.hist = _slot(0, "hist", FillMode.INLINE)
        self.beh = _slot(1, "beh", FillMode.PROJECTED)
        self.answer = _slot(
            2, "answer", FillMode.INLINE, width=Width(WidthKind.STATIC, 3)
        )
        self.plan = _plan(
            segments=(Static((10, 11)), self.hist, Static((12,)), self.beh),
            response_segments=(self.answer,),
            projected=(self.beh,),
        )
        # a SID history as a multi-value sequence: items per row, codes per item
        self.raw = {
            "hist.values": np.array([0, 4, 8, 1, 5, 9, 2, 6, 10], dtype=np.int64),
            "hist.lengths": np.array([2, 1], dtype=np.int64),
            "hist.key_lengths": np.array([3, 3, 3], dtype=np.int64),
            "beh.values": np.array([7, 8, 9, 21, 22], dtype=np.int64),
            "beh.lengths": np.array([3, 2], dtype=np.int64),
            "answer.values": np.array([3, 7, 11, 0, 4, 8], dtype=np.int64),
            "answer.lengths": np.array([1, 1], dtype=np.int64),
            "answer.key_lengths": np.array([3, 3], dtype=np.int64),
        }
        self.module = PromptAssembler(
            self.plan,
            _sid_space(),
            features_are_multi_valued={"hist": True, "answer": True},
            plan_hash="a1b2c3d4e5f60718",
        )

    def _assert_matches_reference(self, module, raw):
        out = module(_tensors(raw))
        reference = ReferenceAssembler(self.plan, _sid_space()).forward(raw)
        for key, reference_key in (
            ("input_ids", "prompt_input_ids"),
            ("cu_seqlens", "prompt_cu_seqlens"),
            ("hole_positions", "prompt_hole_positions"),
            ("response_lengths", "prompt_response_lengths"),
        ):
            self.assertEqual(out[key].tolist(), reference[reference_key].tolist(), key)
        return out

    def test_walk_matches_the_reference_implementation(self):
        """The tensor walk and the collator's walk are one specification."""
        self._assert_matches_reference(self.module, self.raw)

    def test_walk_matches_the_reference_on_the_flat_layout(self):
        """A history stored one code per position walks the same way."""
        raw = {
            "hist.values": self.raw["hist.values"],
            "hist.lengths": np.array([6, 3], dtype=np.int64),
            "beh.values": self.raw["beh.values"],
            "beh.lengths": self.raw["beh.lengths"],
            "answer.values": self.raw["answer.values"],
            "answer.lengths": np.array([3, 3], dtype=np.int64),
        }
        module = PromptAssembler(self.plan, _sid_space(), plan_hash="a1b2c3d4e5f60718")
        self._assert_matches_reference(module, raw)

    def test_holes_land_on_sentinels(self):
        """Every recorded hole is a sentinel and every sentinel is recorded."""
        out = self.module(_tensors(self.raw))
        sentinel = _sid_space().sentinel_token_id
        holes = out["hole_positions"]
        self.assertTrue(bool(torch.all(out["input_ids"][holes] == sentinel)))
        self.assertEqual(
            int(torch.sum(out["input_ids"] == sentinel)), int(holes.numel())
        )
        self.assertEqual(out["hole_slot_counts"].tolist(), [5])

    def test_scripting_preserves_every_output(self):
        """The artifact and the eager module are the same function."""
        batch = _tensors(self.raw)
        eager = self.module(batch)
        scripted = torch.jit.script(self.module)(batch)
        for key, value in eager.items():
            self.assertTrue(torch.equal(scripted[key], value), key)

    def test_inline_without_a_sid_space_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no sid_space"):
            PromptAssembler(self.plan)


class FoldTest(unittest.TestCase):
    def setUp(self):
        self.beh = _slot(0, "beh", FillMode.PROJECTED)
        self.plan = _plan(segments=(self.beh,), projected=(self.beh,))
        self.module = PromptAssembler(
            self.plan, _sid_space(), plan_hash="a1b2c3d4e5f60718"
        )

    def _keys(self, values, lengths):
        return self.module(
            _tensors(
                {
                    "beh.values": np.array(values, dtype=np.int64),
                    "beh.lengths": np.array(lengths, dtype=np.int64),
                }
            )
        )["hole_keys"]

    def test_equal_content_folds_equal(self):
        """A key is a function of the hole's inputs and nothing else."""
        self.assertTrue(
            torch.equal(self._keys([5, 6, 7], [3]), self._keys([5, 6, 7], [3]))
        )

    def test_different_content_folds_apart(self):
        """The whole point: a changed input must not reuse the cached KV."""
        first = self._keys([5, 6, 7], [3])
        second = self._keys([5, 6, 8], [3])
        self.assertEqual(first[:2].tolist(), second[:2].tolist())
        self.assertNotEqual(int(first[2]), int(second[2]))

    def test_the_plan_hash_salts_the_keys(self):
        """Two plans must not cross-match in a shared prefix cache."""
        other = PromptAssembler(self.plan, _sid_space(), plan_hash="ffffffffffffffff")
        batch = _tensors(
            {
                "beh.values": np.array([5, 6, 7], dtype=np.int64),
                "beh.lengths": np.array([3], dtype=np.int64),
            }
        )
        self.assertFalse(
            torch.equal(self.module(batch)["hole_keys"], other(batch)["hole_keys"])
        )

    def test_a_permuted_multi_value_item_does_not_collide(self):
        """``[a, b, c]`` and ``[c, b, a]`` are different items, in different bands."""
        module = PromptAssembler(
            self.plan,
            _sid_space(),
            features_are_multi_valued={"beh": True},
            plan_hash="a1b2c3d4e5f60718",
        )

        def keys(values):
            return module(
                _tensors(
                    {
                        "beh.values": np.array(values, dtype=np.int64),
                        "beh.lengths": np.array([1], dtype=np.int64),
                        "beh.key_lengths": np.array([3], dtype=np.int64),
                    }
                )
            )["hole_keys"]

        self.assertNotEqual(keys([1, 5, 9]).tolist(), keys([9, 5, 1]).tolist())

    def test_two_members_exchanging_values_do_not_collide(self):
        """Without the member index a two-member slot is order-blind."""
        slot = _slot(0, "pair", FillMode.PROJECTED, members=("a", "b"))
        plan = _plan(segments=(slot,), projected=(slot,))
        module = PromptAssembler(plan, _sid_space(), plan_hash="a1b2c3d4e5f60718")

        def keys(first, second):
            return module(
                _tensors(
                    {
                        "a.values": np.array(first, dtype=np.int64),
                        "a.lengths": np.array([1], dtype=np.int64),
                        "b.values": np.array(second, dtype=np.int64),
                        "b.lengths": np.array([1], dtype=np.int64),
                    }
                )
            )["hole_keys"]

        self.assertNotEqual(keys([3], [9]).tolist(), keys([9], [3]).tolist())

    def test_a_dense_member_folds_its_bit_pattern(self):
        """A float member contributes the parsed input verbatim, so it is stable."""
        slot = _slot(0, "vec", FillMode.PROJECTED, sequence=False)
        plan = _plan(segments=(slot,), projected=(slot,))
        module = PromptAssembler(
            plan, _sid_space(), features_are_dense={"vec": True}, plan_hash="a1b2"
        )

        def keys(rows):
            return module(
                {
                    "vec.values": torch.tensor(rows, dtype=torch.float32),
                    "vec.lengths": torch.ones(len(rows), dtype=torch.int64),
                }
            )["hole_keys"]

        self.assertEqual(keys([[0.5, 1.0]]).tolist(), keys([[0.5, 1.0]]).tolist())
        self.assertNotEqual(keys([[0.5, 1.0]]).tolist(), keys([[1.0, 0.5]]).tolist())

    @unittest.skipIf(not torch.cuda.is_available(), "no GPU")
    def test_the_fold_is_bit_identical_across_devices(self):
        """Integer addition cannot depend on the order a device reduces in."""
        batch = _tensors(
            {
                "beh.values": np.arange(64, dtype=np.int64),
                "beh.lengths": np.array([32, 32], dtype=np.int64),
            }
        )
        on_cpu = self.module(batch)["hole_keys"]
        on_gpu = self.module({k: v.cuda() for k, v in batch.items()})["hole_keys"]
        self.assertTrue(torch.equal(on_cpu, on_gpu.cpu()))


class FrontEndTest(unittest.TestCase):
    def setUp(self):
        self.slot = _slot(0, "beh", FillMode.PROJECTED)
        self.plan = _plan(segments=(Static((10,)), self.slot), projected=(self.slot,))
        self.batch = _tensors(
            {
                "beh.values": np.array([1, 2, 3], dtype=np.int64),
                "beh.lengths": np.array([3], dtype=np.int64),
            }
        )

    def _table(self):
        torch.manual_seed(0)
        return SlotTable(["beh.values"], ["beh.lengths"], [""], [16], [8], True)

    def _front_end(self, tables, embed_keys=(("beh",),)):
        torch.manual_seed(1)
        assembler = PromptAssembler(self.plan, _sid_space(), plan_hash="a1b2")
        projection = torch.nn.Linear(8, 16)
        return PromptFrontEnd(
            assembler,
            [projection],
            [list(keys) for keys in embed_keys],
            tables=tables,
            vocab_hash="vocab",
            plan_hash="plan",
            bundle_uuid="test-bundle",
        )

    def test_in_module_tables_produce_one_embedding_per_hole(self):
        """Without a host lookup stage, the artifact carries the tables."""
        out = self._front_end([self._table()])(self.batch)
        self.assertEqual(tuple(out["slot_embeds"].shape), (3, 16))
        self.assertEqual(int(out["hole_positions"].numel()), 3)

    def test_both_lookup_shapes_agree_given_the_same_rows(self):
        """Where the lookup happens must not change what the model sees."""
        table = self._table()
        with_tables = self._front_end([table])
        host = self._front_end(None)
        # a host stage hands over rows and item counts, not ids and .lengths
        batch = {
            "beh.values": self.batch["beh.values"],
            "beh": table(self.batch),
            frontend.host_lengths_key("beh"): self.batch["beh.lengths"],
        }
        self.assertTrue(
            torch.allclose(
                with_tables(self.batch)["slot_embeds"], host(batch)["slot_embeds"]
            )
        )
        self.assertTrue(
            torch.equal(with_tables(self.batch)["hole_keys"], host(batch)["hole_keys"])
        )

    def test_the_device_argument_is_optional(self):
        """The processor passes a device like ScriptWrapper; sglang does not."""
        module = torch.jit.script(self._front_end([self._table()]).eval())
        implicit = module(self.batch)
        explicit = module(self.batch, torch.device("cpu"))
        for key, value in implicit.items():
            self.assertTrue(torch.equal(explicit[key], value), key)

    def test_the_artifact_reloads_with_its_identity(self):
        """A loader pairs the front-end with prompt.json by these attributes."""
        path = os.path.join(make_test_dir(), "scripted_model.pt")
        torch.jit.script(self._front_end([self._table()]).eval()).save(path)
        loaded = torch.jit.load(path)
        self.assertEqual(
            (loaded.vocab_hash, loaded.plan_hash, loaded.bundle_uuid),
            ("vocab", "plan", "test-bundle"),
        )
        out = loaded(self.batch)
        self.assertEqual(tuple(out["slot_embeds"].shape), (3, 16))

    def test_pattern_i_exports_empty_hole_streams(self):
        """With no projected slot the artifact degenerates, and still exists."""
        hist = _slot(0, "hist", FillMode.INLINE)
        plan = _plan(segments=(Static((10,)), hist))
        module = torch.jit.script(
            PromptFrontEnd(PromptAssembler(plan, _sid_space()), [], []).eval()
        )
        out = module(
            _tensors(
                {
                    "hist.values": np.array([0, 5, 10], dtype=np.int64),
                    "hist.lengths": np.array([3], dtype=np.int64),
                }
            )
        )
        self.assertEqual(out["input_ids"].tolist(), [10, 1000, 1005, 1010])
        self.assertEqual(int(out["hole_positions"].numel()), 0)
        self.assertEqual(tuple(out["slot_embeds"].shape), (0, 0))


if __name__ == "__main__":
    unittest.main()
