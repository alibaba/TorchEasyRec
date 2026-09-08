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
from torch import nn

from tzrec.prompt.assembler import HOLE_SLOT_COUNTS, PromptAssembler
from tzrec.prompt.hole_keys import PROMPT_HOLE_KEYS, HoleKeyBuilder, mix64
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
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.test_util import gpu_unavailable, mark_ci_scope


def _slot(
    name,
    feature_names=None,
    group_type=FeatureGroupType.JAGGED_SEQUENCE,
    slot_id=0,
) -> SlotSeg:
    """A PROJECTED slot over an ad-hoc plan."""
    return SlotSeg(
        slot_id=slot_id,
        name=name,
        feature_names=tuple(feature_names) if feature_names is not None else (name,),
        group_type=group_type,
        output_key=".sequence"
        if group_type == FeatureGroupType.JAGGED_SEQUENCE
        else "",
        fill=FillMode.PROJECTED,
        width=Width(WidthKind.BOUNDED, 30),
    )


def _plan(segments) -> PromptPlan:
    return PromptPlan(
        segments=tuple(segments),
        response_segments=(),
        max_length=0,
        max_total_length=None,
        max_holes=0,
        logits_suffix_len=None,
        static_prefix_len=0,
        projected_slots=tuple(s for s in segments if isinstance(s, SlotSeg)),
    )


def _tensors(raw):
    return {key: torch.as_tensor(np.asarray(value)) for key, value in raw.items()}


class MixTest(unittest.TestCase):
    def test_mix64_matches_a_host_reference(self) -> None:
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


class HoleKeyBuilderTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _plan((_slot("beh"),))
        self.module = HoleKeyBuilder(self.plan)

    def _keys(self, values, lengths):
        return self.module(
            _tensors(
                {
                    "beh.values": np.array(values, dtype=np.int64),
                    "beh.lengths": np.array(lengths, dtype=np.int64),
                }
            )
        )

    def test_equal_content_folds_equal(self) -> None:
        """A key is a function of the hole's inputs and nothing else."""
        self.assertTrue(
            torch.equal(self._keys([5, 6, 7], [3]), self._keys([5, 6, 7], [3]))
        )

    def test_different_content_folds_apart(self) -> None:
        """The whole point: a changed input must not reuse the cached KV."""
        first = self._keys([5, 6, 7], [3])
        second = self._keys([5, 6, 8], [3])
        self.assertEqual(first.dtype, torch.int64)
        self.assertEqual(first[:2].tolist(), second[:2].tolist())
        self.assertNotEqual(int(first[2]), int(second[2]))

    def test_keys_follow_the_assemblers_hole_order(self) -> None:
        """Projected occurrence first, then sample, like ``hole_positions``."""
        a, b = _slot("a", slot_id=0), _slot("b", slot_id=1)
        batch = _tensors(
            {
                "a.values": np.array([1, 2, 3], dtype=np.int64),
                "a.lengths": np.array([1, 2], dtype=np.int64),
                "b.values": np.array([4, 5, 6], dtype=np.int64),
                "b.lengths": np.array([2, 1], dtype=np.int64),
            }
        )
        keys = HoleKeyBuilder(_plan((a, Static((7,)), b)))(batch)
        counts = PromptAssembler(_plan((a, Static((7,)), b)), _SID_SPACE)(batch)[
            HOLE_SLOT_COUNTS
        ]
        self.assertEqual(counts.tolist(), [3, 3])
        self.assertEqual(keys.numel(), 6)
        self.assertTrue(torch.equal(keys[:3], HoleKeyBuilder(_plan((a,)))(batch)))
        self.assertTrue(torch.equal(keys[3:], HoleKeyBuilder(_plan((b,)))(batch)))

    def test_two_slots_holding_the_same_id_do_not_collide(self) -> None:
        """The slot id salts the key, so the same value in another slot differs."""
        batch = _tensors(
            {
                "beh.values": np.array([5], dtype=np.int64),
                "beh.lengths": np.array([1], dtype=np.int64),
            }
        )
        other = HoleKeyBuilder(_plan((_slot("beh", slot_id=1),)))
        self.assertFalse(torch.equal(self.module(batch), other(batch)))

    def test_a_permuted_multi_value_item_does_not_collide(self) -> None:
        """``[a, b, c]`` and ``[c, b, a]`` are different items, in different bands."""

        def keys(values):
            return self.module(
                _tensors(
                    {
                        "beh.values": np.array(values, dtype=np.int64),
                        "beh.lengths": np.array([1], dtype=np.int64),
                        "beh.key_lengths": np.array([3], dtype=np.int64),
                    }
                )
            )

        self.assertNotEqual(keys([1, 5, 9]).tolist(), keys([9, 5, 1]).tolist())

    def test_two_members_exchanging_values_do_not_collide(self) -> None:
        """Without the member index a two-member slot is order-blind."""
        module = HoleKeyBuilder(_plan((_slot("pair", feature_names=("a", "b")),)))

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
            )

        self.assertNotEqual(keys([3], [9]).tolist(), keys([9], [3]).tolist())

    def test_a_dense_member_folds_its_bit_pattern(self) -> None:
        """A float member contributes the parsed input verbatim, so it is stable."""
        module = HoleKeyBuilder(
            _plan((_slot("vec", group_type=FeatureGroupType.DEEP),))
        )

        def keys(rows):
            return module({"vec.values": torch.tensor(rows, dtype=torch.float32)})

        self.assertEqual(keys([[0.5, 1.0]]).tolist(), keys([[0.5, 1.0]]).tolist())
        self.assertNotEqual(keys([[0.5, 1.0]]).tolist(), keys([[1.0, 0.5]]).tolist())
        self.assertEqual(keys([[0.5, 1.0], [0.5, 1.0]]).numel(), 2)

    def test_a_dense_sequence_member_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "no per-item boundary"):
            self.module(
                {
                    "beh.values": torch.tensor([[0.5], [1.0]]),
                    "beh.lengths": torch.tensor([2]),
                }
            )

    def test_no_projected_slot_folds_nothing(self) -> None:
        keys = HoleKeyBuilder(_plan((Static((7,)),)))({"batch_size": torch.tensor(2)})
        self.assertEqual(keys.numel(), 0)
        self.assertEqual(keys.dtype, torch.int64)
        self.assertEqual(PROMPT_HOLE_KEYS, "prompt_hole_keys")

    def test_scripting_preserves_the_keys(self) -> None:
        """The exported front-end folds exactly what the eager module does."""
        batch = _tensors(
            {
                "beh.values": np.array([7, 8, 9, 21, 22], dtype=np.int64),
                "beh.lengths": np.array([3, 2], dtype=np.int64),
            }
        )
        scripted = torch.jit.script(self.module)
        self.assertTrue(torch.equal(scripted(batch), self.module(batch)))

    def test_is_an_fx_leaf(self) -> None:
        """Export traces the wrapper first; the fold must stay one opaque node."""

        class Wrapper(nn.Module):
            def __init__(self, builder):
                super().__init__()
                self.builder = builder

            def forward(self, batch):
                return self.builder(batch)

        batch = _tensors(
            {
                "beh.values": np.array([7, 8, 9], dtype=np.int64),
                "beh.lengths": np.array([3], dtype=np.int64),
            }
        )
        traced = symbolic_trace(Wrapper(self.module))
        self.assertTrue(
            any(
                n.op == "call_module" and n.target == "builder"
                for n in traced.graph.nodes
            )
        )
        self.assertTrue(torch.equal(traced(batch), self.module(batch)))

    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_the_fold_is_bit_identical_across_devices(self) -> None:
        """Integer addition cannot depend on the order a device reduces in."""
        batch = _tensors(
            {
                "beh.values": np.arange(64, dtype=np.int64),
                "beh.lengths": np.array([32, 32], dtype=np.int64),
            }
        )
        on_cpu = self.module(batch)
        on_gpu = self.module({k: v.cuda() for k, v in batch.items()})
        self.assertTrue(torch.equal(on_cpu, on_gpu.cpu()))


_SID_SPACE = ResolvedSidSpace(
    codebook=(4, 4, 4),
    num_levels=3,
    base_vocab_size=1000,
    level_offsets=(0, 4, 8),
    band_lo=(1000, 1004, 1008),
    band_hi=(1003, 1007, 1011),
    target_vocab_size=1152,
    sentinel_token_id=1099,
    eos_token_id=2,
    pad_token_id=3,
)


if __name__ == "__main__":
    unittest.main()
