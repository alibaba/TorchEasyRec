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

from tzrec.prompt.assembler import HOLE_KEYS, PromptAssembler, host_lengths_key
from tzrec.prompt.frontend import SLOT_EMBEDS, PromptFrontEnd, SlotTable
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


def _slot(slot_id, name, fill):
    return SlotSeg(
        slot_id=slot_id,
        name=name,
        feature_names=(name,),
        group_type=FeatureGroupType.JAGGED_SEQUENCE,
        output_key=".sequence",
        fill=fill,
        width=Width(WidthKind.BOUNDED, 30),
    )


def _plan(segments, projected=()):
    return PromptPlan(
        segments=segments,
        response_segments=(),
        max_length=256,
        max_total_length=None,
        max_holes=30,
        logits_suffix_len=4,
        static_prefix_len=1,
        projected_slots=projected,
    )


def _tensors(raw):
    return {key: torch.from_numpy(np.asarray(value)) for key, value in raw.items()}


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
        self.assertEqual(tuple(out[SLOT_EMBEDS].shape), (3, 16))
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
            host_lengths_key("beh"): self.batch["beh.lengths"],
        }
        self.assertTrue(
            torch.allclose(
                with_tables(self.batch)[SLOT_EMBEDS], host(batch)[SLOT_EMBEDS]
            )
        )
        self.assertTrue(
            torch.equal(with_tables(self.batch)[HOLE_KEYS], host(batch)[HOLE_KEYS])
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
        self.assertEqual(tuple(out[SLOT_EMBEDS].shape), (3, 16))

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
        self.assertEqual(tuple(out[SLOT_EMBEDS].shape), (0, 0))


if __name__ == "__main__":
    unittest.main()
