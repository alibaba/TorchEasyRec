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

import torch

from tzrec.models.prompt_generative_qwen import _unpack
from tzrec.prompt.plan import SidSpace


class UnpackTest(unittest.TestCase):
    """The one adapter where padding lives."""

    def test_packs_rows_of_different_lengths(self) -> None:
        # rows of 2 and 3 tokens, hidden size 4
        embeds = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        cu = torch.tensor([0, 2, 5])
        labels = torch.tensor([10, 11, 20, 21, 22])

        padded, mask, out = _unpack(embeds, cu, labels, max_seqlen=3, ignore_index=-100)

        self.assertEqual(padded.shape, (2, 3, 4))
        self.assertEqual(mask.tolist(), [[1, 1, 0], [1, 1, 1]])
        torch.testing.assert_close(padded[0, :2], embeds[:2])
        torch.testing.assert_close(padded[1, :3], embeds[2:])
        # the pad column is zero, and its label is ignored
        torch.testing.assert_close(padded[0, 2], torch.zeros(4))
        self.assertEqual(out.tolist(), [[10, 11, -100], [20, 21, 22]])

    def test_uses_the_given_width_not_the_observed_max(self) -> None:
        # the collator's bucket may exceed the widest row; §7.4 forbids
        # deriving the width on device
        embeds = torch.ones(3, 2)
        cu = torch.tensor([0, 1, 3])
        labels = torch.tensor([1, 2, 3])
        padded, mask, _ = _unpack(embeds, cu, labels, max_seqlen=5, ignore_index=-100)

        self.assertEqual(padded.shape, (2, 5, 2))
        self.assertEqual(mask.sum().item(), 3)

    def test_row_order_survives_the_scatter(self) -> None:
        # mask selects row-major, which must match the packing order
        embeds = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        cu = torch.tensor([0, 1, 4])
        labels = torch.zeros(4, dtype=torch.long)
        padded, _, _ = _unpack(embeds, cu, labels, max_seqlen=3, ignore_index=-100)

        self.assertEqual(padded[0, 0].item(), 1.0)
        self.assertEqual(padded[1, :, 0].tolist(), [2.0, 3.0, 4.0])

    def test_gradient_reaches_the_packed_input(self) -> None:
        embeds = torch.ones(3, 2, requires_grad=True)
        cu = torch.tensor([0, 1, 3])
        labels = torch.zeros(3, dtype=torch.long)
        padded, _, _ = _unpack(embeds, cu, labels, max_seqlen=2, ignore_index=-100)
        padded.sum().backward()

        self.assertIsNotNone(embeds.grad)
        torch.testing.assert_close(embeds.grad, torch.ones(3, 2))


class DetokenizeTest(unittest.TestCase):
    """Both shifts must come back off, in the right order."""

    def _space(self) -> SidSpace:
        return SidSpace(
            codebook=(4, 4, 4),
            num_levels=3,
            base_vocab=1000,
            level_offsets=(0, 4, 8),
            band_lo=(1000, 1004, 1008),
            band_hi=(1003, 1007, 1011),
            target_vocab=1152,
            sentinel_token_id=None,
            eos_token_id=2,
            pad_token_id=3,
        )

    def test_token_ids_become_local_codes(self) -> None:
        space = self._space()
        # one beam row: level 0 code 1, level 1 code 2, level 2 code 3
        tokens = torch.tensor([[1000 + 1, 1000 + 4 + 2, 1000 + 8 + 3]])
        offsets = torch.tensor(space.level_offsets)
        codes = (tokens - space.base_vocab - offsets).view(1, -1, 3)

        self.assertEqual(codes[0, 0].tolist(), [1, 2, 3])
        # every code lands back inside its own codebook
        self.assertTrue(bool(((codes >= 0) & (codes < 4)).all()))

    def test_a_band_edge_maps_to_the_last_code(self) -> None:
        space = self._space()
        tokens = torch.tensor([list(space.band_hi)])
        offsets = torch.tensor(space.level_offsets)
        codes = tokens - space.base_vocab - offsets
        self.assertEqual(codes[0].tolist(), [3, 3, 3])


if __name__ == "__main__":
    unittest.main()
