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


class UnpackTest(unittest.TestCase):
    """The one adapter where padding lives."""

    def test_packs_rows_of_different_lengths(self) -> None:
        # rows of 4 and 5 tokens, with a padded width larger than either row
        embeds = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        cu = torch.tensor([0, 4, 9])
        ignore = -100
        labels = torch.tensor([ignore, ignore, 7, 8, ignore, ignore, ignore, 7, 8])

        padded, mask, out = _unpack(
            embeds, cu, labels, max_seqlen=7, ignore_index=ignore
        )

        self.assertEqual(padded.shape, (2, 7, 2))
        # pads go on the left, so every row ends on a real token
        self.assertEqual(
            mask.tolist(),
            [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]],
        )
        torch.testing.assert_close(padded[0, 3:], embeds[:4])
        torch.testing.assert_close(padded[1, 2:], embeds[4:])
        torch.testing.assert_close(padded[0, :3], torch.zeros(3, 2))
        torch.testing.assert_close(padded[1, :2], torch.zeros(2, 2))
        torch.testing.assert_close(padded[:, -1], torch.stack([embeds[3], embeds[8]]))
        self.assertEqual(out.tolist(), [[ignore] * 5 + [7, 8]] * 2)


if __name__ == "__main__":
    unittest.main()
