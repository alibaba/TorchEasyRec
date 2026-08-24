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

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import GenrecCausalLMModel
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_INPUT_IDS,
    PROMPT_MAX_SEQLEN,
    PROMPT_RESPONSE_LENGTHS,
)


class LeftPadPackedInputsTest(unittest.TestCase):
    """The one adapter where padding lives."""

    def test_packs_rows_of_different_lengths(self) -> None:
        embeds = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        cu = torch.tensor([0, 4, 9])
        input_ids = torch.tensor([1, 2, 7, 8, 3, 4, 5, 7, 8])
        response_lengths = torch.tensor([1, 2])
        ignore = -7
        batch = Batch(
            additional_infos={
                PROMPT_CU_SEQLENS: cu,
                PROMPT_INPUT_IDS: input_ids,
                PROMPT_MAX_SEQLEN: torch.tensor(7),
                PROMPT_RESPONSE_LENGTHS: response_lengths,
            }
        )
        model = GenrecCausalLMModel.__new__(GenrecCausalLMModel)
        torch.nn.Module.__init__(model)
        model._ignore_index = ignore

        padded, mask, out = model._left_pad_packed_inputs(
            embeds, batch, build_labels=True
        )

        self.assertEqual(padded.shape, (2, 7, 2))
        self.assertEqual(
            mask.tolist(),
            [[0, 0, 0, 1, 1, 1, 1], [0, 0, 1, 1, 1, 1, 1]],
        )
        torch.testing.assert_close(padded[0, 3:], embeds[:4])
        torch.testing.assert_close(padded[1, 2:], embeds[4:])
        torch.testing.assert_close(padded[0, :3], torch.zeros(3, 2))
        torch.testing.assert_close(padded[1, :2], torch.zeros(2, 2))
        torch.testing.assert_close(padded[:, -1], torch.stack([embeds[3], embeds[8]]))
        self.assertEqual(
            out.tolist(),
            [[ignore] * 6 + [8], [ignore] * 5 + [7, 8]],
        )


if __name__ == "__main__":
    unittest.main()
