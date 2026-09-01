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
from unittest import mock

import torch
from parameterized import parameterized

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import GenrecCausalLMModel
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_INPUT_IDS,
    PROMPT_MAX_SEQLEN,
    PROMPT_RESPONSE_LENGTHS,
)
from tzrec.tests.prompt_test_util import (
    _CODEBOOK,
    GenrecModelTestBase,
    offset_sid_codes,
)
from tzrec.utils.test_util import parameterized_name_func


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


class GenrecCausalLMModelTest(GenrecModelTestBase):
    """The decode schedule and the training forward, both subclass-owned."""

    @parameterized.expand(
        [
            # every request satisfiable -> the schedule is honoured verbatim
            [[2, 3, 4], [2, 3, 4]],
            # capped by band x surviving prefixes, not by what was asked
            [[6, 12, 12], [4, 12, 12]],
            # a flat schedule is just as valid
            [[2, 2, 2], [2, 2, 2]],
        ],
        name_func=parameterized_name_func,
    )
    def test_beam_widths_are_capped_once_at_init(self, beam_widths, expected) -> None:
        # the kernel takes these already capped, so the derivation lives here
        model = self._model(beam_widths=beam_widths, num_return_sequences=1)
        self.assertEqual(model._capped_widths, expected)
        space = self.compiled_prompt.sid_space
        self.assertEqual(model._bands, list(zip(space.band_lo, space.band_hi)))

    def test_rejects_a_schedule_that_does_not_match_the_codebook(self) -> None:
        with self.assertRaisesRegex(ValueError, "entries but the codebook has"):
            self._model(beam_widths=(2, 2))

    def test_rejects_a_non_positive_beam_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            self._model(beam_widths=(2, 0, 2))

    def test_beam_config_uses_final_capped_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "final capped beam width \\(4\\)"):
            self._model(
                beam_widths=(1, 1, 100),
                num_return_sequences=5,
            )

    def test_training_forward_builds_no_cache(self) -> None:
        model = self._model()
        inner = model.lm.model.forward

        with mock.patch.object(model.lm.model, "forward", side_effect=inner) as spy:
            model.predict(
                self._batch(
                    {
                        "hist.values": torch.tensor(
                            offset_sid_codes([0, 1, 2], _CODEBOOK)
                        ),
                        "hist.lengths": torch.tensor([3]),
                        "answer.values": torch.tensor(
                            offset_sid_codes([1, 2, 3], _CODEBOOK)
                        ),
                        "answer.lengths": torch.tensor([3]),
                    }
                )
            )

        self.assertIs(spy.call_args.kwargs["use_cache"], False)


if __name__ == "__main__":
    unittest.main()
