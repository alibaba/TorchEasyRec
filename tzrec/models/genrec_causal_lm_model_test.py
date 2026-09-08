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
from types import SimpleNamespace

import torch
from parameterized import parameterized
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import GenrecCausalLMModel
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_INPUT_IDS,
    PROMPT_MAX_SEQLEN,
    PROMPT_RESPONSE_LENGTHS,
)
from tzrec.protos.models.genrec_model_pb2 import GenrecModelConfig
from tzrec.tests.prompt_test_util import GenrecModelTestBase
from tzrec.utils.test_util import parameterized_name_func


class LeftPadPackedInputsTest(unittest.TestCase):
    """The one adapter where padding lives."""

    def test_packs_rows_of_different_lengths(self) -> None:
        embeds = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        cu = torch.tensor([0, 4, 9])
        batch = Batch(
            additional_infos={
                PROMPT_CU_SEQLENS: cu,
                PROMPT_MAX_SEQLEN: torch.tensor(7),
            }
        )
        model = GenrecCausalLMModel.__new__(GenrecCausalLMModel)
        torch.nn.Module.__init__(model)

        padded, mask = model._left_pad_packed_inputs(embeds, batch)

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


class _CapturingLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs = {}

    def forward(self, **kwargs):
        self.kwargs = kwargs
        count = kwargs["logits_to_keep"].numel()
        logits = torch.arange(
            count * 5,
            dtype=kwargs["inputs_embeds"].dtype,
            device=kwargs["inputs_embeds"].device,
        ).reshape(1, count, 5)
        return SimpleNamespace(logits=logits)


class _DifferentiableLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.loss_function = ForCausalLMLoss

    def forward(self, **kwargs):
        hidden = kwargs["inputs_embeds"].index_select(1, kwargs["logits_to_keep"])
        return SimpleNamespace(logits=self.lm_head(hidden))


class PackedForwardTest(unittest.TestCase):
    def test_passes_varlen_metadata_and_builds_per_row_labels(self) -> None:
        model = GenrecCausalLMModel.__new__(GenrecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _CapturingLM()
        model._ignore_index = -7
        model._prompt = SimpleNamespace(
            prompt_plan=SimpleNamespace(logits_suffix_len=4)
        )
        embeds = torch.arange(72, dtype=torch.float32).reshape(12, 6)
        input_ids = torch.arange(100, 112)
        batch = Batch(
            additional_infos={
                PROMPT_CU_SEQLENS: torch.tensor([0, 5, 12]),
                PROMPT_INPUT_IDS: input_ids,
                PROMPT_MAX_SEQLEN: torch.tensor(7),
                PROMPT_RESPONSE_LENGTHS: torch.tensor([3, 2]),
            }
        )

        logits, labels = model._forward(embeds, batch)

        kwargs = model.lm.kwargs
        self.assertEqual(kwargs["inputs_embeds"].shape, (1, 12, 6))
        self.assertIsNone(kwargs["attention_mask"])
        self.assertEqual(
            kwargs["position_ids"].tolist(),
            [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]],
        )
        self.assertEqual(kwargs["logits_to_keep"].tolist(), [1, 2, 3, 4, 8, 9, 10, 11])
        self.assertEqual(kwargs["cu_seq_lens_q"].dtype, torch.int32)
        torch.testing.assert_close(kwargs["cu_seq_lens_q"], kwargs["cu_seq_lens_k"])
        self.assertEqual(kwargs["max_length_q"], 7)
        self.assertEqual(kwargs["max_length_k"], 7)
        self.assertIs(kwargs["use_cache"], False)
        self.assertEqual(logits.shape, (2, 4, 5))
        self.assertEqual(
            labels.tolist(),
            [[-7, 102, 103, 104], [-7, -7, 110, 111]],
        )

    def test_loss_and_gradients_cover_only_valid_response_pairs(self) -> None:
        model = GenrecCausalLMModel.__new__(GenrecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _DifferentiableLM(hidden_size=6, vocab_size=32)
        model._ignore_index = -7
        model._prompt = SimpleNamespace(
            prompt_plan=SimpleNamespace(logits_suffix_len=4)
        )
        embeds = torch.randn(
            12, 6, generator=torch.Generator().manual_seed(1), requires_grad=True
        )
        input_ids = torch.arange(4, 16)
        batch = Batch(
            additional_infos={
                PROMPT_CU_SEQLENS: torch.tensor([0, 5, 12]),
                PROMPT_INPUT_IDS: input_ids,
                PROMPT_MAX_SEQLEN: torch.tensor(7),
                PROMPT_RESPONSE_LENGTHS: torch.tensor([3, 2]),
            }
        )

        logits, labels = model._forward(embeds, batch)
        loss = model.loss({"logits": logits, "labels": labels}, batch)["ce_loss"]
        expected = nn.functional.cross_entropy(
            torch.cat((logits[0, :3], logits[1, 1:3])),
            torch.tensor([6, 7, 8, 14, 15]),
        )
        torch.testing.assert_close(loss, expected)

        loss.backward()
        self.assertIsNotNone(embeds.grad)
        grad_norms = embeds.grad.abs().sum(dim=1)
        supervised = torch.tensor([1, 2, 3, 9, 10])
        self.assertTrue(bool(torch.all(grad_norms[supervised] > 0)))
        unsupervised = torch.ones(12, dtype=torch.bool)
        unsupervised[supervised] = False
        torch.testing.assert_close(
            grad_norms[unsupervised], torch.zeros(7), atol=0, rtol=0
        )
        weight_grad = model.lm.lm_head.weight.grad
        self.assertIsNotNone(weight_grad)
        self.assertTrue(bool(torch.isfinite(weight_grad).all()))
        self.assertGreater(float(weight_grad.abs().sum()), 0)


class GenrecCausalLMModelTest(GenrecModelTestBase):
    """The decode schedule and the training forward, both subclass-owned."""

    def _beam_model(
        self, beam_widths=(2, 2, 2), num_return_sequences=2
    ) -> GenrecCausalLMModel:
        model = GenrecCausalLMModel.__new__(GenrecCausalLMModel)
        nn.Module.__init__(model)
        model._prompt = self.compiled_prompt
        common = GenrecModelConfig(num_return_sequences=num_return_sequences)
        common.beam_widths.extend(beam_widths)
        model._read_beam_config(common)
        return model

    @parameterized.expand(
        [
            [[2, 3, 4], [2, 3, 4]],
            [[6, 12, 12], [4, 12, 12]],
            [[2, 2, 2], [2, 2, 2]],
        ],
        name_func=parameterized_name_func,
    )
    def test_beam_widths_are_capped_once_at_init(self, beam_widths, expected) -> None:
        model = self._beam_model(beam_widths=beam_widths, num_return_sequences=1)
        self.assertEqual(model._capped_widths, expected)
        space = self.compiled_prompt.sid_space
        self.assertEqual(model._bands, list(zip(space.band_lo, space.band_hi)))

    def test_rejects_a_schedule_that_does_not_match_the_codebook(self) -> None:
        with self.assertRaisesRegex(ValueError, "entries but the codebook has"):
            self._beam_model(beam_widths=(2, 2))

    def test_rejects_a_non_positive_beam_width(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be >= 1"):
            self._beam_model(beam_widths=(2, 0, 2))

    def test_beam_config_uses_final_capped_capacity(self) -> None:
        with self.assertRaisesRegex(ValueError, "final capped beam width \\(4\\)"):
            self._beam_model(
                beam_widths=(1, 1, 100),
                num_return_sequences=5,
            )


if __name__ == "__main__":
    unittest.main()
