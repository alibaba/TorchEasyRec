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
from types import SimpleNamespace
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss

from tzrec.datasets.utils import Batch
from tzrec.features.feature import FgMode, create_features
from tzrec.models.genrec_causal_lm_model import GenRecCausalLMModel
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
    PromptAssembler,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos import feature_pb2
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import (
    create_genrec_test_model,
    create_genrec_test_tokenizer,
    make_test_dir,
    parameterized_name_func,
)


def _compiled_prompt(test_dir: str) -> CompiledPrompt:
    """Compile the genrec test prompt without building a backbone.

    ``create_genrec_test_model`` also constructs the LM. The decode-schedule
    tests only read the SID space, so they compile the prompt on its own and
    keep ``_read_beam_config`` reachable without an HF backbone.

    Args:
        test_dir (str): scratch directory for the tokenizer.

    Returns:
        CompiledPrompt: the prompt over the ``(4, 4, 4)`` codebook.
    """
    features = create_features(
        [
            feature_pb2.FeatureConfig(
                sequence_raw_feature=feature_pb2.RawFeature(
                    feature_name="hist", expression="user:hist"
                )
            )
        ],
        fg_mode=FgMode.FG_NONE,
    )
    prompt_config = PromptConfig(
        tokenizer_path=create_genrec_test_tokenizer(os.path.join(test_dir, "tok.json")),
        prompt="History : {{hist}} . Predict :",
        response="{{answer}}",
    )
    prompt_config.sid_space.codebook.extend([4, 4, 4])
    return compile_prompt(prompt_config, features, ["answer"])


class LeftPadPackedInputsTest(unittest.TestCase):
    """The one adapter where padding lives."""

    def test_packs_rows_of_different_lengths(self) -> None:
        embeds = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        cu = torch.tensor([0, 4, 9])
        batch = Batch(
            additional_infos={
                CU_SEQLENS: cu,
                MAX_SEQLEN: torch.tensor(7),
            }
        )
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
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
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _CapturingLM()
        model._attn_impl = "sdpa"
        model._ignore_index = -7
        model._prompt = SimpleNamespace(
            prompt_plan=SimpleNamespace(logits_suffix_len=4)
        )
        embeds = torch.arange(72, dtype=torch.float32).reshape(12, 6)
        input_ids = torch.arange(100, 112)
        batch = Batch(
            additional_infos={
                CU_SEQLENS: torch.tensor([0, 5, 12]),
                INPUT_IDS: input_ids,
                MAX_SEQLEN: torch.tensor(7),
                RESPONSE_LENGTHS: torch.tensor([3, 2]),
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

    def test_a_row_shorter_than_the_window_is_rejected(self) -> None:
        """A row under ``logits_suffix_len`` would index into its neighbour."""
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _CapturingLM()
        model._attn_impl = "sdpa"
        model._ignore_index = -7
        model._prompt = SimpleNamespace(
            prompt_plan=SimpleNamespace(logits_suffix_len=4)
        )
        batch = Batch(
            additional_infos={
                CU_SEQLENS: torch.tensor([0, 3, 12], dtype=torch.int32),
                INPUT_IDS: torch.arange(100, 112),
                MAX_SEQLEN: torch.tensor(9),
                RESPONSE_LENGTHS: torch.tensor([3, 3]),
            }
        )

        with self.assertRaisesRegex(ValueError, "at least logits_suffix_len"):
            model._forward(torch.zeros(12, 6), batch)

    def test_fp32_without_autocast_is_rejected_on_the_flash_path(self) -> None:
        """The flash kernel takes bf16/fp16 only; sdpa is happy in fp32."""
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _CapturingLM()
        model._attn_impl = "flash_attention_2"
        model._ignore_index = -7
        model._prompt = SimpleNamespace(
            prompt_plan=SimpleNamespace(logits_suffix_len=4)
        )
        batch = Batch(
            additional_infos={
                CU_SEQLENS: torch.tensor([0, 5, 12], dtype=torch.int32),
                INPUT_IDS: torch.arange(100, 112),
                MAX_SEQLEN: torch.tensor(7),
                RESPONSE_LENGTHS: torch.tensor([3, 2]),
            }
        )

        with self.assertRaisesRegex(ValueError, "needs bf16 or fp16"):
            model._forward(torch.zeros(12, 6, dtype=torch.float32), batch)

        model._attn_impl = "sdpa"
        logits, _ = model._forward(torch.zeros(12, 6, dtype=torch.float32), batch)
        self.assertEqual(logits.shape, (2, 4, 5))

    def test_loss_and_gradients_cover_only_valid_response_pairs(self) -> None:
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
        nn.Module.__init__(model)
        model.lm = _DifferentiableLM(hidden_size=6, vocab_size=32)
        model._attn_impl = "sdpa"
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
                CU_SEQLENS: torch.tensor([0, 5, 12]),
                INPUT_IDS: input_ids,
                MAX_SEQLEN: torch.tensor(7),
                RESPONSE_LENGTHS: torch.tensor([3, 2]),
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


class GenRecCausalLMModelTest(unittest.TestCase):
    """The decode schedule and the training forward, both subclass-owned."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.compiled_prompt = _compiled_prompt(self.test_dir)

    def _beam_model(
        self, beam_widths=(2, 2, 2), num_return_sequences=2
    ) -> GenRecCausalLMModel:
        model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
        nn.Module.__init__(model)
        model._prompt = self.compiled_prompt
        common = GenRecModelConfig(num_return_sequences=num_return_sequences)
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

    def test_training_forward_builds_no_cache(self) -> None:
        model, compiled_prompt = create_genrec_test_model(self.test_dir)
        batch = Batch()
        batch.additional_infos.update(
            PromptAssembler(compiled_prompt.prompt_plan, compiled_prompt.sid_space)(
                {
                    # offset SID codes for the (4, 4, 4) codebook
                    "hist.values": torch.tensor([0, 5, 10]),
                    "hist.lengths": torch.tensor([3]),
                    "answer.values": torch.tensor([1, 6, 11]),
                    "answer.lengths": torch.tensor([3]),
                }
            )
        )
        inner = model.lm.model.forward

        with mock.patch.object(model.lm.model, "forward", side_effect=inner) as spy:
            model.predict(batch)

        self.assertIs(spy.call_args.kwargs["use_cache"], False)


if __name__ == "__main__":
    unittest.main()
