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
from typing import Optional, Sequence

import torch
from parameterized import parameterized
from torch import nn
from transformers.loss.loss_utils import ForCausalLMLoss

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import GenRecCausalLMModel
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
    PromptAssembler,
)
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.utils.test_util import (
    create_genrec_test_model,
    create_genrec_test_prompt,
    flash_attn_unavailable,
    make_test_dir,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


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
        embeds = kwargs["inputs_embeds"]
        count = kwargs["logits_to_keep"].numel()
        return SimpleNamespace(
            logits=torch.zeros(1, count, 5, dtype=embeds.dtype, device=embeds.device)
        )


class _DifferentiableLM(nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(vocab_size=vocab_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.loss_function = ForCausalLMLoss

    def forward(self, **kwargs):
        hidden = kwargs["inputs_embeds"].index_select(1, kwargs["logits_to_keep"])
        return SimpleNamespace(logits=self.lm_head(hidden))


def _stub_model(
    lm: Optional[nn.Module] = None,
    attn_kernel: int = GenRecModelConfig.FLASH_ATTENTION_2,
) -> GenRecCausalLMModel:
    """A model with just the attributes ``_forward`` reads, over a stub LM.

    Defaults to the varlen kernel: these cases are about the packed layout,
    which is the branch only that kernel takes.
    """
    model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
    nn.Module.__init__(model)
    model.lm = _CapturingLM() if lm is None else lm
    model._attn_kernel = attn_kernel
    model._ignore_index = -7
    model._prompt = SimpleNamespace(prompt_plan=SimpleNamespace(logits_suffix_len=4))
    return model


def _varlen_batch(
    cu_seqlens: Sequence[int] = (0, 5, 12),
    max_seqlen: int = 7,
    response_lengths: Sequence[int] = (3, 2),
    first_input_id: int = 100,
    cu_dtype: torch.dtype = torch.int64,
) -> Batch:
    """The four varlen keys ``_forward`` reads, over ``cu_seqlens[-1]`` tokens.

    ``PromptAssembler`` emits ``cu_seqlens`` as int32; the int64 default is
    what makes ``_forward``'s cast to int32 observable.
    """
    return Batch(
        additional_infos={
            CU_SEQLENS: torch.tensor(cu_seqlens, dtype=cu_dtype),
            INPUT_IDS: torch.arange(first_input_id, first_input_id + cu_seqlens[-1]),
            MAX_SEQLEN: torch.tensor(max_seqlen),
            RESPONSE_LENGTHS: torch.tensor(response_lengths),
        }
    )


class PackedForwardTest(unittest.TestCase):
    def test_passes_varlen_metadata_and_builds_per_row_labels(self) -> None:
        model = _stub_model()
        embeds = torch.arange(72, dtype=torch.float32).reshape(12, 6)
        batch = _varlen_batch()

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
        model = _stub_model()
        batch = _varlen_batch(
            cu_seqlens=(0, 3, 12),
            max_seqlen=9,
            response_lengths=(3, 3),
            cu_dtype=torch.int32,
        )

        with self.assertRaisesRegex(ValueError, "at least logits_suffix_len"):
            model._forward(torch.zeros(12, 6), batch)

    def test_loss_and_gradients_cover_only_valid_response_pairs(self) -> None:
        model = _stub_model(_DifferentiableLM(hidden_size=6, vocab_size=32))
        embeds = torch.randn(
            12, 6, generator=torch.Generator().manual_seed(1), requires_grad=True
        )
        batch = _varlen_batch(first_input_id=4)

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


class GenRecCausalLMModelTest(unittest.TestCase):
    """The decode schedule and the training forward, both subclass-owned."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.compiled_prompt, _ = create_genrec_test_prompt(self.test_dir)

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


# offset SID codes for the (4, 4, 4) codebook: level_offsets[l] + code
_HIST_CODES = [0, 5, 10]
_LONG_HIST_CODES = [0, 5, 10, 3, 4, 9]
_ANSWER_CODES = [1, 6, 11]


# a second answer, and a rewrite of _HIST_CODES that keeps its width
_OTHER_ANSWER_CODES = [2, 7, 8]
_REWRITTEN_HIST_CODES = [3, 7, 11]


def _packed_batch(compiled_prompt, hist_rows, answer_rows) -> Batch:
    """Several rows in one batch, packed the way the collator packs them."""
    batch = Batch()
    batch.additional_infos.update(
        PromptAssembler(compiled_prompt.prompt_plan, compiled_prompt.sid_space)(
            {
                "hist.values": torch.tensor(
                    [code for row in hist_rows for code in row]
                ),
                "hist.lengths": torch.tensor([len(row) for row in hist_rows]),
                "answer.values": torch.tensor(
                    [code for row in answer_rows for code in row]
                ),
                "answer.lengths": torch.tensor([len(row) for row in answer_rows]),
            }
        )
    )
    return batch


class _RowIsolationCase:
    """Every row must read exactly as it does alone, whatever the layout.

    The failure this guards against is a row attending into the row before it,
    which the two kernels avoid by different means: the varlen kernel is handed
    ``cu_seq_lens``, every other kernel is handed left-padded rows. Rewriting
    row 0 to different codes of the same width leaves row 1 at the same
    offsets, so its logits have to stay bit identical either way.
    """

    device = torch.device("cpu")
    attn_kernel = GenRecModelConfig.SDPA
    lm_parameter_dtype = GenRecModelConfig.FP32

    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    def test_rows_match_solo_runs_and_backpropagate(self, model_type: str) -> None:
        device = self.device
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            model_type=model_type,
            attn_kernel=self.attn_kernel,
            lm_parameter_dtype=self.lm_parameter_dtype,
            init_seed=0,
        )
        model.to(device)
        model.eval()
        hist_rows = [_HIST_CODES, _LONG_HIST_CODES]
        answer_rows = [_ANSWER_CODES, _OTHER_ANSWER_CODES]
        packed_batch = _packed_batch(compiled_prompt, hist_rows, answer_rows).to(device)

        packed = model.predict(packed_batch)
        with torch.no_grad():
            solos = [
                model.predict(
                    _packed_batch(compiled_prompt, [hist], [answer]).to(device)
                )
                for hist, answer in zip(hist_rows, answer_rows)
            ]
            changed = model.predict(
                _packed_batch(
                    compiled_prompt,
                    [_REWRITTEN_HIST_CODES, hist_rows[1]],
                    answer_rows,
                ).to(device)
            )

        torch.testing.assert_close(
            packed["logits"],
            torch.cat([result["logits"] for result in solos]),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            packed["labels"], torch.cat([result["labels"] for result in solos])
        )
        torch.testing.assert_close(
            packed["logits"][1], changed["logits"][1], atol=0, rtol=0
        )

        loss = model.loss(packed, packed_batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        grad = model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool(torch.isfinite(grad).all()))
        self.assertGreater(float(grad.abs().sum()), 0)


class SdpaRowIsolationTest(_RowIsolationCase, unittest.TestCase):
    """The left-padded layout sdpa is given, which needs no GPU and no wheel."""


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
@unittest.skipIf(*flash_attn_unavailable)
class FlashAttentionRowIsolationTest(_RowIsolationCase, unittest.TestCase):
    """The same rows packed into one stream, through the varlen flash kernel."""

    device = torch.device("cuda")
    attn_kernel = GenRecModelConfig.FLASH_ATTENTION_2
    # the flash kernel takes fp16/bf16 only, and this arm carries no autocast
    lm_parameter_dtype = GenRecModelConfig.BF16


if __name__ == "__main__":
    unittest.main()
