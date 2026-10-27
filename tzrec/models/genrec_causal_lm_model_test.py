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

import json
import os
import time
import unittest
from types import SimpleNamespace
from typing import Optional, Sequence
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
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
from tzrec.utils.logging_util import logger
from tzrec.utils.test_util import (
    GENREC_ANSWER_CODES,
    GENREC_HIST_CODES,
    GENREC_LONG_HIST_CODES,
    create_genrec_test_model,
    create_genrec_test_prompt,
    flash_attn_unavailable,
    make_test_dir,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


class PaddedForwardTest(unittest.TestCase):
    """The layout every kernel but the varlen one is given."""

    def test_a_padded_width_under_the_window_is_rejected(self) -> None:
        """``logits_to_keep`` would return fewer columns than there are labels."""
        model = _stub_model()
        model._attn_kernel = "sdpa"
        batch = _varlen_batch(
            cu_seqlens=(0, 3, 6),
            max_seqlen=3,
            response_lengths=(2, 2),
        )

        with self.assertRaisesRegex(ValueError, "no sample reaches the supervised"):
            model._forward(torch.zeros(6, 6), batch)

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

        padded, mask = model._left_pad(embeds, batch)

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


def _stub_model(lm: Optional[nn.Module] = None) -> GenRecCausalLMModel:
    """A model with just the attributes ``_forward`` reads, over a stub LM.

    Defaults to the varlen kernel: these cases are about the packed layout,
    which is the branch only that kernel takes.
    """
    model = GenRecCausalLMModel.__new__(GenRecCausalLMModel)
    nn.Module.__init__(model)
    model.lm = _CapturingLM() if lm is None else lm
    model._attn_kernel = "flash_attention_2"
    model._ignore_index = -7
    model._prompt = SimpleNamespace(prompt_plan=SimpleNamespace(logits_suffix_len=4))
    return model


def _varlen_batch(
    cu_seqlens: Sequence[int] = (0, 5, 12),
    max_seqlen: int = 7,
    response_lengths: Sequence[int] = (3, 2),
    first_input_id: int = 100,
) -> Batch:
    """The four varlen keys ``_forward`` reads, over ``cu_seqlens[-1]`` tokens.

    ``PromptAssembler`` emits ``cu_seqlens`` as int32; int64 here is what
    makes ``_varlen_logits``' cast to int32 do visible work.
    """
    return Batch(
        additional_infos={
            CU_SEQLENS: torch.tensor(cu_seqlens),
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

    def test_a_row_the_assembler_zeroed_supervises_nothing(self) -> None:
        """A prompt-less row reaches here with ``response_lengths`` at zero.

        ``PromptAssembler`` zeroes it rather than failing the batch, so every
        column of that row's window has to mask out -- including the ones whose
        ``keep`` indices fall in the row before it.
        """
        model = _stub_model()
        batch = _varlen_batch(
            cu_seqlens=(0, 3, 12),
            max_seqlen=9,
            response_lengths=(0, 3),
        )

        _, labels = model._forward(torch.zeros(12, 6), batch)

        self.assertEqual(labels[0].tolist(), [-7, -7, -7, -7])
        self.assertEqual(labels[1].tolist(), [-7, 109, 110, 111])

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


# a second answer, and a rewrite of GENREC_HIST_CODES that keeps its width
_OTHERGENREC_ANSWER_CODES = [2, 7, 8]
_REWRITTENGENREC_HIST_CODES = [3, 7, 11]


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


# the two rows every layout case runs: different histories, different answers
_TWO_ROWS = (
    [GENREC_HIST_CODES, GENREC_LONG_HIST_CODES],
    [GENREC_ANSWER_CODES, _OTHERGENREC_ANSWER_CODES],
)


class LayoutInvarianceTest(unittest.TestCase):
    """Neither batching nor the kernel may change what a row scores.

    Two invariants, one fixture. Row isolation: a row must not attend into the
    row before it, which the kernels avoid by different means -- the varlen
    kernel is handed ``cu_seq_lens``, every other kernel left-padded rows.
    Rewriting row 0 to different codes of the same width leaves row 1 at the
    same offsets, so its logits have to stay bit identical either way. Layout
    equivalence: the two arms must then agree with each other, which comparing
    each against its own solo runs never establishes.
    """

    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def _eval_model(
        self,
        model_type: str,
        attn_kernel: str,
        device: torch.device,
        lm_parameter_dtype: "GenRecModelConfig.ParamDtype",
    ):
        """The same backbone from the same seed, under one kernel."""
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            model_type=model_type,
            attn_kernel=attn_kernel,
            lm_parameter_dtype=lm_parameter_dtype,
            init_seed=0,
        )
        return model.to(device).eval(), compiled_prompt

    def _assert_rows_match_solo_runs(
        self,
        model_type: str,
        attn_kernel: str,
        device: torch.device,
        lm_parameter_dtype: "GenRecModelConfig.ParamDtype",
    ) -> None:
        """Run the isolation invariant under one kernel."""
        model, compiled_prompt = self._eval_model(
            model_type, attn_kernel, device, lm_parameter_dtype
        )
        hist_rows, answer_rows = _TWO_ROWS
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
                    [_REWRITTENGENREC_HIST_CODES, hist_rows[1]],
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

    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    def test_sdpa_rows_match_solo_runs(self, model_type: str) -> None:
        self._assert_rows_match_solo_runs(
            model_type,
            "sdpa",
            torch.device("cpu"),
            GenRecModelConfig.FP32,
        )

    # mark_ci_scope must sit below expand: expand returns None, so tagging
    # above it raises at import
    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    @unittest.skipIf(*nv_gpu_unavailable)
    @unittest.skipIf(*flash_attn_unavailable)
    @mark_ci_scope("gpu")
    def test_flash_rows_match_solo_runs(self, model_type: str) -> None:
        self._assert_rows_match_solo_runs(
            model_type,
            "flash_attention_2",
            torch.device("cuda"),
            # the flash kernel takes fp16/bf16 only, and this carries no autocast
            GenRecModelConfig.BF16,
        )

    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    @unittest.skipIf(*nv_gpu_unavailable)
    @unittest.skipIf(*flash_attn_unavailable)
    @mark_ci_scope("gpu")
    def test_the_packed_and_padded_arms_agree(self, model_type: str) -> None:
        """The kernel is a speed choice, so it must not change the result.

        The two cases above each compare a kernel against solo runs of itself,
        which pins neither arm to the other. Both run BF16 here because the
        flash kernel takes no fp32.
        """
        device = torch.device("cuda")
        outputs = []
        for attn_kernel in (
            "sdpa",
            "flash_attention_2",
        ):
            model, compiled_prompt = self._eval_model(
                model_type, attn_kernel, device, GenRecModelConfig.BF16
            )
            batch = _packed_batch(compiled_prompt, *_TWO_ROWS).to(device)
            with torch.no_grad():
                outputs.append(model.predict(batch))

        padded, packed = outputs
        # the same absolute window either way, so the labels are built once and
        # only the logits path differs
        torch.testing.assert_close(packed["labels"], padded["labels"])
        torch.testing.assert_close(
            packed["logits"], padded["logits"], atol=1e-2, rtol=1e-2
        )


@mark_ci_scope("h20", "gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class MultiTargetTrainingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model, self.compiled = create_genrec_test_model(
            make_test_dir(), attn_kernel="sdpa", init_seed=17
        )
        config = self.model.lm.config
        config.num_attention_heads = 2
        config.num_key_value_heads = 1
        config.head_dim = 16
        self.model.lm = AutoModelForCausalLM.from_config(
            config, attn_implementation="sdpa"
        )
        self.model = self.model.cuda().eval()
        self.histories = [GENREC_HIST_CODES, GENREC_LONG_HIST_CODES]
        self.answers = [
            GENREC_ANSWER_CODES + _OTHERGENREC_ANSWER_CODES,
            GENREC_ANSWER_CODES,
        ]

    def _batch(self, histories=None, answers=None, predict=False):
        histories = self.histories if histories is None else histories
        answers = self.answers if answers is None else answers
        parsed = {
            "hist.values": torch.tensor([v for row in histories for v in row]),
            "hist.lengths": torch.tensor([len(row) for row in histories]),
        }
        if not predict:
            parsed.update(
                {
                    "answer.values": torch.tensor([v for row in answers for v in row]),
                    "answer.lengths": torch.tensor([len(row) for row in answers]),
                }
            )
        batch = Batch()
        batch.additional_infos.update(
            PromptAssembler(
                self.compiled.prompt_plan,
                self.compiled.sid_space,
                include_response=not predict,
            )(parsed)
        )
        return batch.to(torch.device("cuda"))

    def _reference(self):
        batch = self._batch(
            [self.histories[0], self.histories[0], self.histories[1]],
            [GENREC_ANSWER_CODES, _OTHERGENREC_ANSWER_CODES, GENREC_ANSWER_CODES],
        )
        predictions = self.model.predict(batch)
        logits = predictions["logits"][:, :-1]
        labels = predictions["labels"][:, 1:]
        losses = (
            nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="none",
            )
            .reshape(3, 3)
            .mean(-1)
        )
        return logits, labels, ((losses[0] + losses[1]) / 2 + losses[2]) / 2

    def test_logits_loss_and_lm_gradients_match_independent_targets(self) -> None:
        reference_logits, reference_labels, reference_loss = self._reference()
        reference_loss.backward()
        gradients = {
            name: parameter.grad.clone()
            for name, parameter in self.model.named_parameters()
            if parameter.grad is not None
        }
        self.model.zero_grad(set_to_none=True)
        batch = self._batch()
        predictions = self.model.predict(batch)
        loss = self.model.loss(predictions, batch)["ce_loss"]
        torch.testing.assert_close(
            predictions["logits"], reference_logits, atol=2e-5, rtol=2e-4
        )
        torch.testing.assert_close(predictions["labels"], reference_labels)
        torch.testing.assert_close(loss, reference_loss, atol=2e-5, rtol=2e-4)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(
                    parameter.grad, gradients[name], atol=3e-5, rtol=3e-4, msg=name
                )

    def test_other_branch_and_other_parent_are_isolated(self) -> None:
        baseline = self.model.predict(self._batch())["logits"].detach()
        rewritten = self.model.predict(
            self._batch(
                answers=[
                    _REWRITTENGENREC_HIST_CODES + _OTHERGENREC_ANSWER_CODES,
                    GENREC_ANSWER_CODES,
                ]
            )
        )["logits"]
        torch.testing.assert_close(rewritten[1:], baseline[1:], atol=0, rtol=0)

    def test_target_order_does_not_change_parent_loss(self) -> None:
        batch = self._batch()
        first = self.model.loss(self.model.predict(batch), batch)["ce_loss"]
        batch = self._batch(
            answers=[
                _OTHERGENREC_ANSWER_CODES + GENREC_ANSWER_CODES,
                GENREC_ANSWER_CODES,
            ]
        )
        second = self.model.loss(self.model.predict(batch), batch)["ce_loss"]
        torch.testing.assert_close(first, second)

    def test_inference_and_checkpoint_structure_are_unchanged(self) -> None:
        batch = self._batch(predict=True)
        self.model.set_is_inference(True)
        with torch.no_grad():
            before = self.model.predict(batch)["generated_sids"]
        state_shapes = {
            name: value.shape for name, value in self.model.state_dict().items()
        }
        previous_attention = self.model.lm.config._attn_implementation
        self.model.set_is_inference(False)
        self.model.predict(self._batch())
        self.assertEqual(self.model.lm.config._attn_implementation, previous_attention)
        self.assertEqual(
            state_shapes,
            {name: value.shape for name, value in self.model.state_dict().items()},
        )
        self.model.set_is_inference(True)
        with torch.no_grad():
            after = self.model.predict(batch)["generated_sids"]
        torch.testing.assert_close(after, before, atol=0, rtol=0)

    def test_attention_backend_is_restored_after_forward_failure(self) -> None:
        previous_attention = self.model.lm.config._attn_implementation
        with mock.patch.object(
            self.model.lm.base_model,
            "forward",
            side_effect=RuntimeError("test failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.model.predict(self._batch())
        self.assertEqual(self.model.lm.config._attn_implementation, previous_attention)

    def test_incomplete_sid_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of codebook levels"):
            self.model.predict(self._batch(answers=[[0, 4, 8, 1], GENREC_ANSWER_CODES]))

    def test_gradient_checkpointing_is_rejected_for_shared_training(self) -> None:
        self.model.lm.gradient_checkpointing_enable()
        with self.assertRaisesRegex(ValueError, "gradient checkpointing"):
            self.model.predict(self._batch())

    @unittest.skipUnless(
        os.environ.get("TZREC_GENREC_BENCHMARK_BACKBONE"),
        "Optional benchmark requires a local backbone configuration",
    )
    def test_shared_prefix_benchmark(self) -> None:
        config = AutoConfig.from_pretrained(
            os.environ["TZREC_GENREC_BENCHMARK_BACKBONE"]
        )
        self.model.lm = (
            AutoModelForCausalLM.from_config(
                config, attn_implementation="sdpa", dtype=torch.float32
            )
            .cuda()
            .eval()
        )
        histories = [GENREC_HIST_CODES * 300 for _ in range(20)]
        answers = [
            GENREC_ANSWER_CODES + _OTHERGENREC_ANSWER_CODES
            if index < 6
            else GENREC_ANSWER_CODES
            for index in range(20)
        ]
        shared_batch = self._batch(histories, answers)
        expanded_histories, expanded_answers, weights = [], [], []
        for history, answer in zip(histories, answers):
            count = len(answer) // 3
            for start in range(0, len(answer), 3):
                expanded_histories.append(history)
                expanded_answers.append(answer[start : start + 3])
                weights.append(1 / count / len(histories))
        expanded_batch = self._batch(expanded_histories, expanded_answers)
        weights = torch.tensor(weights, device="cuda")
        torch.backends.cuda.enable_cudnn_sdp(False)

        def step(batch, shared):
            self.model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                predictions = self.model.predict(batch)
                if shared:
                    loss = self.model.loss(predictions, batch)["ce_loss"]
                else:
                    logits = predictions["logits"][:, :-1]
                    labels = predictions["labels"][:, 1:]
                    target_loss = (
                        nn.functional.cross_entropy(
                            logits.float().reshape(-1, logits.shape[-1]),
                            labels.reshape(-1),
                            reduction="none",
                        )
                        .reshape(-1, 3)
                        .mean(-1)
                    )
                    loss = (target_loss * weights).sum()
            torch.cuda.synchronize()
            forwarded = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize()
            finished = time.perf_counter()
            return forwarded - started, finished - forwarded, float(loss.detach())

        results = {}
        for name, batch, shared, backend in (
            ("expanded_sdpa", expanded_batch, False, "sdpa"),
            ("expanded_flex", expanded_batch, False, "flex_attention"),
            ("shared_flex", shared_batch, True, "sdpa"),
        ):
            self.model._attn_kernel = backend
            self.model.lm.set_attn_implementation(backend)
            cold_started = time.perf_counter()
            for _ in range(2):
                step(batch, shared)
            warmup_seconds = time.perf_counter() - cold_started
            torch.cuda.reset_peak_memory_stats()
            samples = [step(batch, shared) for _ in range(5)]
            results[name] = {
                "warmup_seconds": warmup_seconds,
                "forward_seconds": sum(sample[0] for sample in samples) / len(samples),
                "backward_seconds": sum(sample[1] for sample in samples) / len(samples),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "loss": samples[-1][2],
                "parents": 20,
                "targets": 26,
            }
        logger.info("GENREC_MULTI_TARGET_BENCHMARK %s", json.dumps(results))
        self.assertAlmostEqual(
            results["expanded_sdpa"]["loss"], results["shared_flex"]["loss"], delta=0.03
        )


if __name__ == "__main__":
    unittest.main()
