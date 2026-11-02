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
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn
from transformers import AutoModelForCausalLM
from transformers.loss.loss_utils import ForCausalLMLoss

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_causal_lm_model import (
    GenRecCausalLMModel,
    _shared_prefix_attention,
)
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
    PromptAssembler,
    build_multi_target_layout,
)
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
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
    atol = 3e-5
    rtol = 3e-4

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
            predictions["logits"], reference_logits, atol=self.atol, rtol=self.rtol
        )
        torch.testing.assert_close(predictions["labels"], reference_labels)
        torch.testing.assert_close(loss, reference_loss, atol=self.atol, rtol=self.rtol)
        loss.backward()
        for name, parameter in self.model.named_parameters():
            if name in gradients:
                torch.testing.assert_close(
                    parameter.grad,
                    gradients[name],
                    atol=self.atol,
                    rtol=self.rtol,
                    msg=name,
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


@mark_ci_scope("h20", "gpu")
@unittest.skipIf(*nv_gpu_unavailable)
@unittest.skipIf(*flash_attn_unavailable)
class HybridMultiTargetTrainingTest(MultiTargetTrainingTest):
    atol = 8e-3
    rtol = 3e-2

    def setUp(self) -> None:
        super().setUp()
        config = self.model.lm.config
        config.hidden_size = 224
        config.intermediate_size = 448
        config.num_attention_heads = 14
        config.num_key_value_heads = 2
        self.model.lm = (
            AutoModelForCausalLM.from_config(
                config,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.bfloat16,
            )
            .cuda()
            .eval()
        )
        self.model._attn_kernel = "flash_attention_2"

    def test_backbone_and_prefix_stay_packed(self):
        batch = self._batch()
        shapes = []
        handles = []
        for layer in self.model.lm.base_model.layers:
            for module in (layer.self_attn.q_proj, layer.mlp):
                handles.append(
                    module.register_forward_pre_hook(
                        lambda module, inputs: shapes.append(inputs[0].shape[:2])
                    )
                )
        try:
            with mock.patch.object(
                self.model, "_left_pad", side_effect=AssertionError("padded backbone")
            ):
                self.model.predict(batch)
        finally:
            for handle in handles:
                handle.remove()
        expected = (1, int(batch.additional_infos[CU_SEQLENS][-1]))
        self.assertTrue(shapes)
        self.assertTrue(all(tuple(shape) == expected for shape in shapes))

    @parameterized.expand(
        [[torch.bfloat16], [torch.float16]], name_func=parameterized_name_func
    )
    def test_hybrid_matches_full_flex_gradients_and_dynamic_layouts(self, dtype):
        self.model.to(dtype=dtype)
        for multiplier, answers in (
            (1, self.answers),
            (43, [GENREC_ANSWER_CODES * 77, []]),
            (3, [GENREC_ANSWER_CODES, GENREC_ANSWER_CODES * 2]),
        ):
            batch = self._batch(
                [history * multiplier for history in self.histories], answers
            )
            self.model.zero_grad(set_to_none=True)
            reference = self.model.predict(batch)
            reference_loss = self.model.loss(reference, batch)["ce_loss"]
            reference_loss.backward()
            gradients = {
                name: parameter.grad.detach().float().clone()
                for name, parameter in self.model.named_parameters()
                if parameter.grad is not None
            }
            self.model.zero_grad(set_to_none=True)
            self.model._attn_kernel = "sdpa"
            try:
                result = self.model.predict(batch)
            finally:
                self.model._attn_kernel = "flash_attention_2"
            loss = self.model.loss(result, batch)["ce_loss"]
            torch.testing.assert_close(
                result["logits"], reference["logits"], atol=self.atol, rtol=self.rtol
            )
            torch.testing.assert_close(
                loss, reference_loss, atol=self.atol, rtol=self.rtol
            )
            loss.backward()
            for name, parameter in self.model.named_parameters():
                if name in gradients:
                    gradient = parameter.grad.float()
                    self.assertTrue(bool(torch.isfinite(gradient).all()), name)
                    error = (gradient - gradients[name]).norm()
                    scale = gradients[name].norm().clamp(min=1e-6)
                    self.assertLess(float(error / scale), 0.04, name)

    def test_unsupported_hybrid_conditions_raise_without_switching_backend(self):
        previous = self.model.lm.config._attn_implementation
        self.model.lm.config.attention_dropout = 0.1
        with self.assertRaisesRegex(ValueError, "zero attention dropout"):
            self.model.predict(self._batch())
        self.assertEqual(self.model.lm.config._attn_implementation, previous)
        self.model.lm.config.attention_dropout = 0
        self.model.float()
        with self.assertRaisesRegex(ValueError, "BF16/FP16"):
            self.model.predict(self._batch())

    def test_fp32_parameters_use_autocast_dtype_at_the_flash_kernel(self):
        from flash_attn import flash_attn_varlen_func

        self.model.float()
        with mock.patch(
            "flash_attn.flash_attn_varlen_func", wraps=flash_attn_varlen_func
        ) as flash:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = self._batch()
                result = self.model.predict(batch)
                loss = self.model.loss(result, batch)["ce_loss"]
            loss.backward()
        for call in flash.call_args_list:
            self.assertEqual(
                [tensor.dtype for tensor in call.args[:3]], [torch.bfloat16] * 3
            )
        self.assertEqual(
            self.model.lm.get_input_embeddings().weight.dtype, torch.float32
        )
        self.assertTrue(bool(torch.isfinite(loss)))

    @parameterized.expand([[1], [3], [4]], name_func=parameterized_name_func)
    def test_attention_matches_explicit_fp32_mask(self, levels):
        from torch.nn.attention.flex_attention import create_block_mask

        lengths = torch.tensor([131 + 2 * levels, 7 + levels], device="cuda")
        width = int(lengths.max()) + 5
        layout = build_multi_target_layout(
            nn.functional.pad(lengths.cumsum(0).int(), (1, 0)),
            torch.tensor([2 * levels, levels], device="cuda"),
            width,
            levels,
            pack_prefix=True,
        )
        block_mask = create_block_mask(
            layout.mask_mod,
            2,
            None,
            2 * levels,
            width,
            device="cuda",
            _compile=True,
        )
        token_count = int(lengths.sum())
        tensors = [
            torch.randn(
                1,
                heads,
                token_count,
                16,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            for heads in (14, 2, 2)
        ]
        output, _ = _shared_prefix_attention(
            self.model.lm.base_model.layers[0].self_attn,
            tensors[0],
            tensors[1],
            tensors[2],
            block_mask,
            shared_prefix_layout=layout,
        )
        output.float().square().mean().backward()
        gradients = [tensor.grad.float().clone() for tensor in tensors]
        reference_tensors = [
            tensor.detach().float().requires_grad_() for tensor in tensors
        ]
        q, k, v = reference_tensors
        boundaries = nn.functional.pad(lengths.cumsum(0), (1, 0))
        token_parents = torch.repeat_interleave(torch.arange(2, device="cuda"), lengths)
        local_positions = (
            torch.arange(token_count, device="cuda") - boundaries[token_parents]
        )[layout.packed_indices]
        token_parents = token_parents[layout.packed_indices]
        prefix_lengths = torch.tensor([131, 7], device="cuda")[token_parents]
        is_prefix = local_positions < prefix_lengths
        branches = (local_positions - prefix_lengths) // levels
        mask = (
            (token_parents[:, None] == token_parents[None, :])
            & (
                is_prefix[None, :]
                | (~is_prefix[:, None] & (branches[:, None] == branches[None, :]))
            )
            & (local_positions[None, :] <= local_positions[:, None])
        )
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            reference = nn.functional.scaled_dot_product_attention(
                q,
                k.repeat_interleave(7, dim=1),
                v.repeat_interleave(7, dim=1),
                attn_mask=mask,
            ).transpose(1, 2)
        torch.testing.assert_close(output.float(), reference, atol=2e-2, rtol=2e-2)
        reference.square().mean().backward()
        for tensor, gradient in zip(reference_tensors, gradients):
            self.assertLess(
                float((tensor.grad - gradient).norm() / tensor.grad.norm()), 0.02
            )


if __name__ == "__main__":
    unittest.main()
