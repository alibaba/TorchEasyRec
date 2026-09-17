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
from typing import Tuple
from unittest import mock

import torch
import torch.fx
from parameterized import parameterized
from torchrec import KeyedJaggedTensor
from transformers import AutoConfig, AutoModelForCausalLM

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import FgMode, create_features
from tzrec.main import _create_model
from tzrec.models.genrec_model import (
    _PARAM_DTYPE,
    SLOT_EMBEDS,
    GenRecFrontEnd,
    project_slots,
)
from tzrec.models.model import BaseModel, ScriptWrapper, TrainWrapper
from tzrec.prompt.assembler import (
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
    PromptAssembler,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.hole_keys import HOLE_KEYS, HoleKeyBuilder
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig, PromptSlot
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    create_genrec_test_model,
    create_genrec_test_tokenizer,
    flash_attn_unavailable,
    make_test_dir,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)

# offset SID codes for the (4, 4, 4) codebook: level_offsets[l] + code
_HIST_CODES = [0, 5, 10]
_LONG_HIST_CODES = [0, 5, 10, 3, 4, 9]
_ANSWER_CODES = [1, 6, 11]


def _hist() -> feature_pb2.FeatureConfig:
    return feature_pb2.FeatureConfig(
        sequence_raw_feature=feature_pb2.RawFeature(
            feature_name="hist", expression="user:hist"
        )
    )


def _projected(name: str, dim: int) -> feature_pb2.FeatureConfig:
    return feature_pb2.FeatureConfig(
        sequence_id_feature=feature_pb2.IdFeature(
            feature_name=name,
            expression=f"user:{name}",
            num_buckets=32,
            embedding_dim=dim,
            sequence_length=2,
        )
    )


def _batch(compiled_prompt, parsed, sparse=None) -> Batch:
    batch = Batch(sparse_features={BASE_DATA_GROUP: sparse} if sparse else {})
    batch.additional_infos.update(
        PromptAssembler(compiled_prompt.prompt_plan, compiled_prompt.sid_space)(parsed)
    )
    return batch


def _projected_batch(compiled_prompt) -> Batch:
    return _batch(
        compiled_prompt,
        {
            "hist.values": torch.tensor(_HIST_CODES).reshape(-1, 1),
            "hist.lengths": torch.tensor([3]),
            "answer.values": torch.tensor(_ANSWER_CODES),
            "answer.lengths": torch.tensor([3]),
            "prof.values": torch.tensor([5, 9]),
            "prof.lengths": torch.tensor([2]),
        },
        sparse=KeyedJaggedTensor.from_lengths_sync(
            keys=["prof"], values=torch.tensor([5, 9]), lengths=torch.tensor([2])
        ),
    )


class BaseGenRecModelTest(unittest.TestCase):
    """Shared causal-LM behavior, reached through its concrete subclass."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.model, self.compiled_prompt = create_genrec_test_model(self.test_dir)

    def test_tokens_to_local_codes_undoes_shifts_and_groups_beams(self) -> None:
        space = self.compiled_prompt.sid_space
        local_codes = torch.tensor(
            [
                [0, 1, 3],
                [3, 0, 2],
                [1, 3, 0],
                [2, 2, 1],
            ]
        )
        tokens = local_codes + torch.tensor(space.level_offsets) + space.base_vocab_size
        codes = self.model._tokens_to_local_codes(tokens, batch_size=2)

        self.assertEqual(codes.shape, (2, 2, space.num_levels))
        self.assertEqual(codes.tolist(), local_codes.reshape(2, 2, -1).tolist())

    def test_rejects_a_model_built_without_a_prompt(self) -> None:
        model_config = ModelConfig()
        model_config.genrec_causal_lm_model.hf_model_name_or_path = os.path.join(
            self.test_dir, "backbone"
        )
        with self.assertRaisesRegex(ValueError, "needs a compiled prompt"):
            _create_model(model_config, [], ["answer"], compiled_prompt=None)

    @parameterized.expand(
        [
            [GenRecModelConfig.SDPA, "sdpa"],
            [GenRecModelConfig.FLASH_ATTENTION_2, "flash_attention_2"],
        ],
        name_func=parameterized_name_func,
    )
    def test_builds_backbone_with_the_configured_kernel_and_dtype(
        self, attn_implementation, expected_impl
    ) -> None:
        # from_config is mocked, so this pins the kwargs init_backbone sends
        # without building a second backbone; setUp still builds a real one.
        stand_in = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.test_dir, "backbone")
        )

        with (
            mock.patch.object(stand_in, "to", wraps=stand_in.to) as to_mock,
            mock.patch.object(
                AutoModelForCausalLM, "from_config", return_value=stand_in
            ) as from_config,
            # the backbone is mocked, so the wheel probe has nothing to check
            mock.patch("tzrec.models.genrec_model.find_spec", return_value=object()),
        ):
            model, _ = create_genrec_test_model(
                self.test_dir,
                lm_parameter_dtype=GenRecModelConfig.BF16,
                attn_implementation=attn_implementation,
            )

        from_config.assert_called_once()
        args, kwargs = from_config.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(args[0].model_type, "qwen2")
        self.assertEqual(
            kwargs,
            {
                "attn_implementation": expected_impl,
                "torch_dtype": torch.bfloat16,
            },
        )
        to_mock.assert_not_called()
        self.assertIs(model.lm, stand_in)

    def test_shared_projection_name_requires_matching_widths(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot share a module"):
            create_genrec_test_model(
                self.test_dir,
                feature_configs=[_hist(), _projected("pa", 8), _projected("pb", 16)],
                prompt="History : {{hist}} . {{pa}} {{pb}} Predict :",
                slots=[
                    PromptSlot(
                        name="pa", feature_names=["pa"], projection_name="shared"
                    ),
                    PromptSlot(
                        name="pb", feature_names=["pb"], projection_name="shared"
                    ),
                ],
            )

    def test_projected_slot_overwrites_sentinels_and_backpropagates(self) -> None:
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("prof", 8)],
            prompt="History : {{hist}} . Predict {{prof}} :",
        )
        # the embedding table is built on meta until something materializes it
        init_parameters(model, device=torch.device("cpu"))
        batch = _projected_batch(compiled_prompt)

        embeds = model.build_input(batch)
        raw = model.lm.get_input_embeddings()(batch.additional_infos[INPUT_IDS])
        holes = batch.additional_infos[HOLE_POSITIONS]
        self.assertGreater(holes.numel(), 0)

        changed = ~torch.isclose(embeds, raw).all(dim=-1)
        self.assertEqual(sorted(changed.nonzero().flatten().tolist()), holes.tolist())

        embeds[holes].sum().backward()
        proj = next(iter(model.projections.values()))
        self.assertIsNotNone(proj.head.weight.grad)

    @parameterized.expand(
        [[GenRecModelConfig.BF16], [GenRecModelConfig.FP16]],
        name_func=parameterized_name_func,
    )
    def test_projected_slot_follows_a_narrow_lm_dtype(self, lm_parameter_dtype) -> None:
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("prof", 8)],
            prompt="History : {{hist}} . Predict {{prof}} :",
            lm_parameter_dtype=lm_parameter_dtype,
        )
        init_parameters(model, device=torch.device("cpu"))
        batch = _projected_batch(compiled_prompt)

        embeds = model.build_input(batch)
        self.assertIs(embeds.dtype, _PARAM_DTYPE[lm_parameter_dtype])

        predictions = model.predict(batch)
        loss = model.loss(predictions, batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        # the projection keeps fp32 masters, so only the spliced values convert
        proj = next(iter(model.projections.values()))
        self.assertIs(proj.head.weight.dtype, torch.float32)
        self.assertGreater(float(proj.head.weight.grad.abs().sum()), 0.0)

    def test_projected_slot_trains_with_fp32_masters_and_bf16_autocast(self) -> None:
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("prof", 8)],
            prompt="History : {{hist}} . Predict {{prof}} :",
        )
        init_parameters(model, device=torch.device("cpu"))
        batch = _projected_batch(compiled_prompt)

        wrapper = TrainWrapper(
            model, device=torch.device("cpu"), mixed_precision="BF16"
        )
        loss, _ = wrapper(batch)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        lm_weight = model.lm.model.layers[0].self_attn.q_proj.weight
        self.assertIs(lm_weight.dtype, torch.float32)
        self.assertIsNotNone(lm_weight.grad)
        self.assertTrue(bool(torch.isfinite(lm_weight.grad).all()))
        proj_weight = next(iter(model.projections.values())).head.weight
        self.assertIs(proj_weight.dtype, torch.float32)
        self.assertIsNotNone(proj_weight.grad)
        self.assertTrue(bool(torch.isfinite(proj_weight.grad).all()))

    def test_metric_averages_the_loss_across_batches(self) -> None:
        self.model.init_metric()
        for value in (1.0, 3.0):
            self.model.update_metric({}, Batch(), {"ce_loss": torch.tensor(value)})

        self.assertAlmostEqual(
            self.model._metric_modules["ce_loss"].compute().item(), 2.0, places=5
        )

    def test_init_from_pretrained_replaces_the_empty_weights(self) -> None:
        base_vocab_size = self.compiled_prompt.sid_space.base_vocab_size
        embeddings = self.model.lm.get_input_embeddings()
        before = embeddings.weight[:base_vocab_size].clone()
        self.model.init_from_pretrained()
        after = embeddings.weight[:base_vocab_size]

        # the checkpoint rows land verbatim; only the appended SID rows are new
        reference = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.test_dir, "backbone")
        )
        expected = reference.get_input_embeddings().weight[:base_vocab_size]
        self.assertFalse(torch.allclose(before, expected))
        torch.testing.assert_close(after, expected)

    def test_model_resizes_to_target_vocab_size(self) -> None:
        rows = self.model.lm.get_input_embeddings().weight.shape[0]
        self.assertEqual(rows, self.compiled_prompt.sid_space.target_vocab_size)
        self.assertGreater(rows, self.compiled_prompt.sid_space.band_hi[-1])

    def test_loss_is_finite_and_backpropagates_into_the_backbone(self) -> None:
        batch = _batch(
            self.compiled_prompt,
            {
                "hist.values": torch.tensor(_LONG_HIST_CODES),
                "hist.lengths": torch.tensor([6]),
                "answer.values": torch.tensor(_ANSWER_CODES),
                "answer.lengths": torch.tensor([3]),
            },
        )
        predictions = self.model.predict(batch)
        loss = self.model.loss(predictions, batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        grad = self.model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool((grad.abs().sum() > 0)))

    def test_training_forward_survives_fx_tracing(self) -> None:
        torch.fx.symbolic_trace(TrainWrapper(self.model))


class GenRecFrontEndTest(unittest.TestCase):
    """The served half of the model, under the same wrapper every export uses."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.model, self.compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("beh", 8)],
            prompt="History : {{hist}} . {{beh}} Predict :",
        )
        init_parameters(self.model, device=torch.device("cpu"))
        # the parsed dict as the data parser emits it: a dense sequence of codes
        # and a sparse behaviour sequence
        self.data = {
            "hist.values": torch.tensor(_LONG_HIST_CODES, dtype=torch.float32).reshape(
                -1, 1
            ),
            "hist.lengths": torch.tensor([6]),
            "beh.values": torch.tensor([3, 9]),
            "beh.lengths": torch.tensor([2]),
        }
        self.wrapped = ScriptWrapper(GenRecFrontEnd(self.model))

    def test_front_end_returns_the_walk_and_the_projected_slots(self) -> None:
        out = self.wrapped(self.data)
        compiled = self.compiled_prompt
        walk = PromptAssembler(
            compiled.prompt_plan, compiled.sid_space, include_response=False
        )(self.data)
        for key in (INPUT_IDS, HOLE_POSITIONS):
            self.assertTrue(torch.equal(out[key], walk[key]), key)
        self.assertTrue(
            torch.equal(out[HOLE_KEYS], HoleKeyBuilder(compiled.prompt_plan)(self.data))
        )
        batch = self.wrapped.get_batch(self.data)
        expected = project_slots(
            self.model.embedding_group,
            compiled.prompt_plan,
            self.model._slot_projections,
            batch,
            int(self.model.lm.config.hidden_size),
        )
        self.assertTrue(torch.allclose(out[SLOT_EMBEDS], expected))
        self.assertEqual(tuple(out[SLOT_EMBEDS].shape), (2, 32))

    def test_front_end_joins_several_projected_slots_in_order(self) -> None:
        """Two slots: ``slot_embeds`` is their projections, occurrence by occurrence."""
        model, compiled = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("beh", 8), _projected("ctx", 8)],
            prompt="History : {{hist}} . {{beh}} then {{ctx}} Predict :",
        )
        init_parameters(model, device=torch.device("cpu"))
        wrapped = ScriptWrapper(GenRecFrontEnd(model))
        data = dict(self.data)
        data["ctx.values"] = torch.tensor([1, 4, 7])
        data["ctx.lengths"] = torch.tensor([3])

        out = wrapped(data)
        counts = out[HOLE_SLOT_COUNTS]
        self.assertEqual(counts.tolist(), [2, 3])
        grouped = model.embedding_group(wrapped.get_batch(data))
        spans = torch.split(out[SLOT_EMBEDS], counts.tolist())
        for seg, proj, span in zip(
            compiled.prompt_plan.projected_slots, model._slot_projections, spans
        ):
            expected = proj(grouped[seg.name + seg.output_key]).reshape(-1, 32)
            self.assertTrue(torch.allclose(span, expected), seg.name)

    def test_front_end_shares_the_checkpoint_names_and_not_the_lm(self) -> None:
        names = set(self.wrapped.state_dict())
        self.assertTrue(any(n.startswith("model.embedding_group.") for n in names))
        self.assertTrue(any(n.startswith("model.projections.") for n in names))
        self.assertFalse(any(".lm." in n for n in names))

    def test_front_end_traces_and_scripts(self) -> None:
        """The assembler is an FX leaf, so the export's trace-then-script works."""
        eager = self.wrapped(self.data)
        scripted = torch.jit.script(symbolic_trace(self.wrapped))
        out = scripted(self.data)
        for key, value in eager.items():
            if value.is_floating_point():
                self.assertTrue(torch.allclose(out[key], value), key)
            else:
                self.assertTrue(torch.equal(out[key], value), key)


# a second answer, and a rewrite of _HIST_CODES that keeps its width
_OTHER_ANSWER_CODES = [2, 7, 8]
_REWRITTEN_HIST_CODES = [3, 7, 11]


def _packed_batch(compiled_prompt, hist_rows, answer_rows) -> Batch:
    """Several rows in one batch, packed the way the collator packs them."""
    return _batch(
        compiled_prompt,
        {
            "hist.values": torch.tensor([code for row in hist_rows for code in row]),
            "hist.lengths": torch.tensor([len(row) for row in hist_rows]),
            "answer.values": torch.tensor(
                [code for row in answer_rows for code in row]
            ),
            "answer.lengths": torch.tensor([len(row) for row in answer_rows]),
        },
    )


def _packed_model(
    test_dir: str, model_type: str, attn_implementation: int, lm_parameter_dtype: int
) -> Tuple[BaseModel, CompiledPrompt]:
    """Build a bf16 genrec model over a tiny ``model_type`` backbone.

    ``create_genrec_test_model`` always writes a Qwen2 backbone, and the packed
    forward only holds where the backbone threads the varlen keyword arguments
    through to its attention, so the architecture is chosen here instead. Only
    the config is written: ``init_backbone`` builds from it and this test never
    restores pretrained weights.

    Args:
        test_dir (str): scratch directory the backbone is written under.
        model_type (str): the hugging-face ``model_type`` of the backbone.
        attn_implementation (int): ``GenRecModelConfig.AttnImpl`` to build with.
        lm_parameter_dtype (int): ``GenRecModelConfig.ParamDtype`` to build in.

    Returns:
        Tuple[BaseModel, CompiledPrompt]: the model and the prompt it was
        built on.
    """
    backbone = os.path.join(test_dir, model_type)
    AutoConfig.for_model(
        model_type,
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    ).save_pretrained(backbone)

    features = create_features([_hist()], fg_mode=FgMode.FG_NONE)
    prompt_config = PromptConfig(
        tokenizer_path=create_genrec_test_tokenizer(os.path.join(test_dir, "tok.json")),
        prompt="History : {{hist}} . Predict :",
        response="{{answer}}",
    )
    prompt_config.sid_space.codebook.extend([4, 4, 4])
    compiled_prompt = compile_prompt(prompt_config, features, ["answer"])

    model_config = ModelConfig()
    lm_config = model_config.genrec_causal_lm_model
    lm_config.hf_model_name_or_path = backbone
    lm_config.common.beam_widths.extend([2, 2, 2])
    lm_config.common.num_return_sequences = 2
    lm_config.common.lm_parameter_dtype = lm_parameter_dtype
    lm_config.common.attn_implementation = attn_implementation
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _create_model(
            model_config, features, ["answer"], compiled_prompt=compiled_prompt
        )
    return model, compiled_prompt


class _PackedRowsCase:
    """A packed batch must read exactly as its rows do one at a time.

    The batch is one concatenated stream with ``cu_seq_lens`` marking the
    boundaries, so the failure this guards against is a row attending into the
    row before it. Rewriting row 0 to different codes of the same width leaves
    row 1 at the same packed offsets: its logits have to stay bit identical.
    """

    device = torch.device("cpu")
    attn_implementation = GenRecModelConfig.SDPA
    lm_parameter_dtype = GenRecModelConfig.FP32

    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def _assert_packed_matches_solo(self, model_type: str) -> None:
        device = self.device
        model, compiled_prompt = _packed_model(
            self.test_dir,
            model_type,
            self.attn_implementation,
            self.lm_parameter_dtype,
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


class PackedSdpaAttentionTest(_PackedRowsCase, unittest.TestCase):
    """The packed forward under sdpa, which needs no GPU and no wheel."""

    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    def test_packed_rows_match_solo_runs_and_backpropagate(
        self, model_type: str
    ) -> None:
        self._assert_packed_matches_solo(model_type)


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
@unittest.skipIf(*flash_attn_unavailable)
class PackedFlashAttentionTest(_PackedRowsCase, unittest.TestCase):
    """The same rows through the varlen flash kernel."""

    device = torch.device("cuda")
    attn_implementation = GenRecModelConfig.FLASH_ATTENTION_2
    # the flash kernel takes fp16/bf16 only, and this arm carries no autocast
    lm_parameter_dtype = GenRecModelConfig.BF16

    @parameterized.expand([["qwen2"], ["qwen3"]], name_func=parameterized_name_func)
    def test_packed_rows_match_solo_runs_and_backpropagate(
        self, model_type: str
    ) -> None:
        self._assert_packed_matches_solo(model_type)
