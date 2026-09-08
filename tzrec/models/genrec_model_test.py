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
from parameterized import parameterized
from torchrec import KeyedJaggedTensor
from transformers import AutoModelForCausalLM

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_model import (
    _PARAM_DTYPE,
    SLOT_EMBEDS,
    GenRecFrontEnd,
    project_slots,
)
from tzrec.models.model import ScriptWrapper
from tzrec.prompt.assembler import (
    HOLE_POSITIONS,
    INPUT_IDS,
    PROMPT_HOLE_POSITIONS,
    PROMPT_INPUT_IDS,
    PromptAssembler,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.hole_keys import HOLE_KEYS, HoleKeyBuilder
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import (
    _CODEBOOK,
    _HIST,
    GenRecModelTestBase,
    create_prompt_feature,
    offset_sid_codes,
    projected_feature,
)
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    parameterized_name_func,
)


class BaseGenRecModelTest(GenRecModelTestBase):
    """Shared causal-LM behavior, reached through its concrete subclass."""

    def test_tokens_to_local_codes_undoes_shifts_and_groups_beams(self) -> None:
        model = self._model()
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
        codes = model._tokens_to_local_codes(tokens, batch_size=2)

        self.assertEqual(codes.shape, (2, 2, space.num_levels))
        self.assertEqual(codes.tolist(), local_codes.reshape(2, 2, -1).tolist())

    def test_rejects_a_model_built_without_a_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs a compiled prompt"):
            self._model(compiled_prompt=None)

    def test_shared_projection_name_requires_matching_widths(self) -> None:
        features = [
            create_prompt_feature(_HIST),
            create_prompt_feature(projected_feature("pa", 8)),
            create_prompt_feature(projected_feature("pb", 16)),
        ]
        cfg = PromptConfig(
            tokenizer_path=self.tok,
            prompt="History : {{hist}} . {{pa}} {{pb}} Predict :",
            response="{{answer}}",
        )
        cfg.sid_space.codebook.extend(_CODEBOOK)
        for name in ("pa", "pb"):
            slot = cfg.slots.add(name=name, projection_name="shared")
            slot.feature_names.append(name)
        compiled_prompt = compile_prompt(cfg, features, ["answer"])

        with self.assertRaisesRegex(ValueError, "cannot share a module"):
            self._model(features=features, compiled_prompt=compiled_prompt)

    def test_projected_slot_overwrites_sentinels_and_backpropagates(self) -> None:
        features = [
            create_prompt_feature(_HIST),
            create_prompt_feature(projected_feature("prof", 8)),
        ]
        compiled_prompt = self._compile(
            features,
            template="History : {{hist}} . Predict {{prof}} :",
            response="{{answer}}",
        )
        model = self._model(features=features, compiled_prompt=compiled_prompt)
        # the embedding table is built on meta until something materializes it
        init_parameters(model, device=torch.device("cpu"))
        batch = self._batch(
            {
                "hist.values": torch.tensor(
                    offset_sid_codes([0, 1, 2], _CODEBOOK)
                ).reshape(-1, 1),
                "hist.lengths": torch.tensor([3]),
                "answer.values": torch.tensor(offset_sid_codes([1, 2, 3], _CODEBOOK)),
                "answer.lengths": torch.tensor([3]),
                "prof.values": torch.tensor([5, 9]),
                "prof.lengths": torch.tensor([2]),
            },
            compiled_prompt=compiled_prompt,
            sparse=KeyedJaggedTensor.from_lengths_sync(
                keys=["prof"],
                values=torch.tensor([5, 9]),
                lengths=torch.tensor([2]),
            ),
        )

        embeds = model.build_input(batch)
        raw = model.lm.get_input_embeddings()(batch.additional_infos[PROMPT_INPUT_IDS])
        holes = batch.additional_infos[PROMPT_HOLE_POSITIONS]
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
        features = [
            create_prompt_feature(_HIST),
            create_prompt_feature(projected_feature("prof", 8)),
        ]
        compiled_prompt = self._compile(
            features,
            template="History : {{hist}} . Predict {{prof}} :",
            response="{{answer}}",
        )
        model = self._model(
            features=features,
            compiled_prompt=compiled_prompt,
            lm_parameter_dtype=lm_parameter_dtype,
        )
        init_parameters(model, device=torch.device("cpu"))
        batch = self._batch(
            {
                "hist.values": torch.tensor(
                    offset_sid_codes([0, 1, 2], _CODEBOOK)
                ).reshape(-1, 1),
                "hist.lengths": torch.tensor([3]),
                "answer.values": torch.tensor(offset_sid_codes([1, 2, 3], _CODEBOOK)),
                "answer.lengths": torch.tensor([3]),
                "prof.values": torch.tensor([5, 9]),
                "prof.lengths": torch.tensor([2]),
            },
            compiled_prompt=compiled_prompt,
            sparse=KeyedJaggedTensor.from_lengths_sync(
                keys=["prof"],
                values=torch.tensor([5, 9]),
                lengths=torch.tensor([2]),
            ),
        )

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

    def test_metric_averages_the_loss_across_batches(self) -> None:
        model = self._model()
        model.init_metric()
        for value in (1.0, 3.0):
            model.update_metric({}, Batch(), {"ce_loss": torch.tensor(value)})

        self.assertAlmostEqual(
            model._metric_modules["ce_loss"].compute().item(), 2.0, places=5
        )

    def test_init_from_pretrained_replaces_the_empty_weights(self) -> None:
        model = self._model()
        base_vocab_size = self.compiled_prompt.sid_space.base_vocab_size
        before = model.lm.get_input_embeddings().weight[:base_vocab_size].clone()
        model.init_from_pretrained()
        after = model.lm.get_input_embeddings().weight[:base_vocab_size]

        # the checkpoint rows land verbatim; only the appended SID rows are new
        reference = AutoModelForCausalLM.from_pretrained(self.backbone)
        expected = reference.get_input_embeddings().weight[:base_vocab_size]
        self.assertFalse(torch.allclose(before, expected))
        torch.testing.assert_close(after, expected)


class GenRecFrontEndTest(GenRecModelTestBase):
    """The served half of the model, under the same wrapper every export uses."""

    def setUp(self) -> None:
        super().setUp()
        self.features = [
            create_prompt_feature(_HIST),
            create_prompt_feature(projected_feature("beh", 8)),
        ]
        self.prompt_config = PromptConfig(
            tokenizer_path=self.tok,
            prompt="History : {{hist}} . {{beh}} Predict :",
            response="{{answer}}",
        )
        self.prompt_config.sid_space.codebook.extend(_CODEBOOK)
        self.compiled_prompt = compile_prompt(
            self.prompt_config, self.features, ["answer"]
        )
        self.model = self._model()
        init_parameters(self.model, device=torch.device("cpu"))
        # the parsed dict as the data parser emits it: a dense sequence of codes
        # and a sparse behaviour sequence
        self.data = {
            "hist.values": torch.tensor(
                offset_sid_codes([0, 1, 2, 3, 0, 1], _CODEBOOK), dtype=torch.float32
            ).reshape(-1, 1),
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
            self.assertTrue(torch.allclose(out[key].float(), value.float()), key)


if __name__ == "__main__":
    unittest.main()
