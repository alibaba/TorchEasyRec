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

import dataclasses
import unittest
from unittest import mock

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor
from transformers import AutoModelForCausalLM

from tzrec.datasets.utils import Batch
from tzrec.models.genrec_model import _PARAM_DTYPE
from tzrec.models.model import TrainWrapper
from tzrec.prompt.assembler import (
    PROMPT_HOLE_POSITIONS,
    PROMPT_INPUT_IDS,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.protos.models.genrec_model_pb2 import GenrecModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import (
    _CODEBOOK,
    _HIST,
    GenrecModelTestBase,
    create_prompt_feature,
    offset_sid_codes,
    projected_feature,
)
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class BaseGenrecModelTest(GenrecModelTestBase):
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

    def test_rejects_a_prompt_that_declares_no_sid_space(self) -> None:
        # compile_prompt refuses this config, so the model precondition is
        # reachable only by constructing a prompt directly
        compiled_prompt = dataclasses.replace(self.compiled_prompt, sid_space=None)

        with self.assertRaisesRegex(ValueError, "declares no sid_space"):
            self._model(compiled_prompt=compiled_prompt)

    def test_builds_backbone_with_flash_attention_2_and_target_dtype(self) -> None:
        stand_in = AutoModelForCausalLM.from_pretrained(self.backbone)

        with (
            mock.patch.object(stand_in, "to", wraps=stand_in.to) as to_mock,
            mock.patch.object(
                AutoModelForCausalLM, "from_config", return_value=stand_in
            ) as from_config,
        ):
            model = self._model(lm_parameter_dtype=GenrecModelConfig.BF16)

        from_config.assert_called_once()
        args, kwargs = from_config.call_args
        self.assertEqual(len(args), 1)
        self.assertEqual(args[0].model_type, "qwen2")
        self.assertEqual(
            kwargs,
            {
                "attn_implementation": "flash_attention_2",
                "torch_dtype": torch.bfloat16,
            },
        )
        to_mock.assert_not_called()
        self.assertIs(model.lm, stand_in)

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
        [[GenrecModelConfig.BF16], [GenrecModelConfig.FP16]],
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
        device = torch.device("cuda")
        init_parameters(model, device=device)
        model.to(device)
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
        ).to(device)

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
        device = torch.device("cuda")
        init_parameters(model, device=device)
        model.to(device)
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
        ).to(device)

        wrapper = TrainWrapper(model, device=device, mixed_precision="BF16")
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


if __name__ == "__main__":
    unittest.main()
