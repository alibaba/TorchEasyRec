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

import numpy as np
import torch
from google.protobuf import text_format
from tokenizers import Tokenizer, models, pre_tokenizers
from torchrec import KeyedJaggedTensor
from transformers import AutoModelForCausalLM

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import FgMode, create_features
from tzrec.main import _create_model
from tzrec.prompt.assembler import (
    PROMPT_HOLE_POSITIONS,
    PROMPT_INPUT_IDS,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import assemble_into
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir

_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]


def _tokenizer(path: str) -> str:
    tok = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(_WORDS)}, unk_token="<unk>")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(path)
    return path


def _feature(text: str):
    fc = feature_pb2.FeatureConfig()
    text_format.Merge(text, fc)
    return create_features([fc], fg_mode=FgMode.FG_NONE)[0]


def _offset(codes):
    """Shift local codes into the flat space, as the SID tool's column does."""
    offsets = np.cumsum([0] + _CODEBOOK[:-1])
    return (np.asarray(codes).reshape(-1, len(_CODEBOOK)) + offsets).reshape(-1)


_HIST = 'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'
_ANSWER = 'sequence_raw_feature { feature_name: "answer" expression: "item:answer" }'


def _projected(name: str, dim: int) -> str:
    return (
        f'sequence_id_feature {{ feature_name: "{name}" expression: "user:{name}" '
        f"num_buckets: 32 embedding_dim: {dim} sequence_length: 2 }}"
    )


class BasePromptGenerativeModelTest(unittest.TestCase):
    """The backbone-agnostic half, reached through its one concrete subclass."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.backbone = os.path.join(self.test_dir, "backbone")
        create_tiny_causal_lm(64).save_pretrained(self.backbone)
        self.tok = _tokenizer(os.path.join(self.test_dir, "tok.json"))
        self.features = [_feature(_HIST), _feature(_ANSWER)]
        self.prompt = self._compile(self.features)

    def _compile(self, features, template="History : {{hist}} . Predict :", **kwargs):
        cfg = PromptConfig(tokenizer=self.tok, prompt=template, **kwargs)
        cfg.sid_space.codebook.extend(_CODEBOOK)
        return compile_prompt(cfg, features, model_dir=self.test_dir)

    def _model(self, features=None, prompt=-1):
        model_config = ModelConfig()
        qwen = model_config.prompt_generative_qwen
        qwen.hf_model_id = self.backbone
        qwen.common.beam_widths.extend([2, 2, 2])
        qwen.common.num_return_sequences = 2
        return _create_model(
            model_config,
            self.features if features is None else features,
            ["answer"],
            prompt=self.prompt if prompt == -1 else prompt,
        )

    def _batch(self, parsed, prompt=None, sparse=None):
        streams = assemble_into(prompt or self.prompt, parsed)
        batch = Batch(sparse_features={BASE_DATA_GROUP: sparse} if sparse else {})
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch

    def test_tokens_to_local_codes_undoes_shifts_and_groups_beams(self) -> None:
        model = self._model()
        space = self.prompt.sid_space
        local_codes = torch.tensor(
            [
                [0, 1, 3],
                [3, 0, 2],
                [1, 3, 0],
                [2, 2, 1],
            ]
        )
        tokens = local_codes + torch.tensor(space.level_offsets) + space.base_vocab
        codes = model._tokens_to_local_codes(tokens, batch_size=2)

        self.assertEqual(codes.shape, (2, 2, space.num_levels))
        self.assertEqual(codes.tolist(), local_codes.reshape(2, 2, -1).tolist())

    def test_rejects_a_model_built_without_a_prompt(self) -> None:
        with self.assertRaisesRegex(ValueError, "needs a compiled prompt"):
            self._model(prompt=None)

    def test_rejects_a_prompt_that_declares_no_sid_space(self) -> None:
        cfg = PromptConfig(tokenizer=self.tok, prompt="History : {{hist}} .")
        prompt = compile_prompt(cfg, self.features, model_dir=self.test_dir)
        self.assertIsNone(prompt.sid_space)

        with self.assertRaisesRegex(ValueError, "declares no sid_space"):
            self._model(prompt=prompt)

    def test_shared_projection_name_requires_matching_widths(self) -> None:
        features = [
            _feature(_HIST),
            _feature(_ANSWER),
            _feature(_projected("pa", 8)),
            _feature(_projected("pb", 16)),
        ]
        cfg = PromptConfig(
            tokenizer=self.tok,
            prompt="History : {{hist}} . {{pa}} {{pb}} Predict :",
            response="{{answer}}",
        )
        cfg.sid_space.codebook.extend(_CODEBOOK)
        for name in ("pa", "pb"):
            slot = cfg.slots.add(name=name, projection_name="shared")
            slot.feature_names.append(name)
        prompt = compile_prompt(cfg, features, model_dir=self.test_dir)

        with self.assertRaisesRegex(ValueError, "cannot share a module"):
            self._model(features=features, prompt=prompt)

    def test_projected_slot_overwrites_sentinels_and_backpropagates(self) -> None:
        features = [_feature(_HIST), _feature(_ANSWER), _feature(_projected("prof", 8))]
        prompt = self._compile(
            features,
            template="History : {{hist}} . Predict {{prof}} :",
            response="{{answer}}",
        )
        model = self._model(features=features, prompt=prompt)
        # the embedding table is built on meta until something materializes it
        init_parameters(model, device=torch.device("cpu"))
        batch = self._batch(
            {
                "hist.values": torch.tensor(_offset([0, 1, 2])).reshape(-1, 1),
                "hist.lengths": torch.tensor([3]),
                "answer.values": torch.tensor(_offset([1, 2, 3])),
                "answer.lengths": torch.tensor([3]),
                "prof.values": torch.tensor([5, 9]),
                "prof.lengths": torch.tensor([2]),
            },
            prompt=prompt,
            sparse=KeyedJaggedTensor.from_lengths_sync(
                keys=["prof"],
                values=torch.tensor([5, 9]),
                lengths=torch.tensor([2]),
            ),
        )

        embeds = model._prompt_embeds(batch)
        raw = model.lm.get_input_embeddings()(batch.additional_infos[PROMPT_INPUT_IDS])
        holes = batch.additional_infos[PROMPT_HOLE_POSITIONS]
        self.assertGreater(holes.numel(), 0)

        changed = ~torch.isclose(embeds, raw).all(dim=-1)
        self.assertEqual(sorted(changed.nonzero().flatten().tolist()), holes.tolist())

        embeds[holes].sum().backward()
        proj = next(iter(model.projections.values()))
        self.assertIsNotNone(proj.head.weight.grad)

    def test_loss_surfaces_the_ce_computed_in_predict(self) -> None:
        model = self._model()
        value = torch.tensor(1.25)

        self.assertEqual(model.loss({"loss": value}, Batch()), {"ce_loss": value})

    def test_metric_averages_the_loss_across_batches(self) -> None:
        model = self._model()
        model.init_metric()
        for value in (1.0, 3.0):
            model.update_metric({"loss": torch.tensor(value)}, Batch())

        self.assertAlmostEqual(
            model._metric_modules["ce_loss"].compute().item(), 2.0, places=5
        )

    def test_save_assets_co_locates_the_prompt_contract(self) -> None:
        model = self._model()
        target = os.path.join(self.test_dir, "ckpt")
        os.makedirs(target, exist_ok=True)
        model.save_assets(target)

        self.assertTrue(os.path.isdir(os.path.join(target, "prompt")))
        self.assertTrue(os.listdir(os.path.join(target, "prompt")))

    def test_init_from_pretrained_replaces_the_empty_weights(self) -> None:
        model = self._model()
        base_vocab = self.prompt.sid_space.base_vocab
        before = model.lm.get_input_embeddings().weight[:base_vocab].clone()
        model.init_from_pretrained()
        after = model.lm.get_input_embeddings().weight[:base_vocab]

        # the checkpoint rows land verbatim; only the appended SID rows are new
        reference = AutoModelForCausalLM.from_pretrained(self.backbone)
        expected = reference.get_input_embeddings().weight[:base_vocab]
        self.assertFalse(torch.allclose(before, expected))
        torch.testing.assert_close(after, expected)


if __name__ == "__main__":
    unittest.main()
