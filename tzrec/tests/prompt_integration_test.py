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
from transformers import Qwen2Config

from tzrec.datasets.utils import Batch
from tzrec.features.feature import FgMode, create_features
from tzrec.main import _create_model
from tzrec.prompt.assembler import assemble_into
from tzrec.prompt.compile import compile_prompt
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import make_test_dir

_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]


def _tiny_backbone(path: str) -> str:
    """A two-layer Qwen saved locally, so no download is needed."""
    Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    ).save_pretrained(path)
    return path


def _tokenizer(path: str) -> str:
    tok = Tokenizer(
        models.WordLevel(vocab={w: i for i, w in enumerate(_WORDS)}, unk_token="<unk>")
    )
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(path)
    return path


def _features():
    text = (
        'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }',
        'sequence_raw_feature { feature_name: "answer" expression: "item:answer" }',
    )
    out = []
    for one in text:
        fc = feature_pb2.FeatureConfig()
        text_format.Merge(one, fc)
        out.append(create_features([fc], fg_mode=FgMode.FG_NONE)[0])
    return out


def _offset(codes):
    """Shift local codes into the flat space, as the SID tool's column does."""
    offsets = np.cumsum([0] + _CODEBOOK[:-1])
    return (np.asarray(codes).reshape(-1, len(_CODEBOOK)) + offsets).reshape(-1)


class PromptStackIntegrationTest(unittest.TestCase):
    """compile -> assemble -> model, on the real code path."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.backbone = _tiny_backbone(os.path.join(self.test_dir, "backbone"))
        self.tok = _tokenizer(os.path.join(self.test_dir, "tok.json"))
        self.features = _features()

        cfg = PromptConfig(
            tokenizer=self.tok,
            prompt="History : {{hist}} . Predict :",
            response="{{answer}}",
        )
        cfg.sid_space.codebook.extend(_CODEBOOK)
        self.prompt = compile_prompt(cfg, self.features, model_dir=self.test_dir)

    def _model(self):
        model_config = ModelConfig()
        qwen = model_config.prompt_generative_qwen
        qwen.hf_model_id = self.backbone
        qwen.common.beam_widths.extend([2, 2, 2])
        qwen.common.num_return_sequences = 2
        return _create_model(
            model_config, self.features, ["answer"], prompt=self.prompt
        )

    def _batch(self, hist, answer):
        parsed = {
            "hist.values": torch.tensor(_offset(hist)),
            "hist.lengths": torch.tensor([len(hist)]),
            "answer.values": torch.tensor(_offset(answer)),
            "answer.lengths": torch.tensor([len(answer)]),
        }
        streams = assemble_into(self.prompt, parsed)
        batch = Batch()
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch

    def test_compiles_a_usable_space(self) -> None:
        space = self.prompt.sid_space
        self.assertEqual(space.num_levels, 3)
        self.assertEqual(space.sid_vocab_size, 12)
        # the atoms sit immediately above the base vocabulary
        self.assertEqual(space.band_lo[0], space.base_vocab)
        self.assertEqual(space.band_hi[-1], space.base_vocab + 11)

    def test_model_resizes_to_target_vocab(self) -> None:
        model = self._model()
        rows = model.lm.get_input_embeddings().weight.shape[0]
        self.assertEqual(rows, self.prompt.sid_space.target_vocab)
        # every SID atom has a row
        self.assertGreater(rows, self.prompt.sid_space.band_hi[-1])

    def test_forward_produces_a_finite_loss(self) -> None:
        model = self._model()
        batch = self._batch([0, 1, 2, 3, 0, 1], [1, 2, 3])
        out = model.predict(batch)

        self.assertIn("loss", out)
        self.assertTrue(bool(torch.isfinite(out["loss"])))

    def test_loss_backpropagates_into_the_backbone(self) -> None:
        model = self._model()
        batch = self._batch([0, 1, 2, 3, 0, 1], [1, 2, 3])
        model.predict(batch)["loss"].backward()

        grad = model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool((grad.abs().sum() > 0)))

    def test_assembled_stream_matches_the_template(self) -> None:
        batch = self._batch([0, 1, 2], [1, 2, 3])
        ids = batch.additional_infos["prompt_input_ids"]
        space = self.prompt.sid_space
        # "History :" + 3 history atoms + "." + "Predict :" + 3 answer atoms
        self.assertEqual(ids.numel(), 2 + 3 + 1 + 2 + 3)
        sid_rows = ids[ids >= space.base_vocab]
        self.assertEqual(sid_rows.numel(), 6)

    def test_labels_supervise_only_the_answer(self) -> None:
        batch = self._batch([0, 1, 2], [1, 2, 3])
        labels = batch.additional_infos["prompt_labels"]
        supervised = labels[labels != -100]
        self.assertEqual(supervised.numel(), 3)

    def test_raw_codes_are_rejected_before_the_model_sees_them(self) -> None:
        parsed = {
            # not offset: level 1 and 2 fall below their bands
            "hist.values": torch.tensor([1, 2, 3]),
            "hist.lengths": torch.tensor([3]),
            "answer.values": torch.tensor(_offset([1, 2, 3])),
            "answer.lengths": torch.tensor([3]),
        }
        with self.assertRaisesRegex(ValueError, "offset_codebook column"):
            assemble_into(self.prompt, parsed)


if __name__ == "__main__":
    unittest.main()
