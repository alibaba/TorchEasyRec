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
import torch.fx

from tzrec.datasets.utils import Batch
from tzrec.main import _create_model
from tzrec.models.prompt_generative_qwen import _unpack
from tzrec.prompt.assembler import (
    PROMPT_CU_SEQLENS,
    PROMPT_LABELS,
    PROMPT_MAX_SEQLEN,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.tests.prompt_test_util import (
    assemble_into,
    create_prompt_feature,
    create_prompt_tokenizer,
    offset_sid_codes,
)
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir

_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]


class PromptStackIntegrationTest(unittest.TestCase):
    """compile -> assemble -> model, on the real code path."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.backbone = os.path.join(self.test_dir, "backbone")
        create_tiny_causal_lm(64).save_pretrained(self.backbone)
        self.tok = create_prompt_tokenizer(
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )
        self.features = [
            create_prompt_feature(text)
            for text in (
                'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }',
                'sequence_raw_feature { feature_name: "answer" '
                'expression: "item:answer" }',
            )
        ]

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
        return self._batch_rows([(hist, answer)])

    def _batch_rows(self, rows):
        hist = [h for h, _ in rows]
        answer = [a for _, a in rows]
        parsed = {
            "hist.values": torch.tensor(
                offset_sid_codes([c for h in hist for c in h], _CODEBOOK)
            ),
            "hist.lengths": torch.tensor([len(h) for h in hist]),
            "answer.values": torch.tensor(
                offset_sid_codes([c for a in answer for c in a], _CODEBOOK)
            ),
            "answer.lengths": torch.tensor([len(a) for a in answer]),
        }
        streams = assemble_into(self.prompt, parsed)
        batch = Batch()
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch

    def test_every_row_is_supervised_whatever_its_length(self) -> None:
        # a short row must not lose its answer to padding: the loss keeps a
        # fixed-width suffix, so both rows have to contribute equally
        batch = self._batch_rows(
            [([0, 1, 2], [1, 2, 3]), ([0, 1, 2, 3, 0, 1], [2, 3, 0])]
        )
        infos = batch.additional_infos
        cu = infos[PROMPT_CU_SEQLENS]
        lengths = (cu[1:] - cu[:-1]).tolist()
        self.assertNotEqual(lengths[0], lengths[1], "rows must differ to be a test")

        _, _, labels = _unpack(
            torch.ones(int(cu[-1]), 1),
            cu,
            infos[PROMPT_LABELS],
            int(infos[PROMPT_MAX_SEQLEN]),
            -100,
        )
        window = labels[:, -self.prompt.prompt_plan.logits_suffix_len :]
        supervised = (window != -100).sum(dim=1).tolist()
        self.assertEqual(supervised[0], supervised[1])
        self.assertEqual(supervised[0], self.prompt.sid_space.num_levels)

    def test_model_resizes_to_target_vocab(self) -> None:
        model = self._model()
        rows = model.lm.get_input_embeddings().weight.shape[0]
        self.assertEqual(rows, self.prompt.sid_space.target_vocab)
        self.assertGreater(rows, self.prompt.sid_space.band_hi[-1])

    def test_loss_is_finite_and_backpropagates_into_the_backbone(self) -> None:
        model = self._model()
        batch = self._batch([0, 1, 2, 3, 0, 1], [1, 2, 3])
        loss = model.predict(batch)["loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        grad = model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool((grad.abs().sum() > 0)))

    def test_training_forward_survives_fx_tracing(self) -> None:
        # TrainPipelineSparseDist symbolically traces the model whenever a
        # sharded module exists, and the padded forward reads the collator's
        # width as a host int
        model = self._model()

        class _Wrapper(torch.nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, batch):
                return self.inner.predict(batch)

        torch.fx.symbolic_trace(_Wrapper(model))


if __name__ == "__main__":
    unittest.main()
