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
from parameterized import parameterized
from transformers import AutoConfig

from tzrec.datasets.utils import Batch
from tzrec.models.model import TrainWrapper
from tzrec.protos.models.genrec_model_pb2 import GenrecModelConfig
from tzrec.tests.prompt_test_util import (
    GenrecModelTestBase,
    assemble_into,
    offset_sid_codes,
)
from tzrec.utils.test_util import (
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)

_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class PromptStackIntegrationTest(GenrecModelTestBase):
    """compile -> assemble -> model, on the real code path."""

    def _batch_from_codes(self, hist, answer):
        return self._batch_from_rows([hist], [answer])

    def _batch_from_rows(self, hist_rows, answer_rows):
        parsed = {
            "hist.values": torch.from_numpy(
                np.concatenate([offset_sid_codes(row, _CODEBOOK) for row in hist_rows])
            ),
            "hist.lengths": torch.tensor([len(row) for row in hist_rows]),
            "answer.values": torch.from_numpy(
                np.concatenate(
                    [offset_sid_codes(row, _CODEBOOK) for row in answer_rows]
                )
            ),
            "answer.lengths": torch.tensor([len(row) for row in answer_rows]),
        }
        streams = assemble_into(self.compiled_prompt, parsed)
        batch = Batch()
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch

    def test_written_digests_satisfy_the_restore_guard(self) -> None:
        from tzrec.prompt.persist import check_prompt_assets
        from tzrec.utils.hf_export_util import write_hf_assets

        model = self._model()
        ckpt = os.path.join(self.test_dir, "model.ckpt-1")
        write_hf_assets(model, ckpt)

        check_prompt_assets(self.compiled_prompt, ckpt)
        self.assertTrue(os.path.exists(os.path.join(ckpt, "hf_export_meta.json")))

    def test_model_resizes_to_target_vocab_size(self) -> None:
        model = self._model()
        rows = model.lm.get_input_embeddings().weight.shape[0]
        self.assertEqual(rows, self.compiled_prompt.sid_space.target_vocab_size)
        self.assertGreater(rows, self.compiled_prompt.sid_space.band_hi[-1])

    @parameterized.expand(
        [["qwen2"], ["qwen3"]],
        name_func=parameterized_name_func,
    )
    def test_packed_rows_match_solo_runs_and_backpropagate(
        self, model_type: str
    ) -> None:
        device = torch.device("cuda")
        backbone = os.path.join(self.test_dir, model_type)
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
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            model = self._model(
                lm_parameter_dtype=GenrecModelConfig.BF16,
                hf_model_name_or_path=backbone,
            ).to(device)
        model.eval()
        hist_rows = [[0, 1, 2], [3, 0, 1, 2, 3, 0]]
        answer_rows = [[1, 2, 3], [2, 3, 0]]
        packed_batch = self._batch_from_rows(hist_rows, answer_rows).to(device)

        packed = model.predict(packed_batch)
        with torch.no_grad():
            solos = [
                model.predict(self._batch_from_codes(hist, answer).to(device))
                for hist, answer in zip(hist_rows, answer_rows)
            ]
            changed = model.predict(
                self._batch_from_rows([[3, 3, 3], hist_rows[1]], answer_rows).to(device)
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

    def test_training_forward_survives_fx_tracing(self) -> None:
        model = self._model()

        torch.fx.symbolic_trace(TrainWrapper(model))


if __name__ == "__main__":
    unittest.main()
