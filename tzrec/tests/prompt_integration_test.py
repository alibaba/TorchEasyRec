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
from tzrec.models.model import TrainWrapper
from tzrec.tests.prompt_test_util import (
    GenrecModelTestBase,
    assemble_into,
    offset_sid_codes,
)

_CODEBOOK = [4, 4, 4]
_WORDS = ["History", "Predict", ":", ".", "<unk>", "<|im_end|>"]


class PromptStackIntegrationTest(GenrecModelTestBase):
    """compile -> assemble -> model, on the real code path."""

    def _batch_from_codes(self, hist, answer):
        parsed = {
            "hist.values": torch.tensor(offset_sid_codes(hist, _CODEBOOK)),
            "hist.lengths": torch.tensor([len(hist)]),
            "answer.values": torch.tensor(offset_sid_codes(answer, _CODEBOOK)),
            "answer.lengths": torch.tensor([len(answer)]),
        }
        streams = assemble_into(self.compiled_prompt, parsed)
        batch = Batch()
        batch.additional_infos.update(
            {k: torch.from_numpy(np.asarray(v)) for k, v in streams.items()}
        )
        return batch

    def test_written_digests_satisfy_the_restore_guard(self) -> None:
        # write_hf_assets is the only writer and check_prompt_assets the only
        # reader, so a round trip is what proves they agree on the location
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

    def test_loss_is_finite_and_backpropagates_into_the_backbone(self) -> None:
        model = self._model()
        batch = self._batch_from_codes([0, 1, 2, 3, 0, 1], [1, 2, 3])
        predictions = model.predict(batch)
        loss = model.loss(predictions, batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        grad = model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool((grad.abs().sum() > 0)))

    def test_training_forward_survives_fx_tracing(self) -> None:
        # TrainPipelineSparseDist symbolically traces the model whenever a
        # sharded module exists. Trace TrainWrapper itself, not predict alone:
        # it iterates predictions and losses, which a leaf returning a bare
        # Proxy cannot survive.
        model = self._model()

        torch.fx.symbolic_trace(TrainWrapper(model))


if __name__ == "__main__":
    unittest.main()
