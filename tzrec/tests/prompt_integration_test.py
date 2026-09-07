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
import unittest
from unittest import mock

import numpy as np
import torch
import torch.fx

from tzrec.datasets.utils import Batch
from tzrec.main import export
from tzrec.models.model import TrainWrapper
from tzrec.prompt.assembler import PromptAssembler
from tzrec.tests.prompt_test_util import (
    GenrecModelTestBase,
    assemble_into,
    export_tiny_genrec,
    offset_sid_codes,
)
from tzrec.utils.test_util import make_test_dir

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
        model = self._model()

        torch.fx.symbolic_trace(TrainWrapper(model))


def _serving_batch():
    """One request as the front-end reads it: an INLINE history, one behaviour."""
    return {
        "hist.values": torch.tensor(offset_sid_codes([0, 1, 2, 3, 0, 1], _CODEBOOK)),
        "hist.lengths": torch.tensor([6]),
        "beh.values": torch.tensor([3, 9]),
        "beh.lengths": torch.tensor([2]),
    }


class GenrecExportIntegrationTest(unittest.TestCase):
    """checkpoint -> export -> the artifacts a serving runtime loads."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def test_export_writes_a_loadable_serving_directory(self) -> None:
        exported = export_tiny_genrec(self.test_dir)
        for name in (
            "config.json",
            "model.safetensors",
            "prompt/prompt.json",
            "prompt/tokenizer/tokenizer.json",
            "prompt/tokenizer/tokenizer_config.json",
            "frontend/scripted_model.pt",
            "frontend/fg.json",
            "frontend/pipeline.config",
            "frontend/model_acc.json",
        ):
            self.assertTrue(
                os.path.exists(os.path.join(exported.export_dir, name)), name
            )
        self.assertFalse(
            os.path.exists(os.path.join(exported.export_dir, "frontend/sparse"))
        )

        # the collator and the scripted front-end are two call sites of one walk
        batch = _serving_batch()
        compiled = exported.compiled_prompt
        collator = PromptAssembler(
            compiled.prompt_plan, compiled.sid_space, include_response=False
        ).forward({k: v.numpy() for k, v in batch.items()})
        front_end = torch.jit.load(
            os.path.join(exported.export_dir, "frontend", "scripted_model.pt")
        )
        out = front_end(batch, torch.device("cpu"))
        self.assertEqual(
            out["input_ids"].tolist(), collator["prompt_input_ids"].tolist()
        )
        self.assertEqual(
            out["cu_seqlens"].tolist(), collator["prompt_cu_seqlens"].tolist()
        )
        self.assertEqual(
            out["hole_positions"].tolist(), collator["prompt_hole_positions"].tolist()
        )
        self.assertEqual(tuple(out["slot_embeds"].shape), (2, 32))

    def test_distributed_embedding_export_writes_the_processor_shape(self) -> None:
        exported = export_tiny_genrec(self.test_dir)
        dist_dir = os.path.join(self.test_dir, "export_dist")
        with mock.patch.dict(os.environ, {"USE_DISTRIBUTED_EMBEDDING": "1"}):
            export(
                exported.config_path, dist_dir, checkpoint_path=exported.checkpoint_dir
            )
        frontend_dir = os.path.join(dist_dir, "frontend")
        sparse_dir = os.path.join(frontend_dir, "sparse")
        for name in (
            "sparse_embeddings-00-of-01.npz",
            "sparse_embeddings-00-of-01.json",
            "sparse_embedding.json",
            "sparse_features.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(sparse_dir, name)), name)
        with open(os.path.join(frontend_dir, "model_acc.json"), "r") as f:
            acc = json.load(f)
        self.assertEqual(acc["DISTRIBUTED_EMBEDDING"], "1")
        self.assertEqual(acc["INPUT_TILE"], "3")
        with open(os.path.join(frontend_dir, "dense_meta.json"), "r") as f:
            dense_meta = json.load(f)
        self.assertEqual(dense_meta, {"sequence__ec": ["beh__ec", "beh__lengths"]})
        with open(os.path.join(dist_dir, "prompt", "prompt.json"), "r") as f:
            self.assertEqual(json.load(f)["frontend"]["lookup"], "host")

        # a processor simulator: look the ids up in the exported tables the way
        # the distributed-embedding stage does, then feed the dense stage
        batch = _serving_batch()
        with open(os.path.join(sparse_dir, "sparse_features.json"), "r") as f:
            table_name = json.load(f)["beh__ec"]["embedding_name"]
        with np.load(os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz")) as npz:
            table = npz[table_name]
        host = dict(batch)
        host["beh"] = torch.from_numpy(
            table[batch["beh.values"].numpy()].astype(np.float32)
        )
        host["beh__lengths"] = batch["beh.lengths"]
        got = torch.jit.load(os.path.join(frontend_dir, "scripted_model.pt"))(host)
        expected = torch.jit.load(
            os.path.join(exported.export_dir, "frontend", "scripted_model.pt")
        )(batch)
        self.assertTrue(
            torch.allclose(got["slot_embeds"], expected["slot_embeds"], atol=1e-6)
        )
        self.assertTrue(torch.equal(got["hole_keys"], expected["hole_keys"]))
        self.assertEqual(got["input_ids"].tolist(), expected["input_ids"].tolist())


if __name__ == "__main__":
    unittest.main()
