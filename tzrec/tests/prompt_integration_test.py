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
import subprocess
import sys
import unittest

import numpy as np
import torch
import torch.fx

from tzrec.datasets.utils import Batch
from tzrec.models.model import TrainWrapper
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    HOLE_POSITIONS,
    INPUT_IDS,
    PromptAssembler,
)
from tzrec.prompt.hole_keys import HOLE_KEYS, HoleKeyBuilder
from tzrec.tests.prompt_test_util import (
    GenrecModelTestBase,
    assemble_into,
    export_tiny_genrec,
    offset_sid_codes,
)
from tzrec.utils.test_util import gpu_unavailable, make_test_dir, mark_ci_scope

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
        batch = Batch()
        batch.additional_infos.update(assemble_into(self.compiled_prompt, parsed))
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


class GenrecExportIntegrationTest(unittest.TestCase):
    """checkpoint -> export -> the artifacts an LLM engine and a processor load."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()

    def test_export_writes_one_serving_directory(self) -> None:
        exported = export_tiny_genrec(self.test_dir, env={"QUANT_EMB": "0"})
        for name in (
            "scripted_model.pt",
            "fg.json",
            "pipeline.config",
            "model_acc.json",
            "config.json",
            "model.safetensors",
            "prompt/prompt.json",
            "prompt/tokenizer/tokenizer.json",
            "prompt/tokenizer/tokenizer_config.json",
        ):
            self.assertTrue(
                os.path.exists(os.path.join(exported.export_dir, name)), name
            )
        self.assertFalse(os.path.exists(os.path.join(exported.export_dir, "sparse")))

        # the exported artifact and the collator are two call sites of one walk;
        # only the artifact folds hole keys
        data = exported.sample_data
        compiled = exported.compiled_prompt
        collator = PromptAssembler(
            compiled.prompt_plan, compiled.sid_space, include_response=False
        )(data)
        front_end = torch.jit.load(
            os.path.join(exported.export_dir, "scripted_model.pt")
        )
        out = front_end(data, torch.device("cpu"))
        for key in (INPUT_IDS, CU_SEQLENS, HOLE_POSITIONS):
            self.assertTrue(torch.equal(out[key], collator[key]), key)
        self.assertTrue(
            torch.equal(out[HOLE_KEYS], HoleKeyBuilder(compiled.prompt_plan)(data))
        )
        self.assertNotIn(HOLE_KEYS, collator)
        self.assertEqual(
            tuple(out["slot_embeds"].shape), (int(out[HOLE_POSITIONS].numel()), 32)
        )

    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_distributed_embedding_export_writes_the_processor_shape(self) -> None:
        exported = export_tiny_genrec(self.test_dir, env={"QUANT_EMB": "0"})
        dist_dir = os.path.join(self.test_dir, "export_dist")
        # its own process: the planner wants a fresh nccl group on the device
        env = dict(os.environ)
        env.update(
            {
                "PYTHONPATH": ".",
                "USE_DISTRIBUTED_EMBEDDING": "1",
                "QUANT_EMB": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": os.environ.get("MASTER_PORT", "29511"),
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "1",
            }
        )
        log_path = os.path.join(self.test_dir, "export_dist.log")
        with open(log_path, "w") as log:
            code = subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "tzrec.export",
                    "--pipeline_config_path",
                    exported.config_path,
                    "--export_dir",
                    dist_dir,
                    "--checkpoint_path",
                    exported.checkpoint_dir,
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        with open(log_path, "r") as log:
            self.assertEqual(code, 0, log.read()[-4000:])
        sparse_dir = os.path.join(dist_dir, "sparse")
        for name in (
            "sparse_embeddings-00-of-01.npz",
            "sparse_embedding.json",
            "sparse_features.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(sparse_dir, name)), name)
        with open(os.path.join(dist_dir, "model_acc.json"), "r") as f:
            self.assertEqual(json.load(f)["DISTRIBUTED_EMBEDDING"], "1")
        with open(os.path.join(dist_dir, "dense_meta.json"), "r") as f:
            dense_meta = json.load(f)
        self.assertEqual(dense_meta["sequence__ec"], ["beh__ec", "beh__lengths"])

        # a processor simulator: one request is one user, looked up in the
        # exported tables the way the distributed-embedding stage does, then
        # fed to the dense stage input-tiled with one candidate
        sample = exported.sample_data
        request = {
            "hist.values": sample["hist.values"][: int(sample["hist.lengths"][0])],
            "hist.lengths": sample["hist.lengths"][:1],
            "beh.values": sample["beh.values"][: int(sample["beh.lengths"][0])],
            "beh.lengths": sample["beh.lengths"][:1],
        }
        with open(os.path.join(sparse_dir, "sparse_features.json"), "r") as f:
            table_name = json.load(f)["beh__ec"]["embedding_name"]
        with np.load(os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz")) as npz:
            table = npz[table_name]
        data = dict(request)
        data["beh"] = torch.from_numpy(
            table[request["beh.values"].numpy()].astype(np.float32)
        )
        data["beh__lengths"] = request["beh.lengths"]
        data["batch_size"] = torch.tensor(1)
        # the planner exported on the GPU; the processor loads onto its device
        got = torch.jit.load(
            os.path.join(dist_dir, "scripted_model.pt"), map_location="cpu"
        )(data)
        expected = torch.jit.load(
            os.path.join(exported.export_dir, "scripted_model.pt")
        )(request)
        self.assertTrue(
            torch.allclose(got["slot_embeds"], expected["slot_embeds"], atol=1e-6)
        )
        self.assertTrue(torch.equal(got["hole_keys"], expected["hole_keys"]))
        self.assertEqual(got["input_ids"].tolist(), expected["input_ids"].tolist())


if __name__ == "__main__":
    unittest.main()
