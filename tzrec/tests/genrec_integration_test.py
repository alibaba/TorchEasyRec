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

import glob
import json
import os
import shutil
import unittest
from unittest import mock

import numpy as np
import torch
from google.protobuf import text_format
from pyarrow import parquet as pq

from tzrec.main import _create_features
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
    PromptAssembler,
)
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.hole_keys import HOLE_KEYS, HoleKeyBuilder
from tzrec.tests import utils
from tzrec.utils import config_util
from tzrec.utils.test_util import (
    create_genrec_test_tokenizer,
    create_tiny_causal_lm,
    gpu_unavailable,
    make_test_dir,
    mark_ci_scope,
)

_MOCK_CONFIG = "tzrec/tests/configs/genrec_causal_lm_model_mock.config"
_BUNDLE_UUID = "bundle-test"
_CODEBOOK = [4, 4, 4]
_BEH = (
    'sequence_id_feature { feature_name: "beh" expression: "user:beh" '
    "num_buckets: 32 embedding_dim: 8 sequence_length: 2 }"
)


class GenRecIntegrationTest(unittest.TestCase):
    """train_eval -> eval -> export in subprocesses, then read the export back."""

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir()
        # the tiny LM trains on one rank, and one rank writes one sparse shard
        patcher = mock.patch.dict(os.environ, {"TEST_NPROC_PER_NODE": "1"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _prepare_config(self) -> str:
        """Write the tiny backbone, tokenizer, manifest and data; return the config."""
        backbone = os.path.join(self.test_dir, "backbone")
        create_tiny_causal_lm(64).save_pretrained(backbone)
        tokenizer = create_genrec_test_tokenizer(
            os.path.join(self.test_dir, "tok.json")
        )
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": _CODEBOOK, "bundle_uuid": _BUNDLE_UUID}, f)
        self.data_glob = utils.create_mock_prompt_data(
            os.path.join(self.test_dir, "data"), _CODEBOOK
        )

        config = config_util.load_pipeline_config(_MOCK_CONFIG)
        config.train_input_path = self.data_glob
        config.eval_input_path = self.data_glob
        config.model_config.genrec_causal_lm_model.hf_model_name_or_path = backbone
        config.prompt_config.tokenizer_path = tokenizer
        config.prompt_config.sid_space.manifest_path = manifest
        text_format.Merge(_BEH, config.feature_configs.add())
        config.prompt_config.prompt = "History : {{hist}} . {{beh}} Predict :"
        config_path = os.path.join(self.test_dir, "genrec.config")
        config_util.save_message(config, config_path)
        return config_path

    def _train_eval_export(self) -> str:
        """Run the pipeline; return the trained ``pipeline.config`` path."""
        config_path = self._prepare_config()
        self.success = utils.test_train_eval(config_path, self.test_dir)
        trained = os.path.join(self.test_dir, "pipeline.config")
        if self.success:
            self.success = utils.test_eval(trained, self.test_dir)
        if self.success:
            self.success = utils.test_export(
                trained, self.test_dir, env_str="QUANT_EMB=0"
            )
        self.assertTrue(self.success)
        self.assertTrue(
            os.path.exists(os.path.join(self.test_dir, "train/eval_result.txt"))
        )
        return trained

    def _request(self, columns, rows: int = 4):
        """The parsed dict a served front-end reads, from the first mock rows."""
        table = pq.read_table(sorted(glob.glob(self.data_glob))[0]).slice(0, rows)
        out = {}
        for column in columns:
            lists = table.column(column).to_pylist()
            out[column + ".values"] = torch.tensor(
                [v for row in lists for v in row], dtype=torch.int64
            )
            out[column + ".lengths"] = torch.tensor([len(row) for row in lists])
        return out

    def test_genrec_train_eval_export(self):
        trained = self._train_eval_export()
        export_dir = os.path.join(self.test_dir, "export")
        for name in (
            "scripted_model.pt",
            "fg.json",
            "pipeline.config",
            "model_acc.json",
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(export_dir, name)), name)
        self.assertFalse(os.path.exists(os.path.join(export_dir, "sparse")))

        # the artifact is the collator's walk plus the serving-only fold
        config = config_util.load_pipeline_config(trained)
        features = _create_features(list(config.feature_configs), config.data_config)
        compiled = compile_prompt(config.prompt_config, features, ["answer"])
        data = self._request(["hist", "beh"])
        front_end = torch.jit.load(os.path.join(export_dir, "scripted_model.pt"))
        # an LLM engine calls it with the batch alone; a processor adds a device
        out = front_end(data)
        with_device = front_end(data, torch.device("cpu"))
        walk = PromptAssembler(
            compiled.prompt_plan, compiled.sid_space, include_response=False
        )(data)
        for key in (INPUT_IDS, CU_SEQLENS, HOLE_POSITIONS, HOLE_SLOT_COUNTS):
            self.assertTrue(torch.equal(out[key], walk[key]), key)
            self.assertTrue(torch.equal(with_device[key], walk[key]), key)
        self.assertTrue(
            torch.equal(out[HOLE_KEYS], HoleKeyBuilder(compiled.prompt_plan)(data))
        )
        positions = out[HOLE_POSITIONS]
        self.assertGreater(positions.numel(), 0)
        self.assertTrue(
            bool(
                torch.all(
                    out[INPUT_IDS][positions] == compiled.sid_space.sentinel_token_id
                )
            )
        )
        self.assertEqual(tuple(out["slot_embeds"].shape), (int(positions.numel()), 32))
        self.assertEqual(out["slot_embeds"].dtype, torch.float32)

        # what the engine loads beside it
        with open(os.path.join(export_dir, "config.json"), "r") as f:
            hf_config = json.load(f)
        self.assertEqual(hf_config["architectures"], ["GenRecForCausalLM"])
        self.assertEqual(hf_config["model_type"], "genrec")
        self.assertEqual(
            hf_config["text_config"]["architectures"], ["Qwen2ForCausalLM"]
        )
        self.assertEqual(hf_config["text_config"]["model_type"], "qwen2")
        self.assertEqual(hf_config["vocab_size"], compiled.sid_space.target_vocab_size)
        self.assertEqual(hf_config["eos_token_id"], compiled.sid_space.eos_token_id)
        self.assertEqual(hf_config["pad_token_id"], compiled.sid_space.pad_token_id)
        self.assertNotIn("sid_space", hf_config)
        from transformers import AutoTokenizer

        # the index builder derives the token base from the first SID token
        tokenizer = AutoTokenizer.from_pretrained(export_dir)
        self.assertEqual(
            tokenizer.convert_tokens_to_ids("<|sid_0|>"),
            compiled.sid_space.base_vocab_size,
        )
        self.assertEqual(tokenizer.decode([compiled.sid_space.band_lo[0]]), "<|sid_0|>")
        self.assertEqual(
            tokenizer.convert_ids_to_tokens(compiled.sid_space.sentinel_token_id),
            "<|pg_hole|>",
        )
        self.assertEqual(tokenizer.eos_token_id, compiled.sid_space.eos_token_id)

    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_genrec_export_distributed_embedding(self):
        trained = self._train_eval_export()
        dist_dir = os.path.join(self.test_dir, "export_dist")
        self.success = utils.test_export(
            trained,
            self.test_dir,
            export_dir=dist_dir,
            env_str="USE_DISTRIBUTED_EMBEDDING=1 QUANT_EMB=0",
        )
        self.assertTrue(self.success)
        sparse_dir = os.path.join(dist_dir, "sparse")
        for name in (
            "sparse_embeddings-00-of-01.npz",
            "sparse_embedding.json",
            "sparse_features.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(sparse_dir, name)), name)
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            self.assertTrue(os.path.exists(os.path.join(dist_dir, name)), name)
        with open(os.path.join(dist_dir, "model_acc.json"), "r") as f:
            self.assertEqual(json.load(f)["DISTRIBUTED_EMBEDDING"], "1")
        with open(os.path.join(dist_dir, "dense_meta.json"), "r") as f:
            self.assertEqual(json.load(f)["sequence__ec"], ["beh__ec", "beh__lengths"])

        # a processor simulator: one request is one user, looked up in the
        # exported tables the way the distributed-embedding stage does, then
        # fed to the dense stage input-tiled with one candidate
        request = self._request(["hist", "beh"], rows=1)
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
            os.path.join(self.test_dir, "export", "scripted_model.pt")
        )(request)
        self.assertTrue(
            torch.allclose(got["slot_embeds"], expected["slot_embeds"], atol=1e-6)
        )
        self.assertTrue(torch.equal(got["hole_keys"], expected["hole_keys"]))
        self.assertEqual(got["input_ids"].tolist(), expected["input_ids"].tolist())


if __name__ == "__main__":
    unittest.main()
