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
import hashlib
import json
import os
import shutil
import unittest
from unittest import mock

import numpy as np
import pyarrow as pa
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
from tzrec.protos.pipeline_pb2 import EasyRecConfig
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
# the template's words plus what the tokenizer needs; token ids are drawn below it
_WORDS = (
    "User",
    "Context",
    "History",
    "Tags",
    "Title",
    "Predict",
    ":",
    ".",
    "<unk>",
    "<|im_end|>",
)
# every slot member the served front-end reads raw, DEEP, dense and jagged alike
_MEMBERS = [
    "hist__sid",
    "title",
    "age",
    "city",
    "home_city",
    "context_id",
    "score",
    "tags__tag_a",
    "tags__tag_b",
    "tags__text",
]


def _md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


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
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": _CODEBOOK, "bundle_uuid": _BUNDLE_UUID}, f)
        self.data_glob = utils.create_mock_prompt_data(
            os.path.join(self.test_dir, "data"), _CODEBOOK, num_words=len(_WORDS)
        )

        # parse rather than load: loading would default the tokenize features'
        # vocab_file to the placeholder tokenizer_path before it is replaced
        config = EasyRecConfig()
        with open(_MOCK_CONFIG) as f:
            text_format.Merge(f.read(), config)
        config.train_input_path = self.data_glob
        config.eval_input_path = self.data_glob
        config.model_config.genrec_causal_lm_model.hf_model_name_or_path = backbone
        config.prompt_config.tokenizer_path = tokenizer
        config.prompt_config.sid_space.manifest_path = manifest
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
        """The parsed dict a served front-end reads, from the first mock rows.

        A multi-value sequence column is one list per item; it also carries
        ``key_lengths``, the codes each item holds.
        """
        table = pq.read_table(sorted(glob.glob(self.data_glob))[0]).slice(0, rows)
        out = {}
        for column in columns:
            arrow = table.column(column)
            lists = arrow.to_pylist()
            if pa.types.is_floating(arrow.type.value_type):
                # a dense member is one row of values with no lengths
                out[column + ".values"] = torch.tensor(lists, dtype=torch.float32)
                continue
            out[column + ".lengths"] = torch.tensor([len(row) for row in lists])
            items = [v for row in lists for v in row]
            if items and isinstance(items[0], list):
                out[column + ".key_lengths"] = torch.tensor([len(i) for i in items])
                items = [v for item in items for v in item]
            out[column + ".values"] = torch.tensor(items, dtype=torch.int64)
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
            "generation_config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(export_dir, name)), name)
        self.assertFalse(os.path.exists(os.path.join(export_dir, "sparse")))

        config = config_util.load_pipeline_config(trained)
        # the HF assets are composed at export time; checkpoints carry none
        for ckpt in glob.glob(os.path.join(config.model_dir, "model.ckpt-*")):
            for name in (
                "config.json",
                "generation_config.json",
                "hf_export_meta.json",
            ):
                self.assertFalse(
                    os.path.exists(os.path.join(ckpt, name)), f"{ckpt}/{name}"
                )

        # the loaded config carries the injected vocabulary, and the export
        # ships it as an FG asset beside fg.json
        title = next(
            fc.tokenize_feature
            for fc in config.feature_configs
            if fc.WhichOneof("feature") == "tokenize_feature"
        )
        self.assertEqual(title.vocab_file, config.prompt_config.tokenizer_path)
        with open(os.path.join(export_dir, "fg.json")) as f:
            fg_title = next(
                fg
                for fg in json.load(f)["features"]
                if fg.get("feature_name") == "title"
            )
        self.assertEqual(
            _md5(os.path.join(export_dir, fg_title["vocab_file"])),
            _md5(config.prompt_config.tokenizer_path),
        )

        # the artifact is the collator's walk plus the serving-only fold
        features = _create_features(list(config.feature_configs), config.data_config)
        compiled = compile_prompt(config.prompt_config, features, ["answer"])
        data = self._request(_MEMBERS)
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
        # profile and ctx are DEEP, one hole per row; hist_tags has two items
        self.assertEqual(out[HOLE_SLOT_COUNTS].tolist(), [4, 4, 8])
        # the two DEEP slots have equal width and share one projection module
        plan = compiled.projection_plan
        self.assertEqual(sorted(plan.projections), ["hist_tags", "user_proj"])
        self.assertEqual(plan.slot_to_module[0], plan.slot_to_module[1])
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

        # decode runs only against a checkpoint: the exported front-end has no LM
        predict_dir = os.path.join(self.test_dir, "predict")
        self.success = utils.test_predict_checkpoint(
            trained,
            self.data_glob,
            predict_dir,
            "answer",
            "generated_sids",
            self.test_dir,
        )
        self.assertTrue(self.success)
        predicted = pq.read_table(predict_dir)
        self.assertIn("answer", predicted.column_names)
        sids = predicted.column("generated_sids").to_pylist()
        self.assertEqual(len(sids), 8)
        for row in sids:
            # num_return_sequences beams of one local code per level
            self.assertEqual([len(beam) for beam in row], [3, 3])
            for beam in row:
                self.assertTrue(all(0 <= c < n for c, n in zip(beam, _CODEBOOK)), beam)

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
            dense_meta = json.load(f)
        # collections form per dim, so tags__text joins tag_a's; the dense
        # score member has no sparse-stage entry and rides on its raw values;
        # all-user DEEP groups are the input-tiled `_user` variant
        self.assertEqual(
            dense_meta,
            {
                "sequence__ec": [
                    "tags__tag_a__ec",
                    "tags__tag_a__lengths",
                    "tags__text__ec",
                    "tags__text__lengths",
                    "tags__tag_b__ec",
                    "tags__tag_b__lengths",
                ],
                "profile__ctx__ebc_user": [
                    "age__ebc",
                    "city__ebc",
                    "home_city__ebc",
                    "context_id__ebc",
                ],
            },
        )

        # a processor simulator: one request is one user, looked up in the
        # exported tables the way the distributed-embedding stage does, then
        # fed to the dense stage input-tiled with one candidate
        request = self._request(_MEMBERS, rows=1)
        with open(os.path.join(sparse_dir, "sparse_features.json"), "r") as f:
            sparse_features = json.load(f)
        data = dict(request)
        with np.load(os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz")) as npz:

            def looked_up(key):
                name = key.rsplit("__", 1)[0]
                table = npz[sparse_features[key]["embedding_name"]]
                rows = table[request[name + ".values"].numpy()].astype(np.float32)
                return torch.from_numpy(rows)

            for key in dense_meta["sequence__ec"][0::2]:
                name = key[: -len("__ec")]
                data[name] = looked_up(key)
                # a multi-value member is fed unpooled; the dense graph reduces it
                lengths = request[name + ".lengths"]
                if name + ".key_lengths" in request:
                    lengths = torch.segment_reduce(
                        request[name + ".key_lengths"].float(), "sum", lengths=lengths
                    ).long()
                data[name + "__lengths"] = lengths
            # every DEEP sparse member holds one id, so a row lookup is its pooling
            data["profile__ctx__ebc_user"] = torch.cat(
                [looked_up(key) for key in dense_meta["profile__ctx__ebc_user"]], dim=1
            )
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
