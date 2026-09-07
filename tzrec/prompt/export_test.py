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

import torch
from torchrec import KeyedJaggedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.prompt.export import (
    LOOKUP_ARTIFACT,
    LOOKUP_HOST,
    build_front_end,
    build_slot_tables,
    host_embed_keys,
    write_composite_config,
    write_front_end_dir,
)
from tzrec.prompt.persist import write_serving_contract
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.tests.prompt_test_util import (
    _HIST,
    GenrecModelTestBase,
    create_prompt_feature,
    offset_sid_codes,
    projected_feature,
)
from tzrec.utils.state_dict_util import init_parameters

_CAT = (
    'id_feature { feature_name: "cat" expression: "user:cat" num_buckets: 16 '
    'embedding_dim: 8 pooling: "mean" }'
)
_VEC = 'raw_feature { feature_name: "vec" expression: "user:vec" value_dim: 4 }'
_TEMPLATE = "History : {{hist}} . {{beh}} {{cat}} Predict :"


class PromptExportTest(GenrecModelTestBase):
    """Builds the serving pieces from an in-memory model with projected slots."""

    def setUp(self) -> None:
        super().setUp()
        self.features = [
            create_prompt_feature(_HIST),
            create_prompt_feature(projected_feature("beh", 8)),
            create_prompt_feature(_CAT),
        ]
        self.compiled_prompt = self._compile(self.features, template=_TEMPLATE)
        self.model = self._model()
        init_parameters(self.model, torch.device("cpu"))
        self.raw = {
            "hist.values": torch.tensor(
                offset_sid_codes([0, 1, 2, 3, 0, 1], [4, 4, 4])
            ),
            "hist.lengths": torch.tensor([6]),
            "beh.values": torch.tensor([3, 9]),
            "beh.lengths": torch.tensor([2]),
            "cat.values": torch.tensor([5]),
            "cat.lengths": torch.tensor([1]),
        }
        sparse = KeyedJaggedTensor(
            keys=["beh", "cat"],
            values=torch.tensor([3, 9, 5]),
            lengths=torch.tensor([2, 1]),
        )
        self.grouped = self.model.embedding_group(
            Batch(sparse_features={BASE_DATA_GROUP: sparse})
        )

    def _expected_slot_embeds(self) -> torch.Tensor:
        projections = self.model._slot_projections
        return torch.cat(
            [
                projections[0](self.grouped["beh.sequence"]),
                projections[1](self.grouped["cat"]),
            ]
        )

    def test_slot_tables_reproduce_the_embedding_group(self) -> None:
        """The carried tables look up exactly what the training model did."""
        tables = build_slot_tables(self.model, self.compiled_prompt, self.features)
        self.assertEqual(len(tables), 2)
        self.assertTrue(
            torch.allclose(tables[0](self.raw), self.grouped["beh.sequence"])
        )
        self.assertTrue(torch.allclose(tables[1](self.raw), self.grouped["cat"]))

    def test_the_artifact_shape_matches_the_model_on_the_same_batch(self) -> None:
        front_end, meta, dense_meta = build_front_end(
            self.model, self.compiled_prompt, self.features, carry_tables=True
        )
        out = front_end(self.raw)
        self.assertTrue(
            torch.allclose(out["slot_embeds"], self._expected_slot_embeds(), atol=1e-6)
        )
        self.assertEqual(out["hole_slot_counts"].tolist(), [2, 1])
        self.assertEqual(meta["lookup"], LOOKUP_ARTIFACT)
        self.assertIsNone(dense_meta)
        self.assertEqual(
            meta["inputs"],
            [
                "hist.values",
                "hist.lengths",
                "beh.values",
                "beh.lengths",
                "cat.values",
                "cat.lengths",
            ],
        )
        self.assertEqual(front_end.vocab_hash, self.compiled_prompt.vocab_hash)

    def test_the_host_shape_reads_the_processor_keys(self) -> None:
        """A host lookup stage hands over rows named as dense_meta describes."""
        artifact, _, _ = build_front_end(
            self.model, self.compiled_prompt, self.features, carry_tables=True
        )
        host, meta, dense_meta = build_front_end(
            self.model, self.compiled_prompt, self.features, carry_tables=False
        )
        self.assertEqual(meta["lookup"], LOOKUP_HOST)
        self.assertEqual(
            dense_meta,
            {"sequence__ec": ["beh__ec", "beh__lengths"], "cat__ebc": ["cat__ebc"]},
        )
        self.assertEqual(
            host_embed_keys(self.compiled_prompt)[0], [["beh"], ["cat__ebc"]]
        )
        batch = dict(self.raw)
        batch["beh"] = self.grouped["beh.sequence"]
        batch["beh__lengths"] = self.raw["beh.lengths"]
        batch["cat__ebc"] = self.grouped["cat"]
        expected = artifact(self.raw)
        got = torch.jit.script(host.eval())(batch)
        self.assertTrue(torch.allclose(got["slot_embeds"], expected["slot_embeds"]))
        self.assertTrue(torch.equal(got["hole_keys"], expected["hole_keys"]))
        self.assertIn("beh__lengths", meta["inputs"])
        self.assertIn("cat__ebc", meta["inputs"])

    def test_a_member_without_a_plain_table_cannot_be_carried(self) -> None:
        features = [create_prompt_feature(_HIST), create_prompt_feature(_VEC)]
        compiled = self._compile(features, template="History : {{hist}} {{vec}} :")
        model = self._model(features=features, compiled_prompt=compiled)
        init_parameters(model, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "USE_DISTRIBUTED_EMBEDDING=1"):
            build_slot_tables(model, compiled, features)

    def test_write_composite_config_is_idempotent(self) -> None:
        path = os.path.join(self.test_dir, "config.json")
        backbone = {
            "model_type": "qwen2",
            "architectures": ["Qwen2ForCausalLM"],
            "hidden_size": 32,
            "vocab_size": 77,
        }
        with open(path, "w") as f:
            json.dump(backbone, f)
        write_composite_config(self.test_dir)
        write_composite_config(self.test_dir)
        with open(path, "r") as f:
            composite = json.load(f)
        self.assertEqual(composite["architectures"], ["PromptGenRecForCausalLM"])
        self.assertEqual(composite["model_type"], "prompt_genrec")
        self.assertEqual(composite["text_config"], backbone)
        self.assertEqual(composite["hidden_size"], 32)

    def test_write_front_end_dir_lays_out_a_tzrec_model_dir(self) -> None:
        front_end, _, _ = build_front_end(
            self.model, self.compiled_prompt, self.features, carry_tables=True
        )
        frontend_dir = write_front_end_dir(
            front_end,
            EasyRecConfig(model_dir="unused"),
            self.features,
            self.test_dir,
            dense_meta={"sequence__ec": ["beh__ec", "beh__lengths"]},
        )
        for name in (
            "scripted_model.pt",
            "fg.json",
            "pipeline.config",
            "model_acc.json",
            "dense_meta.json",
        ):
            self.assertTrue(os.path.exists(os.path.join(frontend_dir, name)), name)
        with open(os.path.join(frontend_dir, "model_acc.json"), "r") as f:
            self.assertEqual(json.load(f)["SPARSE_INT64"], "1")
        with open(os.path.join(frontend_dir, "fg.json"), "r") as f:
            names = [feature["feature_name"] for feature in json.load(f)["features"]]
        self.assertEqual(names, ["hist", "beh", "cat"])
        loaded = torch.jit.load(os.path.join(frontend_dir, "scripted_model.pt"))
        out = loaded(self.raw, torch.device("cpu"))
        self.assertTrue(
            torch.allclose(out["slot_embeds"], self._expected_slot_embeds(), atol=1e-6)
        )

    def test_serving_contract_lists_what_the_runtime_reads(self) -> None:
        _, meta, _ = build_front_end(
            self.model, self.compiled_prompt, self.features, carry_tables=True
        )
        path = write_serving_contract(self.compiled_prompt, self.test_dir, meta)
        with open(path, "r") as f:
            contract = json.load(f)
        space = self.compiled_prompt.sid_space
        self.assertEqual(contract["sid_space"]["band_lo"], list(space.band_lo))
        self.assertEqual(
            contract["sid_space"]["base_vocab_size"], space.base_vocab_size
        )
        self.assertEqual(contract["sid_space"]["bundle_uuid"], "")
        self.assertEqual(
            contract["decode"],
            [
                {
                    "level": level,
                    "band_lo": space.band_lo[level],
                    "band_hi": space.band_hi[level],
                }
                for level in range(3)
            ],
        )
        self.assertEqual(contract["vocab_hash"], self.compiled_prompt.vocab_hash)
        self.assertEqual(contract["plan_hash"], self.compiled_prompt.plan_hash)
        self.assertEqual(contract["frontend"], meta)
        self.assertEqual(contract["static_prefix_len"], 2)

    def test_a_static_run_in_the_response_is_a_forced_token(self) -> None:
        compiled = self._compile(
            self.features, template=_TEMPLATE, response="Predict : {{answer}}"
        )
        path = write_serving_contract(compiled, self.test_dir, {})
        with open(path, "r") as f:
            decode = json.load(f)["decode"]
        self.assertEqual(len(decode), 5)
        self.assertEqual(decode[0], {"token_id": 1})
        self.assertEqual(decode[1], {"token_id": 2})
        self.assertIn("band_lo", decode[2])


if __name__ == "__main__":
    unittest.main()
