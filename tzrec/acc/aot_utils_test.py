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
import shutil
import unittest
from typing import Dict, Tuple
from unittest import mock

import torch
from parameterized import parameterized
from torch import nn
from torch.utils._pytree import tree_flatten, treespec_dumps, treespec_loads

from tzrec.acc import aot_utils
from tzrec.acc.aot_utils import (
    _aoti_compile_cfg,
    _build_dynamic_shapes,
    _pad_empty_sparse_values,
    export_model_aot,
    export_unified_model_aot,
    load_model_aot,
)
from tzrec.datasets.data_parser import DataParser, _infer_tile_size
from tzrec.features.feature import FgMode, create_features
from tzrec.models.model import ScriptWrapper
from tzrec.models.multi_tower_din import MultiTowerDIN
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    gpu_unavailable,
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)


def _create_din_input_tile_features_and_groups(
    include_multival: bool = False,
):
    """Build test feature configs and groups for input-tile + DIN AOTI tests."""
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="u_w",
                expression="user:u_w",
                num_buckets=32,
                embedding_dim=4,
                weighted=True,
                pooling="mean",
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="u_dense",
                expression="user:u_dense",
                value_dim=2,
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="i_id",
                expression="item:i_id",
                num_buckets=32,
                embedding_dim=4,
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="i_dense",
                expression="item:i_dense",
                value_dim=2,
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="u_seq",
                sequence_length=10,
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="id",
                            expression="user:u_seq__id",
                            num_buckets=32,
                            embedding_dim=4,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="raw",
                            expression="user:u_seq__raw",
                            value_dim=2,
                        )
                    ),
                ],
            )
        ),
    ]
    if include_multival:
        feature_cfgs[-1].sequence_feature.features.append(
            feature_pb2.SeqFeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="mv",
                    expression="user:u_seq__mv",
                    num_buckets=32,
                    embedding_dim=4,
                    value_dim=0,
                )
            )
        )
    features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
    seq_names = ["u_seq__id", "u_seq__raw"] + (
        ["u_seq__mv"] if include_multival else []
    )
    feature_groups = [
        model_pb2.FeatureGroupConfig(
            group_name="deep",
            feature_names=["u_w", "u_dense", "i_id", "i_dense"],
            group_type=model_pb2.DEEP,
        ),
        model_pb2.FeatureGroupConfig(
            group_name="seq",
            feature_names=["i_id"] + seq_names,
            group_type=model_pb2.SEQUENCE,
        ),
    ]
    return features, feature_groups


class AotUtilsTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    @parameterized.expand(
        [
            ("no_tile", "0", False),
            ("input_tile_2", "2", False),
            ("input_tile_3", "3", False),
            ("input_tile_3_online", "3", True),
        ],
        name_func=parameterized_name_func,
    )
    def test_build_dynamic_shapes_input_tile(self, _name, input_tile, online):
        features, feature_groups = _create_din_input_tile_features_and_groups(
            include_multival=True
        )
        u_bs = 1 if online else 3
        i_bs = 4
        data = {
            "batch_size": torch.full((1,), 4 if online else 1, dtype=torch.int64),
            "u_w.values": torch.tensor([1, 2], dtype=torch.int64),
            "u_w.lengths": torch.full((u_bs,), 1, dtype=torch.int64),
            "u_w.weights": torch.tensor([1.0, 2.0], dtype=torch.float32),
            "u_dense.values": torch.randn(u_bs, 2),
            "i_id.values": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
            "i_id.lengths": torch.ones(i_bs, dtype=torch.int64),
            "i_dense.values": torch.randn(i_bs, 2),
            "u_seq__id.values": torch.tensor([1, 2, 3], dtype=torch.int64),
            "u_seq__id.lengths": torch.full((u_bs,), 1, dtype=torch.int64),
            "u_seq__raw.values": torch.randn(3, 2),
            "u_seq__raw.lengths": torch.full((u_bs,), 1, dtype=torch.int64),
            "u_seq__mv.values": torch.tensor([4, 5, 6, 7], dtype=torch.int64),
            "u_seq__mv.lengths": torch.full((u_bs,), 1, dtype=torch.int64),
            "u_seq__mv.key_lengths": torch.tensor([1, 2, 1], dtype=torch.int64),
        }

        with mock.patch.dict(os.environ, INPUT_TILE=input_tile, clear=False):
            dyn = _build_dynamic_shapes(data, features, feature_groups)

        self.assertEqual(dyn["batch_size"], {})
        # Sequence dense raw values share total sequence length Dim with sparse sequence
        # values, min=0.
        self.assertEqual(dyn["u_seq__id.values"], dyn["u_seq__raw.values"])
        self.assertEqual(dyn["u_seq__id.values"][0].min, 0)
        self.assertNotEqual(dyn["u_seq__mv.values"], dyn["u_seq__id.values"])
        self.assertEqual(dyn["u_seq__mv.key_lengths"], dyn["u_seq__id.values"])
        self.assertEqual(dyn["u_seq__mv.key_lengths"][0].min, 0)
        self.assertEqual(dyn["u_w.weights"], dyn["u_w.values"])

        item_dim = dyn["i_id.lengths"][0]
        self.assertEqual(dyn["i_dense.values"][0], item_dim)
        if input_tile == "0":
            self.assertEqual(dyn["u_w.lengths"][0], item_dim)
            self.assertEqual(dyn["u_dense.values"][0], item_dim)
            self.assertEqual(dyn["u_seq__id.lengths"][0], item_dim)
        elif online:
            self.assertEqual(dyn["u_w.lengths"], {})
            self.assertEqual(dyn["u_dense.values"], {})
            self.assertEqual(dyn["u_seq__id.lengths"], {})
        else:
            user_dim = dyn["u_w.lengths"][0]
            self.assertNotEqual(user_dim, item_dim)
            self.assertEqual(dyn["u_dense.values"][0], user_dim)
            self.assertEqual(dyn["u_seq__id.lengths"][0], user_dim)
            self.assertEqual(dyn["u_seq__raw.lengths"][0], user_dim)

    def test_build_dynamic_shapes_hsk_and_jagged(self):
        feat_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    num_buckets=16,
                    embedding_dim=4,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    sequence_length=4,
                    hard_search_key="item:cat",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="id",
                                expression="item:id",
                                num_buckets=16,
                                embedding_dim=4,
                            )
                        )
                    ],
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="uih",
                    sequence_length=8,
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            raw_feature=feature_pb2.RawFeature(
                                feature_name="vec",
                                expression="user:uih__vec",
                                value_dim=3,
                            )
                        )
                    ],
                )
            ),
        ]
        features = create_features(feat_cfgs, fg_mode=FgMode.FG_DAG)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="jagged",
                feature_names=["uih__vec"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="hsk",
                feature_names=["click_seq__id"],
                group_type=model_pb2.SEQUENCE,
            ),
        ]
        data = {
            "batch_size": torch.tensor(4, dtype=torch.int64),
            "user_id.values": torch.tensor([1, 2], dtype=torch.int64),
            "user_id.lengths": torch.tensor([1, 1], dtype=torch.int64),
            "click_seq__id.values": torch.tensor([1, 2, 3], dtype=torch.int64),
            "click_seq__id.lengths": torch.tensor([2, 1], dtype=torch.int64),
            "click_seq__inverse_indices": torch.tensor([0, 1, 0, 1], dtype=torch.int64),
            "uih__vec.values": torch.randn(3, 3),
            "uih__vec.lengths": torch.tensor([2, 1], dtype=torch.int64),
        }
        with mock.patch.dict(
            os.environ,
            INPUT_TILE="3",
            HARD_SEARCH_INPUT_TILE="1",
            ENABLE_AOT="2",
            clear=False,
        ):
            dyn = _build_dynamic_shapes(data, features, feature_groups)

        self.assertEqual(dyn["click_seq__id.values"][0].min, 0)
        self.assertEqual(dyn["uih__vec.values"][0].min, 1)
        self.assertEqual(
            dyn["click_seq__inverse_indices"][0].__name__,
            "batch",
        )
        self.assertEqual(
            dyn["click_seq__id.lengths"][0].__name__, "hsk_click_seq_batch"
        )
        self.assertEqual(dyn["user_id.lengths"][0].__name__, "user_batch")

        # Single unique HSK row cannot be dynamically tiled, must become static {}.
        data["click_seq__id.lengths"] = torch.tensor([3], dtype=torch.int64)
        with mock.patch.dict(
            os.environ,
            INPUT_TILE="3",
            HARD_SEARCH_INPUT_TILE="1",
            ENABLE_AOT="2",
            clear=False,
        ):
            dyn_single_hsk = _build_dynamic_shapes(data, features, feature_groups)
        self.assertEqual(dyn_single_hsk["click_seq__id.lengths"], {})

    def test_pad_empty_sparse_values(self):
        features, _ = _create_din_input_tile_features_and_groups(include_multival=True)
        # Sequence of length 1 for u_seq (e.g. INPUT_TILE_3_ONLINE=1 single user row)
        # and empty non-sequence weighted sparse u_w.
        raw_data = {
            "u_w.values": torch.empty(0, dtype=torch.int64),
            "u_w.lengths": torch.tensor([0], dtype=torch.int64),
            "u_w.weights": torch.empty(0, dtype=torch.float32),
            "u_seq__id.values": torch.tensor([5], dtype=torch.int64),
            "u_seq__id.lengths": torch.tensor([1], dtype=torch.int64),
            "u_seq__raw.values": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
            "u_seq__raw.lengths": torch.tensor([1], dtype=torch.int64),
            "u_seq__mv.values": torch.empty(0, dtype=torch.int64),
            "u_seq__mv.lengths": torch.tensor([1], dtype=torch.int64),
            "u_seq__mv.key_lengths": torch.tensor([0], dtype=torch.int64),
        }
        padded = _pad_empty_sparse_values(raw_data, features)
        self.assertGreaterEqual(padded["u_w.values"].numel(), 2)
        self.assertGreaterEqual(padded["u_w.weights"].numel(), 2)
        self.assertEqual(int(padded["u_w.lengths"].sum()), padded["u_w.values"].numel())
        for suffix in ("id", "raw", "mv"):
            self.assertEqual(int(padded[f"u_seq__{suffix}.lengths"].sum()), 2)
        self.assertEqual(padded["u_seq__id.values"].shape, (2,))
        self.assertEqual(padded["u_seq__raw.values"].shape, (2, 2))
        self.assertEqual(padded["u_seq__mv.key_lengths"].shape, (2,))
        self.assertGreaterEqual(padded["u_seq__mv.values"].numel(), 2)
        self.assertEqual(
            int(padded["u_seq__mv.key_lengths"].sum()),
            padded["u_seq__mv.values"].numel(),
        )

        # Empty sequence sample (total_sequence_length == 0 ->
        # sequence_length_padding == 2)
        empty_sequence_data = {
            "u_seq__id.values": torch.empty(0, dtype=torch.int64),
            "u_seq__id.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__raw.values": torch.empty((0, 2), dtype=torch.float32),
            "u_seq__raw.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__mv.values": torch.empty(0, dtype=torch.int64),
            "u_seq__mv.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__mv.key_lengths": torch.empty(0, dtype=torch.int64),
        }
        padded_zero = _pad_empty_sparse_values(empty_sequence_data, features)
        for suffix in ("id", "raw", "mv"):
            self.assertEqual(int(padded_zero[f"u_seq__{suffix}.lengths"].sum()), 2)
        self.assertEqual(padded_zero["u_seq__id.values"].shape, (2,))
        self.assertEqual(padded_zero["u_seq__raw.values"].shape, (2, 2))
        self.assertEqual(padded_zero["u_seq__mv.key_lengths"].shape, (2,))
        self.assertGreaterEqual(padded_zero["u_seq__mv.values"].numel(), 2)
        self.assertEqual(
            int(padded_zero["u_seq__mv.key_lengths"].sum()),
            padded_zero["u_seq__mv.values"].numel(),
        )

    def test_resolve_tile_size_fallbacks(self):
        # 1) All candidate features are hard-search -> falls back to inverse_indices.
        hsk_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    num_buckets=16,
                    embedding_dim=4,
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="click_seq",
                    sequence_length=4,
                    hard_search_key="item:cat_a",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="id",
                                expression="item:id",
                                num_buckets=16,
                                embedding_dim=4,
                            )
                        )
                    ],
                )
            ),
        ]
        with mock.patch.dict(
            os.environ,
            {"INPUT_TILE": "3", "HARD_SEARCH_INPUT_TILE": "1", "ENABLE_AOT": "2"},
            clear=False,
        ):
            parser_hsk = DataParser(create_features(hsk_cfgs, fg_mode=FgMode.FG_DAG))
            self.assertEqual(
                parser_hsk._tile_keys,
                ("click_seq__inverse_indices", "user_id.lengths"),
            )
            self.assertEqual(
                parser_hsk._resolve_tile_size(
                    {
                        "user_id.lengths": torch.tensor([1], dtype=torch.int64),
                        "click_seq__inverse_indices": torch.tensor(
                            [0, 0, 0], dtype=torch.int64
                        ),
                        "batch_size": torch.tensor(3, dtype=torch.int64),
                    }
                ),
                3,
            )
            with self.assertRaises(ValueError):
                parser_hsk._resolve_tile_size(
                    {
                        "user_id.lengths": torch.tensor([1, 1], dtype=torch.int64),
                        "click_seq__inverse_indices": torch.tensor(
                            [0, 0, 0], dtype=torch.int64
                        ),
                    }
                )
            with self.assertRaises(ValueError):
                parser_hsk._resolve_tile_size(
                    {
                        "user_id.lengths": torch.empty(0, dtype=torch.int64),
                        "click_seq__inverse_indices": torch.tensor(
                            [0, 0], dtype=torch.int64
                        ),
                    }
                )

        # 2) Only user features -> warns and falls back to batch_size.item().
        user_only_cfgs = [hsk_cfgs[0]]
        with mock.patch.dict(
            os.environ, {"INPUT_TILE": "2", "ENABLE_AOT": "2"}, clear=False
        ):
            parser_user_only = DataParser(
                create_features(user_only_cfgs, fg_mode=FgMode.FG_DAG)
            )
            self.assertIsNone(parser_user_only._tile_keys)
            with self.assertLogs("tzrec", level="WARNING") as logs:
                tile = parser_user_only._resolve_tile_size(
                    {"batch_size": torch.tensor(4, dtype=torch.int64)}
                )
            self.assertEqual(tile, 4)
            self.assertTrue(
                any("falling back to batch_size.item()" in msg for msg in logs.output)
            )

        # 3) Non-unified export keeps reading batch_size without warning.
        item_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="item_id",
                expression="item:item_id",
                num_buckets=16,
                embedding_dim=4,
            )
        )
        for env in ({"INPUT_TILE": "2"}, {"INPUT_TILE": "3", "ENABLE_AOT": "1"}):
            with self.subTest(env=env):
                with mock.patch.dict(os.environ, env, clear=False):
                    parser = DataParser(
                        create_features([hsk_cfgs[0], item_cfg], fg_mode=FgMode.FG_DAG)
                    )
                    self.assertIsNone(parser._tile_keys)
                    with self.assertNoLogs("tzrec", level="WARNING"):
                        tile = parser._resolve_tile_size(
                            {
                                "user_id.lengths": torch.tensor([1, 1]),
                                "item_id.lengths": torch.tensor([1, 1]),
                                "batch_size": torch.tensor(5, dtype=torch.int64),
                            }
                        )
                    self.assertEqual(tile, 5)

    def test_infer_tile_size_scripted_validation(self):
        scripted = torch.jit.script(_infer_tile_size)
        self.assertEqual(scripted(torch.ones(6), torch.ones(1)), 6)
        self.assertEqual(scripted(torch.ones(6), torch.ones(6)), 1)
        for user_bs in (0, 4, 7):
            with self.subTest(user_bs=user_bs):
                with self.assertRaises(torch.jit.Error):
                    scripted(torch.ones(6), torch.ones(user_bs))

    @parameterized.expand(
        [
            ("input_tile_2", "2", "0"),
            ("input_tile_3", "3", "0"),
            ("input_tile_3_online", "3", "1"),
        ],
        name_func=parameterized_name_func,
    )
    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_unified_aot_input_tile_din(self, _name, input_tile, online):
        device = torch.device("cuda:0")
        with mock.patch.dict(
            os.environ,
            {
                "INPUT_TILE": input_tile,
                "INPUT_TILE_3_ONLINE": online,
                "ENABLE_AOT": "2",
            },
        ):
            features, feature_groups = _create_din_input_tile_features_and_groups(
                include_multival=True
            )
            model_config = model_pb2.ModelConfig(
                feature_groups=feature_groups,
                multi_tower_din=rank_model_pb2.MultiTowerDIN(
                    towers=[
                        tower_pb2.Tower(
                            input="deep", mlp=module_pb2.MLP(hidden_units=[8])
                        )
                    ],
                    din_towers=[
                        tower_pb2.DINTower(
                            input="seq", attn_mlp=module_pb2.MLP(hidden_units=[8])
                        )
                    ],
                    final=module_pb2.MLP(hidden_units=[4]),
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            )
            inner = MultiTowerDIN(model_config, features, labels=[])
            init_parameters(inner, device)
            wrapped = ScriptWrapper(inner).to(device).eval()

            # Sample export batch intentionally uses a sequence of length 1 in
            # online mode to cover sequence-length padding against 0/1 specialization.
            if online == "1":
                export_data = {
                    "batch_size": torch.tensor(3, dtype=torch.int64),
                    "u_w.values": torch.tensor([1, 2], dtype=torch.int64),
                    "u_w.lengths": torch.tensor([2], dtype=torch.int64),
                    "u_w.weights": torch.tensor([1.0, 3.0], dtype=torch.float32),
                    "u_dense.values": torch.randn(1, 2),
                    "u_seq__id.values": torch.tensor([5], dtype=torch.int64),
                    "u_seq__id.lengths": torch.tensor([1], dtype=torch.int64),
                    "u_seq__raw.values": torch.tensor([[0.5, -0.5]]),
                    "u_seq__raw.lengths": torch.tensor([1], dtype=torch.int64),
                    "u_seq__mv.values": torch.empty(0, dtype=torch.int64),
                    "u_seq__mv.lengths": torch.tensor([1], dtype=torch.int64),
                    "u_seq__mv.key_lengths": torch.tensor([0], dtype=torch.int64),
                    "i_id.values": torch.tensor([1, 2, 3], dtype=torch.int64),
                    "i_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                    "i_dense.values": torch.randn(3, 4)[:, :2],
                }
            else:
                export_data = {
                    "batch_size": torch.tensor(1, dtype=torch.int64),
                    "u_w.values": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
                    "u_w.lengths": torch.tensor([2, 1, 1], dtype=torch.int64),
                    "u_w.weights": torch.tensor(
                        [1.0, 3.0, 1.0, 1.0], dtype=torch.float32
                    ),
                    "u_dense.values": torch.randn(3, 2),
                    "u_seq__id.values": torch.tensor([5, 6, 7], dtype=torch.int64),
                    "u_seq__id.lengths": torch.tensor([1, 2, 0], dtype=torch.int64),
                    "u_seq__raw.values": torch.randn(3, 2),
                    "u_seq__raw.lengths": torch.tensor([1, 2, 0], dtype=torch.int64),
                    "u_seq__mv.values": torch.tensor([8, 9, 10], dtype=torch.int64),
                    "u_seq__mv.lengths": torch.tensor([1, 2, 0], dtype=torch.int64),
                    "u_seq__mv.key_lengths": torch.tensor([2, 0, 1], dtype=torch.int64),
                    "i_id.values": torch.tensor([1, 2, 3], dtype=torch.int64),
                    "i_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                    "i_dense.values": torch.randn(3, 4)[:, :2],
                }
            self.assertFalse(export_data["i_dense.values"].is_contiguous())

            save_dir = os.path.join(self.test_dir, f"export_{_name}")
            export_unified_model_aot(wrapped, export_data, save_dir)
            aoti_model = load_model_aot(save_dir, device)

            # 1) Online tiled input (1-D batch_size tensor matching Processor).
            online_data = {
                "batch_size": torch.full((1,), 5, dtype=torch.int64),
                "u_w.values": torch.tensor([2, 3], dtype=torch.int64),
                "u_w.lengths": torch.tensor([2], dtype=torch.int64),
                "u_w.weights": torch.tensor([1.0, 1.0], dtype=torch.float32),
                "u_dense.values": torch.randn(1, 2),
                "u_seq__id.values": torch.tensor([4, 5, 6], dtype=torch.int64),
                "u_seq__id.lengths": torch.tensor([3], dtype=torch.int64),
                "u_seq__raw.values": torch.randn(3, 2),
                "u_seq__raw.lengths": torch.tensor([3], dtype=torch.int64),
                "u_seq__mv.values": torch.tensor([2, 3, 4], dtype=torch.int64),
                "u_seq__mv.lengths": torch.tensor([3], dtype=torch.int64),
                "u_seq__mv.key_lengths": torch.tensor([2, 0, 1], dtype=torch.int64),
                "i_id.values": torch.tensor([1, 2, 3, 4, 5], dtype=torch.int64),
                "i_id.lengths": torch.tensor([1, 1, 1, 1, 1], dtype=torch.int64),
                "i_dense.values": torch.randn(5, 2),
            }
            with torch.no_grad():
                expected_online = wrapped(online_data, device)
                actual_online = aoti_model(online_data, device)
            self.assertEqual(expected_online["probs"].shape, (5,))
            for key, val in expected_online.items():
                torch.testing.assert_close(actual_online[key], val)

            # 2) Online cold-start empty user features + empty sequence.
            cold_data = {
                "batch_size": torch.full((1,), 3, dtype=torch.int64),
                "u_w.values": torch.empty(0, dtype=torch.int64),
                "u_w.lengths": torch.tensor([0], dtype=torch.int64),
                "u_w.weights": torch.empty(0, dtype=torch.float32),
                "u_dense.values": torch.randn(1, 2),
                "u_seq__id.values": torch.empty(0, dtype=torch.int64),
                "u_seq__id.lengths": torch.tensor([0], dtype=torch.int64),
                "u_seq__raw.values": torch.empty((0, 2), dtype=torch.float32),
                "u_seq__raw.lengths": torch.tensor([0], dtype=torch.int64),
                "u_seq__mv.values": torch.empty(0, dtype=torch.int64),
                "u_seq__mv.lengths": torch.tensor([0], dtype=torch.int64),
                "u_seq__mv.key_lengths": torch.empty(0, dtype=torch.int64),
                "i_id.values": torch.tensor([7, 8, 9], dtype=torch.int64),
                "i_id.lengths": torch.tensor([1, 1, 1], dtype=torch.int64),
                "i_dense.values": torch.randn(3, 2),
            }
            with torch.no_grad():
                expected_cold = wrapped(cold_data, device)
                actual_cold = aoti_model(cold_data, device)
                actual_cold_cuda = aoti_model(
                    {k: v.to(device) for k, v in cold_data.items()}, device
                )
            for key, val in expected_cold.items():
                torch.testing.assert_close(actual_cold[key], val)
                torch.testing.assert_close(actual_cold_cuda[key], val)

            bad_tile = dict(online_data)
            bad_tile["u_w.lengths"] = torch.tensor([1, 1], dtype=torch.int64)
            with self.assertRaises(ValueError):
                aoti_model(bad_tile, device)
            # 3) Offline batch_size=1 with u_bs == i_bs == 4 and an extra ignored key.
            if online == "0":
                offline_data = {
                    "batch_size": torch.tensor(1, dtype=torch.int64),
                    "u_w.values": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
                    "u_w.lengths": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "u_w.weights": torch.tensor(
                        [1.0, 2.0, 1.0, 3.0], dtype=torch.float32
                    ),
                    "u_dense.values": torch.randn(4, 5)[:, 1:3],
                    "u_seq__id.values": torch.tensor([1, 2], dtype=torch.int64),
                    "u_seq__id.lengths": torch.tensor([2, 0, 0, 0], dtype=torch.int64),
                    "u_seq__raw.values": torch.randn(2, 2),
                    "u_seq__raw.lengths": torch.tensor([2, 0, 0, 0], dtype=torch.int64),
                    "u_seq__mv.values": torch.tensor([5, 6, 7], dtype=torch.int64),
                    "u_seq__mv.lengths": torch.tensor([2, 0, 0, 0], dtype=torch.int64),
                    "u_seq__mv.key_lengths": torch.tensor([1, 2], dtype=torch.int64),
                    "i_id.values": torch.tensor([1, 2, 3, 4], dtype=torch.int64),
                    "i_id.lengths": torch.tensor([1, 1, 1, 1], dtype=torch.int64),
                    "i_dense.values": torch.randn(4, 5)[:, 1:3],
                }
                self.assertFalse(offline_data["u_dense.values"].is_contiguous())
                self.assertFalse(offline_data["i_dense.values"].is_contiguous())
                with torch.no_grad():
                    expected_offline = wrapped(offline_data, device)
                    offline_with_extra = dict(offline_data, unused_extra=torch.ones(4))
                    actual_offline = aoti_model(offline_with_extra, device)
                for key, val in expected_offline.items():
                    torch.testing.assert_close(actual_offline[key], val)

    @parameterized.expand(["2", "3"], name_func=parameterized_name_func)
    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_unified_aot_hard_search_input_tile(self, tile):
        device = torch.device("cuda:0")
        with mock.patch.dict(
            os.environ,
            {
                "INPUT_TILE": tile,
                "HARD_SEARCH_INPUT_TILE": "1",
                "ENABLE_AOT": "2",
            },
        ):
            feature_cfgs = [
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="user_id",
                        expression="user:user_id",
                        num_buckets=16,
                        embedding_dim=4,
                    )
                ),
                feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="item_id",
                        expression="item:item_id",
                        num_buckets=16,
                        embedding_dim=4,
                    )
                ),
                feature_pb2.FeatureConfig(
                    sequence_feature=feature_pb2.SequenceFeature(
                        sequence_name="click_seq",
                        sequence_length=4,
                        hard_search_key="item:cat_a",
                        features=[
                            feature_pb2.SeqFeatureConfig(
                                id_feature=feature_pb2.IdFeature(
                                    feature_name="id",
                                    expression="item:id",
                                    num_buckets=16,
                                    embedding_dim=4,
                                )
                            )
                        ],
                    )
                ),
            ]
            features = create_features(feature_cfgs, fg_mode=FgMode.FG_DAG)
            feature_groups = [
                model_pb2.FeatureGroupConfig(
                    group_name="deep",
                    feature_names=["user_id", "item_id"],
                    group_type=model_pb2.DEEP,
                ),
                model_pb2.FeatureGroupConfig(
                    group_name="hsk",
                    feature_names=["item_id", "click_seq__id"],
                    group_type=model_pb2.SEQUENCE,
                ),
            ]

            class TinyHskModel(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.features = features
                    self.feature_groups = feature_groups
                    self.embedding_group = EmbeddingGroup(
                        features, feature_groups, device=device
                    )

                def predict(self, batch):
                    grouped = self.embedding_group(batch)
                    return {
                        "score": grouped["deep"].sum(dim=1)
                        + grouped["hsk.sequence"].sum(dim=(1, 2)),
                        "lengths": grouped["hsk.sequence_length"],
                    }

            inner = TinyHskModel()
            init_parameters(inner, device)
            wrapped = ScriptWrapper(inner).to(device).eval()

            def _make_inputs(lengths, indices):
                total = sum(lengths)
                count = len(indices)
                return {
                    "batch_size": torch.tensor(count, dtype=torch.int64),
                    "user_id.values": torch.tensor([2, 3], dtype=torch.int64),
                    "user_id.lengths": torch.tensor([2], dtype=torch.int64),
                    "item_id.values": torch.arange(1, count + 1, dtype=torch.int64),
                    "item_id.lengths": torch.ones(count, dtype=torch.int64),
                    "click_seq__id.values": torch.arange(
                        1, total + 1, dtype=torch.int64
                    ),
                    "click_seq__id.lengths": torch.tensor(lengths, dtype=torch.int64),
                    "click_seq__inverse_indices": torch.tensor(
                        indices, dtype=torch.int64
                    ),
                }

            save_dir = os.path.join(self.test_dir, f"hsk_tile_{tile}")
            export_unified_model_aot(
                wrapped, _make_inputs([2, 0, 1], [2, 0, 2, 1, 0]), save_dir
            )
            aoti_model = load_model_aot(save_dir, device)

            lengths_cases = ([2, 0, 1], [0], [3, 1], [0, 0])
            indices_cases = ([2, 0, 2, 1, 0], [0, 0, 0], [1, 0, 1], [1, 0, 1])
            for lengths, indices in zip(lengths_cases, indices_cases):
                with self.subTest(lengths=lengths):
                    data = _make_inputs(lengths, indices)
                    with torch.no_grad():
                        expected = wrapped(data, device)
                        actual = aoti_model(data, device)
                    for key, val in expected.items():
                        torch.testing.assert_close(actual[key], val)

            bad_hsk = _make_inputs([2, 0, 1], [0, 1])
            with self.assertRaises(ValueError):
                aoti_model(bad_hsk, device)

    def test_load_model_aot_device_validation_and_index_normalization(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a CUDA device"):
            load_model_aot(self.test_dir, torch.device("cpu"))

        _, in_spec = tree_flatten(({"a": torch.zeros(1)}, {}))
        _, out_spec = tree_flatten({"p": torch.zeros(1)})
        fake_pkg = mock.Mock()
        fake_pkg.loader.get_call_spec.return_value = (
            treespec_dumps(in_spec),
            treespec_dumps(out_spec),
        )
        with (
            mock.patch.object(aot_utils, "is_unified_aot_predict", return_value=True),
            mock.patch.object(
                torch._inductor, "aoti_load_package", return_value=fake_pkg
            ) as load_pkg_mock,
        ):
            load_model_aot(self.test_dir, torch.device("cuda"))
            load_pkg_mock.assert_called_once_with(
                os.path.join(self.test_dir, "aoti", "aoti_model.pt2"),
                device_index=-1,
            )

    def test_export_aot_requires_cuda_before_makedirs(self) -> None:
        with mock.patch("torch.cuda.is_available", return_value=False):
            for name, fn in (
                (
                    "legacy",
                    lambda d: export_model_aot(mock.Mock(), mock.Mock(), {}, {}, d),
                ),
                ("unified", lambda d: export_unified_model_aot(mock.Mock(), {}, d)),
            ):
                with self.subTest(case=name):
                    save_dir = os.path.join(self.test_dir, f"no_cuda_{name}")
                    with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
                        fn(save_dir)
                    self.assertFalse(os.path.exists(save_dir))

    def test_aoti_compile_cfg_portable_cpu(self) -> None:
        with mock.patch.dict(os.environ, {"AOTI_PORTABLE_CPU": "1"}, clear=True):
            cfg = _aoti_compile_cfg()
            self.assertIn("cpp.march", cfg)
            self.assertIn("cpp.simdlen", cfg)

        with mock.patch.dict(
            os.environ,
            {"AOTI_CPU_MARCH": "x86-64-v3", "AOTI_CPU_SIMDLEN": "128"},
            clear=True,
        ):
            cfg = _aoti_compile_cfg()
            self.assertEqual(cfg["cpp.march"], "x86-64-v3")
            self.assertEqual(cfg["cpp.simdlen"], 128)

    def test_build_dynamic_shapes_warns_static_and_tracks_non_seq_hsk(self) -> None:
        id_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="u_id",
                expression="user:u_id",
                num_buckets=16,
                embedding_dim=4,
            )
        )
        hsk_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="hsk_item",
                expression="item:hsk_item",
                num_buckets=16,
                embedding_dim=4,
            )
        )
        feats = create_features([id_cfg, hsk_cfg])
        feats[1]._is_hard_search_feat = True
        data = {
            "u_id.values": torch.tensor([1, 2], dtype=torch.int64),
            "u_id.lengths": torch.tensor([2], dtype=torch.int64),
            "hsk_item.values": torch.tensor([1, 2], dtype=torch.int64),
            "hsk_item.lengths": torch.tensor([1, 1], dtype=torch.int64),
        }
        with (
            mock.patch.dict(
                os.environ,
                {"INPUT_TILE": "3", "INPUT_TILE_3_ONLINE": "0"},
                clear=True,
            ),
            mock.patch.object(
                aot_utils,
                "get_hard_search_input_tile_features",
                return_value={"hsk_item": "user_ctx"},
            ),
            mock.patch.object(aot_utils.logger, "warning") as warn_log,
        ):
            shapes = _build_dynamic_shapes(data, feats, [])

        self.assertEqual(shapes["u_id.lengths"], {})
        self.assertEqual(shapes["hsk_item.lengths"][0].__name__, "hsk_user_ctx_batch")
        self.assertTrue(
            any(
                "degraded to static" in str(call.args[0])
                for call in warn_log.call_args_list
            )
        )

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_export_unified_model_aot_prunes_unused_keys(self) -> None:
        id_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="item_id",
                expression="item:item_id",
                num_buckets=16,
                embedding_dim=4,
            )
        )
        feats = create_features([id_cfg])

        class _UnusedBatchSizeModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = feats
                self.feature_groups = []
                self.emb = nn.Embedding(16, 4)

            def set_is_inference(self, _is_inference: bool) -> None:
                return None

            def forward(self, batch, _device=None):
                vals = batch["item_id.values"]
                lens = batch["item_id.lengths"]
                emb = self.emb(vals)
                return {"out": emb.sum() + lens.sum().float()}

        sample_data = {
            "item_id.values": torch.tensor([1, 2], dtype=torch.int64),
            "item_id.lengths": torch.tensor([1, 1], dtype=torch.int64),
            "batch_size": torch.tensor(2, dtype=torch.int64),
        }
        save_dir = os.path.join(self.test_dir, "unified_pruned")
        with mock.patch.dict(os.environ, {"ENABLE_AOT": "2"}):
            export_unified_model_aot(_UnusedBatchSizeModel(), sample_data, save_dir)
            loaded = load_model_aot(save_dir, torch.device("cuda"))

        in_spec = treespec_loads(loaded.model.loader.get_call_spec()[0])
        key_order = list(in_spec.child(0).child(0).context)
        self.assertNotIn("batch_size", key_order)
        self.assertEqual(set(key_order), {"item_id.values", "item_id.lengths"})
        cold_in = {
            "item_id.values": torch.empty(0, dtype=torch.int64),
            "item_id.lengths": torch.tensor([0], dtype=torch.int64),
        }
        out = loaded(cold_in, torch.device("cuda:0"))
        self.assertEqual(float(out["out"].item()), 0.0)

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_export_model_aot_padded_sequence_empty_and_single_len(self) -> None:
        class _PaddedSparseStage(nn.Module):
            def forward(
                self,
                data: Dict[str, torch.Tensor],
                device: torch.device,
            ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
                out = {"seq": data["seq"].to(device)}
                return out, out

        class _PaddedDenseStage(nn.Module):
            def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
                activation = (data["seq"] * 2 + 1).relu()
                return {"activation": activation, "pooled": activation.sum(dim=1)}

        sparse_model = _PaddedSparseStage()
        dense_model = _PaddedDenseStage().cuda()
        sample_data = {"seq": torch.randn(3, 5, 1)}
        save_dir = os.path.join(self.test_dir, "padded_two_stage")
        export_model_aot(
            sparse_model,
            dense_model,
            sample_data,
            {"seq_tensor_names": ["seq"]},
            save_dir,
        )
        with open(os.path.join(save_dir, "model_acc.json"), "w") as f:
            json.dump({"ENABLE_AOT": "1"}, f)

        device = torch.device("cuda:0")
        loaded = load_model_aot(save_dir, device)
        for seq_len in (0, 1, 5):
            test_in = {"seq": torch.randn(3, seq_len, 1, device=device)}
            expected = dense_model(test_in)
            actual = loaded(test_in, device)
            for key in ("activation", "pooled"):
                torch.testing.assert_close(actual[key], expected[key])


if __name__ == "__main__":
    unittest.main()
