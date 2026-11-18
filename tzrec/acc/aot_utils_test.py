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
import shutil
import unittest
from unittest import mock

import torch
from parameterized import parameterized

from tzrec.acc.aot_utils import (
    _build_dynamic_shapes,
    _pad_empty_sparse_values,
    export_unified_model_aot,
    load_model_aot,
)
from tzrec.datasets.data_parser import DataParser
from tzrec.features.feature import FgMode, create_features
from tzrec.models.model import ScriptWrapper
from tzrec.models.multi_tower_din import MultiTowerDIN
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import feature_pb2, model_pb2, module_pb2, tower_pb2
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
        # Sequence dense raw values share step Dim with sparse sequence values, min=0.
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
        # Step count == 1 for u_seq (e.g. INPUT_TILE_3_ONLINE=1 single user row)
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

        # 0-step sequence sample (total_steps == 0 -> pad_steps == 2)
        zero_step_data = {
            "u_seq__id.values": torch.empty(0, dtype=torch.int64),
            "u_seq__id.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__raw.values": torch.empty((0, 2), dtype=torch.float32),
            "u_seq__raw.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__mv.values": torch.empty(0, dtype=torch.int64),
            "u_seq__mv.lengths": torch.tensor([0], dtype=torch.int64),
            "u_seq__mv.key_lengths": torch.empty(0, dtype=torch.int64),
        }
        padded_zero = _pad_empty_sparse_values(zero_step_data, features)
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
        with mock.patch.dict(os.environ, {"INPUT_TILE": "2"}, clear=False):
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
            )
            inner = MultiTowerDIN(model_config, features, labels=[])
            init_parameters(inner, device)
            wrapped = ScriptWrapper(inner).to(device).eval()

            # Sample export batch intentionally uses a 1-step user sequence in
            # online mode to cover step-padding against 0/1 specialization.
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


if __name__ == "__main__":
    unittest.main()
