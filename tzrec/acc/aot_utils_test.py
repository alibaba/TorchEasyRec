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
import operator
import os
import shutil
import unittest
from typing import Dict, Optional, Tuple
from unittest import mock

import torch
from parameterized import parameterized
from torch import fx, nn
from torch.utils._pytree import tree_flatten, treespec_dumps, treespec_loads
from torchrec import JaggedTensor, KeyedTensor

from tzrec.acc import aot_utils
from tzrec.acc.aot_utils import (
    _aoti_compile_cfg,
    _build_distributed_dense_dynamic_shapes,
    _build_dynamic_shapes,
    _pad_distributed_dense_sample_inputs,
    _pad_empty_sparse_values,
    _prepare_distributed_dense_sample_inputs,
    export_distributed_dense_model_aot,
    export_model_aot,
    export_unified_model_aot,
    load_model_aot,
)
from tzrec.datasets.data_parser import DataParser, _infer_tile_size
from tzrec.features.feature import FgMode, create_features
from tzrec.models.genrec_model import GenRecFrontEnd
from tzrec.models.model import ScriptWrapper
from tzrec.models.multi_tower import MultiTower
from tzrec.models.multi_tower_din import MultiTowerDIN
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2, tower_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.protos.prompt_pb2 import PromptSlot
from tzrec.utils.export_util import (
    _run_dense_graph_sanity_check,
    build_dense_graph_module,
    finalize_dense_export,
)
from tzrec.utils.fx_util import eliminate_dense_dead_code
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    GENREC_LONG_HIST_CODES,
    create_genrec_test_model,
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
                (
                    "dist_dense",
                    lambda d: export_distributed_dense_model_aot(
                        mock.Mock(),
                        mock.Mock(),
                        {},
                        {},
                        {},
                        {},
                        d,
                        torch.device("cuda:0"),
                    ),
                ),
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


class DistributedDenseAotUtilsTest(unittest.TestCase):
    def setUp(self):
        self.test_dir = make_test_dir()
        self.addCleanup(shutil.rmtree, self.test_dir, ignore_errors=True)

    def _build_distributed_dense_sample_fixture(
        self, user_bs: int = 1, item_bs: int = 2
    ):
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_w",
                    expression="user:u_w",
                    num_buckets=16,
                    embedding_dim=4,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="i_id",
                    expression="item:i_id",
                    num_buckets=16,
                    embedding_dim=4,
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
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="u_seq",
                    sequence_length=4,
                    hard_search_key="item:i_id",
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="id",
                                expression="user:u_seq__id",
                                num_buckets=16,
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
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="mv",
                                expression="user:u_seq__mv",
                                num_buckets=16,
                                embedding_dim=4,
                                value_dim=0,
                            )
                        ),
                    ],
                )
            ),
            feature_pb2.FeatureConfig(
                sequence_feature=feature_pb2.SequenceFeature(
                    sequence_name="j_seq",
                    sequence_length=4,
                    features=[
                        feature_pb2.SeqFeatureConfig(
                            id_feature=feature_pb2.IdFeature(
                                feature_name="id",
                                expression="item:j_seq__id",
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
                feature_names=["u_w", "i_id", "u_dense"],
                group_type=model_pb2.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="seq",
                feature_names=[
                    "i_id",
                    "u_seq__id",
                    "u_seq__raw",
                    "u_seq__mv",
                ],
                group_type=model_pb2.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="jseq",
                feature_names=["j_seq__id"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            ),
        ]

        expected_keys = [
            "deep__ebc",
            "deep__ebc_user",
            "i_id@seq",
            "i_id@seq__lengths",
            "u_seq__id",
            "u_seq__id__lengths",
            "u_seq__mv",
            "u_seq__mv__lengths",
            "u_seq__mv.lengths",
            "u_seq__mv.key_lengths",
            "u_seq__raw.values",
            "u_seq__raw.lengths",
            "j_seq__id",
            "j_seq__id__lengths",
            "u_dense.values",
            "u_seq__inverse_indices",
            "batch_size",
        ]

        class DummyDenseGraph(torch.nn.Module):
            def forward(self, data):
                return (
                    data["deep__ebc"]
                    + data["deep__ebc_user"]
                    + data["i_id@seq"]
                    + data["i_id@seq__lengths"].float().unsqueeze(-1)
                    + data["u_seq__id"]
                    + data["u_seq__id__lengths"].float().unsqueeze(-1)
                    + data["u_seq__mv"]
                    + data["u_seq__mv__lengths"].float().unsqueeze(-1)
                    + data["u_seq__mv.lengths"].float().unsqueeze(-1)
                    + data["u_seq__mv.key_lengths"].float().unsqueeze(-1)
                    + data["u_seq__raw.values"]
                    + data["u_seq__raw.lengths"].float().unsqueeze(-1)
                    + data["j_seq__id"]
                    + data["j_seq__id__lengths"].float().unsqueeze(-1)
                    + data["u_dense.values"]
                    + data["u_seq__inverse_indices"].float().unsqueeze(-1)
                    + data["batch_size"].float()
                )

        gm = torch.fx.symbolic_trace(DummyDenseGraph())
        data = {
            "u_w.lengths": torch.ones(user_bs, dtype=torch.int64),
            "u_seq__mv.lengths": torch.tensor([0, 0], dtype=torch.int64),
            "u_seq__mv.key_lengths": torch.empty(0, dtype=torch.int64),
            "u_seq__raw.values": torch.empty((0, 2), dtype=torch.float32),
            "u_seq__raw.lengths": torch.tensor([0, 0], dtype=torch.int64),
            "u_dense.values": torch.randn(user_bs, 5)[:, :2],
            "u_seq__inverse_indices": torch.arange(item_bs, dtype=torch.int64) % 2,
            "batch_size": torch.tensor(item_bs // user_bs, dtype=torch.int64),
            "i_id.values": torch.zeros(item_bs, dtype=torch.int64),
        }
        sparse_kts = {
            "deep__ebc": KeyedTensor(
                keys=["i_id"], length_per_key=[4], values=torch.randn(item_bs, 4)
            ),
            # EmbeddingGroupImpl marks __ebc_user after .tile(tile_size, 1).
            "deep__ebc_user": KeyedTensor(
                keys=["u_w"], length_per_key=[4], values=torch.randn(item_bs, 4)
            ),
        }
        seq_jts = {
            "i_id@seq": JaggedTensor(
                values=torch.empty((0, 4)),
                lengths=torch.zeros(item_bs, dtype=torch.int64),
            ),
            "u_seq__id": JaggedTensor(
                values=torch.empty((0, 4)),
                lengths=torch.tensor([0, 0], dtype=torch.int64),
            ),
            "u_seq__mv": JaggedTensor(
                values=torch.empty((0, 4)),
                lengths=torch.tensor([0, 0], dtype=torch.int64),
            ),
            "j_seq__id": JaggedTensor(
                values=torch.randn(1, 4),
                lengths=torch.cat(
                    [
                        torch.ones(1, dtype=torch.int64),
                        torch.zeros(item_bs - 1, dtype=torch.int64),
                    ]
                ),
            ),
        }
        dense_graph_config = {
            "deep__ebc": ["i_id__ebc"],
            "deep__ebc_user": ["u_w__ebc_user"],
            "sequence__ec": [
                "i_id@seq__ec",
                "i_id@seq__lengths",
                "u_seq__id__ec",
                "u_seq__id__lengths",
                "u_seq__mv__ec",
                "u_seq__mv__lengths",
                "j_seq__id__ec",
                "j_seq__id__lengths",
            ],
        }
        return (
            features,
            feature_groups,
            gm,
            data,
            sparse_kts,
            seq_jts,
            dense_graph_config,
            expected_keys,
        )

    def test_prepare_distributed_dense_sample_inputs_and_dynamic_shapes(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        (
            features,
            feature_groups,
            gm,
            data,
            sparse_kts,
            seq_jts,
            dense_graph_config,
            expected_keys,
        ) = self._build_distributed_dense_sample_fixture()

        with mock.patch.dict(
            os.environ,
            {
                "USE_DISTRIBUTED_EMBEDDING": "1",
                "INPUT_TILE": "3",
                "INPUT_TILE_3_ONLINE": "1",
                "HARD_SEARCH_INPUT_TILE": "1",
            },
        ):
            sample_data = _prepare_distributed_dense_sample_inputs(
                gm=gm,
                data=data,
                sparse_kts=sparse_kts,
                seq_jts=seq_jts,
                features=features,
                feature_groups=feature_groups,
                device=device,
            )
            self.assertEqual(set(sample_data.keys()), set(expected_keys))
            self.assertEqual(sample_data["deep__ebc_user"].shape[0], 1)
            self.assertTrue(sample_data["u_dense.values"].is_contiguous())
            self.assertEqual(sample_data["i_id@seq__lengths"].dtype, torch.int32)
            self.assertEqual(sample_data["u_seq__id__lengths"].dtype, torch.int32)
            self.assertEqual(sample_data["u_seq__mv__lengths"].dtype, torch.int32)
            self.assertEqual(sample_data["j_seq__id__lengths"].dtype, torch.int32)
            self.assertEqual(sample_data["i_id@seq"].shape[0], 2)
            self.assertEqual(sample_data["u_seq__id"].shape[0], 2)
            self.assertEqual(sample_data["u_seq__raw.values"].shape[0], 2)
            self.assertEqual(sample_data["u_seq__mv.key_lengths"].shape[0], 2)
            self.assertEqual(sample_data["u_seq__mv"].shape[0], 2)
            self.assertEqual(sample_data["j_seq__id"].shape[0], 2)
            self.assertEqual(
                int(sample_data["u_seq__mv.lengths"].sum()),
                sample_data["u_seq__mv.key_lengths"].numel(),
            )
            self.assertEqual(
                int(sample_data["u_seq__mv.key_lengths"].sum()),
                sample_data["u_seq__mv"].shape[0],
            )
            self.assertEqual(
                int(sample_data["u_seq__mv__lengths"].sum()),
                sample_data["u_seq__mv"].shape[0],
            )
            self.assertEqual(
                int(sample_data["u_seq__id__lengths"].sum()),
                sample_data["u_seq__id"].shape[0],
            )
            self.assertEqual(
                int(sample_data["u_seq__raw.lengths"].sum()),
                sample_data["u_seq__raw.values"].shape[0],
            )
            self.assertEqual(
                int(sample_data["i_id@seq__lengths"].sum()),
                sample_data["i_id@seq"].shape[0],
            )
            self.assertEqual(
                int(sample_data["j_seq__id__lengths"].sum()),
                sample_data["j_seq__id"].shape[0],
            )

            shapes = _build_distributed_dense_dynamic_shapes(
                sample_data=sample_data,
                dense_graph_config=dense_graph_config,
                features=features,
                feature_groups=feature_groups,
            )
            self.assertEqual(shapes["batch_size"], {})
            self.assertEqual(shapes["deep__ebc_user"], {})
            self.assertEqual(shapes["u_dense.values"], {})
            self.assertEqual(shapes["u_seq__inverse_indices"][0].__name__, "batch")
            self.assertEqual(
                shapes["deep__ebc"][0].__name__,
                "batch",
            )
            self.assertEqual(
                shapes["u_seq__id"][0],
                shapes["u_seq__raw.values"][0],
            )
            self.assertEqual(
                shapes["u_seq__id"][0],
                shapes["u_seq__mv.key_lengths"][0],
            )
            self.assertNotEqual(
                shapes["u_seq__mv"][0], shapes["u_seq__mv.key_lengths"][0]
            )
            self.assertEqual(shapes["u_seq__id"][0].min, 0)
            self.assertEqual(shapes["u_seq__mv"][0].min, 0)
            self.assertEqual(shapes["i_id@seq"][0].min, 0)
            self.assertEqual(shapes["j_seq__id"][0].min, 1)
            self.assertIn("hsk_u_seq_batch", shapes["u_seq__id__lengths"][0].__name__)
            self.assertEqual(
                shapes["u_seq__id__lengths"][0],
                shapes["u_seq__raw.lengths"][0],
            )

        # Offline tiled warm-up (user_bs=2, item_bs=4) under online=0 must also
        # slice __ebc_user back to user_bs=2 so it shares user_batch.
        (
            features2,
            feature_groups2,
            gm2,
            data2,
            sparse_kts2,
            seq_jts2,
            dense_graph_config2,
            _,
        ) = self._build_distributed_dense_sample_fixture(user_bs=2, item_bs=4)
        with mock.patch.dict(
            os.environ,
            {
                "USE_DISTRIBUTED_EMBEDDING": "1",
                "INPUT_TILE": "3",
                "INPUT_TILE_3_ONLINE": "0",
                "HARD_SEARCH_INPUT_TILE": "1",
            },
        ):
            sample_data2 = _prepare_distributed_dense_sample_inputs(
                gm=gm2,
                data=data2,
                sparse_kts=sparse_kts2,
                seq_jts=seq_jts2,
                features=features2,
                feature_groups=feature_groups2,
                device=device,
            )
            self.assertEqual(sample_data2["deep__ebc_user"].shape[0], 2)
            shapes2 = _build_distributed_dense_dynamic_shapes(
                sample_data=sample_data2,
                dense_graph_config=dense_graph_config2,
                features=features2,
                feature_groups=feature_groups2,
            )
            self.assertEqual(shapes2["deep__ebc_user"][0].__name__, "user_batch")
            self.assertEqual(shapes2["deep__ebc_user"], shapes2["u_dense.values"])

    def test_prepare_and_dynamic_shapes_explicit_tile_args_without_env(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        (
            features,
            feature_groups,
            gm,
            data,
            sparse_kts,
            seq_jts,
            dense_graph_config,
            expected_keys,
        ) = self._build_distributed_dense_sample_fixture(user_bs=2, item_bs=4)

        with (
            mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "HARD_SEARCH_INPUT_TILE": "1",
                },
            ),
            mock.patch(
                "tzrec.acc.aot_utils.get_hard_search_input_tile_features",
                side_effect=AssertionError(
                    "explicit hard_search_features must bypass env lookup"
                ),
            ),
        ):
            os.environ.pop("INPUT_TILE", None)
            os.environ.pop("INPUT_TILE_3_ONLINE", None)
            sample_data = _prepare_distributed_dense_sample_inputs(
                gm=gm,
                data=data,
                sparse_kts=sparse_kts,
                seq_jts=seq_jts,
                features=features,
                feature_groups=feature_groups,
                device=device,
                input_tile=True,
                input_tile_3_online=True,
                hard_search_features={
                    "u_seq__id": "u_seq",
                    "u_seq__raw": "u_seq",
                    "u_seq__mv": "u_seq",
                },
            )
            self.assertEqual(set(sample_data.keys()), set(expected_keys))
            self.assertEqual(sample_data["deep__ebc_user"].shape[0], 2)

            shapes = _build_distributed_dense_dynamic_shapes(
                sample_data=sample_data,
                dense_graph_config=dense_graph_config,
                features=features,
                feature_groups=feature_groups,
                input_tile=True,
                hard_search_features={
                    "u_seq__id": "u_seq",
                    "u_seq__raw": "u_seq",
                    "u_seq__mv": "u_seq",
                },
            )
            self.assertNotIn("INPUT_TILE", os.environ)
            self.assertEqual(shapes["deep__ebc_user"][0].__name__, "user_batch")
            self.assertEqual(shapes["u_dense.values"][0].__name__, "user_batch")
            self.assertEqual(shapes["deep__ebc_user"], shapes["u_dense.values"])
            self.assertIn("hsk_u_seq_batch", shapes["u_seq__id__lengths"][0].__name__)
            self.assertEqual(
                shapes["u_seq__id__lengths"][0],
                shapes["u_seq__raw.lengths"][0],
            )

    def test_pad_distributed_dense_multivalue_sequence_with_zero_tokens(self):
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
        (
            features,
            feature_groups,
            gm,
            data,
            sparse_kts,
            seq_jts,
            dense_graph_config,
            _,
        ) = self._build_distributed_dense_sample_fixture(user_bs=2, item_bs=4)

        # Sequence has 2 total elements on sample 1 ([0, 2]) so sequence-length
        # padding does not fire, while multi-value token counts are all zero
        # (key_lengths=[0, 0]).
        data["u_seq__mv.lengths"] = torch.tensor([0, 2], dtype=torch.int64)
        data["u_seq__mv.key_lengths"] = torch.tensor([0, 0], dtype=torch.int64)
        data["u_seq__mv.values"] = torch.empty(0, dtype=torch.int64)
        data["u_seq__raw.values"] = torch.zeros((2, 2), dtype=torch.float32)
        data["u_seq__raw.lengths"] = torch.tensor([0, 2], dtype=torch.int64)
        seq_jts["u_seq__id"] = JaggedTensor(
            values=torch.zeros((2, 4), dtype=torch.float32),
            lengths=torch.tensor([0, 2], dtype=torch.int64),
        )
        seq_jts["u_seq__mv"] = JaggedTensor(
            values=torch.empty((0, 4), dtype=torch.float32),
            lengths=torch.tensor([0, 0], dtype=torch.int64),
        )

        with mock.patch.dict(
            os.environ,
            {
                "USE_DISTRIBUTED_EMBEDDING": "1",
                "INPUT_TILE": "3",
                "INPUT_TILE_3_ONLINE": "0",
                "HARD_SEARCH_INPUT_TILE": "1",
            },
        ):
            sample_data = _prepare_distributed_dense_sample_inputs(
                gm=gm,
                data=data,
                sparse_kts=sparse_kts,
                seq_jts=seq_jts,
                features=features,
                feature_groups=feature_groups,
                device=device,
            )
            self.assertEqual(sample_data["u_seq__mv.lengths"].tolist(), [0, 2])
            self.assertEqual(sample_data["u_seq__mv.key_lengths"].tolist(), [2, 0])
            self.assertEqual(sample_data["u_seq__mv"].shape[0], 2)
            self.assertEqual(sample_data["u_seq__mv__lengths"].tolist(), [0, 2])
            self.assertEqual(
                int(sample_data["u_seq__mv.key_lengths"].sum()),
                sample_data["u_seq__mv"].shape[0],
            )
            self.assertEqual(
                int(sample_data["u_seq__mv__lengths"].sum()),
                sample_data["u_seq__mv"].shape[0],
            )

            # Also verify step-3 padding when raw .values is present (GenRec case).
            raw_sample = dict(sample_data)
            raw_sample["u_seq__mv"] = torch.empty(
                (0, 4), dtype=torch.float32, device=device
            )
            raw_sample["u_seq__mv__lengths"] = torch.tensor(
                [0, 0], dtype=torch.int32, device=device
            )
            raw_sample["u_seq__mv.key_lengths"] = torch.tensor(
                [0, 0], dtype=torch.int64, device=device
            )
            raw_sample["u_seq__mv.values"] = torch.empty(
                0, dtype=torch.int64, device=device
            )
            padded = _pad_distributed_dense_sample_inputs(
                raw_sample, seq_jts, features, feature_groups
            )
            self.assertEqual(padded["u_seq__mv.key_lengths"].tolist(), [2, 0])
            self.assertEqual(padded["u_seq__mv"].shape[0], 2)
            self.assertEqual(padded["u_seq__mv.values"].shape[0], 2)
            self.assertEqual(padded["u_seq__mv__lengths"].tolist(), [0, 2])

            shapes = _build_distributed_dense_dynamic_shapes(
                sample_data=sample_data,
                dense_graph_config=dense_graph_config,
                features=features,
                feature_groups=feature_groups,
            )
            self.assertEqual(
                shapes["u_seq__id"][0],
                shapes["u_seq__mv.key_lengths"][0],
            )
            self.assertNotEqual(
                shapes["u_seq__mv"][0],
                shapes["u_seq__mv.key_lengths"][0],
            )

    @parameterized.expand(
        [
            ("offline_tiled", "0"),
            ("online_tiled", "1"),
        ],
        name_func=parameterized_name_func,
    )
    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_user_ebc_tiling_multibatch(self, _name, online):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "INPUT_TILE_3_ONLINE": online,
                    "ENABLE_AOT": "2",
                },
            ):
                cfgs = [
                    feature_pb2.FeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="u_id",
                            expression="user:u_id",
                            num_buckets=16,
                            embedding_dim=4,
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
                            num_buckets=16,
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
                ]
                feats = create_features(cfgs, fg_mode=FgMode.FG_DAG)
                fgs = [
                    model_pb2.FeatureGroupConfig(
                        group_name="deep",
                        feature_names=["u_id", "u_dense", "i_id", "i_dense"],
                        group_type=model_pb2.DEEP,
                    )
                ]
                mc = model_pb2.ModelConfig(
                    feature_groups=fgs,
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
                    multi_tower=rank_model_pb2.MultiTower(
                        towers=[
                            tower_pb2.Tower(
                                input="deep", mlp=module_pb2.MLP(hidden_units=[8])
                            )
                        ],
                        final=module_pb2.MLP(hidden_units=[4]),
                    ),
                )
                inner = MultiTower(mc, feats, labels=[])
                init_parameters(inner, device)
                wrapped = ScriptWrapper(inner).to(device).eval()
                export_data = {
                    "batch_size": torch.tensor(2, dtype=torch.int64, device=device),
                    "u_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                    "u_id.lengths": torch.ones(2, dtype=torch.int64, device=device),
                    "u_dense.values": torch.randn(2, 2, device=device),
                    "i_id.values": torch.zeros(4, dtype=torch.int64, device=device),
                    "i_id.lengths": torch.ones(4, dtype=torch.int64, device=device),
                    "i_dense.values": torch.randn(4, 2, device=device),
                }
                gm, full_graph, dense_cfg = build_dense_graph_module(
                    wrapped, export_data, device
                )
                save_dir = os.path.join(self.test_dir, f"user_ebc_tile_{_name}")
                finalize_dense_export(
                    wrapped,
                    full_graph,
                    gm,
                    export_data,
                    device,
                    save_dir,
                    dense_cfg,
                )
                aoti_model = torch._inductor.aoti_load_package(
                    os.path.join(save_dir, "aoti", "aoti_model.pt2")
                )
                in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                key_order = list(in_spec.child(0).child(0).context)
                gm_eval = gm.to(device).eval()
                for u_bs, i_bs in ((1, 5), (3, 6), (5, 5)):
                    inputs = {
                        "deep__ebc": torch.randn(i_bs, 4, device=device),
                        "deep__ebc_user": torch.randn(u_bs, 4, device=device),
                        "u_dense.values": torch.randn(u_bs, 2, device=device),
                        "i_dense.values": torch.randn(i_bs, 2, device=device),
                    }
                    inputs = {k: inputs[k] for k in key_order}
                    with torch.no_grad():
                        expected = gm_eval(inputs, device)
                        actual = aoti_model(inputs)
                    torch.testing.assert_close(
                        actual["probs"], expected["probs"], atol=1e-5, rtol=1e-5
                    )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @parameterized.expand(
        [
            ("fp32", "", "0"),
            ("bf16", "BF16", "0"),
            ("fp32_online", "", "1"),
        ],
        name_func=parameterized_name_func,
    )
    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_din_accuracy_and_call_spec(self, _name, mp, online):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "INPUT_TILE_3_ONLINE": online,
                    "ENABLE_AOT": "2",
                },
            ):
                features, feature_groups = _create_din_input_tile_features_and_groups(
                    include_multival=True
                )
                model_config = model_pb2.ModelConfig(
                    feature_groups=feature_groups,
                    losses=[
                        loss_pb2.LossConfig(
                            binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                        )
                    ],
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

                if online == "1":
                    export_data = {
                        "batch_size": torch.tensor(2, dtype=torch.int64, device=device),
                        "u_w.values": torch.zeros(2, dtype=torch.int64, device=device),
                        "u_w.lengths": torch.tensor(
                            [2], dtype=torch.int64, device=device
                        ),
                        "u_w.weights": torch.tensor(
                            [1.0, 2.0], dtype=torch.float32, device=device
                        ),
                        "u_dense.values": torch.randn(1, 2, device=device),
                        "u_seq__id.values": torch.zeros(
                            2, dtype=torch.int64, device=device
                        ),
                        "u_seq__id.lengths": torch.tensor(
                            [2], dtype=torch.int64, device=device
                        ),
                        "u_seq__raw.values": torch.randn(2, 2, device=device),
                        "u_seq__raw.lengths": torch.tensor(
                            [2], dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.values": torch.zeros(
                            2, dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.lengths": torch.tensor(
                            [2], dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.key_lengths": torch.tensor(
                            [1, 1], dtype=torch.int64, device=device
                        ),
                        "i_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                        "i_id.lengths": torch.tensor(
                            [1, 1], dtype=torch.int64, device=device
                        ),
                        "i_dense.values": torch.randn(2, 2, device=device),
                    }
                else:
                    export_data = {
                        "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                        "u_w.values": torch.zeros(3, dtype=torch.int64, device=device),
                        "u_w.lengths": torch.tensor(
                            [1, 2], dtype=torch.int64, device=device
                        ),
                        "u_w.weights": torch.tensor(
                            [1.0, 2.0, 1.0], dtype=torch.float32, device=device
                        ),
                        "u_dense.values": torch.randn(2, 2, device=device),
                        "u_seq__id.values": torch.zeros(
                            3, dtype=torch.int64, device=device
                        ),
                        "u_seq__id.lengths": torch.tensor(
                            [1, 2], dtype=torch.int64, device=device
                        ),
                        "u_seq__raw.values": torch.randn(3, 2, device=device),
                        "u_seq__raw.lengths": torch.tensor(
                            [1, 2], dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.values": torch.zeros(
                            3, dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.lengths": torch.tensor(
                            [1, 2], dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.key_lengths": torch.tensor(
                            [2, 0, 1], dtype=torch.int64, device=device
                        ),
                        "i_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                        "i_id.lengths": torch.tensor(
                            [1, 1], dtype=torch.int64, device=device
                        ),
                        "i_dense.values": torch.randn(2, 2, device=device),
                    }
                gm, full_graph, dense_graph_config = build_dense_graph_module(
                    wrapped, export_data, device, mixed_precision=mp or None
                )
                save_dir = os.path.join(self.test_dir, f"dist_din_{_name}")
                finalize_dense_export(
                    wrapped,
                    full_graph,
                    gm,
                    export_data,
                    device,
                    save_dir,
                    dense_graph_config,
                    mixed_precision=mp or None,
                )

                pt2_path = os.path.join(save_dir, "aoti", "aoti_model.pt2")
                self.assertTrue(os.path.exists(pt2_path))
                with open(os.path.join(save_dir, "gm_dense.code")) as f:
                    gm_code = f.read()
                for raw_key in (
                    "u_w.values",
                    "u_w.lengths",
                    "u_w.weights",
                    "u_seq__id.values",
                    "u_seq__id.lengths",
                    "u_seq__mv.values",
                    "i_id.values",
                    "i_id.lengths",
                ):
                    self.assertNotIn(raw_key, gm_code)

                aoti_model = torch._inductor.aoti_load_package(pt2_path)
                in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                key_order = list(in_spec.child(0).child(0).context)
                spec_keys = set(key_order)
                ebc_item_key = next(
                    k
                    for k in dense_graph_config
                    if k != "sequence__ec" and not k.endswith("_user")
                )
                ebc_user_key = next(
                    k for k in dense_graph_config if k.endswith("_user")
                )
                for raw_key in (
                    "u_w.values",
                    "u_w.lengths",
                    "u_w.weights",
                    "u_seq__id.values",
                    "u_seq__id.lengths",
                    "u_seq__mv.values",
                    "i_id.values",
                    "i_id.lengths",
                ):
                    self.assertNotIn(raw_key, spec_keys)

                def _run_case(user_bs, item_bs, seq_lens, mv_key_lens, query_rows):
                    total_steps = sum(seq_lens)
                    total_mv = sum(mv_key_lens)
                    case_inputs = {
                        ebc_item_key: torch.randn(item_bs, 4, device=device),
                        ebc_user_key: torch.randn(user_bs, 4, device=device),
                        "i_dense.values": torch.randn(item_bs, 2, device=device),
                        "u_dense.values": torch.randn(user_bs, 2, device=device),
                        "i_id": torch.randn(query_rows, 4, device=device),
                        "i_id__lengths": torch.ones(
                            item_bs, dtype=torch.int32, device=device
                        )
                        if query_rows == item_bs
                        else torch.zeros(item_bs, dtype=torch.int32, device=device),
                        "u_seq__id": torch.randn(total_steps, 4, device=device),
                        "u_seq__id__lengths": torch.tensor(
                            seq_lens, dtype=torch.int32, device=device
                        ),
                        "u_seq__mv": torch.randn(total_mv, 4, device=device),
                        "u_seq__mv__lengths": torch.tensor(
                            seq_lens, dtype=torch.int32, device=device
                        ),
                        "u_seq__mv.lengths": torch.tensor(
                            seq_lens, dtype=torch.int64, device=device
                        ),
                        "u_seq__mv.key_lengths": torch.tensor(
                            mv_key_lens, dtype=torch.int64, device=device
                        ),
                        "u_seq__raw.values": torch.randn(total_steps, 2, device=device),
                        "u_seq__raw.lengths": torch.tensor(
                            seq_lens, dtype=torch.int64, device=device
                        ),
                        "batch_size": torch.tensor(
                            item_bs // user_bs, dtype=torch.int64, device=device
                        ),
                    }
                    case_inputs = {k: case_inputs[k] for k in key_order}
                    gm_eval = gm.to(device).eval()
                    with torch.no_grad():
                        with torch.autocast(
                            device_type="cuda",
                            dtype=torch.bfloat16,
                            enabled=(mp == "BF16"),
                        ):
                            expected = gm_eval(case_inputs, device)
                        actual = aoti_model(case_inputs)
                    self.assertIn("probs", expected)
                    tol = (
                        {"atol": 2e-2, "rtol": 2e-2}
                        if mp == "BF16"
                        else {"atol": 1e-5, "rtol": 1e-5}
                    )
                    for k, val in expected.items():
                        torch.testing.assert_close(actual[k], val, **tol)

                # Online (User=1, Item=7), Offline (User=5, Item=5),
                # and empty sequence cases.
                _run_case(
                    user_bs=1,
                    item_bs=7,
                    seq_lens=[3],
                    mv_key_lens=[2, 0, 1],
                    query_rows=7,
                )
                if online == "0":
                    _run_case(
                        user_bs=5,
                        item_bs=5,
                        seq_lens=[2, 0, 1, 0, 1],
                        mv_key_lens=[1, 2, 0, 1],
                        query_rows=5,
                    )
                _run_case(
                    user_bs=1, item_bs=4, seq_lens=[0], mv_key_lens=[], query_rows=0
                )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_hard_search_input_tile(self):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "INPUT_TILE_3_ONLINE": "0",
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
                            hard_search_key="item:item_id",
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

                export_data = {
                    "batch_size": torch.tensor(5, dtype=torch.int64, device=device),
                    "user_id.values": torch.zeros(1, dtype=torch.int64, device=device),
                    "user_id.lengths": torch.tensor(
                        [1], dtype=torch.int64, device=device
                    ),
                    "item_id.values": torch.zeros(5, dtype=torch.int64, device=device),
                    "item_id.lengths": torch.ones(5, dtype=torch.int64, device=device),
                    "click_seq__id.values": torch.zeros(
                        3, dtype=torch.int64, device=device
                    ),
                    "click_seq__id.lengths": torch.tensor(
                        [2, 0, 1], dtype=torch.int64, device=device
                    ),
                    "click_seq__inverse_indices": torch.tensor(
                        [2, 0, 2, 1, 0], dtype=torch.int64, device=device
                    ),
                }
                gm, full_graph, dense_graph_config = build_dense_graph_module(
                    wrapped, export_data, device
                )
                save_dir = os.path.join(self.test_dir, "dist_hsk")
                finalize_dense_export(
                    wrapped,
                    full_graph,
                    gm,
                    export_data,
                    device,
                    save_dir,
                    dense_graph_config,
                )

                pt2_path = os.path.join(save_dir, "aoti", "aoti_model.pt2")
                self.assertTrue(os.path.exists(pt2_path))
                aoti_model = torch._inductor.aoti_load_package(pt2_path)
                in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                key_order = list(in_spec.child(0).child(0).context)
                spec_keys = set(key_order)
                ebc_item_key = next(
                    k
                    for k in dense_graph_config
                    if k != "sequence__ec" and not k.endswith("_user")
                )
                ebc_user_key = next(
                    k for k in dense_graph_config if k.endswith("_user")
                )
                for raw_key in (
                    "user_id.values",
                    "item_id.values",
                    "click_seq__id.values",
                    "click_seq__id.lengths",
                ):
                    self.assertNotIn(raw_key, spec_keys)

                gm_eval = gm.to(device).eval()
                for user_bs, seq_lens, indices in (
                    (1, [2, 0, 1], [2, 0, 2, 1, 0]),
                    (1, [0, 0], [0, 1, 0, 1]),
                ):
                    item_bs = len(indices)
                    total_steps = sum(seq_lens)
                    case_inputs = {
                        ebc_item_key: torch.randn(item_bs, 4, device=device),
                        ebc_user_key: torch.randn(user_bs, 4, device=device),
                        "item_id": torch.randn(item_bs, 4, device=device),
                        "item_id__lengths": torch.ones(
                            item_bs, dtype=torch.int32, device=device
                        ),
                        "click_seq__id": torch.randn(total_steps, 4, device=device),
                        "click_seq__id__lengths": torch.tensor(
                            seq_lens, dtype=torch.int32, device=device
                        ),
                        "click_seq__inverse_indices": torch.tensor(
                            indices, dtype=torch.int64, device=device
                        ),
                        "batch_size": torch.tensor(
                            item_bs // user_bs, dtype=torch.int64, device=device
                        ),
                    }
                    case_inputs = {k: case_inputs[k] for k in key_order}
                    with torch.no_grad():
                        expected = gm_eval(case_inputs, device)
                        actual = aoti_model(case_inputs)
                    for k, val in expected.items():
                        torch.testing.assert_close(actual[k], val, atol=1e-5, rtol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_genrec_frontend(self):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "ENABLE_AOT": "2",
                },
            ):
                hist_cfg = feature_pb2.FeatureConfig(
                    sequence_raw_feature=feature_pb2.RawFeature(
                        feature_name="hist", expression="user:hist"
                    )
                )
                beh_cfg = feature_pb2.FeatureConfig(
                    sequence_id_feature=feature_pb2.IdFeature(
                        feature_name="beh",
                        expression="user:beh",
                        num_buckets=32,
                        embedding_dim=8,
                        sequence_length=2,
                    )
                )
                model, _ = create_genrec_test_model(
                    os.path.join(self.test_dir, "genrec"),
                    feature_configs=[hist_cfg, beh_cfg],
                    prompt="History : {{hist}} . {{beh}} Predict :",
                )
                init_parameters(model, device=device)
                wrapped = ScriptWrapper(GenRecFrontEnd(model)).to(device).eval()

                export_data = {
                    "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                    "hist.values": torch.tensor(
                        GENREC_LONG_HIST_CODES * 2, dtype=torch.float32, device=device
                    ).reshape(-1, 1),
                    "hist.lengths": torch.tensor(
                        [6, 6], dtype=torch.int64, device=device
                    ),
                    "beh.values": torch.zeros(4, dtype=torch.int64, device=device),
                    "beh.lengths": torch.tensor(
                        [2, 2], dtype=torch.int64, device=device
                    ),
                }
                gm, full_graph, dense_graph_config = build_dense_graph_module(
                    wrapped, export_data, device
                )
                save_dir = os.path.join(self.test_dir, "dist_genrec")
                finalize_dense_export(
                    wrapped,
                    full_graph,
                    gm,
                    export_data,
                    device,
                    save_dir,
                    dense_graph_config,
                )

                pt2_path = os.path.join(save_dir, "aoti", "aoti_model.pt2")
                self.assertTrue(os.path.exists(pt2_path))
                aoti_model = torch._inductor.aoti_load_package(pt2_path)
                in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                key_order = list(in_spec.child(0).child(0).context)

                gm_eval = gm.to(device).eval()
                for bs in (1, 3):
                    beh_len = 2 * bs
                    case_inputs = {
                        "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                        "hist.values": torch.tensor(
                            GENREC_LONG_HIST_CODES * bs,
                            dtype=torch.float32,
                            device=device,
                        ).reshape(-1, 1),
                        "hist.lengths": torch.full(
                            (bs,), 6, dtype=torch.int64, device=device
                        ),
                        "beh.values": torch.arange(
                            1, beh_len + 1, dtype=torch.int64, device=device
                        ),
                        "beh.lengths": torch.full(
                            (bs,), 2, dtype=torch.int64, device=device
                        ),
                        "beh": torch.randn(beh_len, 8, device=device),
                        "beh__lengths": torch.full(
                            (bs,), 2, dtype=torch.int32, device=device
                        ),
                    }
                    case_inputs = {k: case_inputs[k] for k in key_order}
                    with torch.no_grad():
                        expected = gm_eval(case_inputs, device)
                        actual = aoti_model(case_inputs)
                    for k, val in expected.items():
                        torch.testing.assert_close(actual[k], val, atol=1e-5, rtol=1e-5)
                for bs in (1, 3):
                    cold_case = {
                        "beh": torch.empty((0, 8), dtype=torch.float32, device=device),
                        "beh.values": torch.empty(0, dtype=torch.int64, device=device),
                        "beh.lengths": torch.zeros(
                            bs, dtype=torch.int64, device=device
                        ),
                        "beh__lengths": torch.zeros(
                            bs, dtype=torch.int32, device=device
                        ),
                        "hist.values": torch.empty(
                            (0, 1), dtype=torch.float32, device=device
                        ),
                        "hist.lengths": torch.zeros(
                            bs, dtype=torch.int64, device=device
                        ),
                    }
                    aot_inputs = {k: cold_case[k] for k in key_order}
                    with torch.no_grad():
                        expected = gm_eval(cold_case, device)
                        actual = aoti_model(aot_inputs)
                    for k, val in expected.items():
                        torch.testing.assert_close(actual[k], val, atol=1e-5, rtol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_genrec_mixed_user_item(self):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            for include_item_seq in (True, False):
                with (
                    self.subTest(include_item_seq=include_item_seq),
                    mock.patch.dict(
                        os.environ,
                        {
                            "USE_DISTRIBUTED_EMBEDDING": "1",
                            "INPUT_TILE": "3",
                            "INPUT_TILE_3_ONLINE": "0",
                            "ENABLE_AOT": "2",
                        },
                    ),
                ):
                    u_id = feature_pb2.FeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="u_id",
                            expression="user:u_id",
                            num_buckets=16,
                            embedding_dim=4,
                        )
                    )
                    i_id = feature_pb2.FeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="i_id",
                            expression="item:i_id",
                            num_buckets=16,
                            embedding_dim=4,
                        )
                    )
                    feat_cfgs = [u_id, i_id]
                    prompt = "User : {{mix}} Predict :"
                    export_data = {
                        "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                        "u_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                        "u_id.lengths": torch.ones(2, dtype=torch.int64, device=device),
                        "i_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                        "i_id.lengths": torch.ones(2, dtype=torch.int64, device=device),
                    }
                    if include_item_seq:
                        feat_cfgs.append(
                            feature_pb2.FeatureConfig(
                                sequence_id_feature=feature_pb2.IdFeature(
                                    feature_name="i_seq",
                                    expression="item:i_seq",
                                    num_buckets=16,
                                    embedding_dim=4,
                                    sequence_length=2,
                                )
                            )
                        )
                        prompt = "User : {{mix}} Seq : {{i_seq}} Predict :"
                        export_data["i_seq.values"] = torch.zeros(
                            3, dtype=torch.int64, device=device
                        )
                        export_data["i_seq.lengths"] = torch.tensor(
                            [2, 1], dtype=torch.int64, device=device
                        )

                    case_dir = os.path.join(
                        self.test_dir, f"genrec_mixed_{int(include_item_seq)}"
                    )
                    model, _ = create_genrec_test_model(
                        case_dir,
                        feature_configs=feat_cfgs,
                        prompt=prompt,
                        slots=[PromptSlot(name="mix", feature_names=["u_id", "i_id"])],
                    )
                    init_parameters(model, device=device)
                    wrapped = ScriptWrapper(GenRecFrontEnd(model)).to(device).eval()
                    gm, full_graph, dense_graph_config = build_dense_graph_module(
                        wrapped, export_data, device
                    )
                    sparse_kts, seq_jts = _run_dense_graph_sanity_check(
                        wrapped, full_graph, gm, export_data, device
                    )
                    sample_data = _prepare_distributed_dense_sample_inputs(
                        gm=gm,
                        data=export_data,
                        sparse_kts=sparse_kts,
                        seq_jts=seq_jts,
                        features=wrapped.features,
                        feature_groups=wrapped.feature_groups,
                        device=device,
                    )
                    dyn_shapes = _build_distributed_dense_dynamic_shapes(
                        sample_data=sample_data,
                        gm=gm,
                        dense_graph_config=dense_graph_config,
                        features=wrapped.features,
                        feature_groups=wrapped.feature_groups,
                    )
                    batch_dim = dyn_shapes["i_id.lengths"][0]
                    self.assertEqual(batch_dim.__name__, "batch")
                    self.assertEqual(dyn_shapes["u_id.lengths"][0], batch_dim)
                    self.assertEqual(dyn_shapes["mix__ebc"][0], batch_dim)
                    self.assertEqual(dyn_shapes["mix__ebc_user"][0], batch_dim)
                    if include_item_seq:
                        self.assertEqual(dyn_shapes["i_seq.lengths"][0], batch_dim)
                        self.assertEqual(dyn_shapes["i_seq__lengths"][0], batch_dim)

                    save_dir = os.path.join(case_dir, "export")
                    finalize_dense_export(
                        wrapped,
                        full_graph,
                        gm,
                        export_data,
                        device,
                        save_dir,
                        dense_graph_config,
                    )
                    aoti_model = torch._inductor.aoti_load_package(
                        os.path.join(save_dir, "aoti", "aoti_model.pt2")
                    )
                    in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                    key_order = list(in_spec.child(0).child(0).context)
                    gm_eval = gm.to(device).eval()

                    for bs in (1, 3):
                        case_inputs = {
                            "mix__ebc": torch.randn(bs, 4, device=device),
                            "mix__ebc_user": torch.randn(bs, 4, device=device),
                            "u_id.values": torch.ones(
                                bs, dtype=torch.int64, device=device
                            ),
                            "u_id.lengths": torch.ones(
                                bs, dtype=torch.int64, device=device
                            ),
                            "i_id.values": torch.ones(
                                bs, dtype=torch.int64, device=device
                            ),
                            "i_id.lengths": torch.ones(
                                bs, dtype=torch.int64, device=device
                            ),
                        }
                        if include_item_seq:
                            case_inputs.update(
                                {
                                    "i_seq.values": torch.ones(
                                        2 * bs, dtype=torch.int64, device=device
                                    ),
                                    "i_seq.lengths": torch.full(
                                        (bs,), 2, dtype=torch.int64, device=device
                                    ),
                                    "i_seq": torch.randn(2 * bs, 4, device=device),
                                    "i_seq__lengths": torch.full(
                                        (bs,), 2, dtype=torch.int32, device=device
                                    ),
                                }
                            )
                        case_inputs = {k: case_inputs[k] for k in key_order}
                        with torch.no_grad():
                            expected = gm_eval(case_inputs, device)
                            actual = aoti_model(case_inputs)
                        for k, val in expected.items():
                            torch.testing.assert_close(
                                actual[k], val, atol=1e-5, rtol=1e-5
                            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_aot_genrec_empty_anchor(self):
        device = torch.device("cuda:0")
        prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            with mock.patch.dict(
                os.environ,
                {
                    "USE_DISTRIBUTED_EMBEDDING": "1",
                    "INPUT_TILE": "3",
                    "INPUT_TILE_3_ONLINE": "0",
                    "ENABLE_AOT": "2",
                },
            ):
                u_id = feature_pb2.FeatureConfig(
                    id_feature=feature_pb2.IdFeature(
                        feature_name="u_id",
                        expression="user:u_id",
                        num_buckets=16,
                        embedding_dim=4,
                    )
                )
                model, _ = create_genrec_test_model(
                    os.path.join(self.test_dir, "genrec_static"),
                    feature_configs=[u_id],
                    prompt="Predict :",
                )
                init_parameters(model, device=device)
                wrapped = ScriptWrapper(GenRecFrontEnd(model)).to(device).eval()
                export_data = {
                    "batch_size": torch.tensor(1, dtype=torch.int64, device=device),
                    "u_id.values": torch.zeros(2, dtype=torch.int64, device=device),
                    "u_id.lengths": torch.ones(2, dtype=torch.int64, device=device),
                }
                gm, full_graph, dense_graph_config = build_dense_graph_module(
                    wrapped, export_data, device
                )
                save_dir = os.path.join(self.test_dir, "dist_genrec_static")
                finalize_dense_export(
                    wrapped,
                    full_graph,
                    gm,
                    export_data,
                    device,
                    save_dir,
                    dense_graph_config,
                )
                aoti_model = torch._inductor.aoti_load_package(
                    os.path.join(save_dir, "aoti", "aoti_model.pt2")
                )
                in_spec = treespec_loads(aoti_model.loader.get_call_spec()[0])
                key_order = list(in_spec.child(0).child(0).context)
                self.assertEqual(key_order, ["batch_size"])
                gm_eval = gm.to(device).eval()
                for bs in (1, 3):
                    case_inputs = {
                        "batch_size": torch.tensor(bs, dtype=torch.int64, device=device)
                    }
                    with torch.no_grad():
                        expected = gm_eval(case_inputs, device)
                        actual = aoti_model(case_inputs)
                    for k, val in expected.items():
                        torch.testing.assert_close(actual[k], val, atol=1e-5, rtol=1e-5)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
            torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32

    def test_jagged_sequence_min_bound_with_and_without_genrec_frontend(
        self,
    ) -> None:
        j_seq = feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="j_seq",
                sequence_length=4,
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="id",
                            expression="user:j_seq",
                            num_buckets=16,
                            embedding_dim=4,
                        )
                    )
                ],
            )
        )
        feats = create_features([j_seq])
        groups = [
            model_pb2.FeatureGroupConfig(
                group_name="jagged_seq",
                feature_names=["j_seq__id"],
                group_type=model_pb2.JAGGED_SEQUENCE,
            )
        ]

        class _PlainMod(nn.Module):
            def forward(self, data):
                return data["j_seq__id"]

        class PromptAssembler(nn.Module):
            def forward(self, x):
                return x

        sample_data = {
            "j_seq__id": torch.randn(3, 4),
            "j_seq__id.values": torch.tensor([1, 2, 3], dtype=torch.int64),
            "j_seq__id.lengths": torch.tensor([2, 1], dtype=torch.int64),
        }
        dense_meta = {"sequence__ec": ["j_seq__id__ec", "j_seq__id__lengths"]}

        plain_gm = fx.symbolic_trace(_PlainMod())
        shapes_plain = _build_distributed_dense_dynamic_shapes(
            sample_data=sample_data,
            gm=plain_gm,
            dense_graph_config=dense_meta,
            features=feats,
            feature_groups=groups,
            input_tile=False,
        )
        self.assertEqual(shapes_plain["j_seq__id"][0].min, 1)

        genrec_gm = fx.symbolic_trace(_PlainMod())
        genrec_gm.assembler = PromptAssembler()
        shapes_genrec = _build_distributed_dense_dynamic_shapes(
            sample_data=sample_data,
            gm=genrec_gm,
            dense_graph_config=dense_meta,
            features=feats,
            feature_groups=groups,
            input_tile=False,
        )
        self.assertEqual(shapes_genrec["j_seq__id"][0].min, 0)

    @mark_ci_scope("gpu")
    @unittest.skipIf(*gpu_unavailable)
    def test_distributed_dense_dead_mv_lengths(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        seq_mv_cfg = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="seq_mv",
                expression="user:seq_mv",
                num_buckets=16,
                embedding_dim=4,
                value_dim=0,
            )
        )
        mv_feats = create_features([seq_mv_cfg])

        class _DeadLengthsModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.features = mv_feats
                self.feature_groups = []
                self.head = nn.Linear(4, 1)

            def forward(self, data, dev: Optional[torch.device] = None):
                return {"out": self.head(data["seq_mv"]).sum(dim=1)}

        dead_mod = _DeadLengthsModel().to(device).eval()
        dead_gm = fx.symbolic_trace(dead_mod)
        input_node = next(n for n in dead_gm.graph.nodes if n.op == "placeholder")
        with dead_gm.graph.inserting_after(input_node):
            dead_gm.graph.call_function(
                operator.getitem, args=(input_node, "seq_mv__lengths")
            )
        dead_gm.recompile()
        self.assertNotIn(
            "seq_mv__lengths",
            aot_utils._extract_distributed_dense_consumed_keys(dead_gm),
        )
        eliminate_dense_dead_code(dead_gm.graph)
        dead_gm.recompile()
        dead_save_dir = os.path.join(self.test_dir, "dead_mv_lengths")
        with mock.patch.dict(os.environ, {"ENABLE_AOT": "2", "INPUT_TILE": "0"}):
            export_distributed_dense_model_aot(
                model=ScriptWrapper(dead_mod),
                dense_model_traced=dead_gm,
                data={
                    "seq_mv": torch.randn(3, 4, device=device),
                    "seq_mv__lengths": torch.tensor([2, 1], dtype=torch.int64),
                },
                sparse_kts={},
                seq_jts={
                    "seq_mv": JaggedTensor(
                        values=torch.randn(3, 4, device=device),
                        lengths=torch.tensor([2, 1], dtype=torch.int64, device=device),
                    )
                },
                dense_graph_config={"sequence__ec": ["seq_mv__ec", "seq_mv__lengths"]},
                save_dir=dead_save_dir,
                device=device,
            )
        dead_aoti = torch._inductor.aoti_load_package(
            os.path.join(dead_save_dir, "aoti", "aoti_model.pt2")
        )
        dead_spec = treespec_loads(dead_aoti.loader.get_call_spec()[0])
        self.assertEqual(list(dead_spec.child(0).child(0).context), ["seq_mv"])


if __name__ == "__main__":
    unittest.main()
