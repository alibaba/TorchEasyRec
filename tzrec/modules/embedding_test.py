# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import io
import os
import subprocess
import sys
import unittest
from collections import OrderedDict
from typing import Dict
from unittest import mock
from unittest.mock import patch

import torch
from parameterized import param, parameterized
from torch import nn
from torchrec import JaggedTensor, KeyedJaggedTensor, KeyedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.model import ScriptWrapper
from tzrec.models.multi_tower import MultiTower
from tzrec.modules.embedding import (
    EMPTY_KJT,
    EmbeddingGroup,
    EmbeddingGroupImpl,
    HardSearchInputTileExpander,
    SequenceEmbeddingGroupImpl,
    _pool_sequence_values,
    build_embedding_bag_configs,
    build_embedding_collection_configs,
)
from tzrec.protos import (
    feature_pb2,
    loss_pb2,
    model_pb2,
    module_pb2,
    seq_encoder_pb2,
    tower_pb2,
)
from tzrec.protos.models import rank_model_pb2
from tzrec.utils import dynamicemb_util
from tzrec.utils.fx_util import fx_mark_seq_ec_jt, symbolic_trace
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    TestGraphType,
    ci_resources,
    create_test_module,
    mark_ci_scope,
    parameterized_name_func,
)


def _create_test_features(has_zch=False):
    cat_a_kwargs = {}
    cat_b_kwargs = {}
    if has_zch:
        cat_a_kwargs["zch"] = feature_pb2.ZeroCollisionHash(
            zch_size=100, lfu=feature_pb2.LFU_EvictionPolicy()
        )
        cat_b_kwargs["zch"] = feature_pb2.ZeroCollisionHash(
            zch_size=1000, lru=feature_pb2.LRU_EvictionPolicy()
        )
    else:
        cat_a_kwargs["num_buckets"] = 100
        cat_b_kwargs["num_buckets"] = 1000
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a", embedding_dim=16, **cat_a_kwargs
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b", embedding_dim=8, **cat_b_kwargs
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_c", embedding_dim=12, num_buckets=1000
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(feature_name="int_a")
        ),
    ]
    features = create_features(feature_cfgs)
    return features


def _create_test_sequence_features(has_zch=False, has_mulval=False, pooling_type=None):
    cat_a_kwargs = {}
    cat_b_kwargs = {}
    if has_zch:
        cat_a_kwargs["zch"] = feature_pb2.ZeroCollisionHash(
            zch_size=100, lfu=feature_pb2.LFU_EvictionPolicy()
        )
        cat_b_kwargs["zch"] = feature_pb2.ZeroCollisionHash(
            zch_size=1000, lru=feature_pb2.LRU_EvictionPolicy()
        )
    else:
        cat_a_kwargs["num_buckets"] = 100
        cat_b_kwargs["num_buckets"] = 1000
    if has_mulval:
        cat_a_kwargs["value_dim"] = 0
        cat_a_kwargs["pooling"] = pooling_type
    feature_cfgs = [
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a",
                embedding_dim=16,
                expression="item:cat_a",
                **cat_a_kwargs,
            )
        ),
        feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b",
                embedding_dim=8,
                expression="item:cat_b",
                **cat_b_kwargs,
            )
        ),
        feature_pb2.FeatureConfig(
            raw_feature=feature_pb2.RawFeature(
                feature_name="int_a", expression="item:int_a"
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="click_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_a",
                            expression="item:cat_a",
                            embedding_dim=16,
                            **cat_a_kwargs,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_b",
                            expression="item:cat_b",
                            embedding_dim=8,
                            **cat_b_kwargs,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="int_a",
                            expression="item:int_a",
                        )
                    ),
                ],
            )
        ),
        feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="buy_seq",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat_a",
                            expression="item:cat_a",
                            embedding_dim=16,
                            **cat_a_kwargs,
                        )
                    ),
                    feature_pb2.SeqFeatureConfig(
                        raw_feature=feature_pb2.RawFeature(
                            feature_name="int_a",
                            expression="item:int_a",
                        )
                    ),
                ],
            )
        ),
    ]
    features = create_features(feature_cfgs)
    return features


class _EGScriptWrapper(nn.Module):
    """Embedding Group inference wrapper for jit.script."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self._module = module

    def forward(
        self,
        sparse_features: Dict[str, KeyedJaggedTensor],
        dense_features: Dict[str, KeyedTensor],
        sequence_dense_features: Dict[str, JaggedTensor],
        tile_size: int = -1,
    ):
        return self._module(
            Batch(
                sparse_features=sparse_features,
                dense_features=dense_features,
                sequence_dense_features=sequence_dense_features,
                tile_size=tile_size,
            )
        )


def _feature_group(
    group_name,
    feature_names,
    group_type=model_pb2.FeatureGroupType.DEEP,
    embedding_name_suffix="",
):
    """One feature group config."""
    return model_pb2.FeatureGroupConfig(
        group_name=group_name,
        feature_names=feature_names,
        group_type=group_type,
        embedding_name_suffix=embedding_name_suffix,
    )


def _create_shared_table_feature_configs(
    pooling_a=None, unique_a=None, pooling_b=None, unique_b=None
):
    """Two id features of one shared table, with optional measured factors."""
    configs = []
    for feature_name, pooling_factor, unique_id_factor in (
        ("cat_a", pooling_a, unique_a),
        ("cat_b", pooling_b, unique_b),
    ):
        f = feature_pb2.IdFeature(
            feature_name=feature_name,
            embedding_name="shared",
            embedding_dim=4,
            num_buckets=100,
        )
        if pooling_factor is not None:
            f.embedding_constraints.pooling_factor = pooling_factor
        if unique_id_factor is not None:
            f.embedding_constraints.unique_id_factor = unique_id_factor
        configs.append(feature_pb2.FeatureConfig(id_feature=f))
    return configs


class SequencePoolingTest(unittest.TestCase):
    @parameterized.expand(
        [
            (pooling, values, lengths, expected)
            for pooling in ("sum", "mean")
            for values, lengths, expected in (
                ([], [], []),
                ([], [0], [[0.0, 0.0]]),
                ([[2.0, 4.0]], [0, 1, 0], [[0.0, 0.0], [2.0, 4.0], [0.0, 0.0]]),
            )
        ],
        name_func=parameterized_name_func,
    )
    def test_empty_sequence_and_query_segments(
        self, pooling, values, lengths, expected
    ):
        values = torch.tensor(values, dtype=torch.float32).reshape(-1, 2)
        lengths = torch.tensor(lengths, dtype=torch.int64)
        expected = torch.tensor(expected, dtype=torch.float32).reshape(-1, 2)
        result = _pool_sequence_values(values, lengths, pooling)
        torch.testing.assert_close(result, expected)
        self.assertTrue(torch.isfinite(result).all())


@ci_resources(slots=1)
class EmbeddingGroupTest(unittest.TestCase):
    def tearDown(self):
        os.environ.pop("INPUT_TILE", None)

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_embedding_group_impl(self, graph_type, has_zch=False) -> None:
        features = _create_test_features(has_zch=has_zch)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="wide",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.WIDE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("wide"), [4, 4])
        self.assertEqual(embedding_group.group_dims("deep"), [16, 8, 1])
        self.assertEqual(embedding_group.group_total_dim("wide"), 8)
        self.assertEqual(embedding_group.group_total_dim("deep"), 25)
        wide_feature_dims = OrderedDict({"cat_a": 4, "cat_b": 4})
        deep_feature_dims = OrderedDict({"cat_a": 16, "cat_b": 8, "int_a": 1})
        self.assertDictEqual(
            embedding_group.group_feature_dims("wide"), wide_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("deep"), deep_feature_dims
        )
        if has_zch and graph_type != TestGraphType.NORMAL:
            embedding_group.eval()
        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        result = embedding_group(
            sparse_feature,
            dense_feature,
            EMPTY_KJT,
        )

        self.assertEqual(result["wide"].size(), (2, 8))
        self.assertEqual(result["deep"].size(), (2, 25))

    @staticmethod
    def _create_shared_table_embedding_group(feature_names, configs):
        """One DEEP group over the given shared-table feature configs."""
        features = create_features(configs)
        embedding_group = EmbeddingGroupImpl(
            features,
            [
                model_pb2.FeatureGroupConfig(
                    group_name="deep",
                    feature_names=feature_names,
                    group_type=model_pb2.FeatureGroupType.DEEP,
                )
            ],
            device=torch.device("cpu"),
        )
        return embedding_group

    @parameterized.expand(
        [
            param(
                "both_factors",
                pooling_a=2.5,
                pooling_b=1.5,
                expected_factors=[2.5, 1.5],
            ),
            param(
                "order_follows_the_group",
                pooling_a=2.5,
                pooling_b=1.5,
                feature_names=["cat_b", "cat_a"],
                expected_factors=[1.5, 2.5],
            ),
            param("unset_contributes_one", pooling_a=2.5, expected_factors=[2.5, 1.0]),
        ],
        name_func=parameterized_name_func,
    )
    def test_shared_table_preserves_feature_order_and_pooling_factors(
        self,
        name,
        pooling_a=None,
        pooling_b=None,
        feature_names=None,
        expected_factors=None,
    ) -> None:
        # one raw id factor per input feature, in group order, unset 1.0
        configs = _create_shared_table_feature_configs(
            pooling_a=pooling_a, pooling_b=pooling_b
        )
        embedding_group = self._create_shared_table_embedding_group(
            feature_names or ["cat_a", "cat_b"], configs
        )
        const = embedding_group.parameter_constraints()["ebc.shared"]
        self.assertEqual(const.pooling_factors, expected_factors)
        self.assertEqual(const.feature_names, feature_names or ["cat_a", "cat_b"])

    def test_shared_table_counts_a_reused_feature_once(self) -> None:
        # the same feature referenced twice by one group is one input
        configs = _create_shared_table_feature_configs(pooling_a=2.5, pooling_b=1.5)
        embedding_group = self._create_shared_table_embedding_group(
            ["cat_a", "cat_b", "cat_a"], configs
        )
        const = embedding_group.parameter_constraints()["ebc.shared"]
        self.assertEqual(const.pooling_factors, [2.5, 1.5])
        self.assertEqual(const.feature_names, ["cat_a", "cat_b"])

    @parameterized.expand(
        [
            param("max_of_members", unique_a=20.0, unique_b=30.0, expected=30.0),
            param(
                "unmeasured_members_contribute_nothing",
                unique_a=20.0,
                unique_b=None,
                expected=20.0,
            ),
            param(
                "measured_below_the_raw_sum",
                pooling_a=10.0,
                pooling_b=20.0,
                unique_a=3.0,
                expected=3.0,
            ),
            param("no_member_measured", unique_a=None, unique_b=None, expected=None),
        ],
        name_func=parameterized_name_func,
    )
    def test_shared_table_unique_id_factor_aggregation(
        self,
        name,
        pooling_a=None,
        pooling_b=None,
        unique_a=None,
        unique_b=None,
        expected=None,
    ) -> None:
        # configured values aggregate with max; a table whose members set
        # none keeps the value unset - nothing is fabricated in its place
        configs = _create_shared_table_feature_configs(
            pooling_a=pooling_a,
            pooling_b=pooling_b,
            unique_a=unique_a,
            unique_b=unique_b,
        )
        embedding_group = self._create_shared_table_embedding_group(
            ["cat_a", "cat_b"], configs
        )
        const = embedding_group.parameter_constraints()["ebc.shared"]
        if expected is None:
            self.assertIsNone(getattr(const, "unique_id_factor", None))
        else:
            self.assertEqual(const.unique_id_factor, expected)

    def test_shared_table_preserves_sequence_feature_identity(self) -> None:
        # two sequence groups whose sub-features share the bare config name
        # and one table are two inputs; the constraint identity is the batch
        # name, so neither collapses into the other
        configs = []
        for sequence_name, pooling_factor in (("click_seq", 2.5), ("buy_seq", 1.5)):
            wrapper = feature_pb2.SequenceFeature(
                sequence_name=sequence_name,
                sequence_length=50,
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="item_id",
                            embedding_name="shared",
                            embedding_dim=4,
                            num_buckets=100,
                        )
                    )
                ],
            )
            wrapper.features[
                0
            ].id_feature.embedding_constraints.pooling_factor = pooling_factor
            configs.append(feature_pb2.FeatureConfig(sequence_feature=wrapper))
        features = create_features(configs)
        seq_group = SequenceEmbeddingGroupImpl(
            features,
            [
                model_pb2.FeatureGroupConfig(
                    group_name="click",
                    feature_names=["click_seq__item_id"],
                    group_type=model_pb2.FeatureGroupType.SEQUENCE,
                ),
                model_pb2.FeatureGroupConfig(
                    group_name="buy",
                    feature_names=["buy_seq__item_id"],
                    group_type=model_pb2.FeatureGroupType.SEQUENCE,
                ),
            ],
            device=torch.device("cpu"),
        )
        const = seq_group.parameter_constraints()["ec_dict.4.shared"]
        self.assertEqual(const.pooling_factors, [2.5, 1.5])
        self.assertEqual(
            const.feature_names, ["click_seq__item_id", "buy_seq__item_id"]
        )

    def test_build_embedding_bag_configs_ordinary(self) -> None:
        features = _create_test_features()
        table_configs = build_embedding_bag_configs(
            features, [_feature_group("deep", ["cat_a", "cat_b", "int_a"])]
        )
        self.assertEqual(
            [
                (c.name, list(c.feature_names), c.embedding_dim)
                for c in table_configs.ebc.values()
            ],
            [("cat_a_emb", ["cat_a"], 16), ("cat_b_emb", ["cat_b"], 8)],
        )
        self.assertFalse(table_configs.mc_ebc)
        self.assertFalse(table_configs.ebc_user)
        self.assertFalse(table_configs.mc_ebc_user)
        self.assertFalse(any(table_configs.shared_feature_flag.values()))

    def test_build_embedding_bag_configs_zch(self) -> None:
        features = _create_test_features(has_zch=True)
        table_configs = build_embedding_bag_configs(
            features, [_feature_group("deep", ["cat_a", "cat_b"])]
        )
        self.assertFalse(table_configs.ebc)
        self.assertEqual(
            [(c.name, list(c.feature_names)) for c in table_configs.mc_ebc.values()],
            [("cat_a_emb", ["cat_a"]), ("cat_b_emb", ["cat_b"])],
        )

    @unittest.skipUnless(
        dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
    )
    @mark_ci_scope("gpu")
    def test_dynamicemb_constraint_sees_the_wide_table(self) -> None:
        # the constraint is built from the table config the builder resized
        # and renamed, not from a fresh feature config
        f = feature_pb2.IdFeature(
            feature_name="cat_a", embedding_dim=16, num_buckets=100
        )
        f.dynamicemb.max_capacity = 100_000
        # NORMAL's default std_dev comes from the table's dim, so the mode
        # makes the wide resize observable
        f.dynamicemb.initializer_args.mode = "NORMAL"
        features = create_features([feature_pb2.FeatureConfig(id_feature=f)])
        wide = model_pb2.FeatureGroupConfig(
            group_name="wide",
            feature_names=["cat_a"],
            group_type=model_pb2.FeatureGroupType.WIDE,
        )
        embedding_group = EmbeddingGroupImpl(
            features, [wide], wide_embedding_dim=4, device=torch.device("cpu")
        )
        const = embedding_group.parameter_constraints()["ebc.cat_a_emb_wide"]
        self.assertTrue(const.use_dynamicemb)
        # default initialization is uniform with bounds from the table dim
        initializer_args = const.dynamicemb_options.initializer_args
        self.assertEqual(str(initializer_args.mode), "DynamicEmbInitializerMode.NORMAL")
        self.assertAlmostEqual(initializer_args.std_dev, 1.0 / (4**0.5), places=6)

    @parameterized.expand(
        [("pooled", False), ("sequence", True)],
        name_func=parameterized_name_func,
    )
    @unittest.skipUnless(dynamicemb_util.has_dynamicemb, "dynamicemb is not installed.")
    @mark_ci_scope("gpu")
    def test_dynamicemb_capacity_is_normalized_before_tensor_creation(
        self, _name, sequence
    ) -> None:
        cfg = feature_pb2.IdFeature(
            feature_name="cat_a",
            embedding_name="aligned",
            embedding_dim=4,
            num_buckets=100,
        )
        cfg.dynamicemb.max_capacity = 100003
        features = create_features([feature_pb2.FeatureConfig(id_feature=cfg)])
        group_cfg = model_pb2.FeatureGroupConfig(
            group_name="group",
            feature_names=["cat_a"],
            group_type=(
                model_pb2.FeatureGroupType.SEQUENCE
                if sequence
                else model_pb2.FeatureGroupType.DEEP
            ),
        )
        factory = SequenceEmbeddingGroupImpl if sequence else EmbeddingGroupImpl
        with mock.patch.dict("os.environ", {"WORLD_SIZE": "4"}):
            group = factory(features, [group_cfg], device=torch.device("meta"))
        collection = group.ec_dict["4"] if sequence else group.ebc
        self.assertEqual(next(collection.parameters()).shape, (100352, 4))
        constraint = next(iter(group.parameter_constraints().values()))
        self.assertEqual(constraint.dynamicemb_options.max_capacity, 25088)

    def test_build_embedding_bag_configs_wide_resizes_and_renames(self) -> None:
        features = _create_test_features()
        wide = model_pb2.FeatureGroupConfig(
            group_name="wide",
            feature_names=["cat_a"],
            group_type=model_pb2.FeatureGroupType.WIDE,
        )
        table_configs = build_embedding_bag_configs(
            features, [wide], wide_embedding_dim=8
        )
        self.assertEqual(
            [
                (c.name, list(c.feature_names), c.embedding_dim)
                for c in table_configs.ebc.values()
            ],
            [("cat_a_emb_wide", ["cat_a"], 8)],
        )

    def test_build_embedding_bag_configs_suffix_makes_another_table(self) -> None:
        # the same feature in a plain and a suffixed group feeds two tables
        # and is marked shared between them
        features = _create_test_features()
        groups = [
            _feature_group("deep", ["cat_a"]),
            _feature_group("deep_x", ["cat_a"], embedding_name_suffix="x"),
        ]
        table_configs = build_embedding_bag_configs(features, groups)
        self.assertEqual(
            [(c.name, list(c.feature_names)) for c in table_configs.ebc.values()],
            [("cat_a_emb", ["cat_a"]), ("cat_a_emb_x", ["cat_a"])],
        )

    @unittest.skipUnless(
        dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
    )
    @mark_ci_scope("gpu")
    def test_build_embedding_bag_configs_dynamicemb(self) -> None:
        # a dynamicemb feature's table resolves like any other
        f = feature_pb2.IdFeature(
            feature_name="dyn_a", embedding_dim=16, num_buckets=100
        )
        f.dynamicemb.max_capacity = 100_000
        features = create_features([feature_pb2.FeatureConfig(id_feature=f)])
        table_configs = build_embedding_bag_configs(
            features, [_feature_group("deep", ["dyn_a"])]
        )
        self.assertEqual(
            [
                (c.name, list(c.feature_names), c.embedding_dim)
                for c in table_configs.ebc.values()
            ],
            [("dyn_a_emb", ["dyn_a"], 16)],
        )

    def test_build_embedding_collection_configs_groups_by_dim(self) -> None:
        features = _create_test_sequence_features()
        table_configs = build_embedding_collection_configs(
            features,
            [
                _feature_group(
                    "seq",
                    ["cat_a", "cat_b", "click_seq__cat_a", "click_seq__cat_b"],
                    group_type=model_pb2.FeatureGroupType.SEQUENCE,
                )
            ],
        )
        self.assertEqual(
            [(c.name, list(c.feature_names)) for c in table_configs.ec[16].values()],
            [("cat_a_emb", ["cat_a"]), ("click_seq__cat_a_emb", ["click_seq__cat_a"])],
        )
        self.assertEqual(
            [(c.name, list(c.feature_names)) for c in table_configs.ec[8].values()],
            [("cat_b_emb", ["cat_b"]), ("click_seq__cat_b_emb", ["click_seq__cat_b"])],
        )
        self.assertFalse(table_configs.mc_ec)

    def test_build_embedding_collection_configs_shared_table(self) -> None:
        # two sequence sub-features of one table keep one entry each, and a
        # shared feature's batch name carries the table
        configs = []
        for sequence_name in ("click_seq", "buy_seq"):
            wrapper = feature_pb2.SequenceFeature(
                sequence_name=sequence_name,
                sequence_length=50,
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name=sequence_name + "_tag",
                            embedding_name="shared",
                            embedding_dim=4,
                            num_buckets=100,
                        )
                    )
                ],
            )
            configs.append(feature_pb2.FeatureConfig(sequence_feature=wrapper))
        features = create_features(configs)
        table_configs = build_embedding_collection_configs(
            features,
            [
                _feature_group(
                    "seq",
                    ["click_seq__click_seq_tag", "buy_seq__buy_seq_tag"],
                    group_type=model_pb2.FeatureGroupType.SEQUENCE,
                )
            ],
        )
        self.assertEqual(
            [(c.name, list(c.feature_names)) for c in table_configs.ec[4].values()],
            [("shared", ["click_seq__click_seq_tag", "buy_seq__buy_seq_tag"])],
        )
        self.assertEqual(
            table_configs.emb_name_to_feature_to_shared_name["shared"],
            {
                "click_seq__click_seq_tag": "click_seq__click_seq_tag",
                "buy_seq__buy_seq_tag": "buy_seq__buy_seq_tag",
            },
        )

    def test_data_groups_build_separate_embedding_tables(self) -> None:
        # features of one group must share a data group; different data
        # groups build separate embedding collections of their own
        features = _create_test_features()
        cat_a = next(f for f in features if f.name == "cat_a")
        cat_a.data_group = "user"
        embedding_group = EmbeddingGroup(
            features,
            [
                _feature_group("user_deep", ["cat_a"]),
                _feature_group("deep", ["cat_b"]),
            ],
            device=torch.device("cpu"),
        )
        self.assertEqual(set(embedding_group.emb_impls), {BASE_DATA_GROUP, "user"})
        user_tables = [
            c.name
            for c in embedding_group.emb_impls["user"].ebc.embedding_bag_configs()
        ]
        base_tables = [
            c.name
            for c in embedding_group.emb_impls[
                BASE_DATA_GROUP
            ].ebc.embedding_bag_configs()
        ]
        self.assertEqual(user_tables, ["cat_a_emb"])
        self.assertEqual(base_tables, ["cat_b_emb"])

    def test_mixed_data_groups_in_one_group_are_rejected(self) -> None:
        features = _create_test_features()
        cat_a = next(f for f in features if f.name == "cat_a")
        cat_a.data_group = "user"
        with self.assertRaisesRegex(ValueError, "same feature group"):
            EmbeddingGroup(
                features,
                [_feature_group("deep", ["cat_a", "cat_b"])],
                device=torch.device("cpu"),
            )

    @parameterized.expand(
        [
            param("use_weight", use_weight=True),
            param("wo_use_weight", use_weight=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_embedding_group_impl_weighted(self, name, use_weight) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=use_weight,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.ebc.is_weighted(), use_weight)

        values = torch.tensor([1, 2, 3])
        weights = torch.tensor([0.5, 2.0, 1.0])
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a"],
            values=values,
            lengths=torch.tensor([2, 1]),
            weights=weights,
        )
        result = embedding_group(sparse_feature, None, EMPTY_KJT)

        emb = next(iter(embedding_group.ebc.embedding_bags.values())).weight
        if use_weight:
            expected = torch.stack(
                [
                    emb[1] * weights[0] + emb[2] * weights[1],
                    emb[3] * weights[2],
                ]
            )
        else:
            expected = torch.stack([emb[1] + emb[2], emb[3]])
        torch.testing.assert_close(result["deep"], expected)

    def test_embedding_group_impl_weighted_with_mean_pooling(self) -> None:
        # mean pooling of a weighted group is done by sum pooling over the
        # per-sample weights that DataParser normalizes to sum to one per bag.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    pooling="mean",
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertTrue(embedding_group.ebc.is_weighted())
        for emb_bag_config in embedding_group.ebc.embedding_bag_configs():
            self.assertEqual(emb_bag_config.pooling, PoolingType.SUM)

        # weights as DataParser normalizes them: cat_a raw [1, 3], cat_b plain mean
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor([1, 2, 3, 4]),
            lengths=torch.tensor([2, 2]),
            weights=torch.tensor([0.25, 0.75, 0.5, 0.5]),
        )
        result = embedding_group(sparse_feature, None, EMPTY_KJT)

        emb_a = embedding_group.ebc.embedding_bags["cat_a_emb"].weight
        emb_b = embedding_group.ebc.embedding_bags["cat_b_emb"].weight
        expected = torch.cat(
            [
                (0.25 * emb_a[1] + 0.75 * emb_a[2]).unsqueeze(0),
                (0.5 * emb_b[3] + 0.5 * emb_b[4]).unsqueeze(0),
            ],
            dim=1,
        )
        torch.testing.assert_close(result["deep"], expected)

    @parameterized.expand(
        [
            param("sum_neighbor", neighbor_pooling="sum", expected_ebc_weighted=False),
            param("mean_neighbor", neighbor_pooling="mean", expected_ebc_weighted=True),
        ],
        name_func=parameterized_name_func,
    )
    def test_embedding_group_impl_weighted_per_collection(
        self, name, neighbor_pooling, expected_ebc_weighted
    ) -> None:
        # is_weighted is per EmbeddingBagCollection: the weighted feature lives in
        # the zch collection, so the plain one only needs weights when it holds a
        # mean pooled table, whose mean is carried by the normalized weights.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    zch=feature_pb2.ZeroCollisionHash(
                        zch_size=100, lfu=feature_pb2.LFU_EvictionPolicy()
                    ),
                    weighted=True,
                    use_weight=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b",
                    embedding_dim=8,
                    num_buckets=1000,
                    pooling=neighbor_pooling,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.ebc.is_weighted(), expected_ebc_weighted)
        self.assertTrue(embedding_group.mc_ebc._embedding_bag_collection.is_weighted())

    @parameterized.expand(
        [
            param("weighted_first", weighted_first=True),
            param("weighted_second", weighted_first=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_embedding_group_impl_weighted_shared_table(self, name, weighted_first):
        # a shared table is weighted when any of its features needs weight,
        # whichever of them is merged into the config dict first.
        weighted = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_a",
                embedding_dim=8,
                num_buckets=100,
                embedding_name="shared_emb",
                weighted=True,
                use_weight=True,
            )
        )
        plain = feature_pb2.FeatureConfig(
            id_feature=feature_pb2.IdFeature(
                feature_name="cat_b",
                embedding_dim=8,
                num_buckets=100,
                embedding_name="shared_emb",
            )
        )
        feature_cfgs = [weighted, plain] if weighted_first else [plain, weighted]
        names = ["cat_a", "cat_b"] if weighted_first else ["cat_b", "cat_a"]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=names,
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertTrue(embedding_group.ebc.is_weighted())

    def test_embedding_group_impl_weighted_input_tile_emb(self) -> None:
        # under INPUT_TILE=3 the user side gets its own collections, so the
        # weighted flag has to be resolved per collection there as well.
        os.environ["INPUT_TILE"] = "3"
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_w",
                    expression="user:u_w",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="u_z",
                    expression="user:u_z",
                    embedding_dim=8,
                    zch=feature_pb2.ZeroCollisionHash(
                        zch_size=100, lfu=feature_pb2.LFU_EvictionPolicy()
                    ),
                    pooling="mean",
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="i_s",
                    expression="item:i_s",
                    embedding_dim=8,
                    num_buckets=100,
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["u_w", "u_z", "i_s"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        # u_w carries its own weight, u_z is mean pooled in a weighted data group
        # so its mean rides on the normalized weights
        self.assertTrue(embedding_group.ebc_user.is_weighted())
        mc_ebc_user = embedding_group.mc_ebc_user._embedding_bag_collection
        self.assertTrue(mc_ebc_user.is_weighted())
        for emb_bag_config in mc_ebc_user.embedding_bag_configs():
            self.assertEqual(emb_bag_config.pooling, PoolingType.SUM)
        # the item side holds only an unweighted sum table
        self.assertFalse(embedding_group.ebc.is_weighted())

    def test_embedding_group_impl_weighted_stub_feature(self) -> None:
        # DataParser skips stub features when collecting weight keys, so a stub
        # must not mark its data group weighted either, otherwise the group's
        # mean tables would be rewritten to sum against a kjt with no weights.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="stub_w",
                    expression="user:stub_w",
                    embedding_dim=8,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                    stub_type=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_m",
                    expression="item:cat_m",
                    embedding_dim=8,
                    num_buckets=100,
                    pooling="mean",
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_m"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertFalse(embedding_group.ebc.is_weighted())
        self.assertEqual(
            embedding_group.ebc.embedding_bag_configs()[0].pooling, PoolingType.MEAN
        )

    @parameterized.expand(
        [
            param("sum_dynamicemb", pooling="sum"),
            param("mean_dynamicemb", pooling="mean"),
        ],
        name_func=parameterized_name_func,
    )
    @unittest.skipIf(not dynamicemb_util.has_dynamicemb, "dynamicemb not available.")
    @mark_ci_scope("gpu")
    def test_embedding_group_impl_weighted_with_dynamicemb(self, name, pooling) -> None:
        # dynamicemb only pools with weights in sum mode, and it reads the kjt
        # weights only when the collection is weighted, so both must hold for
        # every table of a weighted data group.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a",
                    embedding_dim=16,
                    num_buckets=100,
                    weighted=True,
                    use_weight=True,
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b",
                    embedding_dim=8,
                    pooling=pooling,
                    dynamicemb=feature_pb2.DynamicEmbedding(max_capacity=1024),
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        embedding_group = EmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertTrue(embedding_group.ebc.is_weighted())
        for emb_bag_config in embedding_group.ebc.embedding_bag_configs():
            self.assertEqual(emb_bag_config.pooling, PoolingType.SUM)

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False, None],
            [TestGraphType.FX_TRACE, False, False, None],
            [TestGraphType.JIT_SCRIPT, False, False, None],
            [TestGraphType.NORMAL, True, False, None],
            [TestGraphType.FX_TRACE, True, False, None],
            [TestGraphType.JIT_SCRIPT, True, False, None],
            [TestGraphType.NORMAL, False, True, "sum"],
            [TestGraphType.FX_TRACE, False, True, "sum"],
            [TestGraphType.JIT_SCRIPT, False, True, "sum"],
            [TestGraphType.NORMAL, False, True, "mean"],
            [TestGraphType.FX_TRACE, False, True, "mean"],
            [TestGraphType.JIT_SCRIPT, False, True, "mean"],
        ]
    )
    def test_sequence_embedding_group_impl(
        self, graph_type, has_zch=False, has_mulval=False, pooling_type=None
    ) -> None:
        features = _create_test_sequence_features(has_zch, has_mulval, pooling_type)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_all",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_other",
                feature_names=[
                    "cat_a",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__int_a",
                ],
            ),
            model_pb2.SeqGroupConfig(
                group_name="deep___click_no_query",
                feature_names=["click_seq__cat_a", "click_seq__int_a"],
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(
            embedding_group.group_dims("deep___click_all.sequence"), [16, 8, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_all.query"), [16, 8, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_other.sequence"), [16, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_other.query"), [16, 1]
        )
        self.assertEqual(
            embedding_group.group_dims("deep___click_no_query.sequence"), [16, 1]
        )
        self.assertEqual(embedding_group.group_dims("deep___click_no_query.query"), [])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_all.sequence"), 25
        )
        self.assertEqual(embedding_group.group_total_dim("deep___click_all.query"), 25)
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_other.sequence"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_other.query"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_no_query.sequence"), 17
        )
        self.assertEqual(
            embedding_group.group_total_dim("deep___click_no_query.query"), 0
        )

        if has_zch and graph_type != TestGraphType.NORMAL:
            embedding_group = embedding_group.eval()
        embedding_group = create_test_module(embedding_group, graph_type)

        if has_mulval:
            values = torch.tensor(list(range(30)))
            lengths = torch.tensor([2, 0, 1, 1, 4, 5, 3, 3, 3, 4, 2, 2])
            sequence_mulval_lengths = KeyedJaggedTensor.from_lengths_sync(
                keys=["click_seq__cat_a", "buy_seq__cat_a"],
                values=torch.tensor([1, 0, 3, 1, 2, 2, 1, 2, 2, 2]),
                lengths=torch.tensor([3, 3, 2, 2]),
            )
        else:
            values = torch.tensor(list(range(24)))
            lengths = torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2])
            sequence_mulval_lengths = EMPTY_KJT
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=values,
            lengths=lengths,
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(10)], dtype=torch.float32),
            lengths=torch.tensor([3, 3, 2, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            sequence_mulval_lengths,
            EMPTY_KJT,
            EMPTY_KJT,
        )
        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertFalse(torch.any(torch.isnan(result["click.query"])).item())
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertFalse(torch.any(torch.isnan(result["click.sequence"])).item())
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))
        self.assertEqual(result["deep___click_all.query"].size(), (2, 25))
        self.assertEqual(result["deep___click_all.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["deep___click_all.sequence_length"].size(), (2,))
        self.assertEqual(result["deep___click_other.query"].size(), (2, 17))
        self.assertEqual(result["deep___click_other.sequence"].size(), (2, 3, 17))
        self.assertEqual(result["deep___click_other.sequence_length"].size(), (2,))
        self.assertTrue("deep___click_no_query.query" not in result)
        self.assertEqual(result["deep___click_no_query.sequence"].size(), (2, 3, 17))
        self.assertEqual(result["deep___click_no_query.sequence_length"].size(), (2,))

    def test_sequence_embedding_group_impl_zch_marks_seq_jt(self) -> None:
        """Distributed export splits the sparse part on fx_mark_seq_ec_jt marks."""
        features = _create_test_sequence_features(has_zch=True)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        embedding_group.eval()
        graph = symbolic_trace(embedding_group).graph
        marked = {
            node.args[0]
            for node in graph.nodes
            if node.op == "call_function" and node.target == fx_mark_seq_ec_jt
        }
        self.assertEqual(
            marked, {"cat_a", "cat_b", "click_seq__cat_a", "click_seq__cat_b"}
        )

    @parameterized.expand([[False], [True]], name_func=parameterized_name_func)
    def test_sequence_embedding_group_impl_emb_name_suffix(self, has_zch) -> None:
        features = _create_test_sequence_features(has_zch=has_zch)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name=f"click_{suffix}",
                feature_names=[
                    "cat_a",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
                embedding_name_suffix=suffix,
            )
            for suffix in ["x", "y"]
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        embedding_group.eval()

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "click_seq__cat_a"],
            values=torch.tensor(list(range(10))),
            lengths=torch.tensor([1, 1, 3, 5]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a"],
            values=torch.tensor([[x] for x in range(8)], dtype=torch.float32),
            lengths=torch.tensor([3, 5]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            EMPTY_KJT,
            EMPTY_KJT,
            EMPTY_KJT,
        )
        for suffix in ["x", "y"]:
            self.assertEqual(result[f"click_{suffix}.query"].size(), (2, 17))
            self.assertEqual(result[f"click_{suffix}.sequence"].size(), (2, 5, 17))
        self.assertFalse(
            torch.allclose(result["click_x.query"], result["click_y.query"])
        )

    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
            [TestGraphType.JIT_SCRIPT],
        ]
    )
    def test_jagged_sequence_embedding_group_impl(self, graph_type) -> None:
        features = _create_test_sequence_features()
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.JAGGED_SEQUENCE,
            ),
        ]

        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)

        embedding_group = create_test_module(embedding_group, graph_type)

        values = torch.tensor(list(range(24)))
        lengths = torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2])
        sequence_mulval_lengths = EMPTY_KJT
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=values,
            lengths=lengths,
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(10)], dtype=torch.float32),
            lengths=torch.tensor([3, 3, 2, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            sequence_mulval_lengths,
            EMPTY_KJT,
            EMPTY_KJT,
        )
        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (6, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (4, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_embedding_group(self, graph_type, has_zch) -> None:
        features = _create_test_sequence_features(has_zch)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="wide",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.WIDE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b", "int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="click_seq",
                        feature_names=[
                            "cat_a",
                            "cat_b",
                            "int_a",
                            "click_seq__cat_a",
                            "click_seq__cat_b",
                            "click_seq__int_a",
                        ],
                    ),
                    model_pb2.SeqGroupConfig(
                        group_name="buy_seq",
                        feature_names=[
                            "cat_a",
                            "int_a",
                            "buy_seq__cat_a",
                            "buy_seq__int_a",
                        ],
                    ),
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        din_encoder=seq_encoder_pb2.DINEncoder(
                            input="click_seq",
                            attn_mlp=module_pb2.MLP(hidden_units=[128, 64]),
                        )
                    ),
                    seq_encoder_pb2.SeqEncoderConfig(
                        simple_attention=seq_encoder_pb2.SimpleAttention(
                            input="buy_seq"
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="only_buy",
                group_type=model_pb2.FeatureGroupType.DEEP,
                sequence_groups=[
                    model_pb2.SeqGroupConfig(
                        group_name="only_buy_seq",
                        feature_names=[
                            "cat_a",
                            "int_a",
                            "buy_seq__cat_a",
                            "buy_seq__int_a",
                        ],
                    )
                ],
                sequence_encoders=[
                    seq_encoder_pb2.SeqEncoderConfig(
                        simple_attention=seq_encoder_pb2.SimpleAttention(
                            input="only_buy_seq"
                        )
                    ),
                ],
            ),
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = EmbeddingGroup(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("wide"), [4, 4])
        self.assertEqual(embedding_group.group_dims("deep"), [16, 8, 1, 25, 17])
        self.assertEqual(embedding_group.group_dims("only_buy"), [17])
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("wide"), 8)
        self.assertEqual(embedding_group.group_total_dim("deep"), 67)
        self.assertEqual(embedding_group.group_total_dim("only_buy"), 17)
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)
        wide_feature_dims = {"cat_a": 4, "cat_b": 4}
        deep_feature_dims = {
            "cat_a": 16,
            "cat_b": 8,
            "int_a": 1,
            "deep_seq_encoder_0": 25,
            "deep_seq_encoder_1": 17,
        }
        only_buy_feature_dims = {"only_buy_seq_encoder_0": 17}
        self.assertDictEqual(
            embedding_group.group_feature_dims("wide"), wide_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("deep"), deep_feature_dims
        )
        self.assertDictEqual(
            embedding_group.group_feature_dims("only_buy"), only_buy_feature_dims
        )

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=torch.tensor(list(range(24))),
            lengths=torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(10)], dtype=torch.float32),
            lengths=torch.tensor([3, 3, 2, 2]),
        ).to_dict()

        if has_zch and graph_type != TestGraphType.NORMAL:
            embedding_group = embedding_group.eval()
        if graph_type == TestGraphType.JIT_SCRIPT:
            embedding_group = create_test_module(
                _EGScriptWrapper(embedding_group), graph_type
            )
            result = embedding_group(
                sparse_features={BASE_DATA_GROUP: sparse_feature},
                dense_features={BASE_DATA_GROUP: dense_feature},
                sequence_dense_features=sequence_dense_feature,
            )
        else:
            embedding_group = create_test_module(embedding_group, graph_type)
            result = embedding_group(
                Batch(
                    sparse_features={BASE_DATA_GROUP: sparse_feature},
                    dense_features={BASE_DATA_GROUP: dense_feature},
                    sequence_dense_features=sequence_dense_feature,
                )
            )
        self.assertEqual(result["wide"].size(), (2, 8))
        self.assertEqual(result["deep"].size(), (2, 67))
        self.assertEqual(result["only_buy"].size(), (2, 17))
        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))

    @parameterized.expand(
        [
            # Two DEEP groups, different suffix -> distinct tables.
            param(
                "distinct_suffix_per_tower",
                deep_a_suffix="tower_a",
                deep_b_suffix="tower_b",
                expected_ebc={
                    "cat_a_emb_tower_a",
                    "cat_a_emb_tower_b",
                    "cat_b_emb_tower_a",
                    "cat_b_emb_tower_b",
                },
                forbidden_ebc={"cat_a_emb", "cat_b_emb"},
            ),
            # Two DEEP groups with same suffix -> shared (no unsuffixed leakage).
            param(
                "same_suffix_shares",
                deep_a_suffix="shared",
                deep_b_suffix="shared",
                expected_ebc={"cat_a_emb_shared", "cat_b_emb_shared"},
                forbidden_ebc={"cat_a_emb", "cat_b_emb"},
            ),
            # WIDE + suffix -> "_wide_<suffix>" (additive with _wide).
            param(
                "wide_plus_suffix",
                wide_suffix="tower_a",
                expected_ebc={"cat_a_emb_wide_tower_a", "cat_b_emb_wide_tower_a"},
                forbidden_ebc={"cat_a_emb_wide", "cat_a_emb"},
            ),
            # Nested seq inherits parent suffix when child doesn't set its own.
            param(
                "nested_seq_inherits_parent",
                seq_parent_suffix="tower_a",
                expected_ebc={"cat_a_emb_tower_a"},
                forbidden_ebc={"cat_a_emb"},
                expected_ec={"cat_a_emb_tower_a", "click_seq__cat_a_emb_tower_a"},
                forbidden_ec={"cat_a_emb", "click_seq__cat_a_emb"},
            ),
            # Explicit child suffix overrides parent.
            param(
                "explicit_child_overrides_parent",
                seq_parent_suffix="parent",
                seq_child_suffix="child",
                expected_ebc={"cat_a_emb_parent"},
                expected_ec={"cat_a_emb_child", "click_seq__cat_a_emb_child"},
                forbidden_ec={"cat_a_emb_parent", "click_seq__cat_a_emb_parent"},
            ),
            # Top-level SEQUENCE FeatureGroupConfig with suffix -> suffix
            # applied to both query and sequence-side embedding tables.
            param(
                "toplevel_sequence_with_suffix",
                seq_toplevel_suffix="tower_a",
                expected_ec={
                    "cat_a_emb_tower_a",
                    "cat_b_emb_tower_a",
                    "click_seq__cat_a_emb_tower_a",
                    "click_seq__cat_b_emb_tower_a",
                },
                forbidden_ec={"cat_a_emb", "click_seq__cat_a_emb"},
            ),
        ]
    )
    def test_embedding_name_suffix(
        self,
        name,
        deep_a_suffix=None,
        deep_b_suffix=None,
        wide_suffix=None,
        seq_parent_suffix=None,
        seq_child_suffix=None,
        seq_toplevel_suffix=None,
        expected_ebc=frozenset(),
        forbidden_ebc=frozenset(),
        expected_ec=frozenset(),
        forbidden_ec=frozenset(),
    ) -> None:
        features = _create_test_sequence_features()
        feature_groups = []
        if deep_a_suffix is not None:
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="deep_a",
                    feature_names=["cat_a", "cat_b", "int_a"],
                    group_type=model_pb2.FeatureGroupType.DEEP,
                    embedding_name_suffix=deep_a_suffix,
                )
            )
        if deep_b_suffix is not None:
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="deep_b",
                    feature_names=["cat_a", "cat_b", "int_a"],
                    group_type=model_pb2.FeatureGroupType.DEEP,
                    embedding_name_suffix=deep_b_suffix,
                )
            )
        if wide_suffix is not None:
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="wide",
                    feature_names=["cat_a", "cat_b"],
                    group_type=model_pb2.FeatureGroupType.WIDE,
                    embedding_name_suffix=wide_suffix,
                )
            )
        if seq_parent_suffix is not None:
            seq = model_pb2.SeqGroupConfig(
                group_name="seq",
                feature_names=["cat_a", "click_seq__cat_a"],
            )
            if seq_child_suffix is not None:
                seq.embedding_name_suffix = seq_child_suffix
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="parent",
                    feature_names=["cat_a"],
                    group_type=model_pb2.FeatureGroupType.DEEP,
                    embedding_name_suffix=seq_parent_suffix,
                    sequence_groups=[seq],
                    sequence_encoders=[
                        seq_encoder_pb2.SeqEncoderConfig(
                            simple_attention=seq_encoder_pb2.SimpleAttention(
                                input="seq"
                            )
                        )
                    ],
                )
            )
        if seq_toplevel_suffix is not None:
            feature_groups.append(
                model_pb2.FeatureGroupConfig(
                    group_name="toplevel_seq",
                    feature_names=[
                        "cat_a",
                        "cat_b",
                        "click_seq__cat_a",
                        "click_seq__cat_b",
                    ],
                    group_type=model_pb2.FeatureGroupType.SEQUENCE,
                    embedding_name_suffix=seq_toplevel_suffix,
                )
            )
        embedding_group = EmbeddingGroup(
            features, feature_groups, device=torch.device("cpu")
        )
        ebc_names = {
            cfg.name
            for impl in embedding_group.emb_impls.values()
            for cfg in impl.ebc.embedding_bag_configs()
        }
        ec_names = {
            cfg.name
            for seq_impl in embedding_group.seq_emb_impls.values()
            for ec in seq_impl.ec_dict.values()
            for cfg in ec.embedding_configs()
        }
        self.assertTrue(
            expected_ebc <= ebc_names,
            f"missing from ebc: {expected_ebc - ebc_names}; got {ebc_names}",
        )
        self.assertFalse(
            forbidden_ebc & ebc_names,
            f"forbidden in ebc: {forbidden_ebc & ebc_names}",
        )
        self.assertTrue(
            expected_ec <= ec_names,
            f"missing from ec: {expected_ec - ec_names}; got {ec_names}",
        )
        self.assertFalse(
            forbidden_ec & ec_names,
            f"forbidden in ec: {forbidden_ec & ec_names}",
        )

    def test_embedding_name_suffix_collision_raises(self) -> None:
        # Group A has suffix "x" so cat_a -> "cat_a_emb_x".
        # Group B has no suffix but cat_collide's embedding_name is manually
        # "cat_a_emb_x"; without detection these would silently merge.
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=16, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_collide",
                    embedding_dim=16,
                    num_buckets=100,
                    embedding_name="cat_a_emb_x",
                )
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep_a",
                feature_names=["cat_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
                embedding_name_suffix="x",
            ),
            model_pb2.FeatureGroupConfig(
                group_name="deep_b",
                feature_names=["cat_collide"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        with self.assertRaisesRegex(ValueError, "different embedding_name_suffix"):
            EmbeddingGroup(features, feature_groups, device=torch.device("cpu"))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False],
            [TestGraphType.FX_TRACE, False, False],
            [TestGraphType.JIT_SCRIPT, False, False],
            [TestGraphType.NORMAL, True, False],
            [TestGraphType.FX_TRACE, True, False],
            [TestGraphType.JIT_SCRIPT, True, False],
            [TestGraphType.NORMAL, False, True],
            [TestGraphType.FX_TRACE, False, True],
            [TestGraphType.JIT_SCRIPT, False, True],
        ]
    )
    def test_sequence_embedding_group_impl_input_tile(
        self, graph_type, has_zch=False, has_mulval=False
    ) -> None:
        os.environ["INPUT_TILE"] = "2"
        features = _create_test_sequence_features(
            has_zch=has_zch, has_mulval=has_mulval
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)

        if has_zch and graph_type != TestGraphType.NORMAL:
            embedding_group = embedding_group.eval()
        embedding_group = create_test_module(embedding_group, graph_type)

        if has_mulval:
            values = torch.tensor(list(range(30)))
            lengths = torch.tensor([1, 1, 1, 1, 4, 5, 3, 3, 3, 4, 2, 2])
            sequence_mulval_lengths = KeyedJaggedTensor.from_lengths_sync(
                keys=["click_seq__cat_a", "buy_seq__cat_a"],
                values=torch.tensor([1, 1, 2, 1, 2, 2, 1, 2, 2, 2]),
                lengths=torch.tensor([3, 3, 2, 2]),
            )
        else:
            values = torch.tensor(list(range(24)))
            lengths = torch.tensor([1, 1, 1, 1, 3, 3, 3, 3, 2, 2, 2, 2])
            sequence_mulval_lengths = EMPTY_KJT
        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "cat_a",
                "cat_b",
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=values,
            lengths=lengths,
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([3, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            sequence_mulval_lengths,
            EMPTY_KJT,
            EMPTY_KJT,
            2,
        )

        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False],
            [TestGraphType.FX_TRACE, False, False],
            [TestGraphType.JIT_SCRIPT, False, False],
            [TestGraphType.NORMAL, True, False],
            [TestGraphType.FX_TRACE, True, False],
            [TestGraphType.JIT_SCRIPT, True, False],
            [TestGraphType.NORMAL, False, True],
            [TestGraphType.FX_TRACE, False, True],
            [TestGraphType.JIT_SCRIPT, False, True],
        ]
    )
    def test_sequence_embedding_group_impl_input_tile_emb(
        self, graph_type, has_zch=False, has_mulval=False
    ) -> None:
        os.environ["INPUT_TILE"] = "3"
        features = _create_test_sequence_features(
            has_zch=has_zch, has_mulval=has_mulval
        )
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="click",
                feature_names=[
                    "cat_a",
                    "cat_b",
                    "int_a",
                    "click_seq__cat_a",
                    "click_seq__cat_b",
                    "click_seq__int_a",
                ],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="buy",
                feature_names=["cat_a", "int_a", "buy_seq__cat_a", "buy_seq__int_a"],
                group_type=model_pb2.FeatureGroupType.SEQUENCE,
            ),
        ]
        embedding_group = SequenceEmbeddingGroupImpl(
            features, feature_groups, device=torch.device("cpu")
        )
        self.assertEqual(embedding_group.group_dims("click.sequence"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("click.query"), [16, 8, 1])
        self.assertEqual(embedding_group.group_dims("buy.sequence"), [16, 1])
        self.assertEqual(embedding_group.group_dims("buy.query"), [16, 1])
        self.assertEqual(embedding_group.group_total_dim("click.sequence"), 25)
        self.assertEqual(embedding_group.group_total_dim("click.query"), 25)
        self.assertEqual(embedding_group.group_total_dim("buy.sequence"), 17)
        self.assertEqual(embedding_group.group_total_dim("buy.query"), 17)

        if has_zch and graph_type != TestGraphType.NORMAL:
            embedding_group = embedding_group.eval()
        embedding_group = create_test_module(embedding_group, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor(list(range(4))),
            lengths=torch.tensor([1, 1, 1, 1]),
        )
        if has_mulval:
            values = torch.tensor(list(range(12)))
            lengths = torch.tensor([4, 3, 3, 2])
            sequence_mulval_lengths_user = KeyedJaggedTensor.from_lengths_sync(
                keys=["click_seq__cat_a", "buy_seq__cat_a"],
                values=torch.tensor([1, 1, 2, 1, 2]),
                lengths=torch.tensor([3, 2]),
            )
        else:
            values = torch.tensor(list(range(10)))
            lengths = torch.tensor([3, 3, 2, 2])
            sequence_mulval_lengths_user = EMPTY_KJT
        sparse_feature_user = KeyedJaggedTensor.from_lengths_sync(
            keys=[
                "click_seq__cat_a",
                "click_seq__cat_b",
                "buy_seq__cat_a",
                "buy_seq__cat_b",
            ],
            values=values,
            lengths=lengths,
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        sequence_dense_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["click_seq__int_a", "buy_seq__int_a"],
            values=torch.tensor([[x] for x in range(5)], dtype=torch.float32),
            lengths=torch.tensor([3, 2]),
        ).to_dict()
        result = embedding_group(
            sparse_feature,
            dense_feature,
            sequence_dense_feature,
            EMPTY_KJT,
            sparse_feature_user,
            sequence_mulval_lengths_user,
            2,
        )

        self.assertEqual(result["click.query"].size(), (2, 25))
        self.assertEqual(result["click.sequence"].size(), (2, 3, 25))
        self.assertEqual(result["click.sequence_length"].size(), (2,))
        self.assertEqual(result["buy.query"].size(), (2, 17))
        self.assertEqual(result["buy.sequence"].size(), (2, 2, 17))
        self.assertEqual(result["buy.sequence_length"].size(), (2,))


@ci_resources(slots=1)
@mark_ci_scope("gpu")
class HardSearchEmbeddingGroupTest(unittest.TestCase):
    def _run_in_subprocess(self):
        test_name = (
            f"tzrec.modules.embedding_test.{type(self).__name__}.{self._testMethodName}"
        )
        if os.environ.get("TZREC_HARD_SEARCH_TEST") == test_name:
            return False
        result = subprocess.run(
            [sys.executable, "-m", "unittest", test_name],
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            env=dict(os.environ, TZREC_HARD_SEARCH_TEST=test_name, PYTHONPATH="."),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        return True

    @staticmethod
    def _create_features(pooling=None):
        sparse_kwargs = dict(embedding_name="shared", embedding_dim=2, num_buckets=16)
        if pooling is not None:
            sparse_kwargs.update(value_dim=0, pooling=pooling)
        configs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name=f"{side}_cat",
                    expression=f"{side}:{side}_category",
                    **sparse_kwargs,
                )
            )
            for side in ("user", "item")
        ] + [
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(
                    feature_name=f"{side}_value", expression=f"{side}:{side}_value"
                )
            )
            for side in ("user", "item")
        ]
        for sequence_name in ("history", "hard"):
            sequence_kwargs = {}
            if sequence_name == "hard":
                sequence_kwargs["hard_search_key"] = "item:category"
            configs.append(
                feature_pb2.FeatureConfig(
                    sequence_feature=feature_pb2.SequenceFeature(
                        sequence_name=sequence_name,
                        sequence_length=4,
                        features=[
                            feature_pb2.SeqFeatureConfig(
                                id_feature=feature_pb2.IdFeature(
                                    feature_name="cat",
                                    expression="item:category",
                                    **sparse_kwargs,
                                )
                            ),
                            feature_pb2.SeqFeatureConfig(
                                raw_feature=feature_pb2.RawFeature(
                                    feature_name="value", expression="item:value"
                                )
                            ),
                        ],
                        **sequence_kwargs,
                    )
                )
            )
        return create_features(configs)

    def _create_model(self, pooling=None):
        features = self._create_features(pooling)
        for feature in features:
            if feature.name.startswith("hard__"):
                feature.is_user_feat = True
                self.assertTrue(feature.is_hard_search_feat)
                self.assertFalse(feature.is_user_feat)
            elif feature.is_sequence:
                self.assertFalse(feature.is_hard_search_feat)
                self.assertTrue(feature.is_user_feat)
        group = _feature_group(
            "deep", ["user_cat", "user_value", "item_cat", "item_value"]
        )
        for sequence_name in ("history", "hard"):
            group.sequence_groups.add(
                group_name=sequence_name,
                feature_names=[
                    "user_cat",
                    "item_value",
                    f"{sequence_name}__cat",
                    f"{sequence_name}__value",
                ],
            )
            group.sequence_encoders.add(
                simple_attention=seq_encoder_pb2.SimpleAttention(input=sequence_name)
            )
        model = MultiTower(
            model_config=model_pb2.ModelConfig(
                feature_groups=[group],
                multi_tower=rank_model_pb2.MultiTower(
                    towers=[
                        tower_pb2.Tower(
                            input="deep", mlp=module_pb2.MLP(hidden_units=[4])
                        )
                    ]
                ),
                losses=[
                    loss_pb2.LossConfig(
                        binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                    )
                ],
            ),
            features=features,
            labels=["label"],
        )
        init_parameters(model, device=torch.device("cpu"))
        rows = torch.arange(16, dtype=torch.float32)
        weights = torch.stack((0.25 + rows / 8, 0.5 - rows / 16), dim=1)
        tables = []
        with torch.no_grad():
            for module in model.modules():
                if isinstance(module, (nn.Embedding, nn.EmbeddingBag)):
                    module.weight.copy_(weights)
                    tables.append(module.weight)
            model.towers["deep"].mlp[0].perceptron[0].weight.copy_(
                torch.arange(48, dtype=torch.float32).reshape(4, 12) / 100 - 0.2
            )
            model.towers["deep"].mlp[0].perceptron[0].bias.copy_(
                torch.tensor([-0.1, 0.05, 0.15, -0.2])
            )
            model.output_mlp.weight.copy_(torch.tensor([[0.3, -0.2, 0.5, 0.1]]))
            model.output_mlp.bias.fill_(0.07)
        self.assertEqual(len(tables), 4 if os.environ["INPUT_TILE"] == "3" else 2)
        for table in tables:
            torch.testing.assert_close(table, weights, rtol=0, atol=0)
        return model.eval()

    @staticmethod
    def _create_batch(tile, batch_size=3, pooling=None, all_empty=False):
        has_mulval = pooling is not None
        user_ids = [1, 3, 1] if has_mulval else [1]
        item_ids = (
            [[2, 2], [4], [2, 5], [6, 1]] if has_mulval else [[2], [4], [2], [6]]
        )[:batch_size]
        history_ids = [[1, 1], [3, 5]] if has_mulval else [[1], [3]]
        hard_ids = (
            [[[2, 2], [2, 4]], [], [[4, 4, 1]], [[6, 1], [1, 1], [6]]]
            if has_mulval
            else [[[2], [2]], [], [[4]], [[6], [1], [6]]]
        )[:batch_size]
        hard_values = [[-0.25, 0.75], [], [1.5], [0.125, -0.75, 1.25]][:batch_size]
        if all_empty:
            hard_ids = [[] for _ in range(batch_size)]
            hard_values = [[] for _ in range(batch_size)]
        item_values = [-0.75, 0.5, 1.25, -1.5][:batch_size]
        sparse_rows = {
            "user_cat": [user_ids] * (1 if tile == "3" else batch_size),
            "item_cat": item_ids,
            "history__cat": [sum(history_ids, [])] * (1 if tile == "3" else batch_size),
            "hard__cat": [sum(row, []) for row in hard_ids],
        }
        sparse_features = {}
        sequence_mulval_lengths = {}
        for suffix in ("", "_user") if tile == "3" else ("",):
            rows = {
                key: value
                for key, value in sparse_rows.items()
                if tile != "3"
                or (key in ("user_cat", "history__cat")) == (suffix == "_user")
            }
            sparse_features[BASE_DATA_GROUP + suffix] = (
                KeyedJaggedTensor.from_lengths_sync(
                    keys=list(rows),
                    values=torch.tensor(
                        [
                            value
                            for feature in rows.values()
                            for row in feature
                            for value in row
                        ],
                        dtype=torch.int64,
                    ),
                    lengths=torch.tensor(
                        [len(row) for feature in rows.values() for row in feature],
                        dtype=torch.int64,
                    ),
                )
            )
            if has_mulval:
                sequence_rows = {
                    key: value
                    for key, value in {
                        "history__cat": [history_ids]
                        * (1 if tile == "3" else batch_size),
                        "hard__cat": hard_ids,
                    }.items()
                    if key in rows
                }
                sequence_mulval_lengths[BASE_DATA_GROUP + suffix] = (
                    KeyedJaggedTensor.from_lengths_sync(
                        keys=list(sequence_rows),
                        values=torch.tensor(
                            [
                                len(step)
                                for feature in sequence_rows.values()
                                for row in feature
                                for step in row
                            ],
                            dtype=torch.int64,
                        ),
                        lengths=torch.tensor(
                            [
                                len(row)
                                for feature in sequence_rows.values()
                                for row in feature
                            ],
                            dtype=torch.int64,
                        ),
                    )
                )
        dense_features = {
            BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                keys=["item_value"], tensors=[torch.tensor(item_values).reshape(-1, 1)]
            ),
            BASE_DATA_GROUP + "_user": KeyedTensor.from_tensor_list(
                keys=["user_value"], tensors=[torch.tensor([[0.25]])]
            ),
        }
        if tile == "0":
            dense_features = {
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["user_value", "item_value"],
                    tensors=[
                        torch.full((batch_size, 1), 0.25),
                        torch.tensor(item_values).reshape(-1, 1),
                    ],
                )
            }
        history_count = batch_size if tile == "0" else 1
        batch = Batch(
            sparse_features=sparse_features,
            dense_features=dense_features,
            sequence_dense_features={
                "history__value": JaggedTensor(
                    values=torch.tensor([[0.25], [-0.5]] * history_count),
                    lengths=torch.tensor([2] * history_count),
                ),
                "hard__value": JaggedTensor(
                    values=torch.tensor(
                        [value for row in hard_values for value in row],
                        dtype=torch.float32,
                    ).reshape(-1, 1),
                    lengths=torch.tensor([len(row) for row in hard_values]),
                ),
            },
            sequence_mulval_lengths=sequence_mulval_lengths,
            tile_size=batch_size if tile != "0" else -1,
        )

        def lookup(ids):
            ids = torch.tensor(ids, dtype=torch.float32)
            values = torch.stack((0.25 + ids / 8, 0.5 - ids / 16), dim=1)
            return values.sum(0) if pooling == "sum" else values.mean(0)

        user_embedding = lookup(user_ids).repeat(batch_size, 1)
        query = torch.cat((user_embedding, torch.tensor(item_values).reshape(-1, 1)), 1)
        history = torch.stack(
            [
                torch.cat((lookup(ids), torch.tensor([value])))
                for ids, value in zip(history_ids, [0.25, -0.5])
            ]
        ).repeat(batch_size, 1, 1)
        hard = torch.zeros(batch_size, max(map(len, hard_ids)), 3)
        hard_encoded = torch.zeros(batch_size, 3)
        for i, (row, dense_row) in enumerate(zip(hard_ids, hard_values)):
            if row:
                sequence = torch.stack(
                    [
                        torch.cat((lookup(ids), torch.tensor([value])))
                        for ids, value in zip(row, dense_row)
                    ]
                )
                hard[i, : len(row)] = sequence
                scores = torch.softmax(sequence @ query[i], dim=0)
                hard_encoded[i] = (scores.unsqueeze(1) * sequence).sum(0)
        history_scores = torch.softmax((history * query.unsqueeze(1)).sum(2), dim=1)
        history_encoded = (history_scores.unsqueeze(2) * history).sum(1)
        deep = torch.cat(
            (
                user_embedding,
                torch.full((batch_size, 1), 0.25),
                torch.stack([lookup(ids) for ids in item_ids]),
                torch.tensor(item_values).reshape(-1, 1),
                history_encoded,
                hard_encoded,
            ),
            dim=1,
        )
        expected = {
            "history.query": query,
            "history.sequence": history,
            "history.sequence_length": torch.full((batch_size,), 2, dtype=torch.int64),
            "hard.query": query,
            "hard.sequence": hard,
            "hard.sequence_length": torch.tensor([len(row) for row in hard_ids]),
            "deep": deep,
        }
        tower = torch.relu(
            deep @ (torch.arange(48, dtype=torch.float32).reshape(4, 12) / 100 - 0.2).T
            + torch.tensor([-0.1, 0.05, 0.15, -0.2])
        )
        logits = tower @ torch.tensor([0.3, -0.2, 0.5, 0.1]) + 0.07
        return batch, expected, {"logits": logits, "probs": torch.sigmoid(logits)}

    @parameterized.expand(
        [
            (tile, pooling, online)
            for tile, online in (("2", "0"), ("3", "0"), ("3", "1"))
            for pooling in (None, "mean", "sum")
        ],
        name_func=parameterized_name_func,
    )
    @torch.no_grad()
    def test_hard_search_input_tile_attention(self, tile, pooling, online):
        if self._run_in_subprocess():
            return
        with patch.dict(
            os.environ,
            INPUT_TILE=tile,
            INPUT_TILE_3_ONLINE=online,
            HARD_SEARCH_INPUT_TILE="1",
        ):
            model = ScriptWrapper(self._create_model(pooling)).eval()
            traced = symbolic_trace(model)
            scripted = torch.jit.script(traced)
            buffer = io.BytesIO()
            torch.jit.save(scripted, buffer)
            buffer.seek(0)
            loaded = torch.jit.load(buffer)
        for indices, all_empty in (([2, 0, 2, 1, 0], False), ([1, 1, 1, 1], True)):
            batch, _, _ = self._create_batch("3", 3, pooling, all_empty)
            compact = batch.to_dict()
            index = torch.tensor(indices)
            count = len(indices)
            # Same history with different candidate queries must not share attention.
            compact["item_value.values"] = torch.arange(count).float().view(-1, 1) / 4
            item_ids = [2, 4, 2, 6, 1][:count]
            compact["item_cat.values"] = torch.tensor(item_ids)
            compact["item_cat.lengths"] = torch.ones(count, dtype=torch.int64)
            compact["batch_size"] = torch.tensor(count)
            compact["hard__inverse_indices"] = index
            expanded = dict(compact)
            del expanded["hard__inverse_indices"]
            for name in ("hard__cat", "hard__value"):
                lengths = compact[name + ".lengths"]
                value_lengths = lengths
                if name == "hard__cat" and pooling is not None:
                    key_lengths = compact[name + ".key_lengths"]
                    key_rows = key_lengths.split(lengths.tolist())
                    expanded[name + ".key_lengths"] = torch.cat(
                        [key_rows[i] for i in indices]
                    )
                    value_lengths = torch.tensor([row.sum() for row in key_rows])
                rows = compact[name + ".values"].split(value_lengths.tolist())
                expanded[name + ".values"] = torch.cat([rows[i] for i in indices])
                expanded[name + ".lengths"] = lengths[index]
            with patch.dict(
                os.environ,
                INPUT_TILE="3",
                INPUT_TILE_3_ONLINE="0",
                HARD_SEARCH_INPUT_TILE="0",
            ):
                reference = ScriptWrapper(self._create_model(pooling)).eval()(expanded)
            with patch.dict(
                os.environ,
                INPUT_TILE=tile,
                INPUT_TILE_3_ONLINE=online,
                HARD_SEARCH_INPUT_TILE="1",
            ):
                for runner in (model, traced, scripted, loaded):
                    torch.testing.assert_close(runner(compact), reference)
                parsed = model._data_parser.to_batch(compact)
                self.assertEqual(
                    parsed.sparse_features[BASE_DATA_GROUP]["hard__cat"]
                    .values()
                    .numel(),
                    compact["hard__cat.values"].numel(),
                )
                torch.testing.assert_close(
                    parsed.to_dict()["hard__cat.lengths"], compact["hard__cat.lengths"]
                )

    @parameterized.expand(
        [
            param(pooling=None, all_empty=False),
            param(pooling=None, all_empty=True),
            param(pooling="mean", all_empty=False),
            param(pooling="mean", all_empty=True),
            param(pooling="sum", all_empty=False),
            param(pooling="sum", all_empty=True),
        ],
        name_func=parameterized_name_func,
    )
    @torch.no_grad()
    def test_forward_hard_search_input_tile_numerical(self, pooling, all_empty):
        if self._run_in_subprocess():
            return
        reference = {}
        for tile, online in (("0", "0"), ("2", "0"), ("3", "0"), ("3", "1")):
            with patch.dict(os.environ, INPUT_TILE=tile, INPUT_TILE_3_ONLINE=online):
                model = self._create_model(pooling)
                for batch_size in (3, 4):
                    with self.subTest(tile=tile, online=online, batch_size=batch_size):
                        batch, expected, predictions = self._create_batch(
                            tile, batch_size, pooling, all_empty
                        )
                        grouped = model.embedding_group(batch)
                        self.assertEqual(set(grouped), set(expected))
                        for key, value in expected.items():
                            torch.testing.assert_close(
                                grouped[key], value, rtol=1e-6, atol=1e-6, msg=key
                            )
                        batch, _, _ = self._create_batch(
                            tile, batch_size, pooling, all_empty
                        )
                        actual = model(batch)
                        self.assertEqual(set(actual), set(predictions))
                        if tile == "0":
                            reference[batch_size] = actual
                        for key, value in predictions.items():
                            torch.testing.assert_close(
                                actual[key], value, rtol=1e-6, atol=1e-6, msg=key
                            )
                            if batch_size in reference:
                                torch.testing.assert_close(
                                    actual[key],
                                    reference[batch_size][key],
                                    rtol=1e-6,
                                    atol=1e-6,
                                )

    @parameterized.expand(
        [
            [graph_type, tile, online, pooling]
            for graph_type in (TestGraphType.FX_TRACE, TestGraphType.JIT_SCRIPT)
            for tile, online in (("0", "0"), ("2", "0"), ("3", "0"), ("3", "1"))
            for pooling in (None, "mean")
        ],
        name_func=parameterized_name_func,
    )
    @torch.no_grad()
    def test_forward_hard_search_graph_dynamic_batch(
        self, graph_type, tile, online, pooling
    ):
        if self._run_in_subprocess():
            return
        with patch.dict(os.environ, INPUT_TILE="0", INPUT_TILE_3_ONLINE="0"):
            eager = self._create_model(pooling)
            reference = {}
            for batch_size, all_empty in ((3, False), (4, False), (4, True)):
                batch, _, expected = self._create_batch(
                    "0", batch_size, pooling, all_empty
                )
                reference[batch_size, all_empty] = eager(batch)
                for key in expected:
                    torch.testing.assert_close(
                        reference[batch_size, all_empty][key],
                        expected[key],
                        rtol=1e-6,
                        atol=1e-6,
                    )
        with patch.dict(os.environ, INPUT_TILE=tile, INPUT_TILE_3_ONLINE=online):
            model = symbolic_trace(ScriptWrapper(self._create_model(pooling)))
            if graph_type == TestGraphType.JIT_SCRIPT:
                model = torch.jit.script(model)
            input_tile = "3" if tile == "2" else tile
            batch, _, _ = self._create_batch(input_tile, 3, pooling)
            result = model(batch.to_dict())
            for key in reference[3, False]:
                torch.testing.assert_close(
                    result[key], reference[3, False][key], rtol=1e-6, atol=1e-6
                )
            with io.BytesIO() as buffer:
                if graph_type == TestGraphType.JIT_SCRIPT:
                    torch.jit.save(model, buffer)
                    buffer.seek(0)
                    restored = torch.jit.load(buffer)
                else:
                    restored = model
                for all_empty in (False, True):
                    with self.subTest(batch_size=4, all_empty=all_empty):
                        batch, _, _ = self._create_batch(
                            input_tile, 4, pooling, all_empty
                        )
                        result = restored(batch.to_dict())
                        self.assertEqual(set(result), set(reference[4, all_empty]))
                        for key in reference[4, all_empty]:
                            torch.testing.assert_close(
                                result[key],
                                reference[4, all_empty][key],
                                rtol=1e-6,
                                atol=1e-6,
                                msg=key,
                            )

    @parameterized.expand(
        [
            [names, nested]
            for names in (
                ["history__cat", "hard__cat"],
                ["hard__cat", "history__cat"],
                ["history__value", "hard__value"],
                ["hard__cat", "history__value"],
            )
            for nested in (False, True)
        ],
        name_func=parameterized_name_func,
    )
    def test_build_embedding_collection_configs_mixed_hard_search_history(
        self, names, nested
    ):
        if self._run_in_subprocess():
            return
        features = self._create_features()
        group = (
            model_pb2.SeqGroupConfig(group_name="mixed", feature_names=names)
            if nested
            else _feature_group("mixed", names, model_pb2.SEQUENCE)
        )
        for tile in ("0", "2", "3"):
            with patch.dict(os.environ, INPUT_TILE=tile, INPUT_TILE_3_ONLINE="0"):
                with (
                    self.subTest(tile=tile),
                    self.assertRaisesRegex(ValueError, "[Hh]ard.search"),
                ):
                    build_embedding_collection_configs(features, [group])

    def test_build_embedding_collection_configs_different_hard_search_histories(self):
        features = self._create_features()
        extra = feature_pb2.FeatureConfig(
            sequence_feature=feature_pb2.SequenceFeature(
                sequence_name="other_hard",
                sequence_length=4,
                sequence_delim=";",
                hard_search_key="item:category",
                features=[
                    feature_pb2.SeqFeatureConfig(
                        id_feature=feature_pb2.IdFeature(
                            feature_name="cat",
                            expression="item:category",
                            embedding_dim=2,
                            num_buckets=16,
                        )
                    )
                ],
            )
        )
        features.extend(create_features([extra]))
        group = _feature_group(
            "mixed", ["hard__cat", "other_hard__cat"], model_pb2.SEQUENCE
        )
        with self.assertRaisesRegex(
            ValueError, "different hard-search sequence groups"
        ):
            build_embedding_collection_configs(features, [group])

    def test_repeated_hard_search_history_keeps_candidate_rows(self):
        with patch.dict(os.environ, INPUT_TILE="3", INPUT_TILE_3_ONLINE="1"):
            model = self._create_model()
            batch, _, _ = self._create_batch("3")
            sparse = batch.sparse_features[BASE_DATA_GROUP]
            batch.sparse_features[BASE_DATA_GROUP] = (
                KeyedJaggedTensor.from_lengths_sync(
                    keys=sparse.keys(),
                    values=torch.tensor([2, 4, 2, 2, 2, 4, 2, 2]),
                    lengths=torch.tensor([1, 1, 1, 2, 1, 2]),
                )
            )
            batch.sequence_dense_features["hard__value"] = JaggedTensor(
                values=torch.tensor([[-0.25], [0.75], [1.5], [-0.25], [0.75]]),
                lengths=torch.tensor([2, 1, 2]),
            )
            grouped = model.embedding_group(batch)
            torch.testing.assert_close(
                grouped["hard.sequence_length"], torch.tensor([2, 1, 2])
            )
            torch.testing.assert_close(
                grouped["hard.sequence"][0], grouped["hard.sequence"][2]
            )
            self.assertFalse(
                torch.equal(grouped["hard.sequence"][0], grouped["hard.sequence"][1])
            )

    @parameterized.expand(
        [["hard__cat"], ["hard__value"]], name_func=parameterized_name_func
    )
    def test_build_embedding_collection_configs_hard_search_jagged_sequence(self, name):
        if self._run_in_subprocess():
            return
        features = self._create_features()
        group = _feature_group(
            "jagged", ["user_cat", "item_cat", name], model_pb2.JAGGED_SEQUENCE
        )
        for tile in ("0", "2", "3"):
            with patch.dict(os.environ, INPUT_TILE=tile, INPUT_TILE_3_ONLINE="0"):
                with (
                    self.subTest(tile=tile),
                    self.assertRaisesRegex(ValueError, "[Hh]ard.search"),
                ):
                    build_embedding_collection_configs(features, [group])

    @parameterized.expand(
        [[model_pb2.SEQUENCE], [model_pb2.JAGGED_SEQUENCE]],
        name_func=parameterized_name_func,
    )
    def test_build_embedding_collection_configs_ordinary_history_mixed_queries(
        self, group_type
    ):
        if self._run_in_subprocess():
            return
        features = self._create_features()
        group = _feature_group(
            "ordinary",
            ["user_cat", "item_cat", "history__cat", "history__value"],
            group_type,
        )
        for tile in ("0", "2", "3"):
            with patch.dict(os.environ, INPUT_TILE=tile, INPUT_TILE_3_ONLINE="0"):
                with self.subTest(tile=tile):
                    configs = build_embedding_collection_configs(features, [group])
                    regular = configs.ec[2]["shared"].feature_names
                    if tile == "3":
                        self.assertEqual(regular, ["item_cat"])
                        self.assertEqual(
                            configs.ec_user[2]["shared"].feature_names,
                            ["user_cat", "history__cat"],
                        )
                    else:
                        self.assertEqual(
                            regular, ["user_cat", "item_cat", "history__cat"]
                        )
                        self.assertEqual(dict(configs.ec_user), {})

    def test_hard_search_input_tile_expander(self):
        tables = [
            EmbeddingBagConfig(
                name="t1", embedding_dim=2, num_embeddings=8, feature_names=["a", "b"]
            ),
            EmbeddingBagConfig(
                name="t2", embedding_dim=3, num_embeddings=8, feature_names=["c"]
            ),
        ]
        kt = KeyedTensor(
            keys=["a", "b", "c"],
            length_per_key=[2, 2, 3],
            values=torch.arange(14, dtype=torch.float32).reshape(2, 7),
        )
        passthrough = HardSearchInputTileExpander(tables, {})
        self.assertEqual(passthrough.blocks, [])
        self.assertIs(passthrough(kt), kt)

        single = HardSearchInputTileExpander(
            tables, {"a": "seq_a", "b": "seq_a", "c": "seq_a"}
        )
        self.assertEqual(single.blocks, [(0, 7, "seq_a__inverse_indices")])
        with self.assertRaisesRegex(ValueError, "inverse indices"):
            single(kt)
        expanded = single(kt, {"seq_a__inverse_indices": torch.tensor([1, 0, 1])})
        torch.testing.assert_close(expanded.values(), kt.values()[[1, 0, 1]])

        mixed = HardSearchInputTileExpander(tables, {"a": "seq_a", "b": "seq_a"})
        self.assertEqual(mixed.blocks, [(0, 4, "seq_a__inverse_indices"), (4, 3, "")])
        mixed_out = mixed(
            KeyedTensor(
                keys=["a", "b", "c"],
                length_per_key=[2, 2, 3],
                values=torch.tensor(
                    [
                        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0],
                        [5.0, 6.0, 7.0, 8.0, 40.0, 50.0, 60.0],
                    ]
                ),
            ),
            {"seq_a__inverse_indices": torch.tensor([1, 0])},
        )
        torch.testing.assert_close(
            mixed_out.values(),
            torch.tensor(
                [
                    [5.0, 6.0, 7.0, 8.0, 10.0, 20.0, 30.0],
                    [1.0, 2.0, 3.0, 4.0, 40.0, 50.0, 60.0],
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
