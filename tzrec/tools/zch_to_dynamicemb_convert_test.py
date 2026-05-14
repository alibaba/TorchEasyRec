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

"""Tests for tzrec.tools.zch_to_dynamicemb_convert."""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from parameterized import parameterized
from torch import nn
from torch.distributed import checkpoint as dcp
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)

from tzrec.features.feature import create_features
from tzrec.models.model import TrainWrapper
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import feature_pb2, model_pb2, pipeline_pb2
from tzrec.tools import zch_to_dynamicemb_convert as conv
from tzrec.utils import config_util
from tzrec.utils.dynamicemb_util import has_dynamicemb

_IINFO_MAX = torch.iinfo(torch.int64).max


def _read_int64(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int64)


def _read_float32(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32)


# -----------------------------------------------------------------------------
# Real-component test helpers
# -----------------------------------------------------------------------------


class _TestBaseModel(nn.Module):
    """Shim that satisfies the real ``TrainWrapper.__init__`` contract.

    ``TrainWrapper`` (tzrec/models/model.py:235-236) calls ``init_loss`` and
    ``init_metric`` on its inner module at construction time. We provide
    no-op stubs so tests can wrap a model that only carries an
    ``EmbeddingGroup`` (no real loss / metric wiring).
    """

    def __init__(self, embedding_group: nn.Module) -> None:
        super().__init__()
        self.embedding_group = embedding_group

    def init_loss(self) -> None:
        pass

    def init_metric(self) -> None:
        pass


def _id_feature_with_zch(name, embedding_dim, zch_size, eviction):
    """Build a FeatureConfig with id_feature.zch{} configured.

    ``embedding_name`` is set explicitly to ``name`` so the underlying
    embedding-table name equals the feature name (tzrec otherwise appends
    ``_emb``, see ``tzrec/features/feature.py:613``).
    """
    if eviction == "lfu":
        zch = feature_pb2.ZeroCollisionHash(
            zch_size=zch_size, lfu=feature_pb2.LFU_EvictionPolicy()
        )
    elif eviction == "lru":
        zch = feature_pb2.ZeroCollisionHash(
            zch_size=zch_size, lru=feature_pb2.LRU_EvictionPolicy()
        )
    else:
        raise ValueError(f"unknown eviction policy: {eviction}")
    return feature_pb2.FeatureConfig(
        id_feature=feature_pb2.IdFeature(
            feature_name=name,
            embedding_name=name,
            embedding_dim=embedding_dim,
            zch=zch,
        )
    )


def _id_feature_with_dynamicemb(name, embedding_dim, max_capacity, score_strategy):
    """Build a FeatureConfig with id_feature.dynamicemb{} configured.

    Same ``embedding_name=name`` convention as :func:`_id_feature_with_zch`.
    """
    return feature_pb2.FeatureConfig(
        id_feature=feature_pb2.IdFeature(
            feature_name=name,
            embedding_name=name,
            embedding_dim=embedding_dim,
            dynamicemb=feature_pb2.DynamicEmbedding(
                max_capacity=max_capacity, score_strategy=score_strategy
            ),
        )
    )


def _make_emb_group(feature_cfgs, feature_groups):
    """Build a real EmbeddingGroup on CPU from FeatureConfig protos."""
    features = create_features(feature_cfgs)
    return EmbeddingGroup(features, feature_groups, device=torch.device("cpu"))


def _find_mch_modules_for_table(model, table_name, kind=None):
    """Yield (containing_module_path, mch_module) for owners of ``table_name``.

    ``kind`` filters by wrapper type: ``"pooled"`` (DEEP / WIDE →
    ``ManagedCollisionEmbeddingBagCollection``), ``"sequence"`` (SEQUENCE /
    JAGGED_SEQUENCE → ``ManagedCollisionEmbeddingCollection``), or ``None``
    (no filter).
    """
    if kind == "pooled":
        types = (ManagedCollisionEmbeddingBagCollection,)
    elif kind == "sequence":
        types = (ManagedCollisionEmbeddingCollection,)
    else:
        types = (
            ManagedCollisionEmbeddingBagCollection,
            ManagedCollisionEmbeddingCollection,
        )
    for name, m in model.named_modules():
        if isinstance(m, types):
            mc = m._managed_collision_collection._managed_collision_modules
            if table_name in mc:
                yield name, mc[table_name]


def _find_inner_weights_for_table(model, table_name, kind=None):
    """Yield (containing_module_path, inner_weight_param) for ``table_name``.

    Same ``kind`` filter as :func:`_find_mch_modules_for_table`.
    """
    for name, m in model.named_modules():
        if isinstance(m, ManagedCollisionEmbeddingBagCollection):
            if kind == "sequence":
                continue
            ebc = m._embedding_module
            if table_name in ebc.embedding_bags:
                yield name, ebc.embedding_bags[table_name].weight
        elif isinstance(m, ManagedCollisionEmbeddingCollection):
            if kind == "pooled":
                continue
            ec = m._embedding_module
            if table_name in ec.embeddings:
                yield name, ec.embeddings[table_name].weight


# -----------------------------------------------------------------------------
# Pure helper tests
# -----------------------------------------------------------------------------


class HelperFunctionTests(unittest.TestCase):
    """Parameterized tests for path helpers and eviction-policy classification."""

    @parameterized.expand(
        [
            # Pooled wrappers (one segment).
            ("mc_ebc", "model.eg.gi.mc_ebc", "model.eg.gi"),
            ("ebc", "model.eg.gi.ebc", "model.eg.gi"),
            ("mc_ebc_user", "model.eg.gi.mc_ebc_user", "model.eg.gi"),
            ("ebc_user", "model.eg.gi.ebc_user", "model.eg.gi"),
            # Sequence wrappers (two segments: name.<dim>).
            ("mc_ec_dict", "model.eg.gi.mc_ec_dict.16", "model.eg.gi"),
            ("ec_dict", "model.eg.gi.ec_dict.32", "model.eg.gi"),
            ("mc_ec_dict_user", "model.eg.gi.mc_ec_dict_user.8", "model.eg.gi"),
            ("ec_dict_user", "model.eg.gi.ec_dict_user.4", "model.eg.gi"),
        ]
    )
    def test_strip_collection_suffix(self, _name, input_path, expected):
        self.assertEqual(conv._strip_collection_suffix(input_path), expected)

    def test_strip_collection_suffix_raises_on_unknown(self):
        with self.assertRaises(ValueError):
            conv._strip_collection_suffix("model.eg.gi.not_a_wrapper")

    @parameterized.expand(
        [
            ("one_prefix", "model.embedding_group.gi", "embedding_group.gi"),
            ("two_prefixes", "model.model.embedding_group.gi", "embedding_group.gi"),
            ("no_prefix", "embedding_group.gi", "embedding_group.gi"),
        ]
    )
    def test_normalize_group_key(self, _name, input_path, expected):
        self.assertEqual(conv._normalize_group_key(input_path), expected)

    @parameterized.expand(
        [
            ("lfu", ["_mch_counts"], "lfu"),
            ("lru", ["_mch_last_access_iter"], "lru"),
            (
                "distance_lfu",
                ["_mch_counts", "_mch_last_access_iter"],
                "distance_lfu",
            ),
            ("none", [], "none"),
        ]
    )
    def test_classify_eviction_policy(self, _name, buffers, expected):
        self.assertEqual(conv._classify_eviction_policy(buffers), expected)


class DeriveScoresTests(unittest.TestCase):
    """Parameterized tests for score derivation across all branches."""

    @parameterized.expand(
        [
            # name, metadata (lists of ints), source_policy, target_strategy,
            # valid_mask (list of bools), expected (list of ints or None).
            (
                "lfu_to_lfu_uses_counts",
                {"_mch_counts": [10, 20, 30, 40]},
                "lfu",
                "LFU",
                [True, True, False, True],
                [10, 20, 40],
            ),
            (
                "lru_to_step_uses_last_access",
                {"_mch_last_access_iter": [100, 200, 300, 400]},
                "lru",
                "STEP",
                [True, False, True, True],
                [100, 300, 400],
            ),
            (
                "lfu_to_step_falls_back_to_counts",
                {"_mch_counts": [5, 6, 7, 8]},
                "lfu",
                "STEP",
                [True, True, True, True],
                [5, 6, 7, 8],
            ),
            (
                "lru_to_lfu_falls_back_to_last_access",
                {"_mch_last_access_iter": [1, 2, 3, 4]},
                "lru",
                "LFU",
                [True, True, True, True],
                [1, 2, 3, 4],
            ),
            (
                "distance_lfu_to_lfu_prefers_counts",
                {"_mch_counts": [7, 7, 7, 7], "_mch_last_access_iter": [1, 2, 3, 4]},
                "distance_lfu",
                "LFU",
                [True, True, True, True],
                [7, 7, 7, 7],
            ),
            (
                "distance_lfu_to_step_prefers_last_access",
                {
                    "_mch_counts": [7, 7, 7, 7],
                    "_mch_last_access_iter": [10, 20, 30, 40],
                },
                "distance_lfu",
                "STEP",
                [True, True, True, True],
                [10, 20, 30, 40],
            ),
            (
                "timestamp_target_returns_none",
                {"_mch_counts": [1, 2, 3]},
                "lfu",
                "TIMESTAMP",
                [True, True, True],
                None,
            ),
            (
                "customized_target_returns_none",
                {"_mch_counts": [1, 2]},
                "lfu",
                "CUSTOMIZED",
                [True, True],
                None,
            ),
            (
                "no_eviction_target_returns_none",
                {"_mch_counts": [1, 2, 3]},
                "lfu",
                "NO_EVICTION",
                [True, True, True],
                None,
            ),
            (
                "no_source_metadata_returns_none",
                {},
                "none",
                "STEP",
                [True, True],
                None,
            ),
        ]
    )
    def test_derive_scores(
        self,
        _name,
        metadata,
        source_policy,
        target_strategy,
        valid_mask_list,
        expected_list,
    ):
        metadata_tensors = {
            k: torch.tensor(v, dtype=torch.int64) for k, v in metadata.items()
        }
        valid_mask = torch.tensor(valid_mask_list)
        scores = conv._derive_scores(
            metadata_tensors, source_policy, target_strategy, valid_mask
        )
        if expected_list is None:
            self.assertIsNone(scores)
        else:
            torch.testing.assert_close(
                scores, torch.tensor(expected_list, dtype=torch.int64)
            )


# -----------------------------------------------------------------------------
# Shard-write tests (need dynamicemb for encode_checkpoint_file_path)
# -----------------------------------------------------------------------------


@unittest.skipUnless(has_dynamicemb, "dynamicemb not available")
class ShardWriteTests(unittest.TestCase):
    """Parameterized tests for ``_gather_and_shard_writes`` output files."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_shard_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    @parameterized.expand(
        [
            # name, raw_ids, values_factory(n, dim), scores (or None),
            # world_size, expected_per_rank: dict
            #   {rank: {item -> "present"|"absent"|"empty"}}
            # for items in {keys, values, scores}.
            (
                "world_size_1_with_scores",
                [7, 13, 42, 100],
                lambda n: torch.arange(n * 2, dtype=torch.float32).reshape(n, 2),
                [1, 2, 3, 4],
                1,
                {0: {"keys": "present", "values": "present", "scores": "present"}},
            ),
            (
                "world_size_2_scores_none_omits_score_files",
                [0, 1, 2, 3],
                lambda n: torch.zeros((n, 2), dtype=torch.float32),
                None,  # scores=None
                2,
                {
                    0: {"keys": "present", "values": "present", "scores": "absent"},
                    1: {"keys": "present", "values": "present", "scores": "absent"},
                },
            ),
            (
                "world_size_3_splits_by_modulo",
                [0, 1, 2, 3, 4, 5, 6, 7, 8],
                lambda n: torch.zeros((n, 2), dtype=torch.float32),
                [0] * 9,
                3,
                {
                    0: {"keys": "present", "values": "present", "scores": "present"},
                    1: {"keys": "present", "values": "present", "scores": "present"},
                    2: {"keys": "present", "values": "present", "scores": "present"},
                },
            ),
            (
                "world_size_4_empty_shards_are_zero_byte",
                [0, 4, 8],
                lambda n: torch.zeros((n, 1), dtype=torch.float32),
                [0, 0, 0],
                4,
                {
                    0: {"keys": "present", "values": "present", "scores": "present"},
                    1: {"keys": "empty", "values": "empty", "scores": "empty"},
                    2: {"keys": "empty", "values": "empty", "scores": "empty"},
                    3: {"keys": "empty", "values": "empty", "scores": "empty"},
                },
            ),
        ]
    )
    def test_shard_writes(
        self,
        name,
        raw_ids_list,
        values_factory,
        scores_list,
        world_size,
        expected_per_rank,
    ):
        raw_ids = torch.tensor(raw_ids_list, dtype=torch.int64)
        values = values_factory(len(raw_ids_list))
        scores = (
            None
            if scores_list is None
            else torch.tensor(scores_list, dtype=torch.int64)
        )
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            table_name="t",
            save_path=self.tmp,
            world_size=world_size,
        )
        for rank, items in expected_per_rank.items():
            for item, state in items.items():
                p = os.path.join(
                    self.tmp, f"t_emb_{item}.rank_{rank}.world_size_{world_size}"
                )
                if state == "absent":
                    self.assertFalse(
                        os.path.exists(p),
                        f"[{name}] rank {rank} item {item}: should be absent ({p})",
                    )
                elif state == "empty":
                    self.assertTrue(os.path.exists(p), f"missing {p}")
                    self.assertEqual(os.path.getsize(p), 0, f"{p} not empty")
                else:  # "present"
                    self.assertTrue(os.path.exists(p), f"missing {p}")
                    self.assertGreater(os.path.getsize(p), 0, f"{p} is empty")

        # Additional content check: when world_size > 1 and any rank has keys,
        # verify the modulo invariant.
        if world_size > 1:
            for rank in range(world_size):
                p = os.path.join(
                    self.tmp, f"t_emb_keys.rank_{rank}.world_size_{world_size}"
                )
                if os.path.getsize(p) > 0:
                    arr = _read_int64(p)
                    self.assertTrue(
                        np.all(arr % world_size == rank),
                        f"[{name}] rank {rank} got non-matching keys {arr.tolist()}",
                    )


# -----------------------------------------------------------------------------
# Model-walking tests (real EmbeddingGroup)
# -----------------------------------------------------------------------------


class FindZchTablesTests(unittest.TestCase):
    """Tests for the source-side MCH-discovery walk using a real EmbeddingGroup."""

    def test_finds_mch_tables_with_correct_state_dict_paths(self):
        # Real EmbeddingGroup with two ZCH features inside one DEEP feature_group.
        feature_cfgs = [
            _id_feature_with_zch("cat_a", embedding_dim=4, zch_size=8, eviction="lfu"),
            _id_feature_with_zch("cat_b", embedding_dim=8, zch_size=16, eviction="lru"),
        ]
        groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.DEEP,
            )
        ]
        emb_group = _make_emb_group(feature_cfgs, groups)
        model = TrainWrapper(_TestBaseModel(emb_group))

        tables = conv._find_zch_tables(model)

        # Both tables should live under the same EmbeddingGroupImpl, so same group_key.
        self.assertEqual(len(tables), 2)
        group_keys = {gk for (gk, _) in tables}
        self.assertEqual(len(group_keys), 1, f"got {group_keys}")
        gk = group_keys.pop()
        self.assertIn("emb_impls", gk)

        u = tables[(gk, "cat_a")]
        self.assertEqual(u.embedding_dim, 4)
        self.assertEqual(u.zch_size, 8)
        self.assertEqual(u.eviction_policy, "lfu")
        self.assertEqual(u.metadata_buffer_names, ["_mch_counts"])
        self.assertTrue(u.weight_fqn.startswith("model.embedding_group."))
        self.assertIn(
            ".mc_ebc._embedding_module.embedding_bags.cat_a.weight", u.weight_fqn
        )
        self.assertIn(
            ".mc_ebc._managed_collision_collection._managed_collision_modules.cat_a",
            u.mch_prefix,
        )

        i = tables[(gk, "cat_b")]
        self.assertEqual(i.eviction_policy, "lru")
        self.assertEqual(i.metadata_buffer_names, ["_mch_last_access_iter"])
        self.assertEqual(i.embedding_dim, 8)
        self.assertEqual(i.zch_size, 16)


class DcpRoundTripTests(unittest.TestCase):
    """Save real model state via DCP single-process; load via the helper."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_dcp_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_dcp_subset_round_trip(self):
        feature_cfgs = [
            _id_feature_with_zch(
                "user_id", embedding_dim=4, zch_size=8, eviction="lfu"
            ),
        ]
        groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["user_id"],
                group_type=model_pb2.DEEP,
            )
        ]
        emb_group = _make_emb_group(feature_cfgs, groups)
        model = TrainWrapper(_TestBaseModel(emb_group))

        # Locate and populate the MCH module + inner weight.
        ((mch_path, mch),) = list(_find_mch_modules_for_table(model, "user_id"))
        sorted_raw = mch._buffers["_mch_sorted_raw_ids"].clone()
        sorted_raw[:6] = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int64)
        mch._buffers["_mch_sorted_raw_ids"].copy_(sorted_raw)
        mch._buffers["_mch_remapped_ids_mapping"].copy_(
            torch.tensor([5, 4, 3, 2, 1, 0, 6, 7], dtype=torch.int64)
        )
        counts = torch.tensor([1, 2, 3, 4, 5, 6, 0, 0], dtype=torch.int64)
        mch._buffers["_mch_counts"].copy_(counts)

        ((_, weight),) = list(_find_inner_weights_for_table(model, "user_id"))
        with torch.no_grad():
            for r in range(8):
                weight[r].fill_(float(r))

        dcp_dir = os.path.join(self.tmp, "model")
        os.makedirs(dcp_dir, exist_ok=True)
        dcp.save(model.state_dict(), checkpoint_id=dcp_dir)

        # Use the converter's discovery to locate the canonical FQNs, then
        # round-trip via the loader helper.
        tables = conv._find_zch_tables(model)
        ((gk, emb_name),) = list(tables)
        info = tables[(gk, emb_name)]
        loaded = conv._load_dcp_subset(
            dcp_dir,
            [
                f"{info.mch_prefix}._mch_sorted_raw_ids",
                f"{info.mch_prefix}._mch_remapped_ids_mapping",
                f"{info.mch_prefix}._mch_counts",
                info.weight_fqn,
            ],
        )
        torch.testing.assert_close(
            loaded[f"{info.mch_prefix}._mch_sorted_raw_ids"], sorted_raw
        )
        torch.testing.assert_close(
            loaded[f"{info.mch_prefix}._mch_remapped_ids_mapping"],
            torch.tensor([5, 4, 3, 2, 1, 0, 6, 7], dtype=torch.int64),
        )
        torch.testing.assert_close(loaded[f"{info.mch_prefix}._mch_counts"], counts)
        torch.testing.assert_close(loaded[info.weight_fqn][3], torch.full((4,), 3.0))


# -----------------------------------------------------------------------------
# convert() end-to-end (real EmbeddingGroup + real DynamicEmbParameterConstraints)
# -----------------------------------------------------------------------------


def _make_minimal_pipeline_config():
    """Construct an in-memory EasyRecConfig with the fields convert() reads.

    convert() only touches ``feature_configs``, ``data_config.label_fields``,
    ``data_config.sample_weight_fields``, and ``model_config`` — and we mock
    ``_create_features`` / ``_create_model`` to bypass the real construction
    path. So a default-constructed proto with the three required string
    fields is sufficient.
    """
    return pipeline_pb2.EasyRecConfig(
        train_input_path="", eval_input_path="", model_dir=""
    )


@unittest.skipUnless(has_dynamicemb, "dynamicemb not available")
class ConvertE2ETests(unittest.TestCase):
    """End-to-end ``convert()`` covering the v3 most-complex scenario.

    Exercises DEEP + SEQUENCE groups, shared and unique embedding names, and
    a matrix of source-policy × target-strategy migrations.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_e2e_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # The migration matrix this test covers (4 features, 5 tables, all dim=4,
    # zch_size=8 / max_capacity=64):
    #
    #   feature_name    | zch  | target  | groups        | covers
    #   ----------------+------+---------+---------------+----------------------------
    #   shared_lfu      | lfu  | STEP    | DEEP+SEQUENCE | LFU→STEP proxy; group_key
    #                   |      |         |               |   disambiguation across
    #                   |      |         |               |   pooled + sequence impls
    #   deep_only_lfu   | lfu  | LFU     | DEEP          | LFU→LFU direct counts
    #   deep_only_lru   | lru  | TIMESTAMP | DEEP        | LRU→TIMESTAMP omit scores
    #   seq_only_lru    | lru  | STEP    | SEQUENCE      | LRU→STEP direct last_access

    _DIM = 4
    _ZCH_SIZE = 8
    _MAX_CAP = 64

    def _build_source_model(self):
        feature_cfgs = [
            _id_feature_with_zch("shared_lfu", self._DIM, self._ZCH_SIZE, "lfu"),
            _id_feature_with_zch("deep_only_lfu", self._DIM, self._ZCH_SIZE, "lfu"),
            _id_feature_with_zch("deep_only_lru", self._DIM, self._ZCH_SIZE, "lru"),
            _id_feature_with_zch("seq_only_lru", self._DIM, self._ZCH_SIZE, "lru"),
        ]
        groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["shared_lfu", "deep_only_lfu", "deep_only_lru"],
                group_type=model_pb2.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="seq",
                feature_names=["shared_lfu", "seq_only_lru"],
                group_type=model_pb2.SEQUENCE,
            ),
        ]
        emb_group = _make_emb_group(feature_cfgs, groups)
        return _TestBaseModel(emb_group)

    def _build_target_model(self):
        feature_cfgs = [
            _id_feature_with_dynamicemb("shared_lfu", self._DIM, self._MAX_CAP, "STEP"),
            _id_feature_with_dynamicemb(
                "deep_only_lfu", self._DIM, self._MAX_CAP, "LFU"
            ),
            _id_feature_with_dynamicemb(
                "deep_only_lru", self._DIM, self._MAX_CAP, "TIMESTAMP"
            ),
            _id_feature_with_dynamicemb(
                "seq_only_lru", self._DIM, self._MAX_CAP, "STEP"
            ),
        ]
        groups = [
            model_pb2.FeatureGroupConfig(
                group_name="deep",
                feature_names=["shared_lfu", "deep_only_lfu", "deep_only_lru"],
                group_type=model_pb2.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="seq",
                feature_names=["shared_lfu", "seq_only_lru"],
                group_type=model_pb2.SEQUENCE,
            ),
        ]
        emb_group = _make_emb_group(feature_cfgs, groups)
        return _TestBaseModel(emb_group)

    def _populate_source(self, src_inner):
        """Populate ZCH state on every MCH-wrapped table.

        For ``shared_lfu`` (two instances), the DEEP and SEQUENCE instances
        get *different* value patterns and remapped permutations so any
        cross-group merge bug would surface as a value mismatch.
        """
        wrapped = TrainWrapper(src_inner)

        # raw_ids = [100, 101, 102, 103] for every table (4 valid out of 8 slots).
        # ``kind`` discriminates between the pooled (DEEP) and sequence
        # (SEQUENCE) instances when the same feature appears in both groups.
        plans = [
            # (table_name, kind, eviction, value_base, remapped)
            ("shared_lfu", "pooled", "lfu", 10.0, [3, 2, 1, 0]),
            ("shared_lfu", "sequence", "lfu", 100.0, [0, 1, 2, 3]),
            ("deep_only_lfu", "pooled", "lfu", 1.0, [2, 0, 1, 3]),
            ("deep_only_lru", "pooled", "lru", 5.0, [1, 3, 0, 2]),
            ("seq_only_lru", "sequence", "lru", 50.0, [3, 1, 0, 2]),
        ]

        raw_ids = [100, 101, 102, 103]

        for table_name, kind, eviction, base_val, remapped in plans:
            matches = list(_find_mch_modules_for_table(wrapped, table_name, kind=kind))
            self.assertEqual(len(matches), 1, f"{table_name}/{kind}: {matches}")
            _, mch = matches[0]

            weight_matches = list(
                _find_inner_weights_for_table(wrapped, table_name, kind=kind)
            )
            self.assertEqual(len(weight_matches), 1)
            _, weight = weight_matches[0]

            # _mch_sorted_raw_ids: 4 valid + 4 sentinel.
            sorted_raw = mch._buffers["_mch_sorted_raw_ids"].clone()
            sorted_raw[: len(raw_ids)] = torch.tensor(raw_ids, dtype=torch.int64)
            mch._buffers["_mch_sorted_raw_ids"].copy_(sorted_raw)

            # _mch_remapped_ids_mapping: non-trivial permutation on the 4
            # valid slots; sentinel slots get safe distinct defaults. The
            # output_global_offset is added so values are in the module's
            # local range and validate_state() invariants hold.
            offset = int(getattr(mch, "_output_global_offset", 0))
            remapped_buf = torch.tensor(
                [r + offset for r in remapped]
                + [4 + offset, 5 + offset, 6 + offset, 7 + offset],
                dtype=torch.int64,
            )
            mch._buffers["_mch_remapped_ids_mapping"].copy_(remapped_buf)

            # Eviction metadata.
            if eviction == "lfu":
                counts = torch.zeros(self._ZCH_SIZE, dtype=torch.int64)
                counts[: len(raw_ids)] = torch.tensor([10, 20, 30, 40])
                mch._buffers["_mch_counts"].copy_(counts)
            else:  # lru
                last = torch.zeros(self._ZCH_SIZE, dtype=torch.int64)
                last[: len(raw_ids)] = torch.tensor([101, 202, 303, 404])
                mch._buffers["_mch_last_access_iter"].copy_(last)

            # Inner weight row r -> all base_val + r. Distinct base per table
            # so any cross-merge bug surfaces in value assertions later.
            with torch.no_grad():
                for r in range(self._ZCH_SIZE):
                    weight[r].fill_(base_val + float(r))

    def test_convert_e2e_complex_scenario(self):
        src_inner = self._build_source_model()
        self._populate_source(src_inner)

        # Save source model and a tiny optimizer DCP.
        src_ckpt = os.path.join(self.tmp, "model.ckpt-100")
        os.makedirs(src_ckpt, exist_ok=True)
        src_wrapped = TrainWrapper(src_inner)
        dcp.save(
            src_wrapped.state_dict(), checkpoint_id=os.path.join(src_ckpt, "model")
        )
        dcp.save(
            {"state.linear.weight.exp_avg": torch.full((1, 4), 0.5)},
            checkpoint_id=os.path.join(src_ckpt, "optimizer"),
        )

        # Stub pipeline-config files (convert reads them, but
        # _create_features/_create_model are mocked, so content can be minimal).
        src_cfg_path = os.path.join(self.tmp, "src.config")
        tgt_cfg_path = os.path.join(self.tmp, "tgt.config")
        config_util.save_message(_make_minimal_pipeline_config(), src_cfg_path)
        config_util.save_message(_make_minimal_pipeline_config(), tgt_cfg_path)

        tgt_inner = self._build_target_model()
        # We have already wrapped src_inner once above; convert() will wrap
        # again, but Module re-wrapping is fine -- the state_dict was saved
        # off the once-wrapped form. To avoid double-wrap path drift, return
        # the *inner* models from the mock and let convert()'s real
        # TrainWrapper produce the same single-prefix shape that we saved.
        models = iter([src_inner, tgt_inner])

        save_dir = os.path.join(self.tmp, "out")
        with (
            mock.patch.object(conv, "_create_features", lambda *a, **k: []),
            mock.patch.object(conv, "_create_model", lambda *a, **k: next(models)),
        ):
            conv.convert(
                source_checkpoint_path=src_ckpt,
                source_pipeline_config_path=src_cfg_path,
                target_pipeline_config_path=tgt_cfg_path,
                save_dir=save_dir,
                world_size=2,
            )

        out = os.path.join(save_dir, "model.ckpt-0")

        # ----- meta -----
        with open(os.path.join(out, "meta")) as f:
            meta = json.load(f)
        self.assertTrue(meta["load_model"])
        self.assertTrue(meta["load_optim"])
        self.assertFalse(meta["dynamicemb_load_optim"])
        tables_by_mod_path = meta["dynamicemb_load_table_names"]
        # Exactly two distinct mod_paths: pooled vs sequence group_impls.
        self.assertEqual(len(tables_by_mod_path), 2)
        pooled_mod_paths = [
            p for p in tables_by_mod_path if ".ebc" in p and "ec_dict" not in p
        ]
        seq_mod_paths = [p for p in tables_by_mod_path if "ec_dict" in p]
        self.assertEqual(len(pooled_mod_paths), 1, tables_by_mod_path)
        self.assertEqual(len(seq_mod_paths), 1, tables_by_mod_path)
        pooled_mod_path = pooled_mod_paths[0]
        seq_mod_path = seq_mod_paths[0]
        # 3 tables under pooled, 2 under sequence.
        self.assertEqual(
            sorted(tables_by_mod_path[pooled_mod_path]),
            sorted(["shared_lfu", "deep_only_lfu", "deep_only_lru"]),
        )
        self.assertEqual(
            sorted(tables_by_mod_path[seq_mod_path]),
            sorted(["shared_lfu", "seq_only_lru"]),
        )

        # ----- model/ and optimizer/ byte-copied -----
        self.assertTrue(os.path.exists(os.path.join(out, "model", ".metadata")))
        self.assertTrue(os.path.exists(os.path.join(out, "optimizer", ".metadata")))

        # ----- per-table file presence -----
        # LFU/STEP targets: scores files present. TIMESTAMP target: scores absent.
        score_expected = {
            (pooled_mod_path, "shared_lfu"): True,  # LFU → STEP (proxy)
            (pooled_mod_path, "deep_only_lfu"): True,  # LFU → LFU
            (pooled_mod_path, "deep_only_lru"): False,  # LRU → TIMESTAMP omit
            (seq_mod_path, "shared_lfu"): True,  # LFU → STEP (proxy)
            (seq_mod_path, "seq_only_lru"): True,  # LRU → STEP direct
        }
        for (mod_path, table), should_have_scores in score_expected.items():
            shard_dir = os.path.join(out, "dynamicemb", mod_path)
            for rank in (0, 1):
                k = os.path.join(
                    shard_dir, f"{table}_emb_keys.rank_{rank}.world_size_2"
                )
                v = os.path.join(
                    shard_dir, f"{table}_emb_values.rank_{rank}.world_size_2"
                )
                s = os.path.join(
                    shard_dir, f"{table}_emb_scores.rank_{rank}.world_size_2"
                )
                self.assertTrue(os.path.exists(k), f"missing {k}")
                self.assertTrue(os.path.exists(v), f"missing {v}")
                if should_have_scores:
                    self.assertTrue(os.path.exists(s), f"missing {s}")
                else:
                    self.assertFalse(os.path.exists(s), f"unexpected {s}")

        # ----- content checks: keys are roundrobin-distributed -----
        for (mod_path, table), _ in score_expected.items():
            shard_dir = os.path.join(out, "dynamicemb", mod_path)
            all_keys = []
            for rank in (0, 1):
                p = os.path.join(
                    shard_dir, f"{table}_emb_keys.rank_{rank}.world_size_2"
                )
                arr = _read_int64(p)
                self.assertTrue(np.all(arr % 2 == rank), f"{table}/r{rank}: {arr}")
                all_keys.extend(arr.tolist())
            self.assertEqual(sorted(all_keys), [100, 101, 102, 103])

        # ----- shared_lfu values are distinct between DEEP and SEQUENCE -----
        # Pooled shared_lfu used base_val=10.0; sequence used base_val=100.0.
        for mod_path, expected_base in (
            (pooled_mod_path, 10.0),
            (seq_mod_path, 100.0),
        ):
            shard_dir = os.path.join(out, "dynamicemb", mod_path)
            for rank in (0, 1):
                vpath = os.path.join(
                    shard_dir, f"shared_lfu_emb_values.rank_{rank}.world_size_2"
                )
                vals = _read_float32(vpath).reshape(-1, self._DIM)
                if vals.size == 0:
                    continue
                # weight[r] = base + r. Every entry on a row must equal the same value.
                for row in vals:
                    self.assertEqual(len(set(row.tolist())), 1, row.tolist())
                    v = row[0]
                    self.assertGreaterEqual(
                        v, expected_base, f"{mod_path}/r{rank}: {v} < {expected_base}"
                    )

        # ----- scores for an LFU→LFU table mirror the populated counts -----
        # deep_only_lfu has counts [10, 20, 30, 40] for raw_ids [100, 101, 102, 103].
        for rank in (0, 1):
            keys_p = os.path.join(
                out,
                "dynamicemb",
                pooled_mod_path,
                f"deep_only_lfu_emb_keys.rank_{rank}.world_size_2",
            )
            scores_p = os.path.join(
                out,
                "dynamicemb",
                pooled_mod_path,
                f"deep_only_lfu_emb_scores.rank_{rank}.world_size_2",
            )
            keys = _read_int64(keys_p).tolist()
            scores = _read_int64(scores_p).tolist()
            self.assertEqual(len(keys), len(scores))
            expected_score = {100: 10, 101: 20, 102: 30, 103: 40}
            for k, s in zip(keys, scores):
                self.assertEqual(s, expected_score[k], f"k={k} got score {s}")

        # ----- scores for an LRU→STEP table mirror the populated last_access -----
        # seq_only_lru has last_access [101, 202, 303, 404] for raw_ids [100..103].
        for rank in (0, 1):
            keys_p = os.path.join(
                out,
                "dynamicemb",
                seq_mod_path,
                f"seq_only_lru_emb_keys.rank_{rank}.world_size_2",
            )
            scores_p = os.path.join(
                out,
                "dynamicemb",
                seq_mod_path,
                f"seq_only_lru_emb_scores.rank_{rank}.world_size_2",
            )
            keys = _read_int64(keys_p).tolist()
            scores = _read_int64(scores_p).tolist()
            expected = {100: 101, 101: 202, 102: 303, 103: 404}
            for k, s in zip(keys, scores):
                self.assertEqual(s, expected[k], f"k={k} got score {s}")


if __name__ == "__main__":
    unittest.main()
