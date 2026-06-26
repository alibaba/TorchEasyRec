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

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch
from parameterized import parameterized
from torch import nn
from torch.distributed import checkpoint as dcp

from tzrec.features.feature import create_features
from tzrec.models.model import TrainWrapper
from tzrec.modules.embedding import EmbeddingGroup
from tzrec.protos import feature_pb2, model_pb2
from tzrec.tests import utils
from tzrec.tools.dynamicemb import zch_to_dynamicemb_convert as conv
from tzrec.utils import checkpoint_util, misc_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.test_util import gpu_unavailable, mark_ci_scope

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


_BASE_IMPL_KEY = "__BASE__"


def _pooled_mc_ebc(emb_group):
    """Return the DEEP-group ``ManagedCollisionEmbeddingBagCollection``."""
    return emb_group.emb_impls[_BASE_IMPL_KEY].mc_ebc


def _sequence_mc_ec(emb_group, dim):
    """Return the SEQUENCE-group ``ManagedCollisionEmbeddingCollection`` for ``dim``."""
    return emb_group.seq_emb_impls[_BASE_IMPL_KEY].mc_ec_dict[str(dim)]


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
                "lfu_source_to_step_target_returns_none",
                {"_mch_counts": [5, 6, 7, 8]},
                "lfu",
                "STEP",
                [True, True, True, True],
                None,
            ),
            (
                "lru_source_to_lfu_target_returns_none",
                {"_mch_last_access_iter": [1, 2, 3, 4]},
                "lru",
                "LFU",
                [True, True, True, True],
                None,
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
@mark_ci_scope("gpu")
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

        mc_ebc = _pooled_mc_ebc(emb_group)
        mch = mc_ebc._managed_collision_collection._managed_collision_modules["user_id"]
        weight = mc_ebc._embedding_module.embedding_bags["user_id"].weight

        sorted_raw = mch._buffers["_mch_sorted_raw_ids"].clone()
        sorted_raw[:6] = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int64)
        mch._buffers["_mch_sorted_raw_ids"].copy_(sorted_raw)
        mch._buffers["_mch_remapped_ids_mapping"].copy_(
            torch.tensor([5, 4, 3, 2, 1, 0, 6, 7], dtype=torch.int64)
        )
        counts = torch.tensor([1, 2, 3, 4, 5, 6, 0, 0], dtype=torch.int64)
        mch._buffers["_mch_counts"].copy_(counts)

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
# convert() end-to-end (subprocess-driven, mirrors create_dynamicemb_init_ckpt_test)
# -----------------------------------------------------------------------------


_SRC_CONFIG = "tzrec/tests/configs/zch_to_dynamicemb_convert_src.config"
_TGT_CONFIG = "tzrec/tests/configs/zch_to_dynamicemb_convert_tgt.config"


class ConvertE2ETests(unittest.TestCase):
    """End-to-end test of the converter CLI.

    Trains a ZCH model briefly, runs the converter as a subprocess, then
    confirms the converted checkpoint can be used as a fine-tune source for
    a dynamicemb-flavored training run. Mirrors the existing pattern in
    ``tzrec/tools/dynamicemb/create_dynamicemb_init_ckpt_test.py:43-93``.
    """

    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_zch_to_dyemb_", dir="./tmp")
        os.chmod(self.test_dir, 0o755)

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(
        gpu_unavailable[0] or not has_dynamicemb,
        "dynamicemb not available.",
    )
    @mark_ci_scope("gpu")
    def test_convert_e2e(self):
        # 1. Train a ZCH model briefly to produce a real source checkpoint.
        src_train_dir = os.path.join(self.test_dir, "src_train")
        os.makedirs(src_train_dir, exist_ok=True)
        ok = utils.test_train_eval(_SRC_CONFIG, src_train_dir)
        self.assertTrue(ok, "Source ZCH training failed.")

        src_ckpt_path, _ = checkpoint_util.latest_checkpoint(
            os.path.join(src_train_dir, "train")
        )
        self.assertIsNotNone(src_ckpt_path, "No ZCH checkpoint produced.")

        # 2. Run the converter via the actual CLI, no mocks.
        converted_dir = os.path.join(self.test_dir, "converted")
        cmd_str = (
            "PYTHONPATH=. python -m tzrec.tools.dynamicemb.zch_to_dynamicemb_convert "
            f"--source_checkpoint_path {src_ckpt_path} "
            f"--source_pipeline_config_path {_SRC_CONFIG} "
            f"--target_pipeline_config_path {_TGT_CONFIG} "
            f"--save_dir {converted_dir}"
        )
        ok = misc_util.run_cmd(
            cmd_str,
            os.path.join(self.test_dir, "log_zch_to_dynamicemb_convert.txt"),
            timeout=600,
        )
        self.assertTrue(ok, "Converter subprocess failed.")

        # 3. Verify the converted checkpoint loads as a fine-tune source.
        tgt_train_dir = os.path.join(self.test_dir, "tgt_train")
        os.makedirs(tgt_train_dir, exist_ok=True)
        ok = utils.test_train_eval(
            _TGT_CONFIG,
            tgt_train_dir,
            f"--fine_tune_checkpoint {os.path.join(converted_dir, 'model.ckpt-0')}",
        )
        self.assertTrue(ok, "Fine-tune-resume training with converted ckpt failed.")
        self.success = True


if __name__ == "__main__":
    unittest.main()
