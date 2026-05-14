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
from torch import nn
from torch.distributed import checkpoint as dcp
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
)
from torchrec.modules.mc_modules import (
    LFU_EvictionPolicy,
    LRU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)

from tzrec.protos import pipeline_pb2
from tzrec.tools import zch_to_dynamicemb_convert as conv
from tzrec.utils import config_util
from tzrec.utils.dynamicemb_util import has_dynamicemb

try:
    from dynamicemb.planner import DynamicEmbParameterConstraints
except Exception:
    DynamicEmbParameterConstraints = None  # type: ignore[assignment]

_IINFO_MAX = torch.iinfo(torch.int64).max


def _read_int64(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.int64)


def _read_float32(path):
    with open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.float32)


def _build_mch_ebc(table_specs, device=None):
    """Build a ManagedCollisionEmbeddingBagCollection for given table specs.

    Each spec is a tuple ``(name, zch_size, embedding_dim, policy_kind)`` where
    ``policy_kind`` is one of ``"lfu"`` / ``"lru"``.
    """
    if device is None:
        device = torch.device("cpu")
    emb_configs = []
    mc_modules = {}
    for name, zch_size, dim, policy in table_specs:
        emb_configs.append(
            EmbeddingBagConfig(
                name=name,
                num_embeddings=zch_size,
                embedding_dim=dim,
                feature_names=[name],
            )
        )
        if policy == "lfu":
            ev = LFU_EvictionPolicy()
        elif policy == "lru":
            ev = LRU_EvictionPolicy()
        else:
            raise ValueError(policy)
        mc_modules[name] = MCHManagedCollisionModule(
            zch_size=zch_size,
            device=device,
            eviction_policy=ev,
            eviction_interval=5,
        )
    ebc = EmbeddingBagCollection(emb_configs, device=device)
    mc_coll = ManagedCollisionCollection(mc_modules, emb_configs)
    return ManagedCollisionEmbeddingBagCollection(
        ebc, mc_coll, return_remapped_features=False
    )


class _MiniModel(nn.Module):
    """Wraps the MC-EBC so the state_dict prefix mimics tzrec's layout.

    Final state_dict keys look like::

        embedding_group.embedding_group_impl_x.mc_ebc.<...>

    After wrapping with ``_TrainWrapperLike`` (mirror of tzrec.TrainWrapper)
    the canonical converter-side FQN prefix becomes
    ``model.embedding_group.<...>``.
    """

    def __init__(self, mc_ebc: ManagedCollisionEmbeddingBagCollection) -> None:
        super().__init__()
        self.embedding_group = nn.Module()
        self.embedding_group.embedding_group_impl_x = nn.Module()
        self.embedding_group.embedding_group_impl_x.mc_ebc = mc_ebc


class _TrainWrapperLike(nn.Module):
    """A stand-in for tzrec.models.model.TrainWrapper.

    The real one wraps with ``self.model = module``; mirroring that here
    avoids needing to build an actual tzrec BaseModel subclass for tests.
    """

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.model = inner


# -----------------------------------------------------------------------------
# Pure helper tests
# -----------------------------------------------------------------------------


class HelperFunctionTests(unittest.TestCase):
    """Tests for path-normalization helpers and policy classification."""

    def test_strip_collection_suffix_pooled(self):
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.mc_ebc"), "model.eg.gi"
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.ebc"), "model.eg.gi"
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.mc_ebc_user"), "model.eg.gi"
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.ebc_user"), "model.eg.gi"
        )

    def test_strip_collection_suffix_sequence(self):
        # Two-segment wrapper: ec_dict.<dim>.
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.mc_ec_dict.16"),
            "model.eg.gi",
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.ec_dict.32"),
            "model.eg.gi",
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.mc_ec_dict_user.8"),
            "model.eg.gi",
        )
        self.assertEqual(
            conv._strip_collection_suffix("model.eg.gi.ec_dict_user.4"),
            "model.eg.gi",
        )

    def test_strip_collection_suffix_raises_on_unknown(self):
        with self.assertRaises(ValueError):
            conv._strip_collection_suffix("model.eg.gi.not_a_wrapper")

    def test_normalize_group_key_strips_one_or_two_model_prefixes(self):
        self.assertEqual(
            conv._normalize_group_key("model.embedding_group.gi"),
            "embedding_group.gi",
        )
        self.assertEqual(
            conv._normalize_group_key("model.model.embedding_group.gi"),
            "embedding_group.gi",
        )
        # Already normalized -> idempotent.
        self.assertEqual(
            conv._normalize_group_key("embedding_group.gi"),
            "embedding_group.gi",
        )

    def test_classify_eviction_policy(self):
        self.assertEqual(conv._classify_eviction_policy(["_mch_counts"]), "lfu")
        self.assertEqual(
            conv._classify_eviction_policy(["_mch_last_access_iter"]), "lru"
        )
        self.assertEqual(
            conv._classify_eviction_policy(["_mch_counts", "_mch_last_access_iter"]),
            "distance_lfu",
        )
        self.assertEqual(conv._classify_eviction_policy([]), "none")


class DeriveScoresTests(unittest.TestCase):
    """Tests for the LFU/STEP migration and the skip-cases."""

    def test_lfu_target_uses_counts(self):
        counts = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, False, True])
        scores = conv._derive_scores({"_mch_counts": counts}, "lfu", "LFU", valid_mask)
        torch.testing.assert_close(
            scores, torch.tensor([10, 20, 40], dtype=torch.int64)
        )

    def test_step_target_uses_last_access(self):
        last = torch.tensor([100, 200, 300, 400], dtype=torch.int64)
        valid_mask = torch.tensor([True, False, True, True])
        scores = conv._derive_scores(
            {"_mch_last_access_iter": last}, "lru", "STEP", valid_mask
        )
        torch.testing.assert_close(
            scores, torch.tensor([100, 300, 400], dtype=torch.int64)
        )

    def test_distance_lfu_to_lfu_prefers_counts(self):
        counts = torch.tensor([7, 7, 7, 7], dtype=torch.int64)
        last = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, True, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts, "_mch_last_access_iter": last},
            "distance_lfu",
            "LFU",
            valid_mask,
        )
        torch.testing.assert_close(scores, counts)

    def test_distance_lfu_to_step_prefers_last_access(self):
        counts = torch.tensor([7, 7, 7, 7], dtype=torch.int64)
        last = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, True, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts, "_mch_last_access_iter": last},
            "distance_lfu",
            "STEP",
            valid_mask,
        )
        torch.testing.assert_close(scores, last)

    def test_timestamp_target_returns_none(self):
        counts = torch.tensor([1, 2, 3], dtype=torch.int64)
        scores = conv._derive_scores(
            {"_mch_counts": counts},
            "lfu",
            "TIMESTAMP",
            torch.tensor([True, True, True]),
        )
        self.assertIsNone(scores)

    def test_customized_target_returns_none(self):
        counts = torch.tensor([1, 2], dtype=torch.int64)
        scores = conv._derive_scores(
            {"_mch_counts": counts},
            "lfu",
            "CUSTOMIZED",
            torch.tensor([True, True]),
        )
        self.assertIsNone(scores)

    def test_no_eviction_target_returns_none(self):
        counts = torch.tensor([1, 2, 3], dtype=torch.int64)
        scores = conv._derive_scores(
            {"_mch_counts": counts},
            "lfu",
            "NO_EVICTION",
            torch.tensor([True, True, True]),
        )
        self.assertIsNone(scores)

    def test_step_target_without_metadata_returns_none(self):
        valid_mask = torch.tensor([True, True])
        scores = conv._derive_scores({}, "none", "STEP", valid_mask)
        self.assertIsNone(scores)


# -----------------------------------------------------------------------------
# Shard-write tests (need dynamicemb for encode_checkpoint_file_path)
# -----------------------------------------------------------------------------


@unittest.skipUnless(has_dynamicemb, "dynamicemb not available")
class ShardWriteTests(unittest.TestCase):
    """Tests for _gather_and_shard_writes binary file output."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_shard_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_world_size_1_with_scores(self):
        raw_ids = torch.tensor([7, 13, 42, 100], dtype=torch.int64)
        values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        scores = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            table_name="tbl",
            save_path=self.tmp,
            world_size=1,
        )
        for item in ("keys", "values", "scores"):
            p = os.path.join(self.tmp, f"tbl_emb_{item}.rank_0.world_size_1")
            self.assertTrue(os.path.exists(p), f"missing {p}")
        # Content round-trips:
        keys_back = _read_int64(
            os.path.join(self.tmp, "tbl_emb_keys.rank_0.world_size_1")
        )
        np.testing.assert_array_equal(keys_back, raw_ids.numpy())
        # No opt_values file is written (opt migration is out of scope).
        self.assertFalse(
            os.path.exists(
                os.path.join(self.tmp, "tbl_emb_opt_values.rank_0.world_size_1")
            )
        )

    def test_scores_none_omits_score_file_for_every_rank(self):
        raw_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        values = torch.zeros((4, 2), dtype=torch.float32)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=None,
            table_name="t",
            save_path=self.tmp,
            world_size=2,
        )
        for r in (0, 1):
            self.assertTrue(
                os.path.exists(
                    os.path.join(self.tmp, f"t_emb_keys.rank_{r}.world_size_2")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(self.tmp, f"t_emb_values.rank_{r}.world_size_2")
                )
            )
            self.assertFalse(
                os.path.exists(
                    os.path.join(self.tmp, f"t_emb_scores.rank_{r}.world_size_2")
                ),
                f"scores file should not exist on rank {r}",
            )

    def test_sharded_world_size_3_splits_by_modulo(self):
        raw_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
        values = torch.zeros((9, 2), dtype=torch.float32)
        scores = torch.zeros(9, dtype=torch.int64)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            table_name="t",
            save_path=self.tmp,
            world_size=3,
        )
        for r in range(3):
            keys_path = os.path.join(self.tmp, f"t_emb_keys.rank_{r}.world_size_3")
            self.assertTrue(os.path.exists(keys_path))
            arr = _read_int64(keys_path)
            self.assertTrue(np.all(arr % 3 == r), f"rank {r} got {arr.tolist()}")
            self.assertEqual(len(arr), 3)

    def test_empty_shard_writes_zero_byte_file(self):
        # All raw_ids land on rank 0 under world_size=4 if we choose 0,4,8.
        raw_ids = torch.tensor([0, 4, 8], dtype=torch.int64)
        values = torch.zeros((3, 1), dtype=torch.float32)
        scores = torch.zeros(3, dtype=torch.int64)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            table_name="t",
            save_path=self.tmp,
            world_size=4,
        )
        for r in (1, 2, 3):
            keys_path = os.path.join(self.tmp, f"t_emb_keys.rank_{r}.world_size_4")
            self.assertTrue(os.path.exists(keys_path))
            self.assertEqual(os.path.getsize(keys_path), 0)
        keys_path0 = os.path.join(self.tmp, "t_emb_keys.rank_0.world_size_4")
        arr = _read_int64(keys_path0)
        np.testing.assert_array_equal(arr, raw_ids.numpy())


# -----------------------------------------------------------------------------
# Model-walking tests
# -----------------------------------------------------------------------------


class FindZchTablesTests(unittest.TestCase):
    """Tests for the model-walking helper that discovers MCH-wrapped tables."""

    def test_finds_mch_tables_with_correct_state_dict_paths(self):
        mc_ebc = _build_mch_ebc(
            [
                ("user_id", 32, 4, "lfu"),
                ("item_id", 16, 8, "lru"),
            ]
        )
        model = _TrainWrapperLike(_MiniModel(mc_ebc))
        tables = conv._find_zch_tables(model)

        # Both tables live under the same group_impl, so group_key is shared.
        gk = "embedding_group.embedding_group_impl_x"
        self.assertEqual(set(tables), {(gk, "user_id"), (gk, "item_id")})

        u = tables[(gk, "user_id")]
        self.assertEqual(u.group_key, gk)
        self.assertEqual(u.embedding_dim, 4)
        self.assertEqual(u.zch_size, 32)
        self.assertEqual(u.eviction_policy, "lfu")
        self.assertEqual(u.metadata_buffer_names, ["_mch_counts"])
        self.assertEqual(
            u.weight_fqn,
            "model.embedding_group.embedding_group_impl_x.mc_ebc."
            "_embedding_module.embedding_bags.user_id.weight",
        )
        self.assertEqual(
            u.mch_prefix,
            "model.embedding_group.embedding_group_impl_x.mc_ebc."
            "_managed_collision_collection._managed_collision_modules.user_id",
        )

        i = tables[(gk, "item_id")]
        self.assertEqual(i.eviction_policy, "lru")
        self.assertEqual(i.metadata_buffer_names, ["_mch_last_access_iter"])

    def test_same_emb_name_across_two_groups_does_not_collide(self):
        """Regression for the (group_key, emb_name) match-key migration."""
        # Two MC-EBC instances, each with a table called "shared". Put them
        # under different group_impl paths.
        inner = nn.Module()
        inner.embedding_group = nn.Module()
        inner.embedding_group.eg_a = nn.Module()
        inner.embedding_group.eg_a.mc_ebc = _build_mch_ebc([("shared", 16, 4, "lfu")])
        inner.embedding_group.eg_b = nn.Module()
        inner.embedding_group.eg_b.mc_ebc = _build_mch_ebc([("shared", 32, 8, "lru")])
        model = _TrainWrapperLike(inner)

        tables = conv._find_zch_tables(model)
        gk_a = "embedding_group.eg_a"
        gk_b = "embedding_group.eg_b"
        self.assertEqual(set(tables), {(gk_a, "shared"), (gk_b, "shared")})
        self.assertEqual(tables[(gk_a, "shared")].zch_size, 16)
        self.assertEqual(tables[(gk_a, "shared")].embedding_dim, 4)
        self.assertEqual(tables[(gk_b, "shared")].zch_size, 32)
        self.assertEqual(tables[(gk_b, "shared")].embedding_dim, 8)


# -----------------------------------------------------------------------------
# DCP round-trip
# -----------------------------------------------------------------------------


class DcpRoundTripTests(unittest.TestCase):
    """Save MCH+EBC state via DCP single-process, then load via the helper."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_dcp_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_load_dcp_subset_round_trip(self):
        mc_ebc = _build_mch_ebc([("user_id", 8, 4, "lfu")])
        model = _TrainWrapperLike(_MiniModel(mc_ebc))

        # Manually populate MCH state and EBC weight with known values.
        mch = mc_ebc._managed_collision_collection._managed_collision_modules["user_id"]
        # Slot i represents raw_id = 100+i with count i+1; one slot stays sentinel.
        sorted_raw = mch._buffers["_mch_sorted_raw_ids"].clone()
        sorted_raw[:6] = torch.tensor([100, 101, 102, 103, 104, 105], dtype=torch.int64)
        # Slots 6 and 7 stay at _IINFO_MAX (unused).
        mch._buffers["_mch_sorted_raw_ids"].copy_(sorted_raw)
        # remapped is identity by default; tweak to non-trivial mapping.
        mch._buffers["_mch_remapped_ids_mapping"].copy_(
            torch.tensor([5, 4, 3, 2, 1, 0, 6, 7], dtype=torch.int64)
        )
        counts = torch.tensor([1, 2, 3, 4, 5, 6, 0, 0], dtype=torch.int64)
        mch._buffers["_mch_counts"].copy_(counts)

        ebc = mc_ebc._embedding_module
        weight = ebc.embedding_bags["user_id"].weight
        # Set weight rows to easy-to-check values: row r -> [r, r, r, r].
        with torch.no_grad():
            for r in range(8):
                weight[r].fill_(float(r))

        # Save state_dict via DCP single-process.
        dcp_dir = os.path.join(self.tmp, "model")
        os.makedirs(dcp_dir, exist_ok=True)
        dcp.save(model.state_dict(), checkpoint_id=dcp_dir)

        # Use the converter's loader to read back the three tensors of interest.
        prefix = (
            "model.embedding_group.embedding_group_impl_x.mc_ebc."
            "_managed_collision_collection._managed_collision_modules.user_id"
        )
        weight_fqn = (
            "model.embedding_group.embedding_group_impl_x.mc_ebc."
            "_embedding_module.embedding_bags.user_id.weight"
        )
        loaded = conv._load_dcp_subset(
            dcp_dir,
            [
                f"{prefix}._mch_sorted_raw_ids",
                f"{prefix}._mch_remapped_ids_mapping",
                f"{prefix}._mch_counts",
                weight_fqn,
            ],
        )
        torch.testing.assert_close(loaded[f"{prefix}._mch_sorted_raw_ids"], sorted_raw)
        torch.testing.assert_close(
            loaded[f"{prefix}._mch_remapped_ids_mapping"],
            torch.tensor([5, 4, 3, 2, 1, 0, 6, 7], dtype=torch.int64),
        )
        torch.testing.assert_close(loaded[f"{prefix}._mch_counts"], counts)
        torch.testing.assert_close(loaded[weight_fqn][3], torch.full((4,), 3.0))

        # Now exercise the full conversion math (sans pipeline.config):
        valid_mask = loaded[f"{prefix}._mch_sorted_raw_ids"] != _IINFO_MAX
        raw_ids = loaded[f"{prefix}._mch_sorted_raw_ids"][valid_mask]
        remapped = loaded[f"{prefix}._mch_remapped_ids_mapping"][valid_mask]
        values = loaded[weight_fqn].index_select(0, remapped)
        expected_pairs = [
            (100, 5),
            (101, 4),
            (102, 3),
            (103, 2),
            (104, 1),
            (105, 0),
        ]
        self.assertEqual(raw_ids.tolist(), [p[0] for p in expected_pairs])
        for i, (_, row) in enumerate(expected_pairs):
            torch.testing.assert_close(values[i], torch.full((4,), float(row)))

        # Scores: LFU source + LFU target -> direct copy of counts[valid].
        scores = conv._derive_scores(
            {"_mch_counts": loaded[f"{prefix}._mch_counts"]},
            "lfu",
            "LFU",
            valid_mask,
        )
        torch.testing.assert_close(
            scores, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        )


# -----------------------------------------------------------------------------
# convert() end-to-end (needs dynamicemb because the stubs use
# DynamicEmbParameterConstraints and convert() calls into dynamicemb)
# -----------------------------------------------------------------------------


class _StubDynamicEmbOptions:
    """Minimum surface area to satisfy _find_dynamicemb_tables."""

    def __init__(self, dim, score_strategy_name="LFU", dist_type="roundrobin"):
        self.dim = dim
        self.dist_type = dist_type

        class _S:
            name = score_strategy_name

        self.score_strategy = _S


class _StubEmbeddingGroupImpl(nn.Module):
    """Stand-in exposing parameter_constraints() with dynamicemb constraints.

    Tests use this to feed _find_dynamicemb_tables a controllable target
    model without depending on full tzrec model construction.

    Only instantiable when dynamicemb is installed; tests that build one
    are guarded with ``@unittest.skipUnless(has_dynamicemb, ...)``.
    """

    def __init__(self, table_specs):
        """``table_specs``: list of ``(name, dim, score_strategy_name)``."""
        super().__init__()
        self._table_specs = table_specs
        self.ebc = nn.Module()

    def parameter_constraints(self, prefix=""):
        out = {}
        for spec in self._table_specs:
            if len(spec) == 2:
                name, dim = spec
                score_strategy_name = "LFU"
            else:
                name, dim, score_strategy_name = spec
            options = _StubDynamicEmbOptions(
                dim=dim, score_strategy_name=score_strategy_name
            )
            out[f"{prefix}ebc.{name}"] = DynamicEmbParameterConstraints(
                use_dynamicemb=True,
                dynamicemb_options=options,
            )
        return out


def _make_minimal_pipeline_config():
    """Construct an in-memory EasyRecConfig with the fields convert() reads."""
    cfg = pipeline_pb2.EasyRecConfig(
        train_input_path="",
        eval_input_path="",
        model_dir="",
    )
    cfg.train_config.sparse_optimizer.adagrad_optimizer.lr = 0.01
    cfg.train_config.sparse_optimizer.constant_learning_rate.SetInParent()
    cfg.train_config.dense_optimizer.adagrad_optimizer.lr = 0.01
    cfg.train_config.dense_optimizer.constant_learning_rate.SetInParent()
    return cfg


@unittest.skipUnless(has_dynamicemb, "dynamicemb not available")
class ConvertE2ETests(unittest.TestCase):
    """End-to-end test of ``convert()`` orchestration with patched builders."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_e2e_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _build_and_save_source(self, src_ckpt_dir):
        """Build a 1-table MCH model, populate it, save model + optimizer DCP."""
        mc_ebc = _build_mch_ebc([("user_id", 8, 4, "lfu")])
        src_model = _TrainWrapperLike(_MiniModel(mc_ebc))

        mch = mc_ebc._managed_collision_collection._managed_collision_modules["user_id"]
        sorted_raw = mch._buffers["_mch_sorted_raw_ids"].clone()
        sorted_raw[:4] = torch.tensor([100, 101, 102, 103], dtype=torch.int64)
        mch._buffers["_mch_sorted_raw_ids"].copy_(sorted_raw)
        # Non-identity remapped: raw_id 100 -> row 3, 101 -> 2, 102 -> 1, 103 -> 0.
        mch._buffers["_mch_remapped_ids_mapping"].copy_(
            torch.tensor([3, 2, 1, 0, 4, 5, 6, 7], dtype=torch.int64)
        )
        mch._buffers["_mch_counts"].copy_(
            torch.tensor([10, 20, 30, 40, 0, 0, 0, 0], dtype=torch.int64)
        )

        ebc = mc_ebc._embedding_module
        weight = ebc.embedding_bags["user_id"].weight
        with torch.no_grad():
            for r in range(8):
                weight[r].fill_(float(r))

        os.makedirs(src_ckpt_dir, exist_ok=True)
        dcp.save(
            src_model.state_dict(),
            checkpoint_id=os.path.join(src_ckpt_dir, "model"),
        )

        # Save a synthetic optimizer state so we can verify it's byte-copied.
        # (The converter no longer migrates MCH-EBC opt state into opt_values,
        # but the source optimizer/ DCP is still preserved verbatim.)
        opt_sd = {
            "state.linear.weight.exp_avg": torch.full((1, 4), 0.5),
        }
        dcp.save(opt_sd, checkpoint_id=os.path.join(src_ckpt_dir, "optimizer"))
        return src_model

    def _run_convert(self, src_ckpt, target_table_specs, world_size=2):
        """Helper: stub builders, then call conv.convert with the right cfgs."""
        src_cfg_path = os.path.join(self.tmp, "src.config")
        tgt_cfg_path = os.path.join(self.tmp, "tgt.config")
        config_util.save_message(_make_minimal_pipeline_config(), src_cfg_path)
        config_util.save_message(_make_minimal_pipeline_config(), tgt_cfg_path)

        src_model = _TrainWrapperLike(
            _MiniModel(_build_mch_ebc([("user_id", 8, 4, "lfu")]))
        )
        tgt_inner = nn.Module()
        tgt_inner.embedding_group = nn.Module()
        tgt_inner.embedding_group.embedding_group_impl_x = _StubEmbeddingGroupImpl(
            target_table_specs
        )
        tgt_model = _TrainWrapperLike(tgt_inner)

        models = iter([src_model.model, tgt_model.model])

        def passthrough(m):
            return _TrainWrapperLike(m)

        save_dir = os.path.join(self.tmp, f"out_{world_size}")
        with (
            mock.patch.object(conv, "_create_features", lambda *a, **k: []),
            mock.patch.object(conv, "_create_model", lambda *a, **k: next(models)),
            mock.patch.object(conv, "TrainWrapper", passthrough),
        ):
            conv.convert(
                source_checkpoint_path=src_ckpt,
                source_pipeline_config_path=src_cfg_path,
                target_pipeline_config_path=tgt_cfg_path,
                save_dir=save_dir,
                world_size=world_size,
            )
        return os.path.join(save_dir, "model.ckpt-0")

    def test_convert_lfu_writes_scores_and_no_opt_values(self):
        src_ckpt = os.path.join(self.tmp, "model.ckpt-100")
        self._build_and_save_source(src_ckpt)

        out = self._run_convert(src_ckpt, [("user_id", 4, "LFU")])

        # meta JSON.
        with open(os.path.join(out, "meta")) as f:
            meta = json.load(f)
        self.assertTrue(meta["load_model"])
        self.assertTrue(meta["load_optim"])
        # dynamicemb_load_optim is hard-coded false in v1.
        self.assertFalse(meta["dynamicemb_load_optim"])
        self.assertEqual(
            list(meta["dynamicemb_load_table_names"].values())[0], ["user_id"]
        )
        mod_path = next(iter(meta["dynamicemb_load_table_names"]))
        self.assertIn("ebc", mod_path)

        # model/ and optimizer/ were byte-copied verbatim.
        self.assertTrue(os.path.isdir(os.path.join(out, "model")))
        self.assertTrue(os.path.isdir(os.path.join(out, "optimizer")))
        self.assertTrue(os.path.exists(os.path.join(out, "model", ".metadata")))
        self.assertTrue(os.path.exists(os.path.join(out, "optimizer", ".metadata")))

        # Dynamicemb shards exist for each rank, with content matching
        # raw_id % world_size assignment.
        shard_dir = os.path.join(out, "dynamicemb", mod_path)
        keys_r0 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_keys.rank_0.world_size_2")
        )
        keys_r1 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_keys.rank_1.world_size_2")
        )
        self.assertEqual(sorted(keys_r0.tolist()), [100, 102])
        self.assertEqual(sorted(keys_r1.tolist()), [101, 103])

        # Values for raw_id 100 -> remapped 3 -> weight row 3 -> all 3.0.
        values_r0 = _read_float32(
            os.path.join(shard_dir, "user_id_emb_values.rank_0.world_size_2")
        ).reshape(-1, 4)
        for i, raw_id in enumerate(keys_r0.tolist()):
            expected_row = float({100: 3, 102: 1}[raw_id])
            np.testing.assert_array_equal(
                values_r0[i], np.full((4,), expected_row, dtype=np.float32)
            )

        # Scores: LFU -> LFU. raw_id 100 was at slot 0 with count=10, etc.
        scores_r0 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_scores.rank_0.world_size_2")
        )
        for i, raw_id in enumerate(keys_r0.tolist()):
            expected_count = {100: 10, 102: 30}[raw_id]
            self.assertEqual(int(scores_r0[i]), expected_count)

        # NO opt_values files exist (opt-migration is out of scope in v1).
        for r in (0, 1):
            self.assertFalse(
                os.path.exists(
                    os.path.join(
                        shard_dir,
                        f"user_id_emb_opt_values.rank_{r}.world_size_2",
                    )
                ),
                f"opt_values should not be written for rank {r}",
            )

    def test_convert_timestamp_target_omits_scores(self):
        src_ckpt = os.path.join(self.tmp, "model.ckpt-200")
        self._build_and_save_source(src_ckpt)

        out = self._run_convert(src_ckpt, [("user_id", 4, "TIMESTAMP")])
        mod_path = "model.model.embedding_group.embedding_group_impl_x.ebc"
        shard_dir = os.path.join(out, "dynamicemb", mod_path)

        # Keys and values are present for both ranks.
        for r in (0, 1):
            self.assertTrue(
                os.path.exists(
                    os.path.join(shard_dir, f"user_id_emb_keys.rank_{r}.world_size_2")
                )
            )
            self.assertTrue(
                os.path.exists(
                    os.path.join(shard_dir, f"user_id_emb_values.rank_{r}.world_size_2")
                )
            )
            # No score files for TIMESTAMP target.
            self.assertFalse(
                os.path.exists(
                    os.path.join(shard_dir, f"user_id_emb_scores.rank_{r}.world_size_2")
                ),
                f"TIMESTAMP target should not produce score files (rank {r})",
            )


if __name__ == "__main__":
    unittest.main()
