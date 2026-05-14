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


def _build_mch_ebc(
    table_specs,
    device=None,
):
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

    After wrapping with TrainWrapper(self) and saving, the canonical
    converter-side FQN prefix becomes ``model.embedding_group.<...>``.
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


class PureFunctionTests(unittest.TestCase):
    """Tests for the converter's helper functions that don't need any I/O."""

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

    def test_derive_scores_lfu_target_uses_counts(self):
        counts = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, False, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts}, "lfu", "LFU", valid_mask, 0
        )
        torch.testing.assert_close(
            scores, torch.tensor([10, 20, 40], dtype=torch.int64)
        )

    def test_derive_scores_step_target_uses_last_access(self):
        last = torch.tensor([100, 200, 300, 400], dtype=torch.int64)
        valid_mask = torch.tensor([True, False, True, True])
        scores = conv._derive_scores(
            {"_mch_last_access_iter": last}, "lru", "STEP", valid_mask, 0
        )
        torch.testing.assert_close(
            scores, torch.tensor([100, 300, 400], dtype=torch.int64)
        )

    def test_derive_scores_distance_lfu_to_lfu_prefers_counts(self):
        counts = torch.tensor([7, 7, 7, 7], dtype=torch.int64)
        last = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, True, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts, "_mch_last_access_iter": last},
            "distance_lfu",
            "LFU",
            valid_mask,
            0,
        )
        torch.testing.assert_close(scores, counts)

    def test_derive_scores_distance_lfu_to_step_prefers_last_access(self):
        counts = torch.tensor([7, 7, 7, 7], dtype=torch.int64)
        last = torch.tensor([10, 20, 30, 40], dtype=torch.int64)
        valid_mask = torch.tensor([True, True, True, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts, "_mch_last_access_iter": last},
            "distance_lfu",
            "STEP",
            valid_mask,
            0,
        )
        torch.testing.assert_close(scores, last)

    def test_derive_scores_init_offset_added(self):
        counts = torch.tensor([10, 20], dtype=torch.int64)
        valid_mask = torch.tensor([True, True])
        scores = conv._derive_scores(
            {"_mch_counts": counts}, "lfu", "LFU", valid_mask, init_score_offset=1000
        )
        torch.testing.assert_close(
            scores, torch.tensor([1010, 1020], dtype=torch.int64)
        )

    def test_derive_scores_no_eviction_returns_constant(self):
        valid_mask = torch.tensor([True, True, True])
        scores = conv._derive_scores(
            {"_mch_counts": torch.tensor([1, 2, 3], dtype=torch.int64)},
            "lfu",
            "NO_EVICTION",
            valid_mask,
            init_score_offset=42,
        )
        torch.testing.assert_close(
            scores, torch.tensor([42, 42, 42], dtype=torch.int64)
        )

    def test_derive_scores_missing_metadata_zero_with_warning(self):
        valid_mask = torch.tensor([True, True])
        scores = conv._derive_scores({}, "none", "STEP", valid_mask, 0)
        torch.testing.assert_close(scores, torch.tensor([0, 0], dtype=torch.int64))

    def test_gather_opt_rows_adagrad(self):
        sum_t = torch.tensor(
            [
                [0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
            ],
            dtype=torch.float32,
        )
        remapped = torch.tensor([2, 0, 3], dtype=torch.long)
        out = conv._gather_opt_rows({"sum": sum_t}, "adagrad", remapped, dim=3)
        torch.testing.assert_close(
            out,
            torch.tensor(
                [[2.0, 2.1, 2.2], [0.1, 0.2, 0.3], [3.0, 3.1, 3.2]],
                dtype=torch.float32,
            ),
        )

    def test_gather_opt_rows_adam_concatenates_m_v(self):
        m = torch.full((4, 2), 0.5)
        v = torch.full((4, 2), 1.5)
        remapped = torch.tensor([1, 3], dtype=torch.long)
        out = conv._gather_opt_rows(
            {"exp_avg": m, "exp_avg_sq": v}, "adam", remapped, dim=2
        )
        self.assertEqual(tuple(out.shape), (2, 4))
        torch.testing.assert_close(out[:, :2], torch.full((2, 2), 0.5))
        torch.testing.assert_close(out[:, 2:], torch.full((2, 2), 1.5))

    def test_gather_opt_rows_rowwise_adagrad_reshapes_to_n_x_1(self):
        sum_t = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
        remapped = torch.tensor([2, 0], dtype=torch.long)
        out = conv._gather_opt_rows({"sum": sum_t}, "rowwise_adagrad", remapped, dim=4)
        self.assertEqual(tuple(out.shape), (2, 1))
        torch.testing.assert_close(out, torch.tensor([[30.0], [10.0]]))

    def test_classify_source_optimizer_for_weight_adagrad(self):
        # Synthetic DCP-style metadata: only state.* entries wrapped in
        # TensorStorageMetadata are considered by the classifier.
        from torch.distributed.checkpoint import TensorStorageMetadata

        class _Props:
            def __init__(self, dtype):
                self.dtype = dtype

        opt_meta_typed = {
            "state.model.embedding_group.foo.weight.sum": TensorStorageMetadata(
                properties=_Props(torch.float32),
                size=torch.Size([100, 16]),
                chunks=[],
            ),
        }
        label, names = conv._classify_source_optimizer_for_weight(
            opt_meta_typed,
            weight_fqn="model.embedding_group.foo.weight",
            zch_size=100,
            dim=16,
        )
        self.assertEqual(label, "adagrad")
        self.assertEqual(set(names), {"sum"})

    def test_classify_source_optimizer_for_weight_rowwise_adagrad(self):
        from torch.distributed.checkpoint import TensorStorageMetadata

        class _Props:
            def __init__(self, dtype):
                self.dtype = dtype

        opt_meta_typed = {
            "state.model.embedding_group.foo.weight.sum": TensorStorageMetadata(
                properties=_Props(torch.float32),
                size=torch.Size([100]),
                chunks=[],
            ),
        }
        label, _ = conv._classify_source_optimizer_for_weight(
            opt_meta_typed,
            weight_fqn="model.embedding_group.foo.weight",
            zch_size=100,
            dim=16,
        )
        self.assertEqual(label, "rowwise_adagrad")

    def test_classify_source_optimizer_for_weight_adam(self):
        from torch.distributed.checkpoint import TensorStorageMetadata

        class _Props:
            def __init__(self, dtype):
                self.dtype = dtype

        opt_meta_typed = {
            "state.model.embedding_group.foo.weight.exp_avg": TensorStorageMetadata(
                properties=_Props(torch.float32),
                size=torch.Size([100, 16]),
                chunks=[],
            ),
            "state.model.embedding_group.foo.weight.exp_avg_sq": TensorStorageMetadata(
                properties=_Props(torch.float32),
                size=torch.Size([100, 16]),
                chunks=[],
            ),
        }
        label, names = conv._classify_source_optimizer_for_weight(
            opt_meta_typed,
            weight_fqn="model.embedding_group.foo.weight",
            zch_size=100,
            dim=16,
        )
        self.assertEqual(label, "adam")
        self.assertEqual(set(names), {"exp_avg", "exp_avg_sq"})

    def test_classify_source_optimizer_for_weight_sgd(self):
        # No optimizer state for this weight -> sgd
        from torch.distributed.checkpoint import TensorStorageMetadata

        class _Props:
            def __init__(self, dtype):
                self.dtype = dtype

        opt_meta_typed = {
            "state.other.param.exp_avg": TensorStorageMetadata(
                properties=_Props(torch.float32),
                size=torch.Size([10, 10]),
                chunks=[],
            ),
        }
        label, names = conv._classify_source_optimizer_for_weight(
            opt_meta_typed,
            weight_fqn="model.embedding_group.foo.weight",
            zch_size=100,
            dim=16,
        )
        self.assertEqual(label, "sgd")
        self.assertEqual(names, {})


@unittest.skipUnless(has_dynamicemb, "dynamicemb not available")
class ShardWriteTests(unittest.TestCase):
    """Tests for _gather_and_shard_writes binary file output."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="zch_dyemb_shard_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_sharded_round_trip_world_size_1(self):
        raw_ids = torch.tensor([7, 13, 42, 100], dtype=torch.int64)
        values = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        scores = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
        opt = torch.ones((4, 2), dtype=torch.float32)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            opt_values=opt,
            table_name="tbl",
            save_path=self.tmp,
            world_size=1,
        )
        # Files exist:
        for item in ("keys", "values", "scores", "opt_values"):
            p = os.path.join(self.tmp, f"tbl_emb_{item}.rank_0.world_size_1")
            self.assertTrue(os.path.exists(p), f"missing {p}")
        # Content round-trips:
        keys_back = _read_int64(
            os.path.join(self.tmp, "tbl_emb_keys.rank_0.world_size_1")
        )
        np.testing.assert_array_equal(keys_back, raw_ids.numpy())

    def test_sharded_world_size_3_splits_by_modulo(self):
        # Use raw_ids designed so each rank gets a known subset.
        raw_ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
        values = torch.zeros((9, 2), dtype=torch.float32)
        scores = torch.zeros(9, dtype=torch.int64)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            opt_values=None,
            table_name="t",
            save_path=self.tmp,
            world_size=3,
        )
        for r in range(3):
            keys_path = os.path.join(self.tmp, f"t_emb_keys.rank_{r}.world_size_3")
            self.assertTrue(os.path.exists(keys_path))
            arr = _read_int64(keys_path)
            self.assertTrue(np.all(arr % 3 == r), f"rank {r} got {arr.tolist()}")
            # 9 keys spread across 3 ranks: 3 per rank
            self.assertEqual(len(arr), 3)

    def test_sharded_empty_shard_writes_zero_byte_file(self):
        # All raw_ids land on rank 0 under world_size=4 if we choose 0,4,8.
        raw_ids = torch.tensor([0, 4, 8], dtype=torch.int64)
        values = torch.zeros((3, 1), dtype=torch.float32)
        scores = torch.zeros(3, dtype=torch.int64)
        conv._gather_and_shard_writes(
            raw_ids=raw_ids,
            values=values,
            scores=scores,
            opt_values=None,
            table_name="t",
            save_path=self.tmp,
            world_size=4,
        )
        for r in (1, 2, 3):
            keys_path = os.path.join(self.tmp, f"t_emb_keys.rank_{r}.world_size_4")
            self.assertTrue(os.path.exists(keys_path))
            self.assertEqual(os.path.getsize(keys_path), 0)
        # Rank 0 has the keys:
        keys_path0 = os.path.join(self.tmp, "t_emb_keys.rank_0.world_size_4")
        arr = _read_int64(keys_path0)
        np.testing.assert_array_equal(arr, raw_ids.numpy())


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

        self.assertEqual(set(tables), {"user_id", "item_id"})
        u = tables["user_id"]
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

        i = tables["item_id"]
        self.assertEqual(i.eviction_policy, "lru")
        self.assertEqual(i.metadata_buffer_names, ["_mch_last_access_iter"])
        self.assertEqual(i.embedding_dim, 8)
        self.assertEqual(i.zch_size, 16)


class DcpRoundTripTests(unittest.TestCase):
    """Save MCH+EBC state via DCP single-process, then load via the converter helper."""

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
        # The weight tensor was set to row-constant; check a sample row.
        torch.testing.assert_close(loaded[weight_fqn][3], torch.full((4,), 3.0))

        # Now exercise the full conversion math (sans pipeline.config):
        # Drop sentinel-filled slots and gather (raw_id, embedding, score).
        valid_mask = loaded[f"{prefix}._mch_sorted_raw_ids"] != _IINFO_MAX
        raw_ids = loaded[f"{prefix}._mch_sorted_raw_ids"][valid_mask]
        remapped = loaded[f"{prefix}._mch_remapped_ids_mapping"][valid_mask]
        values = loaded[weight_fqn].index_select(0, remapped)
        # Verify (raw_id, embedding) pairs match what was set up:
        # raw_id 100 -> remapped 5 -> weight row 5 -> [5,5,5,5]
        # raw_id 101 -> remapped 4 -> [4,4,4,4]   etc.
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
            0,
        )
        torch.testing.assert_close(
            scores, torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.int64)
        )


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
        super().__init__()
        # Hold something so named_children doesn't trip; we don't traverse.
        self._table_specs = table_specs
        # Add an inert EBC so the structure looks vaguely model-like
        self.ebc = nn.Module()

    def parameter_constraints(self, prefix=""):
        out = {}
        for name, dim in self._table_specs:
            options = _StubDynamicEmbOptions(dim=dim)
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
    # adagrad_optimizer -> target_opt_label resolves to "adagrad"
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

        # Save a synthetic Adagrad-style optimizer state -- a single
        # `state.<weight_fqn>.sum` tensor of shape (8, 4) filled with 0.5.
        opt_sd = {
            "state.model.embedding_group.embedding_group_impl_x.mc_ebc."
            "_embedding_module.embedding_bags.user_id.weight.sum": torch.full(
                (8, 4), 0.5
            ),
        }
        dcp.save(opt_sd, checkpoint_id=os.path.join(src_ckpt_dir, "optimizer"))
        return src_model

    def test_convert_writes_dynamicemb_shards_and_meta(self):
        src_ckpt = os.path.join(self.tmp, "model.ckpt-100")
        self._build_and_save_source(src_ckpt)

        # Stub source/target pipeline configs and model builders.
        src_cfg_path = os.path.join(self.tmp, "src.config")
        tgt_cfg_path = os.path.join(self.tmp, "tgt.config")
        config_util.save_message(_make_minimal_pipeline_config(), src_cfg_path)
        config_util.save_message(_make_minimal_pipeline_config(), tgt_cfg_path)

        # Source model is the MCH model we just saved; target model has the
        # paired dynamicemb-bound table.
        src_model = _TrainWrapperLike(
            _MiniModel(_build_mch_ebc([("user_id", 8, 4, "lfu")]))
        )
        tgt_inner = nn.Module()
        tgt_inner.embedding_group = nn.Module()
        tgt_inner.embedding_group.embedding_group_impl_x = _StubEmbeddingGroupImpl(
            [("user_id", 4)]
        )
        tgt_model = _TrainWrapperLike(tgt_inner)

        # When convert() calls _create_features then _create_model, we want
        # _create_model to return src_model on first call (for source
        # config) and tgt_model on second call (target config), each before
        # the converter wraps in TrainWrapper. Bypass TrainWrapper wrap by
        # also patching it to be a passthrough.
        models = iter([src_model.model, tgt_model.model])

        def passthrough(m):
            return _TrainWrapperLike(m)

        save_dir = os.path.join(self.tmp, "out")
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
                world_size=2,
                init_score_offset=0,
            )

        out = os.path.join(save_dir, "model.ckpt-0")
        # meta JSON.
        with open(os.path.join(out, "meta")) as f:
            meta = json.load(f)
        self.assertTrue(meta["load_model"])
        self.assertTrue(meta["load_optim"])
        self.assertTrue(meta["dynamicemb_load_optim"])
        self.assertEqual(
            list(meta["dynamicemb_load_table_names"].values())[0], ["user_id"]
        )
        mod_path = next(iter(meta["dynamicemb_load_table_names"]))
        # mod_path should be the doubled-"model." pattern the init script
        # also produces (DynamicEmbLoad's walker prepends "model").
        self.assertIn("ebc", mod_path)

        # model/ and optimizer/ were copied verbatim.
        self.assertTrue(os.path.isdir(os.path.join(out, "model")))
        self.assertTrue(os.path.isdir(os.path.join(out, "optimizer")))
        self.assertTrue(os.path.exists(os.path.join(out, "model", ".metadata")))

        # Dynamicemb shards exist for each rank, with content matching
        # raw_id % world_size assignment.
        shard_dir = os.path.join(out, "dynamicemb", mod_path)
        # 4 keys -> 100,101,102,103. world_size=2: even->rank0, odd->rank1.
        keys_r0 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_keys.rank_0.world_size_2")
        )
        keys_r1 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_keys.rank_1.world_size_2")
        )
        self.assertEqual(sorted(keys_r0.tolist()), [100, 102])
        self.assertEqual(sorted(keys_r1.tolist()), [101, 103])

        # Values for raw_id 100 -> remapped 3 -> weight row 3 -> all 3.0.
        # All keys are stored in roundrobin order matching the .keys file.
        values_r0 = _read_float32(
            os.path.join(shard_dir, "user_id_emb_values.rank_0.world_size_2")
        ).reshape(-1, 4)
        # Use the .keys file to know which row maps to which raw_id.
        for i, raw_id in enumerate(keys_r0.tolist()):
            expected_row = float({100: 3, 102: 1}[raw_id])
            np.testing.assert_array_equal(
                values_r0[i], np.full((4,), expected_row, dtype=np.float32)
            )

        # Scores: LFU -> LFU mapping copies _mch_counts at the valid slots.
        # raw_id 100 was at slot 0 with count=10, 101->20, 102->30, 103->40.
        scores_r0 = _read_int64(
            os.path.join(shard_dir, "user_id_emb_scores.rank_0.world_size_2")
        )
        # In the iter-order we wrote (which matches sorted_raw[valid_mask]),
        # rank0 holds entries where raw_id % 2 == 0.
        for i, raw_id in enumerate(keys_r0.tolist()):
            expected_count = {100: 10, 102: 30}[raw_id]
            self.assertEqual(int(scores_r0[i]), expected_count)

        # opt_values present, shape (n_valid, 4) (AdaGrad: opt_state_dim = dim).
        opt_path = os.path.join(shard_dir, "user_id_emb_opt_values.rank_0.world_size_2")
        self.assertTrue(os.path.exists(opt_path))
        opt_r0 = _read_float32(opt_path).reshape(-1, 4)
        # Source `sum` was all-0.5 so opt rows are all 0.5.
        np.testing.assert_array_almost_equal(opt_r0, np.full(opt_r0.shape, 0.5))


if __name__ == "__main__":
    unittest.main()
