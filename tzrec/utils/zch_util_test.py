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

import unittest

import torch
from torch import nn
from torchrec import JaggedTensor
from torchrec.modules.embedding_configs import EmbeddingBagConfig, EmbeddingConfig
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
    ManagedCollisionEmbeddingCollection,
)
from torchrec.modules.mc_modules import (
    LFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)

from tzrec.utils.zch_util import iter_zch_tables, register_post_zch_event_tracker_fn

_ZCH_SIZE = 4
_EMBEDDING_DIM = 8


def _make_mch(eviction_interval: int = 2) -> MCHManagedCollisionModule:
    return MCHManagedCollisionModule(
        zch_size=_ZCH_SIZE,
        device=torch.device("cpu"),
        eviction_policy=LFU_EvictionPolicy(),
        eviction_interval=eviction_interval,
    )


def _build_pooled_zch_model(mch: nn.Module) -> nn.Module:
    table = EmbeddingBagConfig(
        name="user_emb",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_ZCH_SIZE,
        feature_names=["user_id"],
    )
    model = nn.Module()
    model.mc_ebc = ManagedCollisionEmbeddingBagCollection(
        EmbeddingBagCollection(tables=[table], device=torch.device("meta")),
        ManagedCollisionCollection({"user_emb": mch}, [table]),
    )
    return model


def _build_sequence_zch_model(mch: nn.Module) -> nn.Module:
    table = EmbeddingConfig(
        name="item_emb",
        embedding_dim=_EMBEDDING_DIM,
        num_embeddings=_ZCH_SIZE,
        feature_names=["item_id"],
    )
    model = nn.Module()
    model.mc_ec_dict = nn.ModuleDict(
        {
            str(_EMBEDDING_DIM): ManagedCollisionEmbeddingCollection(
                EmbeddingCollection(tables=[table], device=torch.device("meta")),
                ManagedCollisionCollection({"item_emb": mch}, [table]),
            )
        }
    )
    return model


def _profile(mch: MCHManagedCollisionModule, ids: list) -> None:
    mch.profile(
        {
            "feat": JaggedTensor(
                values=torch.tensor(ids, dtype=torch.int64),
                lengths=torch.tensor([len(ids)], dtype=torch.int64),
            )
        }
    )


class IterZchTablesTest(unittest.TestCase):
    def test_finds_pooled_table(self):
        mch = _make_mch()
        model = _build_pooled_zch_model(mch)
        tables = list(iter_zch_tables(model))
        self.assertEqual(len(tables), 1)
        zch_table = tables[0]
        self.assertEqual(zch_table.wrapper_fqn, "mc_ebc")
        self.assertEqual(zch_table.inner_kind, "embedding_bags")
        self.assertEqual(zch_table.table_name, "user_emb")
        self.assertIs(zch_table.mc_module, mch)
        self.assertIs(zch_table.inner, model.mc_ebc._embedding_module)
        self.assertEqual(
            zch_table.table_fqn, "mc_ebc._embedding_module.embedding_bags.user_emb"
        )
        self.assertEqual(
            zch_table.mc_module_fqn,
            "mc_ebc._managed_collision_collection._managed_collision_modules.user_emb",
        )

    def test_finds_sequence_table(self):
        mch = _make_mch()
        model = _build_sequence_zch_model(mch)
        tables = list(iter_zch_tables(model))
        self.assertEqual(len(tables), 1)
        zch_table = tables[0]
        self.assertEqual(zch_table.wrapper_fqn, f"mc_ec_dict.{_EMBEDDING_DIM}")
        self.assertEqual(zch_table.inner_kind, "embeddings")
        self.assertEqual(
            zch_table.table_fqn,
            f"mc_ec_dict.{_EMBEDDING_DIM}._embedding_module.embeddings.item_emb",
        )

    def test_yields_non_mch_managed_collision_module(self):
        # Callers own the policy for unsupported managed collision modules, so
        # discovery must surface them rather than filter them out.
        mch = _make_mch()
        model = _build_pooled_zch_model(mch)
        model.mc_ebc._managed_collision_collection._managed_collision_modules[
            "user_emb"
        ] = nn.Module()
        tables = list(iter_zch_tables(model))
        self.assertEqual(len(tables), 1)
        self.assertNotIsInstance(tables[0].mc_module, MCHManagedCollisionModule)

    def test_skips_model_without_managed_collision(self):
        model = nn.Module()
        model.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name="user_emb",
                    embedding_dim=_EMBEDDING_DIM,
                    num_embeddings=_ZCH_SIZE,
                    feature_names=["user_id"],
                )
            ],
            device=torch.device("meta"),
        )
        self.assertEqual(list(iter_zch_tables(model)), [])


class ZchEventTrackerFnTest(unittest.TestCase):
    def _register(self, eviction_interval=1):
        mch = _make_mch(eviction_interval=eviction_interval)
        events = []
        register_post_zch_event_tracker_fn(
            mch,
            lambda _module, evicted, admitted: events.append(
                (sorted(evicted.tolist()), sorted(admitted.tolist()))
            ),
        )
        return mch, events

    def test_records_admission_into_free_slot(self):
        mch, events = self._register()
        # zch_size=4 leaves 3 usable slots, so both ids land in free slots and
        # the evicted side is the empty-slot delimiter, not a raw id.
        _profile(mch, [10, 10, 11])
        self.assertEqual(len(events), 1)
        evicted, admitted = events[0]
        self.assertEqual(admitted, [10, 11])
        self.assertEqual(evicted, [torch.iinfo(torch.int64).max] * 2)

    def test_records_eviction_with_its_replacement(self):
        mch, events = self._register()
        _profile(mch, [10, 10, 11])
        _profile(mch, [20, 20])
        # 30 (count 3) outscores 11 (LFU count 1) and takes its slot.
        _profile(mch, [30, 30, 30])
        self.assertEqual(events[-1], ([11], [30]))

    def test_reregister_overrides_without_double_wrapping(self):
        mch, events = self._register()
        other = []
        with self.assertLogs("tzrec", level="WARNING"):
            register_post_zch_event_tracker_fn(
                mch, lambda _module, evicted, admitted: other.append(admitted.tolist())
            )
        _profile(mch, [10, 10, 11])
        self.assertEqual(events, [])
        self.assertEqual(len(other), 1)


if __name__ == "__main__":
    unittest.main()
