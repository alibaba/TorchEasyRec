# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from typing import Dict, List
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch import nn
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharding_spec import EnumerableShardingSpec
from torch.distributed.tensor import DTensor
from torchrec import KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel, ShardingEnv
from torchrec.distributed.embedding import EmbeddingCollectionSharder
from torchrec.distributed.embedding_lookup import GroupedPooledEmbeddingsLookup
from torchrec.distributed.embedding_types import (
    EmbeddingComputeKernel,
    GroupedEmbeddingConfig,
    ShardedEmbeddingTable,
)
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.mc_embeddingbag import (
    ManagedCollisionEmbeddingBagCollectionSharder,
)
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.sharding_plan import (
    construct_module_sharding_plan,
    row_wise,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import (
    ParameterSharding,
    ShardingPlan,
    ShardingType,
)
from torchrec.modules.embedding_configs import (
    EmbeddingBagConfig,
    EmbeddingConfig,
    PoolingType,
)
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)
from torchrec.modules.mc_embedding_modules import (
    ManagedCollisionEmbeddingBagCollection,
)
from torchrec.modules.mc_modules import (
    LFU_EvictionPolicy,
    ManagedCollisionCollection,
    MCHManagedCollisionModule,
)
from torchrec.types import DataType

from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.tests import utils as test_utils
from tzrec.utils import config_util
from tzrec.utils.delta_embedding_dump import (
    _DELTA_DUMP_SCHEMA,
    DeltaEmbeddingDumper,
    ModelDeltaTracker,
    _local_table_weight,
    _table_shard_info_from_config,
    _TableShardInfo,
    _TableWeight,
    _validate_table_shard_info,
    validate_delta_embedding_dump_config,
)
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.test_util import gpu_unavailable, make_test_dir, mark_ci_scope

_SHARDED_TABLE_NAME = "table_1"
_SHARDED_FEATURE_NAME = "feature_1"
_SHARDED_NUM_EMBEDDINGS = 16
_SHARDED_EMBEDDING_DIM = 4
_SHARDED_INPUT_IDS = [0, 2, 8, 9, 15]
_SHARED_TABLE_NAME = "shared_table"
_SHARED_EBC_FEATURE_NAME = "deep_feature"
_SHARED_EC_FEATURE_NAME = "sequence_feature"
_SHARED_EBC_INPUT_IDS = [1, 2]
_SHARED_EC_INPUT_IDS = [5, 6]
_SHARED_EBC_EMBEDDING_DIM = 4
_SHARED_EC_EMBEDDING_DIM = 8


class _DeltaDumpEBCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=_SHARDED_TABLE_NAME,
                    num_embeddings=_SHARDED_NUM_EMBEDDINGS,
                    embedding_dim=_SHARDED_EMBEDDING_DIM,
                    feature_names=[_SHARDED_FEATURE_NAME],
                    pooling=PoolingType.SUM,
                )
            ],
            device=torch.device("meta"),
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        return self.ebc(features).values()


class _SharedTableECAndEBCModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ebc = EmbeddingBagCollection(
            tables=[
                EmbeddingBagConfig(
                    name=_SHARED_TABLE_NAME,
                    num_embeddings=_SHARDED_NUM_EMBEDDINGS,
                    embedding_dim=_SHARED_EBC_EMBEDDING_DIM,
                    feature_names=[_SHARED_EBC_FEATURE_NAME],
                    pooling=PoolingType.SUM,
                )
            ],
            device=torch.device("meta"),
        )
        self.ec = EmbeddingCollection(
            tables=[
                EmbeddingConfig(
                    name=_SHARED_TABLE_NAME,
                    num_embeddings=_SHARDED_NUM_EMBEDDINGS,
                    embedding_dim=_SHARED_EC_EMBEDDING_DIM,
                    feature_names=[_SHARED_EC_FEATURE_NAME],
                )
            ],
            device=torch.device("meta"),
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        pooled = self.ebc(features).values().sum()
        sequence = self.ec(features)[_SHARED_EC_FEATURE_NAME].values().sum()
        return pooled + sequence


class _FakeDynamicTables:
    def __init__(self) -> None:
        self.ids = None
        self.table_ids = None
        self.copy_mode = None

    def find(self, ids, table_ids, copy_mode):
        self.ids = ids.detach().clone()
        self.table_ids = table_ids.detach().clone()
        self.copy_mode = copy_mode
        founds = torch.tensor([True, False, True], device=ids.device)
        values = torch.tensor(
            [
                [1.0, 2.0, 20.0],
                [3.0, 4.0, 40.0],
                [5.0, 6.0, 60.0],
            ],
            device=ids.device,
        )
        return None, None, None, None, None, founds, None, values


def _build_sharded_delta_dump_model(rank: int, world_size: int, ctx):
    torch.manual_seed(2026)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model = _DeltaDumpEBCModel()
    constraints = {
        _SHARDED_TABLE_NAME: ParameterConstraints(
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=[EmbeddingComputeKernel.FUSED.value],
            feature_names=[_SHARDED_FEATURE_NAME],
            pooling_factors=[1.0],
        )
    }
    planner = EmbeddingShardingPlanner(
        topology=Topology(world_size, "cuda"),
        constraints=constraints,
    )
    sharders = [
        EmbeddingBagCollectionSharder(
            fused_params={"optimizer": OptimType.EXACT_ROWWISE_ADAGRAD}
        )
    ]
    plan = planner.collective_plan(model, sharders, ctx.pg)
    return DistributedModelParallel(
        module=model,
        device=device,
        env=ShardingEnv.from_process_group(ctx.pg),
        plan=plan,
        sharders=sharders,
    )


def _sharded_features(rank: int) -> KeyedJaggedTensor:
    device = torch.device(f"cuda:{rank}")
    return KeyedJaggedTensor.from_offsets_sync(
        keys=[_SHARDED_FEATURE_NAME],
        values=torch.tensor(_SHARDED_INPUT_IDS, device=device, dtype=torch.int64),
        offsets=torch.tensor([0, len(_SHARDED_INPUT_IDS)], device=device),
    )


def _shared_table_features(rank: int) -> KeyedJaggedTensor:
    device = torch.device(f"cuda:{rank}")
    values = _SHARED_EBC_INPUT_IDS + _SHARED_EC_INPUT_IDS
    return KeyedJaggedTensor.from_offsets_sync(
        keys=[_SHARED_EBC_FEATURE_NAME, _SHARED_EC_FEATURE_NAME],
        values=torch.tensor(values, device=device, dtype=torch.int64),
        offsets=torch.tensor(
            [0, len(_SHARED_EBC_INPUT_IDS), len(values)],
            device=device,
            dtype=torch.int64,
        ),
    )


def _assert_sharded_dump_file(rank: int, output_path: str, dumper) -> None:
    testcase = unittest.TestCase()
    testcase.assertTrue(os.path.exists(output_path))
    table = pq.read_table(output_path)
    testcase.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
    testcase.assertEqual(table["rank"].to_pylist(), [rank] * table.num_rows)
    testcase.assertEqual(table["world_size"].to_pylist(), [2] * table.num_rows)
    testcase.assertEqual(
        set(table["feature_name"].to_pylist()), {_SHARDED_FEATURE_NAME}
    )
    testcase.assertEqual(set(table["source"].to_pylist()), {"model_delta_tracker"})

    table_weight = dumper._collect_table_weights()[
        f"ebc.embedding_bags.{_SHARDED_TABLE_NAME}"
    ]
    expected_key_ids = [
        key_id
        for key_id in _SHARDED_INPUT_IDS
        if table_weight.shard_info.row_offset
        <= key_id
        < table_weight.shard_info.row_offset + table_weight.shard_info.local_rows
    ]
    actual_key_ids = table["key_id"].to_pylist()
    testcase.assertEqual(sorted(actual_key_ids), expected_key_ids)
    testcase.assertTrue(
        all(key_id >= table_weight.shard_info.row_offset for key_id in actual_key_ids)
    )

    actual_ids = torch.tensor(actual_key_ids, dtype=torch.int64)
    sort_order = torch.argsort(actual_ids)
    local_ids = actual_ids[sort_order] - table_weight.shard_info.row_offset
    actual_embeddings = torch.tensor(
        table["embedding"].to_pylist(), dtype=torch.float32
    )
    expected_embeddings = table_weight.tensor[local_ids.to(table_weight.tensor.device)]
    torch.testing.assert_close(
        actual_embeddings[sort_order],
        expected_embeddings.detach().cpu().to(torch.float32),
    )


def _run_sharded_delta_embedding_dump(rank: int, world_size: int, output_dir: str):
    with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
        model = _build_sharded_delta_dump_model(rank, world_size, ctx)
        dumper = DeltaEmbeddingDumper(
            model,
            DeltaEmbeddingDumpConfig(
                dump_interval_steps=1,
                output_dir=output_dir,
                file_prefix="delta",
            ),
            output_dir,
            torch.device(f"cuda:{rank}"),
        )
        output = model(_sharded_features(rank))
        output.sum().backward()
        output_path = dumper.dump(50)
        unittest.TestCase().assertIsNotNone(output_path)
        _assert_sharded_dump_file(rank, output_path, dumper)
        torch.distributed.barrier()


def _run_shared_table_fqn_delta_embedding_dump(
    rank: int, world_size: int, output_dir: str
):
    with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        model = _SharedTableECAndEBCModel()
        sharders = [
            EmbeddingBagCollectionSharder(),
            EmbeddingCollectionSharder(),
        ]
        planner = EmbeddingShardingPlanner(
            topology=Topology(world_size, "cuda"),
        )
        plan = planner.collective_plan(model, sharders, ctx.pg)
        sharded_model = DistributedModelParallel(
            module=model,
            device=device,
            env=ShardingEnv.from_process_group(ctx.pg),
            plan=plan,
            sharders=sharders,
        )
        dumper = DeltaEmbeddingDumper(
            sharded_model,
            DeltaEmbeddingDumpConfig(
                dump_interval_steps=1,
                output_dir=output_dir,
                file_prefix="delta",
            ),
            output_dir,
            device,
        )

        sharded_model(_shared_table_features(rank)).backward()
        output_path = dumper.dump(1)
        testcase = unittest.TestCase()
        testcase.assertIsNotNone(output_path)
        table = pq.read_table(output_path)

        ebc_fqn = f"ebc.embedding_bags.{_SHARED_TABLE_NAME}"
        ec_fqn = f"ec.embeddings.{_SHARED_TABLE_NAME}"
        testcase.assertEqual(set(table["table_fqn"].to_pylist()), {ebc_fqn, ec_fqn})
        testcase.assertEqual(
            set(dumper._tracker.fqn_to_feature_names),
            {ebc_fqn, ec_fqn},
        )
        table_weights = dumper._collect_table_weights()
        testcase.assertEqual(set(table_weights), {ebc_fqn, ec_fqn})
        testcase.assertEqual(set(dumper._table_shard_infos), {ebc_fqn, ec_fqn})
        testcase.assertEqual(
            dumper._table_shard_infos[ebc_fqn].global_cols,
            _SHARED_EBC_EMBEDDING_DIM,
        )
        testcase.assertEqual(
            dumper._table_shard_infos[ec_fqn].global_cols,
            _SHARED_EC_EMBEDDING_DIM,
        )

        expected_ids = {
            ebc_fqn: _SHARED_EBC_INPUT_IDS,
            ec_fqn: _SHARED_EC_INPUT_IDS,
        }
        expected_features = {
            ebc_fqn: _SHARED_EBC_FEATURE_NAME,
            ec_fqn: _SHARED_EC_FEATURE_NAME,
        }
        for table_fqn in (ebc_fqn, ec_fqn):
            owner_rows = table.filter(pa.compute.equal(table["table_fqn"], table_fqn))
            key_ids = owner_rows["key_id"].to_pylist()
            testcase.assertEqual(key_ids, expected_ids[table_fqn])
            testcase.assertEqual(
                set(owner_rows["feature_name"].to_pylist()),
                {expected_features[table_fqn]},
            )
            expected_embeddings = (
                table_weights[table_fqn]
                .tensor[torch.tensor(key_ids, device=device)]
                .detach()
                .cpu()
                .to(torch.float32)
                .tolist()
            )
            testcase.assertEqual(
                owner_rows["embedding"].to_pylist(),
                expected_embeddings,
            )


_ZCH_TABLE_NAME = "zch_table"
_ZCH_FEATURE_NAME = "zch_feat"
_ZCH_SIZE = 8
_ZCH_EMBEDDING_DIM = 4
_ZCH_INPUT_IDS = [123456, 789012, 555555]


class _DeltaDumpZchEBCModel(nn.Module):
    def __init__(self, eviction_policy=None, eviction_interval: int = 2) -> None:
        super().__init__()
        tables = [
            EmbeddingBagConfig(
                name=_ZCH_TABLE_NAME,
                num_embeddings=_ZCH_SIZE,
                embedding_dim=_ZCH_EMBEDDING_DIM,
                feature_names=[_ZCH_FEATURE_NAME],
                pooling=PoolingType.SUM,
            )
        ]
        self._mc_ebc = ManagedCollisionEmbeddingBagCollection(
            EmbeddingBagCollection(tables=tables, device=torch.device("meta")),
            ManagedCollisionCollection(
                {
                    _ZCH_TABLE_NAME: MCHManagedCollisionModule(
                        zch_size=_ZCH_SIZE,
                        device=torch.device("meta"),
                        eviction_policy=eviction_policy or LFU_EvictionPolicy(),
                        eviction_interval=eviction_interval,
                    )
                },
                tables,
            ),
        )

    def forward(self, features: KeyedJaggedTensor) -> torch.Tensor:
        # The sharded MC-EBC returns (embeddings, remapped_features_or_None).
        output = self._mc_ebc(features)
        if isinstance(output, tuple):
            output = output[0]
        return output.values()


def _build_sharded_zch_delta_dump_model(
    rank: int,
    world_size: int,
    ctx,
    eviction_policy=None,
    eviction_interval: int = 2,
):
    torch.manual_seed(2026)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    model = _DeltaDumpZchEBCModel(
        eviction_policy=eviction_policy, eviction_interval=eviction_interval
    )
    sharder = ManagedCollisionEmbeddingBagCollectionSharder(
        ebc_sharder=EmbeddingBagCollectionSharder(
            fused_params={"optimizer": OptimType.EXACT_ROWWISE_ADAGRAD}
        )
    )
    module_sharding_plan = construct_module_sharding_plan(
        model._mc_ebc,
        per_param_sharding={
            _ZCH_TABLE_NAME: row_wise(compute_kernel=EmbeddingComputeKernel.FUSED.value)
        },
        world_size=world_size,
        device_type="cuda",
        sharder=sharder,
    )
    return DistributedModelParallel(
        module=model,
        device=device,
        env=ShardingEnv.from_process_group(ctx.pg),
        plan=ShardingPlan({"_mc_ebc": module_sharding_plan}),
        sharders=[sharder],
    )


def _run_zch_delta_embedding_dump(rank: int, world_size: int, output_dir: str):
    testcase = unittest.TestCase()
    with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
        device = torch.device(f"cuda:{rank}")
        model = _build_sharded_zch_delta_dump_model(rank, world_size, ctx)
        # tzrec's patched mc-ebc input_dist lazy-initializes on the first
        # forward gated on this flag; tzrec's train wrapper sets it, but this
        # test builds the DMP module directly.
        model.module._mc_ebc._embedding_module._has_uninitialized_input_dist = True
        dumper = DeltaEmbeddingDumper(
            model,
            DeltaEmbeddingDumpConfig(
                dump_interval_steps=10,
                output_dir=output_dir,
                file_prefix="delta",
            ),
            output_dir,
            device,
        )
        testcase.assertEqual(
            set(dumper._zch_modules),
            {f"_mc_ebc._embedding_module.embedding_bags.{_ZCH_TABLE_NAME}"},
        )
        features = KeyedJaggedTensor.from_offsets_sync(
            keys=[_ZCH_FEATURE_NAME],
            values=torch.tensor(_ZCH_INPUT_IDS, device=device, dtype=torch.int64),
            offsets=torch.tensor([0, len(_ZCH_INPUT_IDS)], device=device),
        )
        # The ZCH module coalesces its profiled ids on the second forward, so
        # the first step hashes ids to the fallback row and only the second
        # step's lookup resolves them to real ZCH slots. maybe_dump advances
        # the tracker step without dumping (interval 10 > 3 steps), mirroring
        # the training loop's forward/dump cadence.
        for step in range(1, 4):
            model(features).sum().backward()
            dumper.maybe_dump(step)
        # The per-forward post-odist callback keeps every completed batch
        # compacted; the trailing batch is compacted only by the next forward
        # or the dump's get_unique.
        testcase.assertEqual(
            dumper._tracker.curr_compact_index,
            dumper._tracker.curr_batch_idx - 1,
        )
        output_path = dumper.dump(50)
        testcase.assertIsNotNone(output_path)

        table = pq.read_table(output_path)
        testcase.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
        key_ids = sorted(table["key_id"].to_pylist())
        testcase.assertTrue(set(key_ids).issubset(set(_ZCH_INPUT_IDS)))
        # Recompute the expected rows straight from the ZCH buffers with a
        # forward searchsorted lookup, independent of the dump's inverse map.
        table_fqn = f"_mc_ebc._embedding_module.embedding_bags.{_ZCH_TABLE_NAME}"
        mch = dumper._zch_modules[table_fqn]
        weight = dumper._collect_table_weights()[table_fqn].tensor.to(device)
        raw_ids = torch.tensor(key_ids, device=device, dtype=torch.int64)
        slots = torch.searchsorted(mch._mch_sorted_raw_ids, raw_ids)
        rows = mch._mch_remapped_ids_mapping[slots] - mch._output_global_offset
        expected = weight[rows.long()].detach().cpu().to(torch.float32)
        order = torch.argsort(raw_ids).cpu()
        actual = torch.tensor(
            table["embedding"].to_pylist(), dtype=torch.float32
        ).reshape(-1, _ZCH_EMBEDDING_DIM)
        torch.testing.assert_close(actual[order], expected)

        # Row-wise sharding splits the ZCH slots across ranks, so every input
        # id is dumped by exactly one rank; the union of shards covers all.
        torch.distributed.barrier()
        dumped_ids: set = set()
        for shard in glob.glob(os.path.join(output_dir, "step_50", "*.parquet")):
            dumped_ids.update(pq.read_table(shard)["key_id"].to_pylist())
        testcase.assertEqual(dumped_ids, set(_ZCH_INPUT_IDS))
        torch.distributed.barrier()


# ZCH_SIZE=8 leaves 7 usable rows (the last row is the fallback row), so
# exactly one batch of 7 ids fills the table. LFU admission breaks count
# ties by ascending raw id, so the low X ids win slots over the Y ids and
# the B ids (count 2) evict the X ids (count 1) deterministically.
_ZCH_LIFECYCLE_IDS_X = [11, 12, 13, 14, 15, 16, 17]
_ZCH_LIFECYCLE_IDS_Y = [21, 22, 23, 24, 25, 26, 27]
_ZCH_LIFECYCLE_IDS_B = [31, 32, 33, 34, 35, 36, 37]


def _zch_kjt(rank: int, ids: List[int]) -> KeyedJaggedTensor:
    device = torch.device(f"cuda:{rank}")
    return KeyedJaggedTensor.from_offsets_sync(
        keys=[_ZCH_FEATURE_NAME],
        values=torch.tensor(ids, device=device, dtype=torch.int64),
        offsets=torch.tensor([0, len(ids)], device=device),
    )


def _read_dump_keys(output_dir: str, step: int) -> Dict[int, torch.Tensor]:
    rows_by_key: Dict[int, torch.Tensor] = {}
    shards = glob.glob(
        os.path.join(output_dir, f"delta_step_{step}.parquet")
    ) + glob.glob(
        os.path.join(output_dir, "step_*", f"delta_step_{step}_rank_*.parquet")
    )
    for shard in shards:
        table = pq.read_table(shard)
        for key_id, embedding in zip(
            table["key_id"].to_pylist(), table["embedding"].to_pylist()
        ):
            rows_by_key[int(key_id)] = torch.tensor(embedding, dtype=torch.float32)
    return rows_by_key


def _run_zch_lifecycle_delta_embedding_dump(
    rank: int, world_size: int, output_dir: str
):
    """Drive one admission cycle and one eviction cycle through the dump.

    With eviction_interval=2 the coalesce runs on even forwards. Step 1 looks
    up X (hits the fallback row, only profiled); step 2 looks up Y, whose
    coalesce admits X (same count, lower ids win) but not Y, so X must be
    published by dump 1 although it was never looked up after admission.
    Steps 3-4 look up B twice; the step-4 coalesce evicts X (count 1 < B's
    count 2) and admits B, so dump 2 must publish B's trained rows and X's
    fallback row.
    """
    testcase = unittest.TestCase()
    testcase.assertEqual(world_size, 1)
    with MultiProcessContext(rank=rank, world_size=world_size, backend="nccl") as ctx:
        device = torch.device(f"cuda:{rank}")
        model = _build_sharded_zch_delta_dump_model(rank, world_size, ctx)
        model.module._mc_ebc._embedding_module._has_uninitialized_input_dist = True
        dumper = DeltaEmbeddingDumper(
            model,
            DeltaEmbeddingDumpConfig(
                dump_interval_steps=10,
                output_dir=output_dir,
                file_prefix="delta",
            ),
            output_dir,
            device,
        )
        for step, ids in ((1, _ZCH_LIFECYCLE_IDS_X), (2, _ZCH_LIFECYCLE_IDS_Y)):
            model(_zch_kjt(rank, ids)).sum().backward()
            dumper.maybe_dump(step)
        dumper.dump(10)
        keys_dump1 = _read_dump_keys(output_dir, 10)
        testcase.assertEqual(set(keys_dump1), set(_ZCH_LIFECYCLE_IDS_X))

        for step, ids in ((3, _ZCH_LIFECYCLE_IDS_B), (4, _ZCH_LIFECYCLE_IDS_B)):
            model(_zch_kjt(rank, ids)).sum().backward()
            dumper.maybe_dump(step)
        dumper.dump(20)
        keys_dump2 = _read_dump_keys(output_dir, 20)
        testcase.assertEqual(
            set(keys_dump2), set(_ZCH_LIFECYCLE_IDS_X + _ZCH_LIFECYCLE_IDS_B)
        )

        table_fqn = f"_mc_ebc._embedding_module.embedding_bags.{_ZCH_TABLE_NAME}"
        mch = dumper._zch_modules[table_fqn]
        weight = dumper._collect_table_weights()[table_fqn].tensor.to(device)
        fallback_row = weight[mch._zch_size - 1].detach().cpu().to(torch.float32)
        for key_id in _ZCH_LIFECYCLE_IDS_X:
            torch.testing.assert_close(keys_dump2[key_id], fallback_row)
        raw_ids = torch.tensor(_ZCH_LIFECYCLE_IDS_B, device=device, dtype=torch.int64)
        slots = torch.searchsorted(mch._mch_sorted_raw_ids, raw_ids)
        rows = mch._mch_remapped_ids_mapping[slots] - mch._output_global_offset
        expected_b = weight[rows.long()].detach().cpu().to(torch.float32)
        actual_b = torch.stack([keys_dump2[key_id] for key_id in _ZCH_LIFECYCLE_IDS_B])
        torch.testing.assert_close(actual_b, expected_b)
        torch.distributed.barrier()


class DeltaEmbeddingDumpValidationTest(unittest.TestCase):
    def test_missing_config_skips_runtime_validation(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            validate_delta_embedding_dump_config(None, torch.device("cpu"))

    def test_present_config_allows_multi_gpu_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=10)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            validate_delta_embedding_dump_config(config, torch.device("cuda:0"))

    def test_present_config_requires_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=10)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_delta_embedding_dump_config(config, torch.device("cpu"))

    def test_present_config_requires_positive_interval(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=0)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "dump_interval_steps"):
                validate_delta_embedding_dump_config(config, torch.device("cuda:0"))

    def test_present_config_accepts_minutes_interval(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_minutes=5)
        validate_delta_embedding_dump_config(config, torch.device("cuda:0"))

    def test_present_config_requires_positive_minutes_interval(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_minutes=0)
        with self.assertRaisesRegex(ValueError, "dump_interval_minutes"):
            validate_delta_embedding_dump_config(config, torch.device("cuda:0"))

    def test_init_validates_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=10)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                DeltaEmbeddingDumper(
                    torch.nn.Module(),
                    config,
                    tmp_dir,
                    torch.device("cpu"),
                )

    def _build_zch_collect_model(self, mch_modules):
        inner_module = torch.nn.Module()
        inner_module._table_name_to_config = dict.fromkeys(mch_modules)
        wrapper = torch.nn.Module()
        wrapper._managed_collision_collection = SimpleNamespace(
            _managed_collision_modules=mch_modules
        )
        wrapper._embedding_module = inner_module
        model = torch.nn.Module()
        model.mc_ebc = wrapper
        return model, inner_module

    def test_collect_zch_modules_maps_tracked_table_fqns(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        mch = MCHManagedCollisionModule(
            zch_size=4,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=2,
        )
        model, inner_module = self._build_zch_collect_model({"user_emb": mch})
        dumper._model = model
        table_fqn = "mc_ebc._embedding_module.embedding_bags.user_emb"
        dumper._tracker = SimpleNamespace(
            fqn_to_feature_names={table_fqn: ["user_id"]},
            tracked_modules={"mc_ebc._embedding_module": inner_module},
        )
        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.embedding_bags.{table_name}"
            ),
        ):
            self.assertEqual(dumper._collect_zch_modules(), {table_fqn: mch})

    def test_collect_zch_modules_fails_fast_on_unresolved_table(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        mch = MCHManagedCollisionModule(
            zch_size=4,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=2,
        )
        model, inner_module = self._build_zch_collect_model({"user_emb": mch})
        # The tracker also owns a second table of the same ZCH inner module
        # that the MCH collection does not cover; dumping it plain would emit
        # remapped rows as serving keys.
        inner_module._table_name_to_config["stray_emb"] = None
        dumper._model = model
        dumper._tracker = SimpleNamespace(
            fqn_to_feature_names={
                "mc_ebc._embedding_module.embedding_bags.user_emb": ["user_id"],
                "mc_ebc._embedding_module.embedding_bags.stray_emb": ["stray_id"],
            },
            tracked_modules={"mc_ebc._embedding_module": inner_module},
        )
        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.embedding_bags.{table_name}"
            ),
        ):
            with self.assertRaisesRegex(ValueError, "stray_emb"):
                dumper._collect_zch_modules()

    def test_collect_zch_modules_rejects_non_mch_module(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        wrapper = torch.nn.Module()
        wrapper._managed_collision_collection = SimpleNamespace(
            _managed_collision_modules={"user_emb": torch.nn.Module()}
        )
        wrapper._embedding_module = torch.nn.Module()
        model = torch.nn.Module()
        model.mc_ebc = wrapper
        dumper._model = model
        dumper._tracker = SimpleNamespace(
            fqn_to_feature_names={
                "mc_ebc._embedding_module.embedding_bags.user_emb": ["user_id"]
            }
        )
        with self.assertRaisesRegex(TypeError, "MCHManagedCollisionModule"):
            dumper._collect_zch_modules()

    def test_row_wise_shard_info_uses_row_offset(self):
        table_config = ShardedEmbeddingTable(
            local_rows=16,
            local_cols=8,
            num_embeddings=64,
            embedding_dim=8,
            name="user_emb",
            local_metadata=ShardMetadata(
                shard_offsets=[32, 0],
                shard_sizes=[16, 8],
                placement="rank:1/cuda:1",
            ),
        )
        shard_info = _table_shard_info_from_config(table_config)
        _validate_table_shard_info("user_emb", shard_info)
        self.assertEqual(shard_info.row_offset, 32)
        self.assertEqual(shard_info.global_cols, 8)

    def test_column_wise_shard_info_fails_fast(self):
        table_config = ShardedEmbeddingTable(
            local_rows=64,
            local_cols=4,
            num_embeddings=64,
            embedding_dim=8,
            name="user_emb",
            local_metadata=ShardMetadata(
                shard_offsets=[0, 4],
                shard_sizes=[64, 4],
                placement="rank:0/cuda:0",
            ),
        )
        shard_info = _table_shard_info_from_config(table_config)
        with self.assertRaisesRegex(ValueError, "column-wise"):
            _validate_table_shard_info("user_emb", shard_info)

    def test_dump_rows_include_rank_metadata(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 1
        dumper._world_size = 4
        table_chunks = []
        num_rows = dumper._append_table_chunk(
            table_chunks,
            global_step=10,
            feature_name="user_id",
            table_fqn="model.ebc.user_emb",
            key_ids=torch.tensor([42]),
            embeddings=torch.tensor([[1.0, 2.0]]),
            source="model_delta_tracker",
        )
        self.assertEqual(num_rows, 1)
        self.assertEqual(len(table_chunks), 1)
        table = table_chunks[0]
        self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
        self.assertEqual(table["rank"].to_pylist(), [1])
        self.assertEqual(table["world_size"].to_pylist(), [4])
        self.assertEqual(table["key_id"].to_pylist(), [42])
        self.assertEqual(table["embedding"].to_pylist(), [[1.0, 2.0]])

    def test_write_table_chunks_preserves_parquet_schema(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 0
        dumper._world_size = 1
        table_chunks = []
        dumper._append_table_chunk(
            table_chunks,
            global_step=5,
            feature_name="user_id",
            table_fqn="model.ebc.user_emb",
            key_ids=torch.tensor([7, 8]),
            embeddings=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            source="model_delta_tracker",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "delta.parquet")
            dumper._write_table_chunks(table_chunks, output_path)
            table = pq.read_table(output_path)

        self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
        self.assertEqual(table["key_id"].to_pylist(), [7, 8])
        self.assertEqual(table["embedding"].type, pa.list_(pa.float32()))
        self.assertEqual(table["embedding"].to_pylist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_write_empty_table_chunks_preserves_parquet_schema(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 0
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "delta.parquet")
            dumper._write_table_chunks([], output_path)
            table = pq.read_table(output_path)

        self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
        self.assertEqual(table.num_rows, 0)

    def test_write_table_chunks_leaves_no_partial_shard_on_error(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 0
        writer = mock.MagicMock()
        writer.__enter__.return_value = writer
        writer.write_table.side_effect = RuntimeError("boom mid-write")
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "delta.parquet")
            # The temp file is created by ParquetWriter before the write fails;
            # the error handler must remove it so the dir is left clean.
            open(f"{output_path}.rank0.tmp", "w").close()
            with mock.patch.object(pq, "ParquetWriter", return_value=writer):
                with self.assertRaises(RuntimeError):
                    dumper._write_table_chunks([mock.MagicMock()], output_path)
            # Neither the canonical shard nor the temp file should survive, so
            # a downstream glob(*.parquet) never observes a truncated write.
            self.assertEqual(os.listdir(tmp_dir), [])

    def test_final_dump_skips_boundary_step_to_avoid_overwrite(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = 50
        dumper._interval_secs = None
        dumper._last_dump_step = None
        dumper._world_size = 1
        with mock.patch.object(dumper, "dump") as dump_mock:
            # Boundary steps were already written by maybe_dump; skip them so a
            # trailing empty shard never overwrites the real one.
            self.assertIsNone(dumper.final_dump(50))
            self.assertIsNone(dumper.final_dump(100))
            dump_mock.assert_not_called()

            # Step 0 is not publishable; final_dump returns early. A positive
            # trailing partial interval must still be flushed.
            self.assertIsNone(dumper.final_dump(0))
            dumper.final_dump(73)
            self.assertEqual(
                [call.args[0] for call in dump_mock.call_args_list],
                [73],
            )

    def test_final_dump_syncs_step_across_ranks_before_flush(self):
        # A lagging rank reaches final_dump at a boundary step (50) while the
        # furthest rank stopped at 73. Without syncing, the lagging rank would
        # skip and write no shard, leaving step_73/ ragged. The MAX all_reduce
        # lifts every rank to 73 so all take the same dump-into-step_73 path.
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = 50
        dumper._interval_secs = None
        dumper._last_dump_step = None
        dumper._world_size = 2

        def fake_all_reduce(tensor, op=None):
            self.assertIs(op, torch.distributed.ReduceOp.MAX)
            tensor.fill_(73)

        with (
            mock.patch.object(dumper, "dump") as dump_mock,
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch(
                "torch.tensor",
                side_effect=lambda *a, **k: torch.zeros(1, dtype=torch.long),
            ),
            mock.patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
        ):
            dumper.final_dump(50)
        dump_mock.assert_called_once_with(73)

    def test_final_dump_skips_step_already_dumped_by_time_interval(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = None
        dumper._interval_secs = 60.0
        dumper._last_dump_step = 73
        dumper._world_size = 1
        with mock.patch.object(dumper, "dump") as dump_mock:
            self.assertIsNone(dumper.final_dump(73))
        dump_mock.assert_not_called()

    def test_maybe_dump_uses_checkpoint_aligned_global_step(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = 50
        dumper._interval_secs = None
        dumper._last_dump_step = None
        dumper._rank = 0
        dumper._world_size = 1
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._tracker = mock.MagicMock()
        with mock.patch.object(dumper, "dump") as dump_mock:
            dumper.maybe_dump(49)
            dump_mock.assert_not_called()
            dumper.maybe_dump(50)
            dump_mock.assert_called_once_with(50)
            dumper.maybe_dump(99)
            dump_mock.assert_called_once_with(50)
            dumper.maybe_dump(100)
            self.assertEqual(
                [call.args[0] for call in dump_mock.call_args_list],
                [50, 100],
            )
        self.assertEqual(dumper._tracker.step.call_count, 4)

    def test_maybe_dump_uses_elapsed_time_with_fixed_rate_schedule(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = None
        dumper._interval_secs = 60.0
        dumper._next_dump_time = 160.0
        dumper._last_dump_step = None
        dumper._rank = 0
        dumper._world_size = 1
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._tracker = mock.MagicMock()
        with (
            mock.patch.object(dumper, "dump") as dump_mock,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.time.monotonic",
                side_effect=[159.0, 160.0, 162.0, 221.0, 222.0, 223.0],
            ),
        ):
            dumper.maybe_dump(10)
            dumper.maybe_dump(11)
            dumper.maybe_dump(12)
            dumper.maybe_dump(13)

        self.assertEqual(
            [call.args[0] for call in dump_mock.call_args_list],
            [11, 12],
        )
        # Deadlines advance at a fixed rate from the armed schedule
        # (160 -> 220 -> 280), not from each dump's completion time.
        self.assertEqual(dumper._next_dump_time, 280.0)
        self.assertEqual(dumper._last_dump_step, 12)
        self.assertEqual(dumper._tracker.step.call_count, 4)

    def test_timed_dump_decides_locally_without_collectives(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = None
        dumper._interval_secs = 60.0
        dumper._next_dump_time = 160.0
        dumper._last_dump_step = None
        dumper._rank = 1
        dumper._world_size = 2
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._tracker = mock.MagicMock()
        with (
            mock.patch.object(
                dumper, "dump", return_value="delta.parquet"
            ) as dump_mock,
            mock.patch("torch.distributed.all_reduce") as all_reduce_mock,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.time.monotonic",
                side_effect=[159.0, 160.5, 161.0],
            ),
        ):
            dumper.maybe_dump(10)
            dump_mock.assert_not_called()
            dumper.maybe_dump(11)

        all_reduce_mock.assert_not_called()
        dump_mock.assert_called_once_with(11)
        self.assertEqual(dumper._last_dump_step, 11)
        self.assertEqual(dumper._next_dump_time, 220.0)
        self.assertEqual(dumper._tracker.step.call_count, 2)

    def test_timed_maybe_dump_propagates_local_dump_failure(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = None
        dumper._interval_secs = 60.0
        dumper._next_dump_time = 0.0
        dumper._last_dump_step = None
        dumper._rank = 0
        dumper._world_size = 2
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._tracker = mock.MagicMock()
        dump_error = RuntimeError("local dump failed")
        with (
            mock.patch.object(dumper, "dump", side_effect=dump_error) as dump_mock,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.time.monotonic", return_value=1.0
            ),
        ):
            with self.assertRaises(RuntimeError) as context:
                dumper.maybe_dump(10)

        self.assertIs(context.exception, dump_error)
        dump_mock.assert_called_once_with(10)
        self.assertIsNone(dumper._last_dump_step)
        self.assertEqual(dumper._tracker.step.call_count, 0)

    def test_timed_dump_skips_missed_deadlines_without_burst(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval_steps = None
        dumper._interval_secs = 60.0
        dumper._next_dump_time = 0.0
        dumper._last_dump_step = None
        dumper._rank = 0
        dumper._world_size = 2
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._tracker = mock.MagicMock()
        with (
            mock.patch.object(
                dumper, "dump", return_value="delta.parquet"
            ) as dump_mock,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.time.monotonic",
                return_value=100.0,
            ),
        ):
            dumper.maybe_dump(10)
            dumper.maybe_dump(11)
            dumper.maybe_dump(12)

        dump_mock.assert_called_once_with(10)
        # Deadlines 0 and 60 already elapsed at the dump; skip past them
        # instead of firing a burst of catch-up dumps.
        self.assertEqual(dumper._next_dump_time, 120.0)
        self.assertEqual(dumper._tracker.step.call_count, 3)

    def test_start_initializes_minutes_interval_from_training_start(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._feature_store_enabled = False
        dumper._uploader = None
        dumper._interval_secs = 120.0
        dumper._next_dump_time = None
        with mock.patch(
            "tzrec.utils.delta_embedding_dump.time.monotonic", return_value=100.0
        ):
            dumper.start()
        self.assertEqual(dumper._next_dump_time, 220.0)

    def test_tracker_uses_auto_compact(self):
        tracker = mock.MagicMock()
        tracker.fqn_to_feature_names = {}
        tracker.tracked_modules = {}
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.ModelDeltaTracker",
                return_value=tracker,
            ) as tracker_cls,
        ):
            DeltaEmbeddingDumper(
                torch.nn.Module(),
                DeltaEmbeddingDumpConfig(dump_interval_steps=10),
                tmp_dir,
                torch.device("cuda"),
            )

        self.assertTrue(tracker_cls.call_args.kwargs["auto_compact"])

    def test_minutes_interval_is_converted_to_seconds(self):
        tracker = mock.MagicMock()
        tracker.fqn_to_feature_names = {}
        tracker.tracked_modules = {}
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.ModelDeltaTracker",
                return_value=tracker,
            ),
        ):
            dumper = DeltaEmbeddingDumper(
                torch.nn.Module(),
                DeltaEmbeddingDumpConfig(dump_interval_minutes=2),
                tmp_dir,
                torch.device("cuda"),
            )

        self.assertIsNone(dumper._interval_steps)
        self.assertEqual(dumper._interval_secs, 120.0)

    def test_model_delta_tracker_records_same_table_name_by_owner_fqn(self):
        tracker = object.__new__(ModelDeltaTracker)
        ebc_module = torch.nn.Module()
        ec_module = torch.nn.Module()
        ebc_fqn = "model.ebc.embedding_bags.shared"
        ec_fqn = "model.ec.embeddings.shared"
        tracker._feature_to_fqn_by_module = {
            ebc_module: {"deep_feature": ebc_fqn},
            ec_module: {"sequence_feature": ec_fqn},
        }
        tracker.fqn_to_feature_names = {
            ebc_fqn: ["deep_feature"],
            ec_fqn: ["sequence_feature"],
        }
        tracker.curr_batch_idx = 3
        tracker.store = mock.MagicMock()

        ebc_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["deep_feature"],
            values=torch.tensor([1, 2]),
            offsets=torch.tensor([0, 2]),
        )
        ec_features = KeyedJaggedTensor.from_offsets_sync(
            keys=["sequence_feature"],
            values=torch.tensor([5, 6]),
            offsets=torch.tensor([0, 2]),
        )
        tracker.record_lookup(ebc_features, torch.empty(0), ebc_module)
        tracker.record_lookup(ec_features, torch.empty(0), ec_module)

        self.assertEqual(tracker.store.append.call_count, 2)
        ebc_call, ec_call = tracker.store.append.call_args_list
        self.assertEqual(ebc_call.kwargs["fqn"], ebc_fqn)
        torch.testing.assert_close(
            ebc_call.kwargs["ids"], torch.tensor(_SHARED_EBC_INPUT_IDS)
        )
        self.assertEqual(ec_call.kwargs["fqn"], ec_fqn)
        torch.testing.assert_close(
            ec_call.kwargs["ids"], torch.tensor(_SHARED_EC_INPUT_IDS)
        )
        self.assertEqual(
            tracker.fqn_to_feature_names,
            {
                ebc_fqn: ["deep_feature"],
                ec_fqn: ["sequence_feature"],
            },
        )

    def test_model_delta_tracker_advances_consumer_cursor_and_deletes_read_ids(self):
        tracker = object.__new__(ModelDeltaTracker)
        tracker._delete_on_read = True
        tracker.per_consumer_batch_idx = {"delta": -1}
        tracker.curr_batch_idx = 0
        tracker.store = mock.MagicMock()
        tracker.store.per_fqn_lookups = {
            "model.ebc.embedding_bags.shared": [
                SimpleNamespace(
                    batch_idx=0,
                    ids=torch.tensor([1, 2]),
                    states=None,
                )
            ]
        }

        rows = tracker.get_unique("delta")

        torch.testing.assert_close(
            rows["model.ebc.embedding_bags.shared"].ids,
            torch.tensor([1, 2]),
        )
        tracker.store.compact.assert_called_once_with(-1, 1)
        tracker.store.delete.assert_called_once_with(up_to_idx=1)
        self.assertEqual(tracker.per_consumer_batch_idx["delta"], 1)

        tracker.step()
        tracker.store.reset_mock()
        rows = tracker.get_unique("delta")
        self.assertEqual(rows, {})
        tracker.store.compact.assert_called_once_with(1, 2)
        tracker.store.delete.assert_called_once_with(up_to_idx=2)
        self.assertEqual(tracker.per_consumer_batch_idx["delta"], 2)

    def test_model_delta_tracker_skips_empty_table_in_later_interval(self):
        tracker = ModelDeltaTracker(
            torch.nn.Module(),
            consumers=["delta"],
            delete_on_read=True,
        )
        table_fqn = "model.ebc.embedding_bags.shared"
        tracker.store.append(
            batch_idx=0,
            fqn=table_fqn,
            ids=torch.tensor([2, 1, 2]),
            states=None,
        )

        first_rows = tracker.get_unique("delta")
        torch.testing.assert_close(
            first_rows[table_fqn].ids,
            torch.tensor([1, 2]),
        )
        self.assertEqual(tracker.store.per_fqn_lookups[table_fqn], [])

        tracker.step()
        second_rows = tracker.get_unique("delta")

        self.assertEqual(second_rows, {})
        self.assertEqual(tracker.per_consumer_batch_idx["delta"], 2)

    def test_model_delta_tracker_auto_compacts_once_per_batch(self):
        tracker = object.__new__(ModelDeltaTracker)
        tracker.per_consumer_batch_idx = {"delta": -1}
        tracker.curr_batch_idx = 2
        tracker.curr_compact_index = 0
        tracker.store = mock.MagicMock()

        tracker.trigger_compaction()
        tracker.trigger_compaction()

        tracker.store.compact.assert_called_once_with(-1, 2)
        self.assertEqual(tracker.curr_compact_index, 2)

    def test_model_delta_tracker_clears_single_consumer(self):
        tracker = object.__new__(ModelDeltaTracker)
        tracker.per_consumer_batch_idx = {"delta": -1}
        tracker.store = mock.MagicMock()

        tracker.clear("delta")

        tracker.store.delete.assert_called_once_with()

    def test_collect_table_weights_uses_owner_fqn_keys(self):
        ebc_module = torch.nn.Module()
        ec_module = torch.nn.Module()
        ebc_lookup = mock.Mock(spec=GroupedPooledEmbeddingsLookup)
        ebc_lookup.named_parameters_by_table.return_value = [
            ("shared", torch.tensor([[1.0, 2.0]]))
        ]
        ec_lookup = mock.Mock(spec=GroupedPooledEmbeddingsLookup)
        ec_lookup.named_parameters_by_table.return_value = [
            ("shared", torch.tensor([[3.0, 4.0]]))
        ]
        ebc_module._lookups = [
            ebc_lookup,
        ]
        ec_module._lookups = [
            ec_lookup,
        ]
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._table_shard_infos = {}
        dumper._tracker = SimpleNamespace(
            tracked_modules={
                "model.ebc": ebc_module,
                "model.ec": ec_module,
            }
        )

        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.{table_name}"
            ),
        ):
            table_weights = dumper._collect_table_weights()

        self.assertEqual(
            set(table_weights),
            {"model.ebc.shared", "model.ec.shared"},
        )
        torch.testing.assert_close(
            table_weights["model.ebc.shared"].tensor,
            torch.tensor([[1.0, 2.0]]),
        )
        torch.testing.assert_close(
            table_weights["model.ec.shared"].tensor,
            torch.tensor([[3.0, 4.0]]),
        )

    def test_collect_table_weights_rejects_unsupported_lookup(self):
        sharded_module = torch.nn.Module()
        sharded_module._lookups = [torch.nn.Identity()]
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._table_shard_infos = {}
        dumper._tracker = SimpleNamespace(tracked_modules={"model.ebc": sharded_module})

        with self.assertRaisesRegex(TypeError, "Unsupported embedding lookup"):
            dumper._collect_table_weights()

    def test_local_table_weight_rejects_unsupported_weight(self):
        with self.assertRaisesRegex(TypeError, "Unsupported embedding table value"):
            _local_table_weight(object())

    def test_local_table_weight_materializes_dtensor(self):
        local_tensor = torch.tensor([[1.0, 2.0]])
        dtensor = mock.Mock(spec=DTensor)
        dtensor.to_local.return_value = local_tensor

        table_weight = _local_table_weight(dtensor)

        self.assertIs(table_weight.tensor, local_tensor)
        self.assertEqual(table_weight.shard_info.local_rows, 1)
        self.assertEqual(table_weight.shard_info.local_cols, 2)

    @unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
    @mark_ci_scope("gpu")
    def test_collect_dynamic_modules_uses_owner_fqn_keys(self):
        ebc_module = torch.nn.Module()
        ec_module = torch.nn.Module()
        ebc_dynamic_module = SimpleNamespace(table_names=["shared"])
        ec_dynamic_module = SimpleNamespace(table_names=["shared"])
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._tracker = SimpleNamespace(
            tracked_modules={
                "model.ebc": ebc_module,
                "model.ec": ec_module,
            }
        )

        with (
            mock.patch(
                "dynamicemb.dump_load.get_dynamic_emb_module",
                side_effect=lambda module: (
                    [ebc_dynamic_module]
                    if module is ebc_module
                    else [ec_dynamic_module]
                ),
            ),
            mock.patch(
                "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
                side_effect=lambda module_fqn, _module, table_name: (
                    f"{module_fqn}.{table_name}"
                ),
            ),
        ):
            dynamic_modules = dumper._collect_dynamic_modules()

        self.assertEqual(
            set(dynamic_modules),
            {"model.ebc.shared", "model.ec.shared"},
        )
        self.assertIs(dynamic_modules["model.ebc.shared"], ebc_dynamic_module)
        self.assertIs(dynamic_modules["model.ec.shared"], ec_dynamic_module)

    def test_multi_gpu_output_path_uses_step_underscore_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dumper = object.__new__(DeltaEmbeddingDumper)
            dumper._output_dir = tmp_dir
            dumper._file_prefix = "delta_embedding"
            dumper._rank = 1
            dumper._world_size = 2
            output_path = dumper._output_path(50)
            self.assertEqual(
                output_path,
                os.path.join(
                    tmp_dir,
                    "step_50",
                    "delta_embedding_step_50_rank_1_of_2.parquet",
                ),
            )
            self.assertNotIn("step=50", output_path)

    def test_multi_gpu_dump_writes_empty_shard_when_rank_has_no_delta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dumper = object.__new__(DeltaEmbeddingDumper)
            dumper._output_dir = tmp_dir
            dumper._file_prefix = "delta_embedding"
            dumper._rank = 1
            dumper._world_size = 2
            dumper._feature_store_enabled = False
            dumper._uploader = None
            dumper._retain_local_dump = False
            with (
                mock.patch.object(dumper, "_collect_table_weights", return_value={}),
                mock.patch.object(dumper, "_collect_dynamic_modules", return_value={}),
                mock.patch.object(dumper, "_append_model_delta_rows", return_value=0),
            ):
                output_path = dumper.dump(50)
            table = pq.read_table(output_path)

        self.assertEqual(
            output_path,
            os.path.join(
                tmp_dir,
                "step_50",
                "delta_embedding_step_50_rank_1_of_2.parquet",
            ),
        )
        self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
        self.assertEqual(table.num_rows, 0)

    def test_single_gpu_dump_skips_file_when_rank_has_no_delta(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dumper = object.__new__(DeltaEmbeddingDumper)
            dumper._output_dir = tmp_dir
            dumper._file_prefix = "delta_embedding"
            dumper._rank = 0
            dumper._world_size = 1
            dumper._feature_store_enabled = False
            dumper._uploader = None
            dumper._retain_local_dump = False
            with (
                mock.patch.object(dumper, "_collect_table_weights", return_value={}),
                mock.patch.object(dumper, "_collect_dynamic_modules", return_value={}),
                mock.patch.object(dumper, "_append_model_delta_rows", return_value=0),
            ):
                output_path = dumper.dump(50)

        self.assertIsNone(output_path)

    def test_pause_tracking_suppresses_post_lookup_recording(self):
        lookup_fn = mock.MagicMock()
        odist_fn = mock.MagicMock()
        sharded_module = SimpleNamespace(
            post_lookup_tracker_fn=lookup_fn,
            post_odist_tracker_fn=odist_fn,
        )
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._tracking_pause_depth = 0
        dumper._tracker = SimpleNamespace(tracked_modules={"user_emb": sharded_module})
        dumper._install_tracking_pause_guard()

        sharded_module.post_lookup_tracker_fn("train")
        sharded_module.post_odist_tracker_fn()
        lookup_fn.assert_called_once_with("train")
        odist_fn.assert_called_once_with()

        with dumper.pause_tracking():
            sharded_module.post_lookup_tracker_fn("eval")
            sharded_module.post_odist_tracker_fn()
        lookup_fn.assert_called_once_with("train")
        odist_fn.assert_called_once_with()

        sharded_module.post_lookup_tracker_fn("train2", source="train")
        sharded_module.post_odist_tracker_fn()
        self.assertEqual(
            lookup_fn.call_args_list,
            [
                mock.call("train"),
                mock.call("train2", source="train"),
            ],
        )
        self.assertEqual(odist_fn.call_args_list, [mock.call(), mock.call()])

    def test_collect_table_shard_infos_prefers_grouped_embedding_metadata(self):
        grouped_config = GroupedEmbeddingConfig(
            data_type=DataType.FP32,
            pooling=PoolingType.SUM,
            is_weighted=False,
            has_feature_processor=False,
            compute_kernel=EmbeddingComputeKernel.FUSED,
            embedding_tables=[
                ShardedEmbeddingTable(
                    name="user_emb",
                    local_rows=16,
                    local_cols=8,
                    num_embeddings=64,
                    embedding_dim=8,
                    local_metadata=ShardMetadata(
                        shard_offsets=[32, 0],
                        shard_sizes=[16, 8],
                        placement="rank:1/cuda:1",
                    ),
                )
            ],
        )
        grouped_lookup = mock.Mock(spec=GroupedPooledEmbeddingsLookup)
        grouped_lookup.grouped_configs = [grouped_config]
        owner_module = torch.nn.Module()
        owner_module._table_name_to_config = {
            "user_emb": EmbeddingBagConfig(
                name="user_emb",
                num_embeddings=64,
                embedding_dim=8,
                feature_names=["user_id"],
            )
        }
        owner_module.module_sharding_plan = {}
        owner_module._lookups = [grouped_lookup]
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._tracker = SimpleNamespace(tracked_modules={"model.ebc": owner_module})
        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.{table_name}"
            ),
        ):
            shard_infos = dumper._collect_table_shard_infos()
        table_fqn = "model.ebc.user_emb"
        self.assertTrue(shard_infos[table_fqn].has_shard_metadata)
        self.assertEqual(shard_infos[table_fqn].row_offset, 32)

    def test_collect_table_shard_infos_falls_back_to_sharding_plan(self):
        sharded_module = torch.nn.Module()
        sharded_module._table_name_to_config = {
            "adgroup_id_emb": EmbeddingBagConfig(
                name="adgroup_id_emb",
                num_embeddings=64,
                embedding_dim=8,
                feature_names=["adgroup_id"],
            )
        }
        sharded_module.module_sharding_plan = {
            "adgroup_id_emb": ParameterSharding(
                sharding_type=ShardingType.ROW_WISE.value,
                compute_kernel=EmbeddingComputeKernel.FUSED.value,
                ranks=None,
                sharding_spec=EnumerableShardingSpec(
                    [
                        ShardMetadata(
                            shard_offsets=[0, 0],
                            shard_sizes=[32, 8],
                            placement="rank:0/cuda:0",
                        ),
                        ShardMetadata(
                            shard_offsets=[32, 0],
                            shard_sizes=[32, 8],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            )
        }
        sharded_module._lookups = []
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 0
        dumper._tracker = SimpleNamespace(tracked_modules={"model.ebc": sharded_module})
        table_fqn = "model.ebc.adgroup_id_emb"
        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.{table_name}"
            ),
        ):
            shard_infos = dumper._collect_table_shard_infos()
        self.assertTrue(shard_infos[table_fqn].has_shard_metadata)
        self.assertEqual(shard_infos[table_fqn].row_offset, 0)

        dumper._rank = 1
        with mock.patch(
            "tzrec.utils.delta_embedding_dump._embedding_table_fqn",
            side_effect=lambda module_fqn, _module, table_name: (
                f"{module_fqn}.{table_name}"
            ),
        ):
            shard_infos = dumper._collect_table_shard_infos()
        self.assertTrue(shard_infos[table_fqn].has_shard_metadata)
        self.assertEqual(shard_infos[table_fqn].row_offset, 32)

    def test_row_wise_lookup_outputs_global_key_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        dumper._zch_modules = {}
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        table_fqn = "model.ebc.embedding_bags.user_emb"
        embeddings, key_ids = dumper._lookup_embeddings(
            table_fqn,
            torch.tensor([0, 2]),
            table_weights={
                table_fqn: _TableWeight(
                    tensor=weight,
                    shard_info=_TableShardInfo(
                        row_offset=32,
                        local_rows=4,
                        local_cols=2,
                        global_rows=64,
                        global_cols=2,
                        has_shard_metadata=True,
                    ),
                )
            },
            dynamic_modules={},
        )
        torch.testing.assert_close(embeddings, weight[[0, 2]])
        torch.testing.assert_close(key_ids, torch.tensor([32, 34]))

    def test_lookup_filters_out_of_range_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        dumper._zch_modules = {}
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        table_fqn = "model.ebc.embedding_bags.user_emb"
        embeddings, key_ids = dumper._lookup_embeddings(
            table_fqn,
            torch.tensor([0, 2, 99, -1]),
            table_weights={
                table_fqn: _TableWeight(
                    tensor=weight,
                    shard_info=_TableShardInfo(
                        row_offset=32,
                        local_rows=4,
                        local_cols=2,
                        global_rows=64,
                        global_cols=2,
                        has_shard_metadata=True,
                    ),
                )
            },
            dynamic_modules={},
        )
        torch.testing.assert_close(embeddings, weight[[0, 2]])
        torch.testing.assert_close(key_ids, torch.tensor([32, 34]))

    def test_zch_lookup_maps_local_rows_back_to_raw_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        mch = MCHManagedCollisionModule(
            zch_size=4,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=2,
        )
        # slot -> raw id: slots 0/1 hold raw ids 101/105, slots 2/3 are empty
        # (slot 3 is the fallback row served for unmatched ids).
        mch._mch_sorted_raw_ids.copy_(
            torch.tensor(
                [101, 105, torch.iinfo(torch.int64).max, torch.iinfo(torch.int64).max]
            )
        )
        # slot -> global row, permuted so the slot/row inverse matters.
        mch._mch_remapped_ids_mapping.copy_(torch.tensor([0, 2, 1, 3]))
        dumper._zch_modules = {}
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        table_fqn = "model.mc_ebc._embedding_module.embedding_bags.user_emb"
        dumper._zch_modules[table_fqn] = mch
        embeddings, key_ids = dumper._lookup_embeddings(
            table_fqn,
            # local rows touched by the tracker: rows 0 and 2 resolve to raw
            # ids; row 1 maps to an empty slot and row 3 is the fallback row.
            torch.tensor([0, 2, 1, 3]),
            table_weights={
                table_fqn: _TableWeight(
                    tensor=weight,
                    shard_info=_TableShardInfo(
                        row_offset=0,
                        local_rows=4,
                        local_cols=2,
                        global_rows=4,
                        global_cols=2,
                        has_shard_metadata=True,
                    ),
                )
            },
            dynamic_modules={},
        )
        torch.testing.assert_close(embeddings, weight[[0, 2]])
        torch.testing.assert_close(key_ids, torch.tensor([101, 105]))

    def _build_zch_snapshot_dumper(self, raw_ids_per_slot, mapping, offset, zch_size=4):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 1
        dumper._rank = 0
        mch = MCHManagedCollisionModule(
            zch_size=zch_size,
            device=torch.device("cpu"),
            eviction_policy=LFU_EvictionPolicy(),
            eviction_interval=2,
        )
        mch._mch_sorted_raw_ids.copy_(
            torch.tensor(
                [
                    int(x) if x is not None else torch.iinfo(torch.int64).max
                    for x in raw_ids_per_slot
                ],
                dtype=torch.int64,
            )
        )
        mch._mch_remapped_ids_mapping.copy_(
            torch.tensor(mapping, dtype=torch.int64) + offset
        )
        mch._output_global_offset = offset
        dumper._zch_snapshots = {}
        weight = torch.arange(
            zch_size * _SHARED_EBC_EMBEDDING_DIM, dtype=torch.float32
        ).reshape(zch_size, _SHARED_EBC_EMBEDDING_DIM)
        table_fqn = "model.mc_ebc._embedding_module.embedding_bags.zch_tbl"
        dumper._zch_modules = {table_fqn: mch}
        dumper._tracker = SimpleNamespace(fqn_to_feature_names={table_fqn: ["feat"]})
        table_weights = {
            table_fqn: _TableWeight(
                tensor=weight,
                shard_info=_TableShardInfo(
                    row_offset=offset,
                    local_rows=zch_size,
                    local_cols=_SHARED_EBC_EMBEDDING_DIM,
                    global_rows=zch_size,
                    global_cols=_SHARED_EBC_EMBEDDING_DIM,
                    has_shard_metadata=True,
                ),
            )
        }
        return dumper, mch, weight, table_fqn, table_weights

    def test_zch_delta_publishes_admitted_ids_without_tracker_hits(self):
        dumper, mch, weight, table_fqn, table_weights = self._build_zch_snapshot_dumper(
            raw_ids_per_slot=[101, 102, None, None],
            mapping=[0, 1, 2, 3],
            offset=0,
        )
        # The snapshot holds only 101, so 102 counts as admitted since the
        # previous dump even though the tracker recorded no lookup for it.
        dumper._zch_snapshots[table_fqn] = torch.tensor([101], dtype=torch.int64)
        table_chunks: list = []
        num_rows = dumper._append_zch_delta_rows(
            table_chunks, 5, table_weights, zch_touched={}
        )
        self.assertEqual(num_rows, 1)
        rows = table_chunks[0]
        self.assertEqual(rows["key_id"].to_pylist(), [102])
        self.assertEqual(rows["embedding"].to_pylist(), [weight[1].tolist()])
        self.assertEqual(sorted(dumper._zch_snapshots[table_fqn].tolist()), [101, 102])

    def test_zch_delta_publishes_evicted_ids_with_fallback_row(self):
        dumper, mch, weight, table_fqn, table_weights = self._build_zch_snapshot_dumper(
            raw_ids_per_slot=[102, None, None, None],
            mapping=[1, 0, 2, 3],
            offset=0,
        )
        # 101 was evicted out of the table; the dump must overwrite its stale
        # FeatureStore entry with the fallback row the model now serves.
        dumper._zch_snapshots[table_fqn] = torch.tensor([101, 102], dtype=torch.int64)
        table_chunks: list = []
        num_rows = dumper._append_zch_delta_rows(
            table_chunks, 5, table_weights, zch_touched={}
        )
        # 101 evicted (fallback row), 102 unchanged since the snapshot.
        self.assertEqual(num_rows, 1)
        rows = table_chunks[0]
        self.assertEqual(rows["key_id"].to_pylist(), [101])
        self.assertEqual(
            rows["embedding"].to_pylist(), [weight[mch._zch_size - 1].tolist()]
        )

    def test_zch_delta_first_dump_publishes_all_held_ids(self):
        dumper, mch, weight, table_fqn, table_weights = self._build_zch_snapshot_dumper(
            raw_ids_per_slot=[101, 102, None, None],
            mapping=[0, 1, 2, 3],
            offset=0,
        )
        table_chunks: list = []
        num_rows = dumper._append_zch_delta_rows(
            table_chunks, 5, table_weights, zch_touched={}
        )
        self.assertEqual(num_rows, 2)
        rows = table_chunks[0]
        self.assertEqual(sorted(rows["key_id"].to_pylist()), [101, 102])

    def test_lookup_handles_empty_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        dumper._zch_modules = {}
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        table_fqn = "model.ebc.embedding_bags.user_emb"
        embeddings, key_ids = dumper._lookup_embeddings(
            table_fqn,
            torch.tensor([], dtype=torch.long),
            table_weights={
                table_fqn: _TableWeight(
                    tensor=weight,
                    shard_info=_TableShardInfo(
                        row_offset=32,
                        local_rows=4,
                        local_cols=2,
                        global_rows=64,
                        global_cols=2,
                        has_shard_metadata=True,
                    ),
                )
            },
            dynamic_modules={},
        )
        self.assertEqual(embeddings.shape, (0, 2))
        self.assertEqual(key_ids.shape, (0,))

    def test_row_wise_lookup_requires_shard_metadata(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        dumper._zch_modules = {}
        table_fqn = "model.ebc.embedding_bags.user_emb"
        with self.assertRaisesRegex(ValueError, "shard metadata"):
            dumper._lookup_embeddings(
                table_fqn,
                torch.tensor([0]),
                table_weights={
                    table_fqn: _TableWeight(
                        tensor=torch.zeros(4, 2),
                        shard_info=_TableShardInfo(
                            local_rows=4,
                            local_cols=2,
                            global_rows=64,
                            global_cols=2,
                        ),
                    )
                },
                dynamic_modules={},
            )

    @unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for dynamicemb.")
    @mark_ci_scope("gpu")
    def test_lookup_dynamic_embeddings_filters_missing_ids(self):
        from dynamicemb.types import CopyMode

        torch.cuda.set_device(0)
        dumper = object.__new__(DeltaEmbeddingDumper)
        fake_tables = _FakeDynamicTables()
        dynamic_module = SimpleNamespace(
            table_names=["dyn_table"],
            tables=fake_tables,
            flush=mock.MagicMock(),
            _dynamicemb_options=[SimpleNamespace(dim=2)],
        )

        embeddings, key_ids = dumper._lookup_dynamic_embeddings(
            dynamic_module,
            "model.ec.embeddings.dyn_table",
            torch.tensor([101, 102, 103]),
        )

        dynamic_module.flush.assert_called_once_with()
        self.assertIs(fake_tables.copy_mode, CopyMode.EMBEDDING)
        torch.testing.assert_close(fake_tables.ids.cpu(), torch.tensor([101, 102, 103]))
        torch.testing.assert_close(fake_tables.table_ids.cpu(), torch.tensor([0, 0, 0]))
        torch.testing.assert_close(key_ids.cpu(), torch.tensor([101, 103]))
        torch.testing.assert_close(
            embeddings.cpu(), torch.tensor([[1.0, 2.0], [5.0, 6.0]])
        )

    @unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for dynamicemb.")
    @mark_ci_scope("gpu")
    def test_lookup_dynamic_embeddings_flushes_module_once_per_dump(self):
        torch.cuda.set_device(0)
        dumper = object.__new__(DeltaEmbeddingDumper)
        # One module hosting two tables, reachable under both table_name keys.
        dynamic_module = SimpleNamespace(
            table_names=["dyn_a", "dyn_b"],
            tables=_FakeDynamicTables(),
            flush=mock.MagicMock(),
            _dynamicemb_options=[SimpleNamespace(dim=2), SimpleNamespace(dim=2)],
        )

        flushed_module_ids = set()
        for table_name in ("dyn_a", "dyn_b"):
            dumper._lookup_dynamic_embeddings(
                dynamic_module,
                f"model.ec.embeddings.{table_name}",
                torch.tensor([101, 102, 103]),
                flushed_module_ids,
            )

        # Both tables share the module; flush() flushes all tables, so it runs
        # once per dump rather than once per table.
        dynamic_module.flush.assert_called_once_with()


class DeltaEmbeddingDumpShardedIntegrationTest(MultiProcessTestBase):
    def __init__(self, methodName="runTest") -> None:
        super().__init__(methodName)
        self.world_size = 2

    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    @mark_ci_scope("gpu")
    def test_row_wise_sharded_dump_writes_global_key_ids(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(
                os.environ,
                {
                    "NCCL_DEBUG": "WARN",
                    "FORCED_NCCL_DEBUG": "WARN",
                    "NCCL_DEBUG_SUBSYS": "",
                },
            ),
        ):
            self._run_multi_process_test(
                callable=_run_sharded_delta_embedding_dump,
                world_size=self.world_size,
                output_dir=tmp_dir,
            )
            for rank in range(self.world_size):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(
                            tmp_dir,
                            "step_50",
                            f"delta_step_50_rank_{rank}_of_{self.world_size}.parquet",
                        )
                    )
                )

    @unittest.skipIf(torch.cuda.device_count() < 2, "test requires 2+ GPUs")
    @mark_ci_scope("gpu")
    def test_zch_sharded_dump_writes_raw_ids(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(
                os.environ,
                {
                    "NCCL_DEBUG": "WARN",
                    "FORCED_NCCL_DEBUG": "WARN",
                    "NCCL_DEBUG_SUBSYS": "",
                },
            ),
        ):
            self._run_multi_process_test(
                callable=_run_zch_delta_embedding_dump,
                world_size=self.world_size,
                output_dir=tmp_dir,
            )
            for rank in range(self.world_size):
                self.assertTrue(
                    os.path.exists(
                        os.path.join(
                            tmp_dir,
                            "step_50",
                            f"delta_step_50_rank_{rank}_of_{self.world_size}.parquet",
                        )
                    )
                )

    @unittest.skipIf(torch.cuda.device_count() < 1, "test requires a GPU")
    @mark_ci_scope("gpu")
    def test_zch_lifecycle_dump_publishes_admitted_and_evicted_ids(self):
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch.dict(
                os.environ,
                {
                    "NCCL_DEBUG": "WARN",
                    "FORCED_NCCL_DEBUG": "WARN",
                    "NCCL_DEBUG_SUBSYS": "",
                },
            ),
        ):
            self._run_multi_process_test(
                callable=_run_zch_lifecycle_delta_embedding_dump,
                world_size=1,
                output_dir=tmp_dir,
            )

    @unittest.skipIf(torch.cuda.device_count() < 1, "test requires a GPU")
    @mark_ci_scope("gpu")
    def test_shared_table_name_tracks_ec_and_ebc_fqns_independently(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._run_multi_process_test(
                callable=_run_shared_table_fqn_delta_embedding_dump,
                world_size=1,
                output_dir=tmp_dir,
            )


class DeltaEmbeddingDumpDynamicembIntegrationTest(unittest.TestCase):
    """End-to-end multi-process delta dump over a sharded dynamicemb model.

    Runs the real tzrec train pipeline (torchrun, row-wise sharded dynamicemb
    tables) with delta dump enabled, so the dynamic lookup path
    (``flush()`` + ``tables.find()``) is exercised under genuine multi-rank
    sharding rather than a single-process fake table.
    """

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir(prefix="tzrec_delta_dyn_")

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(
        gpu_unavailable[0] or not has_dynamicemb,
        "dynamicemb or GPU not available.",
    )
    @mark_ci_scope("gpu")
    def test_dynamicemb_multi_gpu_delta_dump_writes_uniform_shards(self):
        world_size = int(os.getenv("TEST_NPROC_PER_NODE", "2"))
        pipeline_config = config_util.load_pipeline_config(
            "tzrec/tests/configs/multi_tower_din_fg_dynamicemb_mock.config"
        )
        # Admit every id immediately so the find() lookup returns embeddings
        # for the touched ids (default frequency admission would hide them).
        for feature_config in pipeline_config.feature_configs:
            feature_type = feature_config.WhichOneof("feature")
            if feature_type is None:
                continue
            feature = getattr(feature_config, feature_type)
            if "dynamicemb" not in feature.DESCRIPTOR.fields_by_name:
                continue
            if feature.HasField("dynamicemb"):
                admission = feature.dynamicemb.WhichOneof("admission_strategy")
                if admission is not None:
                    feature.dynamicemb.ClearField(admission)

        dump_dir = os.path.abspath(os.path.join(self.test_dir, "delta_dump"))
        dump_cfg = pipeline_config.train_config.delta_embedding_dump_config
        dump_cfg.dump_interval_steps = 1
        dump_cfg.output_dir = dump_dir
        dump_cfg.file_prefix = "delta_embedding"
        new_config_path = os.path.join(self.test_dir, "new_pipeline.config")
        config_util.save_message(pipeline_config, new_config_path)

        self.success = test_utils.test_train_eval(
            new_config_path,
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        self.assertTrue(self.success)

        step_dirs = sorted(glob.glob(os.path.join(dump_dir, "step_*")))
        self.assertTrue(step_dirs, f"no delta dump produced under {dump_dir}")

        dumped_real_rows = False
        for step_dir in step_dirs:
            shards = sorted(glob.glob(os.path.join(step_dir, "*.parquet")))
            # Every rank writes a shard even with no delta, so each step dir
            # holds exactly world_size shards (no ragged shard set).
            self.assertEqual(
                len(shards),
                world_size,
                f"{step_dir} has {len(shards)} shards, expected {world_size}",
            )
            for shard in shards:
                table = pq.read_table(shard)
                self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
                if table.num_rows == 0:
                    continue
                dumped_real_rows = True
                self.assertEqual(set(table["world_size"].to_pylist()), {world_size})
                self.assertEqual(
                    set(table["source"].to_pylist()), {"model_delta_tracker"}
                )
                # dynamic lookup must return a real embedding vector per id.
                self.assertTrue(
                    all(len(emb) > 0 for emb in table["embedding"].to_pylist())
                )

        # If no rank ever dumped a real row, the flush()/find() lookup path was
        # not actually exercised and the test would be vacuous.
        self.assertTrue(
            dumped_real_rows,
            "no dynamic delta rows dumped; flush()/find() path not exercised",
        )


class DeltaEmbeddingDumpZchIntegrationTest(unittest.TestCase):
    """End-to-end multi-process delta dump over a sharded ZCH model.

    Runs the real tzrec train pipeline (torchrun, row-wise sharded ZCH
    tables) with delta dump enabled, so the ZCH lookup path (slot inverse
    mapping back to raw feature ids) is exercised under genuine multi-rank
    sharding.
    """

    def setUp(self):
        self.success = False
        self.test_dir = make_test_dir(prefix="tzrec_delta_zch_")

    def tearDown(self):
        if self.success and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    @unittest.skipIf(*gpu_unavailable)
    @mark_ci_scope("gpu")
    def test_zch_multi_gpu_delta_dump_writes_raw_ids(self):
        world_size = int(os.getenv("TEST_NPROC_PER_NODE", "2"))
        pipeline_config = config_util.load_pipeline_config(
            "tzrec/tests/configs/multi_tower_din_zch_fg_mock.config"
        )
        dump_dir = os.path.abspath(os.path.join(self.test_dir, "delta_dump"))
        dump_cfg = pipeline_config.train_config.delta_embedding_dump_config
        dump_cfg.dump_interval_steps = 1
        dump_cfg.output_dir = dump_dir
        dump_cfg.file_prefix = "delta_embedding"
        new_config_path = os.path.join(self.test_dir, "new_pipeline.config")
        config_util.save_message(pipeline_config, new_config_path)

        self.success = test_utils.test_train_eval(
            new_config_path,
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
        )
        self.assertTrue(self.success)

        step_dirs = sorted(glob.glob(os.path.join(dump_dir, "step_*")))
        self.assertTrue(step_dirs, f"no delta dump produced under {dump_dir}")

        dumped_zch_rows = False
        for step_dir in step_dirs:
            shards = sorted(glob.glob(os.path.join(step_dir, "*.parquet")))
            self.assertEqual(
                len(shards),
                world_size,
                f"{step_dir} has {len(shards)} shards, expected {world_size}",
            )
            for shard in shards:
                table = pq.read_table(shard)
                self.assertEqual(table.schema, _DELTA_DUMP_SCHEMA)
                for feature_name, key_id, emb in zip(
                    table["feature_name"].to_pylist(),
                    table["key_id"].to_pylist(),
                    table["embedding"].to_pylist(),
                ):
                    if feature_name not in ("user_id", "item_id"):
                        continue
                    dumped_zch_rows = True
                    self.assertEqual(len(emb), 16)
                    # Mock data generates raw ids in [0, 10000) while the
                    # per-rank ZCH row ranges reach 500000, so raw-id keys
                    # are distinguishable from (wrong) remapped local rows.
                    self.assertGreaterEqual(key_id, 0)
                    self.assertLess(key_id, 10000)

        self.assertTrue(
            dumped_zch_rows,
            "no ZCH delta rows dumped; the ZCH lookup path was not exercised",
        )


if __name__ == "__main__":
    unittest.main()
