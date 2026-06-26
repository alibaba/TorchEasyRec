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
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from fbgemm_gpu.split_embedding_configs import EmbOptimType as OptimType
from torch import nn
from torchrec import KeyedJaggedTensor
from torchrec.distributed import DistributedModelParallel, ShardingEnv
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.embeddingbag import EmbeddingBagCollectionSharder
from torchrec.distributed.planner import (
    EmbeddingShardingPlanner,
    ParameterConstraints,
    Topology,
)
from torchrec.distributed.test_utils.multi_process import (
    MultiProcessContext,
    MultiProcessTestBase,
)
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig, PoolingType
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from tzrec.protos import feature_pb2
from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.tests import utils as test_utils
from tzrec.utils import config_util
from tzrec.utils.delta_embedding_dump import (
    _DELTA_DUMP_SCHEMA,
    DeltaEmbeddingDumper,
    _table_shard_info_from_config,
    _TableShardInfo,
    _TableWeight,
    _validate_table_shard_info,
    validate_delta_embedding_dump_config,
    validate_delta_embedding_dump_no_zch_features,
)
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.test_util import gpu_unavailable, mark_ci_scope

_SHARDED_TABLE_NAME = "table_1"
_SHARDED_FEATURE_NAME = "feature_1"
_SHARDED_NUM_EMBEDDINGS = 16
_SHARDED_EMBEDDING_DIM = 4
_SHARDED_INPUT_IDS = [0, 2, 8, 9, 15]


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

    table_weight = dumper._collect_table_weights()[_SHARDED_TABLE_NAME]
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
            [],
        )
        output = model(_sharded_features(rank))
        output.sum().backward()
        output_path = dumper.dump(50)
        unittest.TestCase().assertIsNotNone(output_path)
        _assert_sharded_dump_file(rank, output_path, dumper)
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

    def test_zch_feature_fails_fast(self):
        feature_configs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    embedding_dim=8,
                    zch=feature_pb2.ZeroCollisionHash(zch_size=1024),
                )
            )
        ]
        with self.assertRaisesRegex(ValueError, "user_id"):
            validate_delta_embedding_dump_no_zch_features(feature_configs)

    def test_dynamicemb_feature_is_allowed(self):
        feature_configs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    embedding_dim=8,
                    dynamicemb=feature_pb2.DynamicEmbedding(max_capacity=1024),
                )
            )
        ]
        validate_delta_embedding_dump_no_zch_features(feature_configs)

    def test_init_validates_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=10)
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "CUDA"):
                DeltaEmbeddingDumper(
                    torch.nn.Module(),
                    config,
                    tmp_dir,
                    torch.device("cpu"),
                    [],
                )

    def test_init_validates_no_zch_features(self):
        config = DeltaEmbeddingDumpConfig(dump_interval_steps=10)
        feature_configs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="user_id",
                    expression="user:user_id",
                    embedding_dim=8,
                    zch=feature_pb2.ZeroCollisionHash(zch_size=1024),
                )
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaisesRegex(ValueError, "user_id"):
                DeltaEmbeddingDumper(
                    torch.nn.Module(),
                    config,
                    tmp_dir,
                    torch.device("cuda"),
                    feature_configs,
                )

    def test_row_wise_shard_info_uses_row_offset(self):
        table_config = SimpleNamespace(
            local_rows=16,
            local_cols=8,
            num_embeddings=64,
            embedding_dim=8,
            local_metadata=SimpleNamespace(
                shard_offsets=[32, 0],
                shard_sizes=[16, 8],
            ),
        )
        shard_info = _table_shard_info_from_config(table_config)
        _validate_table_shard_info("user_emb", shard_info)
        self.assertEqual(shard_info.row_offset, 32)
        self.assertEqual(shard_info.global_cols, 8)

    def test_column_wise_shard_info_fails_fast(self):
        table_config = SimpleNamespace(
            local_rows=64,
            local_cols=4,
            num_embeddings=64,
            embedding_dim=8,
            local_metadata=SimpleNamespace(
                shard_offsets=[0, 4],
                shard_sizes=[64, 4],
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
        dumper._interval = 50
        dumper._world_size = 1
        with mock.patch.object(dumper, "dump") as dump_mock:
            # Boundary steps were already written by maybe_dump; skip them so a
            # trailing empty shard never overwrites the real one.
            self.assertIsNone(dumper.final_dump(50))
            self.assertIsNone(dumper.final_dump(100))
            dump_mock.assert_not_called()

            # Trailing partial interval (and step 0) must still be flushed.
            dumper.final_dump(0)
            dumper.final_dump(73)
            self.assertEqual(
                [call.args[0] for call in dump_mock.call_args_list],
                [0, 73],
            )

    def test_final_dump_syncs_step_across_ranks_before_flush(self):
        # A lagging rank reaches final_dump at a boundary step (50) while the
        # furthest rank stopped at 73. Without syncing, the lagging rank would
        # skip and write no shard, leaving step_73/ ragged. The MAX all_reduce
        # lifts every rank to 73 so all take the same dump-into-step_73 path.
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval = 50
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

    def test_maybe_dump_uses_checkpoint_aligned_global_step(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._interval = 50
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

    def test_tracker_uses_auto_compact(self):
        tracker = mock.MagicMock()
        tracker.table_to_fqn = {}
        tracker.fqn_to_feature_names.return_value = {}
        tracker.get_tracked_modules.return_value = {}
        with (
            tempfile.TemporaryDirectory() as tmp_dir,
            mock.patch(
                "tzrec.utils.delta_embedding_dump.ModelDeltaTrackerTrec",
                return_value=tracker,
            ) as tracker_cls,
        ):
            DeltaEmbeddingDumper(
                torch.nn.Module(),
                DeltaEmbeddingDumpConfig(dump_interval_steps=10),
                tmp_dir,
                torch.device("cuda"),
                [],
            )

        self.assertTrue(tracker_cls.call_args.kwargs["auto_compact"])

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
        dumper._tracker = SimpleNamespace(
            get_tracked_modules=lambda: {"user_emb": sharded_module}
        )
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
        original_config_module = torch.nn.Module()
        original_config_module._table_name_to_config = {
            "user_emb": SimpleNamespace(
                local_rows=16,
                local_cols=8,
                num_embeddings=64,
                embedding_dim=8,
            )
        }
        grouped_config_module = torch.nn.Module()
        grouped_config_module._config = SimpleNamespace(
            embedding_tables=[
                SimpleNamespace(
                    name="user_emb",
                    local_rows=16,
                    local_cols=8,
                    num_embeddings=64,
                    embedding_dim=8,
                    local_metadata=SimpleNamespace(
                        shard_offsets=[32, 0],
                        shard_sizes=[16, 8],
                    ),
                )
            ]
        )
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._model = torch.nn.Sequential(
            original_config_module, grouped_config_module
        )
        shard_infos = dumper._collect_table_shard_infos()
        self.assertTrue(shard_infos["user_emb"].has_shard_metadata)
        self.assertEqual(shard_infos["user_emb"].row_offset, 32)

    def test_collect_table_shard_infos_falls_back_to_sharding_plan(self):
        sharded_module = torch.nn.Module()
        sharded_module._table_name_to_config = {
            "adgroup_id_emb": SimpleNamespace(
                local_rows=16,
                local_cols=8,
                num_embeddings=64,
                embedding_dim=8,
            )
        }
        sharded_module.module_sharding_plan = {
            "adgroup_id_emb": SimpleNamespace(
                ranks=None,
                sharding_spec=SimpleNamespace(
                    shards=[
                        SimpleNamespace(
                            shard_offsets=[0, 0],
                            shard_sizes=[32, 8],
                            placement="rank:0/cuda:0",
                        ),
                        SimpleNamespace(
                            shard_offsets=[32, 0],
                            shard_sizes=[32, 8],
                            placement="rank:1/cuda:1",
                        ),
                    ]
                ),
            )
        }
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._rank = 0
        dumper._model = sharded_module
        shard_infos = dumper._collect_table_shard_infos()
        self.assertTrue(shard_infos["adgroup_id_emb"].has_shard_metadata)
        self.assertEqual(shard_infos["adgroup_id_emb"].row_offset, 0)

        dumper._rank = 1
        shard_infos = dumper._collect_table_shard_infos()
        self.assertTrue(shard_infos["adgroup_id_emb"].has_shard_metadata)
        self.assertEqual(shard_infos["adgroup_id_emb"].row_offset, 32)

    def test_row_wise_lookup_outputs_global_key_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        embeddings, key_ids = dumper._lookup_embeddings(
            "user_emb",
            torch.tensor([0, 2]),
            table_weights={
                "user_emb": _TableWeight(
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
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        embeddings, key_ids = dumper._lookup_embeddings(
            "user_emb",
            torch.tensor([0, 2, 99, -1]),
            table_weights={
                "user_emb": _TableWeight(
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

    def test_lookup_handles_empty_ids(self):
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._world_size = 2
        weight = torch.tensor([[0.0, 0.1], [1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        embeddings, key_ids = dumper._lookup_embeddings(
            "user_emb",
            torch.tensor([], dtype=torch.long),
            table_weights={
                "user_emb": _TableWeight(
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
        with self.assertRaisesRegex(ValueError, "shard metadata"):
            dumper._lookup_embeddings(
                "user_emb",
                torch.tensor([0]),
                table_weights={
                    "user_emb": _TableWeight(
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
            dynamic_module, "dyn_table", torch.tensor([101, 102, 103])
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
                table_name,
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


class DeltaEmbeddingDumpDynamicembIntegrationTest(unittest.TestCase):
    """End-to-end multi-process delta dump over a sharded dynamicemb model.

    Runs the real tzrec train pipeline (torchrun, row-wise sharded dynamicemb
    tables) with delta dump enabled, so the dynamic lookup path
    (``flush()`` + ``tables.find()``) is exercised under genuine multi-rank
    sharding rather than a single-process fake table.
    """

    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_delta_dyn_", dir="./tmp")
        os.chmod(self.test_dir, 0o755)

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


if __name__ == "__main__":
    unittest.main()
