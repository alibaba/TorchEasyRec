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

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tzrec.protos import feature_pb2
from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.utils.delta_embedding_dump import (
    DeltaEmbeddingDumper,
    _table_shard_info_from_config,
    _TableShardInfo,
    _TableWeight,
    _validate_table_shard_info,
    validate_delta_embedding_dump_config,
    validate_delta_embedding_dump_no_zch_features,
)


class DeltaEmbeddingDumpValidationTest(unittest.TestCase):
    def test_disabled_config_skips_runtime_validation(self):
        config = DeltaEmbeddingDumpConfig(enable=False, dump_interval_steps=0)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            validate_delta_embedding_dump_config(config, torch.device("cpu"))

    def test_enabled_config_allows_multi_gpu_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(enable=True, dump_interval_steps=10)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            validate_delta_embedding_dump_config(config, torch.device("cuda:0"))

    def test_enabled_config_requires_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(enable=True, dump_interval_steps=10)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "CUDA"):
                validate_delta_embedding_dump_config(config, torch.device("cpu"))

    def test_enabled_config_requires_positive_interval(self):
        config = DeltaEmbeddingDumpConfig(enable=True, dump_interval_steps=0)
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
        rows = []
        dumper._extend_rows(
            rows,
            global_step=10,
            feature_name="user_id",
            table_fqn="model.ebc.user_emb",
            key_ids=torch.tensor([42]),
            embeddings=torch.tensor([[1.0, 2.0]]),
            source="model_delta_tracker",
        )
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[0]["world_size"], 4)
        self.assertEqual(rows[0]["key_id"], 42)

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

    def test_pause_tracking_suppresses_post_lookup_recording(self):
        record_fn = mock.MagicMock()
        sharded_module = SimpleNamespace(post_lookup_tracker_fn=record_fn)
        dumper = object.__new__(DeltaEmbeddingDumper)
        dumper._tracking_pause_depth = 0
        dumper._tracker = SimpleNamespace(
            get_tracked_modules=lambda: {"user_emb": sharded_module}
        )
        dumper._install_tracking_pause_guard()

        sharded_module.post_lookup_tracker_fn("train")
        record_fn.assert_called_once_with("train")

        with dumper.pause_tracking():
            sharded_module.post_lookup_tracker_fn("eval")
        record_fn.assert_called_once_with("train")

        sharded_module.post_lookup_tracker_fn("train2", source="train")
        self.assertEqual(
            record_fn.call_args_list,
            [
                mock.call("train"),
                mock.call("train2", source="train"),
            ],
        )

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


if __name__ == "__main__":
    unittest.main()
