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
import unittest
from unittest import mock

import torch

from tzrec.protos import feature_pb2
from tzrec.protos.train_pb2 import DeltaEmbeddingDumpConfig
from tzrec.utils.delta_embedding_dump import (
    validate_delta_embedding_dump_config,
    validate_delta_embedding_dump_no_zch_features,
)


class DeltaEmbeddingDumpValidationTest(unittest.TestCase):
    def test_disabled_config_skips_runtime_validation(self):
        config = DeltaEmbeddingDumpConfig(enable=False, dump_interval_steps=0)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            validate_delta_embedding_dump_config(config, torch.device("cpu"))

    def test_enabled_config_requires_single_cuda_device(self):
        config = DeltaEmbeddingDumpConfig(enable=True, dump_interval_steps=10)
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            with self.assertRaisesRegex(ValueError, "single GPU"):
                validate_delta_embedding_dump_config(config, torch.device("cuda:0"))
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            with self.assertRaisesRegex(ValueError, "single GPU"):
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
        zch_feature_names, zch_table_names = (
            validate_delta_embedding_dump_no_zch_features(feature_configs)
        )
        self.assertEqual(zch_feature_names, set())
        self.assertEqual(zch_table_names, set())


if __name__ == "__main__":
    unittest.main()
