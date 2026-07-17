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

import os
import tempfile
import unittest

from tzrec.utils import config_util


class ConfigUtilTest(unittest.TestCase):
    def test_edit_config(self):
        pipeline_config = config_util.load_pipeline_config(
            "examples/multi_tower_taobao.config"
        )
        pipeline_config = config_util.edit_config(
            pipeline_config,
            {
                "train_config.sparse_optimizer.adagrad_optimizer.lr": 0.0001,
                "feature_configs[0].id_feature.num_buckets": 1,
                "feature_configs[1:3].id_feature.num_buckets": 2,
                "feature_configs[id_feature.feature_name=age_level]."
                "id_feature.num_buckets": 3,
            },
        )
        self.assertAlmostEqual(
            pipeline_config.train_config.sparse_optimizer.adagrad_optimizer.lr, 0.0001
        )
        self.assertEqual(pipeline_config.feature_configs[0].id_feature.num_buckets, 1)
        self.assertEqual(pipeline_config.feature_configs[1].id_feature.num_buckets, 2)
        self.assertEqual(pipeline_config.feature_configs[2].id_feature.num_buckets, 2)
        self.assertEqual(
            pipeline_config.feature_configs[4].id_feature.feature_name, "age_level"
        )
        self.assertEqual(pipeline_config.feature_configs[4].id_feature.num_buckets, 3)

    def test_pipeline_artifact_redacts_feature_store_security_token(self):
        pipeline_config = config_util.load_pipeline_config(
            "examples/multi_tower_taobao.config"
        )
        dump_config = pipeline_config.train_config.delta_embedding_dump_config
        feature_store_config = dump_config.feature_store_config
        feature_store_config.region = "cn-test"
        feature_store_config.endpoint = "feature-store.example"
        feature_store_config.project_name = "project_a"
        feature_store_config.feature_entity_name = "embedding_entity"
        feature_store_config.feature_view_name = "shared_embeddings"
        feature_store_config.feature_view_ttl_secs = 86400
        feature_store_config.feature_view_shard_count = 4
        feature_store_config.feature_view_replication_count = 2
        feature_store_config.version = "model_a@export_1"
        feature_store_config.security_token = "SECRET_STS"

        sanitized = config_util.sanitize_pipeline_config_for_artifact(pipeline_config)
        sanitized_fs = (
            sanitized.train_config.delta_embedding_dump_config.feature_store_config
        )
        self.assertEqual(sanitized_fs.project_name, "project_a")
        self.assertEqual(sanitized_fs.feature_entity_name, "embedding_entity")
        self.assertEqual(sanitized_fs.feature_view_name, "shared_embeddings")
        self.assertEqual(sanitized_fs.feature_view_ttl_secs, 86400)
        self.assertEqual(sanitized_fs.feature_view_shard_count, 4)
        self.assertEqual(sanitized_fs.feature_view_replication_count, 2)
        self.assertEqual(sanitized_fs.version, "model_a@export_1")
        self.assertTrue(sanitized_fs.IsInitialized())
        self.assertFalse(sanitized_fs.HasField("security_token"))
        self.assertTrue(feature_store_config.HasField("security_token"))

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "pipeline.config")
            config_util.save_pipeline_config_artifact(pipeline_config, path)
            with open(path) as source:
                artifact_text = source.read()
        self.assertNotIn("SECRET_STS", artifact_text)


if __name__ == "__main__":
    unittest.main()
