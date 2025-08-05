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


if __name__ == "__main__":
    unittest.main()
