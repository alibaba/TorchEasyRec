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
import unittest

from tzrec.protos.module_pb2 import MLP
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import config_util
from tzrec.utils.test_util import make_test_dir


class ConfigUtilTest(unittest.TestCase):
    def test_load_custom_model_any_config(self):
        config_path = os.path.join(make_test_dir(), "custom_model.config")
        with open(config_path, "w") as f:
            f.write(
                """
                model_config {
                  custom_model {
                    class_path: "tzrec_custom.models.CustomRankModel"
                    config {
                      [type.googleapis.com/tzrec.protos.MLP] {
                        hidden_units: 32
                        hidden_units: 16
                      }
                    }
                  }
                }
                """
            )

        pipeline_config = config_util.load_pipeline_config(config_path)
        mlp_config = MLP()
        self.assertTrue(
            pipeline_config.model_config.custom_model.config.Unpack(mlp_config)
        )
        self.assertEqual(list(mlp_config.hidden_units), [32, 16])

    def test_get_inference_batch_size(self):
        pipeline_config = EasyRecConfig()
        pipeline_config.data_config.batch_size = 16

        self.assertEqual(
            config_util.get_inference_batch_size(pipeline_config.data_config), 16
        )

        pipeline_config.data_config.eval_batch_size = 96
        self.assertEqual(
            config_util.get_inference_batch_size(pipeline_config.data_config), 96
        )

    def test_set_inference_batch_size(self):
        pipeline_config = EasyRecConfig()
        pipeline_config.data_config.batch_size = 16
        pipeline_config.data_config.eval_batch_size = 96

        config_util.set_inference_batch_size(pipeline_config.data_config, 7)

        self.assertEqual(pipeline_config.data_config.batch_size, 7)
        self.assertEqual(pipeline_config.data_config.eval_batch_size, 7)

    def test_use_dense_ema(self):
        pipeline_config = EasyRecConfig()
        self.assertFalse(
            config_util.use_dense_ema(
                None,
                pipeline_config.train_config,
            )
        )
        pipeline_config.train_config.dense_optimizer.ema.SetInParent()
        self.assertTrue(
            config_util.use_dense_ema(
                None,
                pipeline_config.train_config,
            )
        )

        for config_field in ("eval_config", "export_config"):
            with self.subTest(config_field=config_field):
                pipeline_config = EasyRecConfig()
                config = getattr(pipeline_config, config_field)

                self.assertFalse(
                    config_util.use_dense_ema(
                        config,
                        pipeline_config.train_config,
                    )
                )
                pipeline_config.train_config.dense_optimizer.ema.SetInParent()
                self.assertTrue(
                    config_util.use_dense_ema(
                        config,
                        pipeline_config.train_config,
                    )
                )

                config.use_dense_ema = False
                self.assertFalse(
                    config_util.use_dense_ema(
                        config,
                        pipeline_config.train_config,
                    )
                )
                config.use_dense_ema = True
                self.assertTrue(
                    config_util.use_dense_ema(
                        config,
                        pipeline_config.train_config,
                    )
                )

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
