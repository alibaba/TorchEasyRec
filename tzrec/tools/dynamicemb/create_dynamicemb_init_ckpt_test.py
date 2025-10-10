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
import random
import shutil
import tempfile
import unittest

import pyarrow as pa
import pyarrow.dataset as ds

from tzrec.tests import utils
from tzrec.utils import config_util, misc_util


class CreateDynamicEmbInitCkptTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if self.success:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    def test_create_dynamicemb_init_ckpt(self):
        pipeline_config = config_util.load_pipeline_config(
            "tzrec/tests/configs/multi_tower_din_fg_dynamicemb_mock.config"
        )
        t = pa.Table.from_arrays(
            [
                pa.array([misc_util.random_name() for _ in range(1000)]),
                pa.array(
                    [
                        ",".join([str(random.random()) for _ in range(16)])
                        for _ in range(1000)
                    ]
                ),
            ],
            names=["id", "emb"],
        )
        ds.write_dataset(
            t,
            os.path.join(self.test_dir, "init_table"),
            format="parquet",
            max_rows_per_file=5000,
            max_rows_per_group=5000,
        )
        pipeline_config.feature_configs[
            0
        ].id_feature.dynamicemb.init_table = os.path.join(
            self.test_dir, "init_table/*.parquet"
        )
        new_config_path = os.path.join(self.test_dir, "new_pipeline.config")
        config_util.save_message(pipeline_config, new_config_path)

        save_dir = os.path.join(self.test_dir, "init_ckpt")
        cmd_str = (
            "PYTHONPATH=. python -m tzrec.tools.dynamicemb.create_dynamicemb_init_ckpt "
            f"--pipeline_config_path {new_config_path} "
            f"--world_size 2 --save_dir {save_dir}"
        )
        self.success = misc_util.run_cmd(
            cmd_str,
            os.path.join(self.test_dir, "log_create_dynamicemb_init_ckpt.txt"),
            timeout=600,
        )
        if self.success:
            self.success = utils.test_train_eval(
                new_config_path,
                self.test_dir,
                f"--fine_tune_checkpoint {os.path.join(save_dir, 'model.ckpt-0')}",
                user_id="user_id",
                item_id="item_id",
            )
        self.assertTrue(self.success)


if __name__ == "__main__":
    unittest.main()
