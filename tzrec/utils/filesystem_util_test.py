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
import shutil
import tempfile
import unittest

from tzrec.tests import utils


class FileSystemUtilTest(unittest.TestCase):
    def setUp(self):
        self.success = False
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")
        os.chmod(self.test_dir, 0o755)

    def tearDown(self):
        if self.success:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)

    def test_local_fsspec_train_eval_export(self):
        self.success = utils.test_train_eval(
            "tzrec/tests/configs/multi_tower_din_fg_mock.config",
            self.test_dir,
            user_id="user_id",
            item_id="item_id",
            args_str=f"--model_dir file://{self.test_dir}/train",
            env_str="USE_FSSPEC=1",
        )
        if self.success:
            self.success = utils.test_eval(
                os.path.join(self.test_dir, "pipeline.config"),
                self.test_dir,
                env_str="USE_FSSPEC=1",
            )
        if self.success:
            self.success = utils.test_export(
                os.path.join(self.test_dir, "pipeline.config"),
                self.test_dir,
                export_dir=f"file://{self.test_dir}/export",
                env_str=f"LOCAL_CACHE_DIR={self.test_dir}/export_cache USE_FSSPEC=1",
            )
        if self.success:
            self.success = utils.test_predict(
                scripted_model_path=f"file://{self.test_dir}/export",
                predict_input_path=os.path.join(self.test_dir, r"eval_data/\*.parquet"),
                predict_output_path=os.path.join(self.test_dir, "predict_result"),
                reserved_columns="clk",
                output_columns="probs",
                test_dir=self.test_dir,
                env_str=f"LOCAL_CACHE_DIR={self.test_dir}/export_cache USE_FSSPEC=1",
            )
        self.assertTrue(self.success)


if __name__ == "__main__":
    unittest.main()
