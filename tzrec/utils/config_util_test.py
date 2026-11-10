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

import os
import unittest

from tzrec.protos import pipeline_pb2
from tzrec.utils import config_util
from tzrec.utils.test_util import make_test_dir


class ConfigUtilTest(unittest.TestCase):
    """Saving and loading pipeline configs."""

    def test_save_message_accepts_a_bare_filename(self) -> None:
        # a path with no directory component writes to the current directory
        # instead of failing to create one
        cwd = os.getcwd()
        try:
            os.chdir(make_test_dir())
            cfg = pipeline_pb2.EasyRecConfig()
            config_util.save_message(cfg, "bare.config")
            self.assertTrue(os.path.exists("bare.config"))
            self.assertIsNotNone(config_util.load_pipeline_config("bare.config"))
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
