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
from unittest import mock

from tzrec.acc import utils


class AccUtilsTest(unittest.TestCase):
    def test_use_distributed_embedding(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(utils.use_distributed_embedding())

        with mock.patch.dict(
            os.environ, {"USE_DISTRIBUTED_EMBEDDING": "1"}, clear=True
        ):
            self.assertTrue(utils.use_distributed_embedding())

    def test_export_acc_config_records_distributed_embedding(self) -> None:
        with mock.patch.dict(
            os.environ, {"USE_DISTRIBUTED_EMBEDDING": "1"}, clear=True
        ):
            acc_config = utils.export_acc_config()

        self.assertIs(acc_config["DISTRIBUTED_EMBEDDING"], True)

    def test_export_acc_config_keeps_distributed_embedding_marker_true(self) -> None:
        with mock.patch.dict(
            os.environ, {"USE_DISTRIBUTED_EMBEDDING": "1"}, clear=True
        ):
            acc_config = utils.export_acc_config(
                additional_export_config={"DISTRIBUTED_EMBEDDING": False}
            )

        self.assertIs(acc_config["DISTRIBUTED_EMBEDDING"], True)

    def test_export_acc_config_omits_distributed_embedding_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            acc_config = utils.export_acc_config()

        self.assertNotIn("DISTRIBUTED_EMBEDDING", acc_config)


if __name__ == "__main__":
    unittest.main()
