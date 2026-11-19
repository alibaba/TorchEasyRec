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

import json
import os
import shutil
import unittest
from unittest import mock

from tzrec.acc import utils
from tzrec.utils.test_util import make_test_dir


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

        self.assertEqual(acc_config["DISTRIBUTED_EMBEDDING"], "1")

    def test_export_acc_config_keeps_distributed_embedding_enabled(self) -> None:
        with mock.patch.dict(
            os.environ, {"USE_DISTRIBUTED_EMBEDDING": "1"}, clear=True
        ):
            acc_config = utils.export_acc_config(
                additional_export_config={"DISTRIBUTED_EMBEDDING": False}
            )

        self.assertEqual(acc_config["DISTRIBUTED_EMBEDDING"], "1")

    def test_export_acc_config_omits_distributed_embedding_by_default(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            acc_config = utils.export_acc_config()

        self.assertNotIn("DISTRIBUTED_EMBEDDING", acc_config)

    def test_is_distributed_embedding_predict(self) -> None:
        test_dir = make_test_dir()
        try:
            for acc_config, expected in [
                ({"DISTRIBUTED_EMBEDDING": "1"}, True),
                ({"SPARSE_INT64": "1"}, False),
            ]:
                with open(os.path.join(test_dir, "model_acc.json"), "w") as f:
                    json.dump(acc_config, f)
                self.assertEqual(
                    utils.is_distributed_embedding_predict(test_dir), expected
                )
        finally:
            shutil.rmtree(test_dir)

    def test_is_hard_search_input_tile(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(utils.is_hard_search_input_tile())

        for tile in ("2", "3"):
            for env in (
                {"INPUT_TILE": tile, "HARD_SEARCH_INPUT_TILE": "1"},
                {"INPUT_TILE": tile, "HARD_SEARCH_INPUT_TILE": "1", "ENABLE_AOT": "2"},
                {
                    "INPUT_TILE": tile,
                    "HARD_SEARCH_INPUT_TILE": "1",
                    "UNIFIED_AOT": "1",
                },
            ):
                with self.subTest(env=env):
                    with mock.patch.dict(os.environ, env, clear=True):
                        self.assertTrue(utils.is_hard_search_input_tile())

        for env in (
            {"HARD_SEARCH_INPUT_TILE": "1"},
            {"INPUT_TILE": "2", "HARD_SEARCH_INPUT_TILE": "1", "ENABLE_AOT": "1"},
            {"INPUT_TILE": "3", "HARD_SEARCH_INPUT_TILE": "1", "ENABLE_TRT": "1"},
            {"INPUT_TILE": "3", "HARD_SEARCH_INPUT_TILE": "1", "USE_RTP": "1"},
        ):
            with self.subTest(invalid_env=env):
                with mock.patch.dict(os.environ, env, clear=True):
                    with self.assertRaises(ValueError):
                        utils.is_hard_search_input_tile()


if __name__ == "__main__":
    unittest.main()
