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

import torch

from tzrec.acc import utils
from tzrec.protos.export_pb2 import ExportConfig
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.protos.train_pb2 import TrainConfig
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

    def test_export_env_scope_does_not_mutate_os_environ(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "USE_DISTRIBUTED_EMBEDDING": "0",
                "INPUT_TILE": "0",
                "INPUT_TILE_3_ONLINE": "0",
                "WORLD_SIZE": "4",
                "RANK": "2",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with utils.export_env_scope(
                    use_distributed_embedding=True,
                    input_tile="3",
                    input_tile_3_online=True,
                    world_size=1,
                    rank=0,
                ):
                    self.assertTrue(utils.use_distributed_embedding())
                    self.assertEqual(utils.get_input_tile(), "3")
                    self.assertTrue(utils.is_input_tile())
                    self.assertTrue(utils.is_input_tile_3_online())
                    self.assertEqual(utils.get_world_size(), 1)
                    self.assertEqual(utils.get_rank(), 0)
                    # Nested scope with None inherits outer scope; explicit
                    # False overrides outer True without touching os.environ.
                    with utils.export_env_scope(
                        use_distributed_embedding=False,
                        input_tile=None,
                    ):
                        self.assertFalse(utils.use_distributed_embedding())
                        self.assertEqual(utils.get_input_tile(), "3")
                        self.assertTrue(utils.is_input_tile_3_online())
                        self.assertEqual(utils.get_world_size(), 1)
                        self.assertEqual(utils.get_rank(), 0)
                    self.assertTrue(utils.use_distributed_embedding())
                    raise RuntimeError("boom")

            self.assertFalse(utils.use_distributed_embedding())
            self.assertEqual(utils.get_input_tile(), "0")
            self.assertFalse(utils.is_input_tile_3_online())
            self.assertEqual(utils.get_world_size(), 4)
            self.assertEqual(utils.get_rank(), 2)
            self.assertEqual(os.environ["USE_DISTRIBUTED_EMBEDDING"], "0")
            self.assertEqual(os.environ["INPUT_TILE"], "0")
            self.assertEqual(os.environ["INPUT_TILE_3_ONLINE"], "0")
            self.assertEqual(os.environ["WORLD_SIZE"], "4")
            self.assertEqual(os.environ["RANK"], "2")

    def test_export_acc_config_records_input_tile_3_online(self) -> None:
        with mock.patch.dict(os.environ, {"INPUT_TILE_3_ONLINE": "1"}, clear=True):
            self.assertEqual(utils.export_acc_config()["INPUT_TILE_3_ONLINE"], "1")
        with mock.patch.dict(os.environ, {"INPUT_TILE_3_ONLINE": "0"}, clear=True):
            self.assertEqual(utils.export_acc_config()["INPUT_TILE_3_ONLINE"], "0")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertNotIn("INPUT_TILE_3_ONLINE", utils.export_acc_config())
            with utils.export_env_scope(
                use_distributed_embedding=True,
                input_tile="3",
                input_tile_3_online=True,
            ):
                acc_cfg = utils.export_acc_config()
                self.assertEqual(acc_cfg["DISTRIBUTED_EMBEDDING"], "1")
                self.assertEqual(acc_cfg["INPUT_TILE"], "3")
                self.assertEqual(acc_cfg["INPUT_TILE_3_ONLINE"], "1")
            with utils.export_env_scope(input_tile_3_online=False):
                self.assertEqual(utils.export_acc_config()["INPUT_TILE_3_ONLINE"], "0")
        with mock.patch.dict(
            os.environ,
            {
                "USE_DISTRIBUTED_EMBEDDING": "1",
                "INPUT_TILE": "3",
                "INPUT_TILE_3_ONLINE": "1",
            },
            clear=True,
        ):
            with utils.export_env_scope(
                use_distributed_embedding=False,
                input_tile="0",
                input_tile_3_online=False,
            ):
                acc_cfg = utils.export_acc_config()
                self.assertNotIn("DISTRIBUTED_EMBEDDING", acc_cfg)
                self.assertEqual(acc_cfg["INPUT_TILE"], "0")
                self.assertEqual(acc_cfg["INPUT_TILE_3_ONLINE"], "0")

        test_dir = make_test_dir()
        try:
            for acc_config, expected in [
                ({"INPUT_TILE_3_ONLINE": "1"}, True),
                ({"INPUT_TILE_3_ONLINE": "0"}, False),
                ({}, False),
            ]:
                with open(os.path.join(test_dir, "model_acc.json"), "w") as f:
                    json.dump(acc_config, f)
                self.assertEqual(
                    utils.is_input_tile_3_online_predict(test_dir), expected
                )
        finally:
            shutil.rmtree(test_dir)

    def test_allow_tf32_for_export_non_mutating_warns(self) -> None:
        prev_cudnn = torch.backends.cudnn.allow_tf32
        prev_matmul = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            cfg = EasyRecConfig(
                train_config=TrainConfig(
                    cudnn_allow_tf32=True, cuda_matmul_allow_tf32=True
                ),
                export_config=ExportConfig(
                    cudnn_allow_tf32=False, cuda_matmul_allow_tf32=False
                ),
            )
            with mock.patch.object(utils.logger, "warning") as warn_log:
                utils.allow_tf32_for_export(cfg, mutate_globals=False)
            self.assertTrue(torch.backends.cudnn.allow_tf32)
            self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
            self.assertEqual(warn_log.call_count, 2)
        finally:
            torch.backends.cudnn.allow_tf32 = prev_cudnn
            torch.backends.cuda.matmul.allow_tf32 = prev_matmul

    def test_portable_cpu_aot_env_helpers(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertFalse(utils.is_portable_cpu_aot())
            self.assertIsNone(utils.get_aot_cpu_march())
        with mock.patch.dict(
            os.environ,
            {"AOTI_PORTABLE_CPU": "1", "AOTI_CPU_MARCH": "x86-64-v3"},
            clear=True,
        ):
            self.assertTrue(utils.is_portable_cpu_aot())
            self.assertEqual(utils.get_aot_cpu_march(), "x86-64-v3")

    def test_canonical_contiguous_fixes_size_one_and_zero_strides(self):
        sliced_one = torch.randn(1, 4)[:, 0:1]
        self.assertTrue(sliced_one.is_contiguous())
        self.assertEqual(sliced_one.stride(), (4, 1))
        fixed = utils.canonical_contiguous(sliced_one)
        self.assertEqual(fixed.stride(), (1, 1))
        torch.testing.assert_close(fixed, sliced_one)

        sliced_zero = torch.empty((0, 8))[:, :4]
        self.assertEqual(utils.canonical_contiguous(sliced_zero).stride(), (4, 1))

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
