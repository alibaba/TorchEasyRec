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

"""Tests for the shared sharded-model runtime."""

import unittest
from unittest import mock

import torch
from parameterized import param, parameterized

from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils.runtime import ShardedModelRuntime
from tzrec.utils.test_util import parameterized_name_func


def _pipeline_config(
    *, model_dir: str = "/tmp/tzrec_runtime_model", constraints: bool = False
) -> EasyRecConfig:
    config = EasyRecConfig(model_dir=model_dir, train_input_path="train")
    config.train_config.num_steps = 1
    if constraints:
        config.train_config.global_embedding_constraints.sharding_types.append(
            "table_wise"
        )
    return config


class ShardedModelRuntimeTest(unittest.TestCase):
    """The runtime owns the job's manager and forwards the sharding setup."""

    def test_train_mode_keeps_retention_and_export_settings(self) -> None:
        config = _pipeline_config()
        config.train_config.keep_checkpoint_max = 3
        manager = mock.Mock()
        with mock.patch(
            "tzrec.utils.runtime.CheckpointManager", return_value=manager
        ) as manager_cls:
            with ShardedModelRuntime.open(config, mode="train") as runtime:
                manager_cls.assert_called_once_with(
                    config.model_dir,
                    keep_checkpoint_max=3,
                    export_config=config.export_config,
                )
                self.assertIs(runtime.checkpoints, manager)
        manager.close.assert_called_once()

    @parameterized.expand(
        [("eval",), ("predict_checkpoint",)], name_func=parameterized_name_func
    )
    def test_non_train_modes_use_read_only_manager(self, mode: str) -> None:
        config = _pipeline_config()
        manager = mock.Mock()
        with mock.patch(
            "tzrec.utils.runtime.CheckpointManager", return_value=manager
        ) as manager_cls:
            with ShardedModelRuntime.open(config, mode=mode) as runtime:
                manager_cls.assert_called_once_with(config.model_dir)
                self.assertIs(runtime.checkpoints, manager)
        manager.close.assert_called_once()

    def test_open_closes_manager_on_setup_error(self) -> None:
        manager = mock.Mock()
        with mock.patch("tzrec.utils.runtime.CheckpointManager", return_value=manager):
            with self.assertRaisesRegex(RuntimeError, "setup boom"):
                with ShardedModelRuntime.open(_pipeline_config(), mode="eval"):
                    raise RuntimeError("setup boom")
        manager.close.assert_called_once()

    @parameterized.expand(
        [
            param("with_constraints_with_plan", constraints=True, with_plan=True),
            param("with_constraints_without_plan", constraints=True, with_plan=False),
            param("without_constraints_with_plan", constraints=False, with_plan=True),
            param(
                "without_constraints_without_plan",
                constraints=False,
                with_plan=False,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_shard_forwards_constraints_and_saved_plan_path(
        self, _name: str, constraints: bool, with_plan: bool
    ) -> None:
        """shard() forwards open()'s constraints and the plan path to the planner.

        A mocked planner proves only what shard() forwards, not that a saved
        plan was actually reused; shard() returns the constructed DMP.
        """
        config = _pipeline_config(constraints=constraints)
        expected_constraints = (
            config.train_config.global_embedding_constraints if constraints else None
        )
        checkpoint_path = "/m/model.ckpt-9" if with_plan else None
        dmp = mock.Mock()
        with (
            mock.patch("tzrec.utils.runtime.CheckpointManager"),
            mock.patch(
                "tzrec.utils.runtime.create_planner", return_value=mock.Mock()
            ) as create_planner,
            mock.patch("tzrec.utils.runtime.get_default_sharders", return_value=[]),
            mock.patch(
                "tzrec.utils.runtime.DistributedModelParallel", return_value=dmp
            ),
        ):
            with ShardedModelRuntime.open(config, mode="train") as runtime:
                self.assertIs(
                    runtime.shard(
                        mock.Mock(),
                        device=torch.device("cpu"),
                        batch_size=8,
                        checkpoint_path=checkpoint_path,
                    ),
                    dmp,
                )
        self.assertEqual(
            create_planner.call_args.kwargs["global_constraints_cfg"],
            expected_constraints,
        )
        self.assertEqual(
            create_planner.call_args.kwargs["ckpt_plan_path"],
            "/m/model.ckpt-9/plan" if with_plan else None,
        )


if __name__ == "__main__":
    unittest.main()
