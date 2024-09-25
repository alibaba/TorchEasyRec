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

import torch
from parameterized import parameterized

from tzrec.modules.task_tower import TaskTower
from tzrec.protos import tower_pb2
from tzrec.protos.module_pb2 import MLP
from tzrec.utils.config_util import config_to_kwargs
from tzrec.utils.test_util import TestGraphType, create_test_module


class TaskTowerTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_task_tower(self, graph_type) -> None:
        task_cgf = tower_pb2.TaskTower(
            tower_name="is_click",
            label_name="is_clk",
            mlp=MLP(hidden_units=[12, 8, 4]),
            num_class=2,
        )
        task_cgf = config_to_kwargs(task_cgf)
        task_tower = TaskTower(32, task_cgf["num_class"], mlp=task_cgf["mlp"])
        task_tower = create_test_module(task_tower, graph_type)
        features = torch.randn(4, 32)
        output = task_tower(features)
        self.assertEqual(list(output.size()), [4, 2])


if __name__ == "__main__":
    unittest.main()
