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


import unittest

import torch
from parameterized import parameterized

from tzrec.modules.activation import Dice, create_activation
from tzrec.utils.test_util import TestGraphType, create_test_module


class ActivationTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dice(self, graph_type) -> None:
        dice = Dice(16)
        dice = create_test_module(dice, graph_type)
        input = torch.randn(4, 16)
        result = dice(input)
        self.assertEqual(result.size(), (4, 16))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dice_seq(self, graph_type) -> None:
        dice = Dice(16, dim=3)
        dice = create_test_module(dice, graph_type)
        input = torch.randn(4, 5, 16)
        result = dice(input)
        self.assertEqual(result.size(), (4, 5, 16))

    def test_create_activation(self):
        act_module = create_activation("nn.ReLU")
        self.assertEqual(act_module.__class__, torch.nn.ReLU)
        act_module = create_activation("torch.nn.ReLU")
        self.assertEqual(act_module.__class__, torch.nn.ReLU)
        act_module = create_activation("Dice", hidden_size=16, dim=3)
        self.assertEqual(act_module.__class__, Dice)
        act_module = create_activation("nn.RReLU(lower=0.1)")
        self.assertEqual(act_module.__class__, torch.nn.RReLU)
        act_module = create_activation("torch.nn.MyReLU")
        self.assertEqual(act_module, None)
        act_module = create_activation("")
        self.assertEqual(act_module, None)


if __name__ == "__main__":
    unittest.main()
