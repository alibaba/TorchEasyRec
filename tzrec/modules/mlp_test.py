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

from tzrec.modules.mlp import MLP
from tzrec.utils.test_util import TestGraphType, create_test_module


class MLPTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, [0.9]],
            [TestGraphType.NORMAL, 0.9],
            [TestGraphType.NORMAL, [0.9, 0.8, 0.7]],
            [TestGraphType.FX_TRACE, [0.9]],
            [TestGraphType.JIT_SCRIPT, [0.9]],
        ]
    )
    def test_mlp(self, graph_type, dropout_ratio) -> None:
        mlp = MLP(
            in_features=16,
            hidden_units=[8, 4, 2],
            activation="nn.ReLU",
            use_bn=True,
            dropout_ratio=dropout_ratio,
        )
        mlp = create_test_module(mlp, graph_type)
        input = torch.randn(4, 16)
        result = mlp(input)
        self.assertEqual(result.size(), (4, 2))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, [0.9]],
            [TestGraphType.NORMAL, 0.9],
            [TestGraphType.NORMAL, [0.9, 0.8, 0.7]],
            [TestGraphType.FX_TRACE, [0.9]],
            [TestGraphType.JIT_SCRIPT, [0.9]],
        ]
    )
    def test_mlp_output_hidden(self, graph_type, dropout_ratio) -> None:
        mlp = MLP(
            in_features=16,
            hidden_units=[8, 4, 2],
            activation="nn.ReLU",
            use_bn=True,
            dropout_ratio=dropout_ratio,
            return_hidden_layer_feature=True,
        )
        mlp = create_test_module(mlp, graph_type)
        input = torch.randn(4, 16)
        result = mlp(input)
        self.assertEqual(result["hidden_layer0"].size(), (4, 8))
        self.assertEqual(result["hidden_layer1"].size(), (4, 4))
        self.assertEqual(result["hidden_layer2"].size(), (4, 2))
        self.assertEqual(result["hidden_layer_end"].size(), (4, 2))

    @parameterized.expand(
        [
            [TestGraphType.NORMAL, [0.9]],
            [TestGraphType.NORMAL, 0.9],
            [TestGraphType.NORMAL, [0.9, 0.8, 0.7]],
            [TestGraphType.FX_TRACE, [0.9]],
            [TestGraphType.JIT_SCRIPT, [0.9]],
        ]
    )
    def test_mlp_seq(self, graph_type, dropout_ratio) -> None:
        mlp = MLP(
            in_features=16,
            hidden_units=[8, 4, 2],
            activation="nn.ReLU",
            use_bn=True,
            dropout_ratio=dropout_ratio,
            dim=3,
        )
        mlp = create_test_module(mlp, graph_type)
        input = torch.randn(4, 2, 16)
        result = mlp(input)
        self.assertEqual(result.size(), (4, 2, 2))


if __name__ == "__main__":
    unittest.main()
