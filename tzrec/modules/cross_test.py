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

from tzrec.modules.cross import Cross, CrossNet, DCNv2Layer, DCNv2Net
from tzrec.utils.test_util import TestGraphType, create_test_module


class CrossTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_layer(self, graph_type) -> None:
        layer = Cross(input_dim=64)
        layer = create_test_module(layer, graph_type)
        x0 = torch.randn(32, 64)
        xl = torch.randn(32, 64)
        result = layer(x0, xl)
        self.assertEqual(result.size(), (32, 64))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_layer_3d(self, graph_type) -> None:
        layer = Cross(input_dim=64)
        layer = create_test_module(layer, graph_type)
        x0 = torch.randn(32, 10, 64)
        xl = torch.randn(32, 10, 64)
        result = layer(x0, xl)
        self.assertEqual(result.size(), (32, 10, 64))


class CrossNetTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_net_single_layer(self, graph_type) -> None:
        net = CrossNet(input_dim=64, num_layers=1)
        self.assertEqual(net.output_dim(), 64)
        net = create_test_module(net, graph_type)
        x = torch.randn(32, 64)
        result = net(x)
        self.assertEqual(result.size(), (32, 64))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_net_multi_layer(self, graph_type) -> None:
        net = CrossNet(input_dim=128, num_layers=3)
        self.assertEqual(net.output_dim(), 128)
        net = create_test_module(net, graph_type)
        x = torch.randn(16, 128)
        result = net(x)
        self.assertEqual(result.size(), (16, 128))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_net_3d_input(self, graph_type) -> None:
        net = CrossNet(input_dim=64, num_layers=2)
        net = create_test_module(net, graph_type)
        x = torch.randn(8, 5, 64)
        result = net(x)
        self.assertEqual(result.size(), (8, 5, 64))


class DCNv2LayerTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_layer(self, graph_type) -> None:
        layer = DCNv2Layer(input_dim=64, low_rank=16)
        layer = create_test_module(layer, graph_type)
        x0 = torch.randn(32, 64)
        xl = torch.randn(32, 64)
        result = layer(x0, xl)
        self.assertEqual(result.size(), (32, 64))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_layer_high_rank(self, graph_type) -> None:
        layer = DCNv2Layer(input_dim=128, low_rank=64)
        layer = create_test_module(layer, graph_type)
        x0 = torch.randn(16, 128)
        xl = torch.randn(16, 128)
        result = layer(x0, xl)
        self.assertEqual(result.size(), (16, 128))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_layer_3d(self, graph_type) -> None:
        layer = DCNv2Layer(input_dim=64, low_rank=32)
        layer = create_test_module(layer, graph_type)
        x0 = torch.randn(8, 10, 64)
        xl = torch.randn(8, 10, 64)
        result = layer(x0, xl)
        self.assertEqual(result.size(), (8, 10, 64))


class DCNv2NetTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_net_single_layer(self, graph_type) -> None:
        net = DCNv2Net(input_dim=64, num_layers=1, low_rank=16)
        self.assertEqual(net.output_dim(), 64)
        net = create_test_module(net, graph_type)
        x = torch.randn(32, 64)
        result = net(x)
        self.assertEqual(result.size(), (32, 64))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_net_multi_layer(self, graph_type) -> None:
        net = DCNv2Net(input_dim=128, num_layers=4, low_rank=32)
        self.assertEqual(net.output_dim(), 128)
        net = create_test_module(net, graph_type)
        x = torch.randn(16, 128)
        result = net(x)
        self.assertEqual(result.size(), (16, 128))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_net_3d_input(self, graph_type) -> None:
        net = DCNv2Net(input_dim=64, num_layers=2, low_rank=24)
        net = create_test_module(net, graph_type)
        x = torch.randn(8, 5, 64)
        result = net(x)
        self.assertEqual(result.size(), (8, 5, 64))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_dcnv2_net_edge_case_low_rank(self, graph_type) -> None:
        # Test with low_rank close to input_dim
        net = DCNv2Net(input_dim=32, num_layers=2, low_rank=30)
        net = create_test_module(net, graph_type)
        x = torch.randn(4, 32)
        result = net(x)
        self.assertEqual(result.size(), (4, 32))


if __name__ == "__main__":
    unittest.main()
