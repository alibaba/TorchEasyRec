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

from tzrec.modules.interaction import CIN, CrossNetV2, InputSENet, InteractionArch
from tzrec.utils.test_util import TestGraphType, create_test_module


class InputSENetTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_input_senet(self, graph_type) -> None:
        se = InputSENet(length_per_key=[4, 8, 8, 16], reduction_ratio=2)
        se = create_test_module(se, graph_type)
        input = torch.randn(4, 36)
        result = se(input)
        self.assertEqual(result.size(), (4, 36))


class InteractionArchTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_interaction_arch(self, graph_type) -> None:
        ia = InteractionArch(feature_num=4)
        self.assertEqual(ia.output_dim(), 6)
        ia = create_test_module(ia, graph_type)
        dense_features = torch.randn([10, 8])
        sparse_features = torch.randn([10, 3, 8])
        result = ia(dense_features, sparse_features)
        self.assertEqual(result.size(), (10, 6))


class CrossNetV2Test(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cross_net_v2(self, graph_type) -> None:
        dcn = CrossNetV2(32, 6, 2)
        dcn = create_test_module(dcn, graph_type)
        features = torch.randn([10, 32])
        result = dcn(features)
        self.assertEqual(result.size(), (10, 32))


class CINTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_cin(self, graph_type) -> None:
        cin = CIN(feature_num=9, cin_layer_size=[10, 15, 20])
        cin = create_test_module(cin, graph_type)
        features = torch.randn([5, 9, 16])
        result = cin(features)
        self.assertEqual(result.size(), (5, 45))


if __name__ == "__main__":
    unittest.main()
