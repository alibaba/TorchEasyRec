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

from tzrec.modules.interaction import InputSENet
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


if __name__ == "__main__":
    unittest.main()
