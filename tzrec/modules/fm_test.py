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

from tzrec.modules.fm import FactorizationMachine
from tzrec.utils.test_util import TestGraphType, create_test_module


class FactorizationMachineTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_fm(self, graph_type) -> None:
        fm = FactorizationMachine()
        fm = create_test_module(fm, graph_type)
        input = torch.randn(4, 2, 16)
        result = fm(input)
        self.assertEqual(result.size(), (4, 16))


if __name__ == "__main__":
    unittest.main()
