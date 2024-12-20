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

from tzrec.modules.intervention import Intervention
from tzrec.utils.test_util import TestGraphType, create_test_module


class InterventionTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, [0.9]],
            [TestGraphType.NORMAL, 0.9],
            [TestGraphType.NORMAL, [0.9, 0.8, 0.7]],
            [TestGraphType.FX_TRACE, [0.9]],
            # [TestGraphType.JIT_SCRIPT, [0.9]],
        ]
    )
    def test_intervention(self, graph_type, dropout_ratio) -> None:
        intervention = Intervention(
            base_dim=16,
            source_dim=8,
            low_rank_dim=4,
            dropout_ratio=0.1,
        )
        intervention = create_test_module(intervention, graph_type)
        base = torch.randn(4, 16)
        source = torch.randn(4, 8)
        result = intervention(base, source)
        self.assertEqual(result.size(), (4, 16))


if __name__ == "__main__":
    unittest.main()
