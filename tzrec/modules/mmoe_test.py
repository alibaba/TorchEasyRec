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

from tzrec.modules.mmoe import MMoE
from tzrec.utils.test_util import TestGraphType, create_test_module


class MMoETest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_fm(self, graph_type) -> None:
        mmoe = MMoE(
            in_features=16,
            expert_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
            num_expert=3,
            num_task=2,
            gate_mlp=dict(hidden_units=[4], activation="nn.ReLU", use_bn=False),
        )
        mmoe = create_test_module(mmoe, graph_type)
        input = torch.randn(4, 16)
        result = mmoe(input)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].size(), (4, 2))


if __name__ == "__main__":
    unittest.main()
