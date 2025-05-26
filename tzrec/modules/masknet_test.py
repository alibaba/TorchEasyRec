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

from tzrec.modules.masknet import MaskNetModule
from tzrec.utils.test_util import TestGraphType, create_test_module


class MaskNetModuleTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_fm(self, graph_type, use_parallel) -> None:
        masknet_module = MaskNetModule(
            feature_dim=16,
            n_mask_blocks=3,
            mask_block=dict(reduction_ratio=2.0, hidden_dim=16),
            top_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
            use_parallel=use_parallel,
        )
        masknet_module = create_test_module(masknet_module, graph_type)
        input = torch.randn(4, 16)
        result = masknet_module(input)
        self.assertEqual(result.size(), (4, 2))


if __name__ == "__main__":
    unittest.main()
