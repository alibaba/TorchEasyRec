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

from tzrec.modules.capsule import CapsuleLayer
from tzrec.protos.module_pb2 import B2ICapsule
from tzrec.utils.test_util import TestGraphType, create_test_module


class CapsuleTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_capsule(self, graph_type) -> None:
        conf = B2ICapsule(
            max_k=5,
            max_seq_len=32,
            high_dim=8,
            num_iters=3,
            routing_logits_scale=1.0,
            routing_logits_stddev=1.0,
            squash_pow=2.0,
            const_caps_num=True,
        )
        cap = CapsuleLayer(conf, input_dim=8)
        cap_test = create_test_module(cap, graph_type)
        input = torch.randn(4, 64, 8)
        seq_len = torch.arange(4) + 32
        result, mask = cap_test(input, seq_len)
        self.assertEqual(result.size(), (4, 5, 8))
        self.assertEqual(mask.size(), (4, 5))


if __name__ == "__main__":
    unittest.main()
