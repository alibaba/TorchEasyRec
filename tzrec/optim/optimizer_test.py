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
from torch.autograd import Variable
from torchrec.optim import KeyedOptimizerWrapper

from tzrec.optim.optimizer import TZRecOptimizer


class TZRecOptimizerTest(unittest.TestCase):
    def test_optimizer(self):
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=0.001)
        )
        optimizer = TZRecOptimizer(keyed_optimizer)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.zero_grad()
        self.assertEqual(param_1.grad, None)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.step()
        torch.testing.assert_close(param_1, torch.tensor([0.9990, 1.9980]))

    def test_optimizer_with_ga(self):
        param_1 = Variable(torch.tensor([1.0, 2.0]), requires_grad=True)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=0.001)
        )
        optimizer = TZRecOptimizer(keyed_optimizer, gradient_accumulation_steps=2)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.zero_grad()
        self.assertEqual(param_1.grad, None)

        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.step()  # do not update
        torch.testing.assert_close(param_1, torch.tensor([1.0, 2.0]))
        optimizer.zero_grad()  # do not zero_grad
        torch.testing.assert_close(param_1.grad, torch.tensor([1.0, 2.0]))

        param_1.grad += torch.tensor([1.0, 2.0])
        optimizer.step()
        torch.testing.assert_close(param_1, torch.tensor([0.9980, 1.9960]))
        optimizer.zero_grad()
        torch.testing.assert_close(param_1.grad, None)


if __name__ == "__main__":
    unittest.main()
