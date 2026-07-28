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
from torch.amp import GradScaler
from torchrec.optim import KeyedOptimizerWrapper

from tzrec.optim.ema import DenseEMA, EMAOptimizer
from tzrec.optim.optimizer import TZRecOptimizer


class TZRecOptimizerTest(unittest.TestCase):
    def test_optimizer(self):
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
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
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
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

    def test_ema_updates_only_on_gradient_accumulation_boundary(self):
        param_1 = torch.tensor([1.0], requires_grad=True)
        dense_ema = DenseEMA({"param_1": param_1}, decay=0.5)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=1.0)
        )
        optimizer = TZRecOptimizer(
            EMAOptimizer(keyed_optimizer, dense_ema),
            gradient_accumulation_steps=2,
        )

        param_1.grad = torch.tensor([-1.0])
        optimizer.step()
        self.assertEqual(dense_ema.n_averaged.item(), 0)
        torch.testing.assert_close(param_1, torch.tensor([1.0]))

        optimizer.step()
        self.assertEqual(dense_ema.n_averaged.item(), 1)
        torch.testing.assert_close(param_1, torch.tensor([2.0]))
        torch.testing.assert_close(
            dense_ema.named_averaged_parameters()["param_1"],
            torch.tensor([2.0]),
        )

    def test_ema_does_not_update_when_grad_scaler_skips_step(self):
        param_1 = torch.tensor([1.0], requires_grad=True)
        dense_ema = DenseEMA({"param_1": param_1}, decay=0.5)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=1.0)
        )
        grad_scaler = GradScaler("cpu")
        optimizer = TZRecOptimizer(
            EMAOptimizer(keyed_optimizer, dense_ema),
            grad_scaler=grad_scaler,
        )

        grad_scaler.scale(param_1.sum() * torch.tensor(float("inf"))).backward()
        optimizer.step()

        self.assertEqual(dense_ema.n_averaged.item(), 0)
        torch.testing.assert_close(param_1, torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
