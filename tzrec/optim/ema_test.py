# Copyright (c) 2026, Alibaba Group;
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
from torchrec.optim import KeyedOptimizerWrapper

from tzrec.optim.ema import DenseEMA, EMAOptimizer
from tzrec.protos import optimizer_pb2


class DenseEMATest(unittest.TestCase):
    def test_config_default(self) -> None:
        dense_optimizer = optimizer_pb2.DenseOptimizer()
        self.assertFalse(dense_optimizer.HasField("ema"))

        dense_optimizer.ema.SetInParent()
        self.assertTrue(dense_optimizer.HasField("ema"))
        self.assertAlmostEqual(dense_optimizer.ema.decay, 0.999)

    def test_update_and_average_parameters(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        dense_ema = DenseEMA({"parameter": parameter}, decay=0.5)

        parameter.data.fill_(2.0)
        dense_ema.update()
        parameter.data.fill_(4.0)
        dense_ema.update()

        self.assertEqual(dense_ema.n_averaged.item(), 2)
        torch.testing.assert_close(
            dense_ema.state_dict()["parameter"],
            torch.tensor([3.0]),
        )
        with dense_ema.average_parameters():
            torch.testing.assert_close(parameter, torch.tensor([3.0]))
        torch.testing.assert_close(parameter, torch.tensor([4.0]))
        torch.testing.assert_close(
            dense_ema.state_dict()["parameter"],
            torch.tensor([3.0]),
        )

    def test_average_parameters_restores_after_error(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        dense_ema = DenseEMA({"parameter": parameter}, decay=0.5)
        parameter.data.fill_(2.0)
        dense_ema.update()
        parameter.data.fill_(4.0)

        with self.assertRaisesRegex(RuntimeError, "evaluation failed"):
            with dense_ema.average_parameters():
                torch.testing.assert_close(parameter, torch.tensor([2.0]))
                raise RuntimeError("evaluation failed")
        torch.testing.assert_close(parameter, torch.tensor([4.0]))

    def test_reset(self) -> None:
        first = torch.nn.Parameter(torch.tensor([1.0]))
        second = torch.nn.Parameter(torch.tensor([2.0]))
        dense_ema = DenseEMA({"first": first, "second": second}, decay=0.5)
        dense_ema.update()
        first.data.fill_(3.0)
        second.data.fill_(4.0)
        dense_ema.reset()

        self.assertEqual(dense_ema.n_averaged.item(), 0)
        torch.testing.assert_close(dense_ema.state_dict()["first"], torch.tensor([3.0]))
        torch.testing.assert_close(
            dense_ema.state_dict()["second"], torch.tensor([4.0])
        )

    def test_invalid_decay(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        with self.assertRaisesRegex(ValueError, "decay"):
            DenseEMA({"parameter": parameter}, decay=1.1)

    def test_optimizer_updates_ema_after_step(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        dense_ema = DenseEMA({"parameter": parameter}, decay=0.5)
        optimizer = KeyedOptimizerWrapper(
            {"parameter": parameter},
            lambda params: torch.optim.SGD(params, lr=1.0),
        )
        optimizer = EMAOptimizer(optimizer, dense_ema)

        parameter.grad = torch.tensor([-1.0])
        optimizer.step()
        parameter.grad = torch.tensor([-2.0])
        optimizer.step()

        torch.testing.assert_close(parameter, torch.tensor([4.0]))
        torch.testing.assert_close(
            dense_ema.state_dict()["parameter"],
            torch.tensor([3.0]),
        )


if __name__ == "__main__":
    unittest.main()
