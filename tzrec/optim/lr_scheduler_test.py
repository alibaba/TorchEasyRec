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


import math
import unittest

import torch

from tzrec.optim import lr_scheduler


class LRSchedulerTest(unittest.TestCase):
    def test_constant_lr(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.ConstantLR(opt)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)

    def test_exponential_decay_lr(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.ExponentialDecayLR(
            opt, decay_size=1, decay_factor=0.7, by_epoch=True
        )
        self.assertTrue(lr.by_epoch)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.007)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0049)

    def test_exponential_decay_lr_with_warmup(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.ExponentialDecayLR(
            opt,
            decay_size=1,
            decay_factor=0.7,
            warmup_size=2,
            warmup_learning_rate=0.005,
        )
        self.assertFalse(lr.by_epoch)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0075)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.007)

    def test_manual_step_lr(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.ManualStepLR(
            opt,
            schedule_sizes=[2, 4, 8],
            learning_rates=[0.008, 0.004, 0.002],
        )
        lr_gts = [0.01, 0.01, 0.008, 0.008, 0.004, 0.004, 0.004, 0.004, 0.002, 0.002]
        for lr_gt in lr_gts:
            lr.step()
            self.assertAlmostEqual(opt.param_groups[0]["lr"], lr_gt)

    def test_manual_step_lr_with_warmup(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.004)
        lr = lr_scheduler.ManualStepLR(
            opt,
            schedule_sizes=[2, 4, 8],
            learning_rates=[0.008, 0.004, 0.002],
            warmup=True,
        )
        lr_gts = [0.006, 0.008, 0.008, 0.008, 0.004, 0.004, 0.004, 0.004, 0.002, 0.002]
        for lr_gt in lr_gts:
            lr.step()
            self.assertAlmostEqual(opt.param_groups[0]["lr"], lr_gt)

    def test_cosine_annealing_lr(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.CosineAnnealingLR(opt, T_max=4, by_epoch=True)
        self.assertTrue(lr.by_epoch)
        # step 0->1: t=1, cos(pi*1/4)
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 1->2: t=2, cos(pi*2/4) = cos(pi/2) = 0 -> lr = 0.005
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.005)
        # step 2->3: t=3, cos(pi*3/4)
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 3 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 3->4: t=4, cos(pi) = -1 -> lr = 0.0
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.0)

    def test_cosine_annealing_lr_with_warmup(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.CosineAnnealingLR(
            opt, T_max=4, warmup_size=2, warmup_learning_rate=0.002
        )
        self.assertFalse(lr.by_epoch)
        # warmup step 0->1: scale=0.5, lr=0.002+(0.01-0.002)*0.5=0.006
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.006)
        # warmup step 1->2: scale=1.0, lr=0.01
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        # cosine step t=1: cos(pi*1/4)
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)

    def test_cosine_annealing_warm_restarts_lr(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.CosineAnnealingWarmRestartsLR(opt, T_0=3)
        # Period of 3, T_mult=1 so fixed period
        # step 1: t=1, cos(pi*1/3)
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 3))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 2: t=2, cos(pi*2/3)
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 2 / 3))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 3: restart, t=0 -> cos(0) = 1 -> lr=0.01
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        # step 4: t=1 again
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 3))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)

    def test_cosine_annealing_warm_restarts_lr_with_T_mult(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.CosineAnnealingWarmRestartsLR(opt, T_0=2, T_mult=2)
        # Period 0: T_i=2, steps 0-1
        # step 1: elapsed=1, T_cur=1, T_i=2
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 2))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 2: elapsed=2, restart, Period 1: T_i=4, T_cur=0
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        # step 3: elapsed=3, T_cur=1, T_i=4
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 4: elapsed=4, T_cur=2, T_i=4
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 2 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 5: elapsed=5, T_cur=3, T_i=4
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 3 / 4))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)
        # step 6: elapsed=6, restart, Period 2: T_i=8, T_cur=0
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)

    def test_cosine_annealing_warm_restarts_lr_with_warmup(self) -> None:
        params = [torch.tensor([1.0, 2.0])]
        opt = torch.optim.Adam(params, lr=0.01)
        lr = lr_scheduler.CosineAnnealingWarmRestartsLR(
            opt, T_0=3, warmup_size=2, warmup_learning_rate=0.002
        )
        # warmup step 0->1: scale=0.5
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.006)
        # warmup step 1->2: scale=1.0
        lr.step()
        self.assertAlmostEqual(opt.param_groups[0]["lr"], 0.01)
        # cosine elapsed=1, T_cur=1, T_i=3
        lr.step()
        expected = 0.5 * 0.01 * (1 + math.cos(math.pi * 1 / 3))
        self.assertAlmostEqual(opt.param_groups[0]["lr"], expected)


if __name__ == "__main__":
    unittest.main()
