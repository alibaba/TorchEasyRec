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


if __name__ == "__main__":
    unittest.main()
