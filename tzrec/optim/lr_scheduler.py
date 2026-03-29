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


import bisect
import math
from typing import List

from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer

from tzrec.utils.load_class import get_register_class_meta

_LR_CLASS_MAP = {}
_meta_cls = get_register_class_meta(_LR_CLASS_MAP)


class BaseLR(LRScheduler, metaclass=_meta_cls):
    """LearningRate Scheduler base class."""

    def __init__(self, optimizer: Optimizer, by_epoch: bool = False) -> None:
        self._by_epoch = by_epoch
        super().__init__(optimizer)

    # pyre-ignore [3]
    def get_lr(self):
        """Calculates the learning rate."""
        lrs = self._get_lr()
        # when base_lr is equal to 0, we freeze lr to 0
        # instead of using the scheduler lr
        return [
            base_lr if base_lr == 0 else lr for base_lr, lr in zip(self.base_lrs, lrs)
        ]

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        raise NotImplementedError

    @property
    def by_epoch(self) -> bool:
        """Schedule by epoch or not."""
        return self._by_epoch


class ConstantLR(BaseLR):
    """Constant LearningRate Scheduler."""

    def __init__(self, optimizer: Optimizer) -> None:
        super().__init__(optimizer, by_epoch=True)

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        return self.base_lrs


class ExponentialDecayLR(BaseLR):
    """Exponential Decay LearningRate Scheduler.

    Args:
        optimizer (Optimizer): an instance of Optimizer.
        decay_size (int): decay steps or epochs.
        decay_factor (float): decay rate.
        staircase (bool): if true, decay the learning rate at discrete intervals.
        warmup_learning_rate (float): warmup start learning rate.
        warmup_size (int): warmup steps or epochs.
        min_learning_rate (float): minimum learning rate.
        by_epoch (bool): schedule by epoch or by step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        decay_size: int,
        decay_factor: float,
        staircase: bool = True,
        warmup_learning_rate: float = 0.0,
        warmup_size: int = 0,
        min_learning_rate: float = 0.0,
        by_epoch: bool = False,
    ) -> None:
        self._decay_size = decay_size
        self._decay_factor = decay_factor
        self._staircase = staircase
        self._warmup_learning_rate = warmup_learning_rate
        self._warmup_size = warmup_size
        self._min_learning_rate = min_learning_rate
        super().__init__(optimizer, by_epoch=by_epoch)

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        step_count = max(self._step_count - 1, 0)
        if step_count < self._warmup_size:
            scale = step_count / self._warmup_size
            lr = [
                (base_lr - self._warmup_learning_rate) * scale
                + self._warmup_learning_rate
                for base_lr in self.base_lrs
            ]
        else:
            p = (step_count - self._warmup_size) / self._decay_size
            if self._staircase:
                p = math.floor(p)
            scale = math.pow(self._decay_factor, p)
            lr = [
                max(base_lr * scale, self._min_learning_rate)
                for base_lr in self.base_lrs
            ]
        return lr


class ManualStepLR(BaseLR):
    """Manual Step LearningRate Scheduler.

    Args:
        optimizer (Optimizer): an instance of Optimizer.
        schedule_steps (list): a list of global steps or epochs at which to
            switch learning.
        learning_rates (list): a list of learning rates corresponding to intervals.
        warmup (bool): whether to linearly interpolate learning rates for steps in
            [0, schedule_steps[0]].
        by_epoch (bool): schedule by epoch or by step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedule_sizes: List[int],
        learning_rates: List[float],
        warmup: bool = False,
        by_epoch: bool = False,
    ) -> None:
        self._schedule_sizes = schedule_sizes
        self._learning_rates = learning_rates
        self._warmup = warmup
        super().__init__(optimizer, by_epoch=by_epoch)

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        step_count = max(self._step_count - 1, 0)
        idx = bisect.bisect_left(self._schedule_sizes, step_count)
        if idx > 0:
            lr = [self._learning_rates[idx - 1] for _ in self.base_lrs]
        elif self._warmup:
            scale = step_count / self._schedule_sizes[0]
            lr = [
                (self._learning_rates[0] - base_lr) * scale + base_lr
                for base_lr in self.base_lrs
            ]
        else:
            lr = self.base_lrs
        return lr


class CosineAnnealingLR(BaseLR):
    """Cosine Annealing LearningRate Scheduler.

    Decays the learning rate following a cosine curve from base_lr to
    min_learning_rate over T_max steps or epochs, with optional linear warmup.

    Args:
        optimizer (Optimizer): an instance of Optimizer.
        T_max (int): total number of steps or epochs for cosine annealing.
        min_learning_rate (float): minimum learning rate.
        warmup_learning_rate (float): warmup start learning rate.
        warmup_size (int): warmup steps or epochs.
        by_epoch (bool): schedule by epoch or by step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        min_learning_rate: float = 0.0,
        warmup_learning_rate: float = 0.0,
        warmup_size: int = 0,
        by_epoch: bool = False,
    ) -> None:
        self._T_max = T_max
        self._min_learning_rate = min_learning_rate
        self._warmup_learning_rate = warmup_learning_rate
        self._warmup_size = warmup_size
        super().__init__(optimizer, by_epoch=by_epoch)

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        step_count = max(self._step_count - 1, 0)
        if step_count < self._warmup_size:
            scale = step_count / self._warmup_size
            return [
                (base_lr - self._warmup_learning_rate) * scale
                + self._warmup_learning_rate
                for base_lr in self.base_lrs
            ]
        t = min(step_count - self._warmup_size, self._T_max)
        cos_scale = 0.5 * (1 + math.cos(math.pi * t / self._T_max))
        return [
            self._min_learning_rate + (base_lr - self._min_learning_rate) * cos_scale
            for base_lr in self.base_lrs
        ]


class CosineAnnealingWarmRestartsLR(BaseLR):
    """Cosine Annealing with Warm Restarts LearningRate Scheduler.

    Decays the learning rate following a cosine curve with periodic restarts.
    After each period, the learning rate resets to base_lr and the period
    length is multiplied by T_mult.

    Args:
        optimizer (Optimizer): an instance of Optimizer.
        T_0 (int): number of steps or epochs for the first cosine period.
        T_mult (int): factor to grow period length after each restart.
        min_learning_rate (float): minimum learning rate.
        warmup_learning_rate (float): warmup start learning rate.
        warmup_size (int): warmup steps or epochs.
        by_epoch (bool): schedule by epoch or by step.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        min_learning_rate: float = 0.0,
        warmup_learning_rate: float = 0.0,
        warmup_size: int = 0,
        by_epoch: bool = False,
    ) -> None:
        self._T_0 = T_0
        self._T_mult = T_mult
        self._min_learning_rate = min_learning_rate
        self._warmup_learning_rate = warmup_learning_rate
        self._warmup_size = warmup_size
        super().__init__(optimizer, by_epoch=by_epoch)

    def _get_lr(self) -> List[float]:
        """Calculates the learning rate."""
        step_count = max(self._step_count - 1, 0)
        if step_count < self._warmup_size:
            scale = step_count / self._warmup_size
            return [
                (base_lr - self._warmup_learning_rate) * scale
                + self._warmup_learning_rate
                for base_lr in self.base_lrs
            ]
        elapsed = step_count - self._warmup_size
        if self._T_mult == 1:
            T_cur = elapsed % self._T_0
            T_i = self._T_0
        else:
            n = math.floor(
                math.log(elapsed / self._T_0 * (self._T_mult - 1) + 1, self._T_mult)
            )
            T_i = self._T_0 * (self._T_mult**n)
            cycle_start = self._T_0 * (self._T_mult**n - 1) // (self._T_mult - 1)
            T_cur = elapsed - cycle_start
        cos_scale = 0.5 * (1 + math.cos(math.pi * T_cur / T_i))
        return [
            self._min_learning_rate + (base_lr - self._min_learning_rate) * cos_scale
            for base_lr in self.base_lrs
        ]
