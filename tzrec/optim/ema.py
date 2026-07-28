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

from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchrec.optim import KeyedOptimizer, OptimizerWrapper

DENSE_EMA_N_AVERAGED = "__n_averaged__"


class DenseEMA:
    """Track exponential moving averages of named dense parameters.

    Args:
        named_parameters: Dense parameters keyed by their model FQN.
        decay: Exponential moving average decay in ``[0, 1]``.
    """

    def __init__(
        self,
        named_parameters: Mapping[str, nn.Parameter],
        decay: float,
    ) -> None:
        self._names = list(named_parameters.keys())
        self._source = nn.ParameterList(named_parameters.values())
        device = self._source[0].device if len(self._source) > 0 else None
        self._averaged_model = AveragedModel(
            self._source,
            device=device,
            multi_avg_fn=get_ema_multi_avg_fn(decay),
            use_buffers=False,
        )
        self._averaged_model.requires_grad_(False)

    @property
    def n_averaged(self) -> torch.Tensor:
        """Number of successful optimizer updates included in the average."""
        return self._averaged_model.n_averaged

    def update(self) -> None:
        """Update EMA parameters from the current dense parameters."""
        self._averaged_model.update_parameters(self._source)

    def named_averaged_parameters(self) -> "OrderedDict[str, torch.Tensor]":
        """Return EMA parameters keyed by their original model FQN."""
        return OrderedDict(
            zip(
                self._names,
                self._averaged_model.module.parameters(),
                strict=True,
            )
        )

    def state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        """Return the distributed-checkpoint state for this EMA."""
        state = self.named_averaged_parameters()
        state[DENSE_EMA_N_AVERAGED] = self.n_averaged
        return state

    @torch.no_grad()
    def reset(self) -> None:
        """Reset EMA so the next update copies the current parameters."""
        self.n_averaged.zero_()
        for source, average in zip(
            self._source,
            self._averaged_model.module.parameters(),
            strict=True,
        ):
            average.copy_(source)

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily expose EMA values through the live model parameters."""
        averaged = list(self._averaged_model.module.parameters())
        swapped = 0
        with torch.no_grad():
            for source, average in zip(self._source, averaged, strict=True):
                temporary = source.detach().clone(memory_format=torch.preserve_format)
                source.copy_(average)
                average.copy_(temporary)
                swapped += 1
        try:
            yield
        finally:
            with torch.no_grad():
                for index in range(swapped - 1, -1, -1):
                    source = self._source[index]
                    average = averaged[index]
                    temporary = source.detach().clone(
                        memory_format=torch.preserve_format
                    )
                    source.copy_(average)
                    average.copy_(temporary)


class EMAOptimizer(OptimizerWrapper):
    """Update Dense EMA after successful optimizer steps.

    Args:
        optimizer: Optimizer to wrap.
        dense_ema: Dense EMA state updated after each successful step.
    """

    def __init__(self, optimizer: KeyedOptimizer, dense_ema: DenseEMA) -> None:
        super().__init__(optimizer)
        self._dense_ema = dense_ema

    def step(self, closure: Any = None) -> None:
        """Run an optimizer step and then update Dense EMA."""
        self._optimizer.step(closure=closure)
        self._dense_ema.update()
