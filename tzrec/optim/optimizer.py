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
from typing import Any, Optional

from torch.amp import GradScaler
from torchrec.optim import KeyedOptimizer, OptimizerWrapper


class TZRecOptimizer(OptimizerWrapper):
    """TorchEasyRec optimizer wrapper.

    For gradient accumulate / gradient scaler etc.

    Args:
        optimizer (KeyedOptimizer): optimizer to wrap.
        grad_scaler (Optional[GradScaler]): gradient scaler.
        gradient_accumulation_steps (int): gradient accumulate steps.
    """

    def __init__(
        self,
        optimizer: KeyedOptimizer,
        grad_scaler: Optional[GradScaler] = None,
        gradient_accumulation_steps: int = 0,
    ) -> None:
        super().__init__(optimizer)
        self._step = 0
        self._grad_scaler = grad_scaler
        self._gradient_accumulation_steps = gradient_accumulation_steps

    def zero_grad(self, set_to_none: bool = False) -> None:
        """Zero gradients."""
        if (
            self._gradient_accumulation_steps > 1
            and self._step % self._gradient_accumulation_steps == 0
        ):
            self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        """Step."""
        self._step += 1
        if (
            self._gradient_accumulation_steps > 1
            and self._step % self._gradient_accumulation_steps == 0
        ):
            if self._grad_scaler is not None:
                self._grad_scaler.step(self._optimizer)
                self._grad_scaler.update()
            else:
                self._optimizer.step(closure=closure)
