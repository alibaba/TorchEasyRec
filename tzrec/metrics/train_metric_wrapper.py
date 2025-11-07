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


import torch
from torch import Tensor, nn
from torchmetrics import Metric

from tzrec.metrics.decay_auc import DecayAUC


class TrainMetricWrapper(nn.Module):
    """Metric wrapper when training.

    Args:
        metric_module (Metric): metric_module.
        decay_rate (float): metric decay rate.
        decay_step (int): decay step for decay,
    """

    def __init__(
        self, metric_module: Metric, decay_rate: float = 0.5, decay_step: int = 100
    ) -> None:
        super().__init__()
        self._decay_rate = decay_rate
        self._decay_step = decay_step
        self._metric_module = metric_module
        self._value = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self._step_total_value = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        self._step_cnt = nn.Parameter(
            torch.tensor(0, dtype=torch.int), requires_grad=False
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric module."""
        self._metric_module.update(preds, target)
        self._step_cnt += 1
        if self._step_cnt % self._decay_step == 0:
            if isinstance(self._metric_module, DecayAUC):
                self._metric_module.decay(self._decay_rate)
                self._value.data = self._metric_module.compute()
            else:
                value = self._metric_module.compute()
                self._metric_module.reset()
                if torch.all(self._value == 0.0):
                    self._value.data = value
                else:
                    self._value.data = (
                        self._decay_rate * self._value + (1 - self._decay_rate) * value
                    )

    def compute(self) -> Tensor:
        """Get metric value."""
        return self._value.data
