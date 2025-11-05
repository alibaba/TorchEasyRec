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

from typing import Any, List, Optional, Union

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import _binary_auroc_compute
from torchmetrics.functional.classification.precision_recall_curve import (
    _binary_precision_recall_curve_format,
    _binary_precision_recall_curve_update,
)


def _adjust_threshold_arg(
    thresholds: Optional[Union[int, List[float], Tensor]] = None,
    device: Optional[torch.device] = None,
) -> Optional[Tensor]:
    """Convert threshold arg for list and int to tensor format."""
    if isinstance(thresholds, int):
        return torch.linspace(0, 1, thresholds, device=device)
    if isinstance(thresholds, list):
        return torch.tensor(thresholds, device=device)
    return thresholds


class DecayAUC(Metric):
    """AUC for train will decay."""

    def __init__(self, thresholds: int = 200, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        thresholds = torch.linspace(0, 1, thresholds)
        self.register_buffer("thresholds", thresholds, persistent=False)
        self.add_state(
            "confmat",
            default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "buffer_confmat",
            default=torch.zeros(len(thresholds), 2, 2, dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update metric buffer states."""
        preds, target, _ = _binary_precision_recall_curve_format(
            preds, target, self.thresholds
        )
        state = _binary_precision_recall_curve_update(preds, target, self.thresholds)
        self.buffer_confmat += state

    def decay(self, decay: float):
        """Update decay state."""
        if torch.any(self.confmat > 0):
            self.confmat = decay * self.confmat + (1 - decay) * self.buffer_confmat
        else:
            self.confmat += self.buffer_confmat
        attr = "buffer_confmat"
        default = self._defaults[attr]
        current_val = getattr(self, attr)
        if isinstance(default, Tensor):
            setattr(self, attr, default.detach().clone().to(current_val.device))
        else:
            setattr(self, attr, [])

    def compute(self) -> Tensor:
        """Compute metric."""
        return _binary_auroc_compute(self.confmat, self.thresholds)
