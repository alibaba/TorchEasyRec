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

import torch
from torchmetrics import Metric


class RelativeL1(Metric):
    """Mean symmetric relative-L1 error ``|t - p| / (max(|t|, |p|) + eps)``.

    A bounded reconstruction-error metric (0 = exact, → 1 = unrelated). It is a
    verbatim port of OpenOneRec's residual-K-Means ``calc_loss`` and is
    deliberately **not** ``torchmetrics.MeanAbsolutePercentageError``, which uses
    the asymmetric ``|t - p| / |t|`` denominator. Aggregation is element-wise
    (count-weighted), so the reported value is the mean over all elements seen.
    """

    higher_is_better = False
    is_differentiable = True

    def __init__(self, epsilon: float = 1e-4, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon
        # float64 sum / long count: float32 loses integer precision past 2**24
        # (~32K rows of a 512-dim embedding) under element-wise aggregation.
        self.add_state(
            "sum_rel",
            default=torch.tensor(0.0, dtype=torch.float64),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate the relative-L1 error between ``preds`` and ``target``.

        Args:
            preds (Tensor): reconstruction, shape (B, D).
            target (Tensor): ground-truth embedding, shape (B, D).
        """
        rel = torch.abs(target - preds) / (
            torch.maximum(torch.abs(target), torch.abs(preds)) + self.epsilon
        )
        self.sum_rel += rel.sum().double()
        self.count += rel.numel()

    def compute(self) -> torch.Tensor:
        """Mean relative-L1 over all elements (NaN before any update)."""
        return self.sum_rel / self.count
