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

# Formula referenced from torchrec/metrics/ne.py (BSD-licensed, Meta).
# Reimplemented here so tzrec owns its metric math surface.

from typing import Any

import torch
from torchmetrics import Metric


class NormalizedEntropy(Metric):
    """Normalized Entropy for binary classification.

    NE is the model's cross-entropy divided by the cross-entropy of a constant
    predictor at the population mean label rate. A value below 1 means the
    model beats that baseline.

    Args:
        eta (float): small epsilon used to clamp predictions and the
            population mean away from 0 and 1.
    """

    def __init__(self, eta: float = 1e-12, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.eta = float(eta)
        for name in (
            "cross_entropy_sum",
            "weighted_num_samples",
            "pos_labels",
            "neg_labels",
        ):
            self.add_state(
                name,
                default=torch.zeros(1, dtype=torch.float64),
                dist_reduce_fx="sum",
            )

    # pyre-ignore [14]
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric.

        Args:
            preds (Tensor): a float 1d-tensor of probabilities in [0, 1].
            target (Tensor): a 0/1 1d-tensor of binary labels.
        """
        labels = target.detach().to(torch.float64)
        preds_f64 = preds.detach().to(torch.float64).clamp(self.eta, 1.0 - self.eta)
        ce = -(
            labels * torch.log2(preds_f64)
            + (1.0 - labels) * torch.log2(1.0 - preds_f64)
        )
        # pyre-ignore [16, 29]
        self.cross_entropy_sum += ce.sum()
        # pyre-ignore [16, 29]
        self.weighted_num_samples += labels.numel()
        # pyre-ignore [16, 29]
        self.pos_labels += labels.sum()
        # pyre-ignore [16, 29]
        self.neg_labels += (1.0 - labels).sum()

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        denom = self.weighted_num_samples.clamp(min=self.eta)
        mean_label = (self.pos_labels / denom).clamp(self.eta, 1.0 - self.eta)
        ce_norm = -(
            self.pos_labels * torch.log2(mean_label)
            + self.neg_labels * torch.log2(1.0 - mean_label)
        )
        return (self.cross_entropy_sum / ce_norm).to(torch.float32).squeeze()
