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


from typing import Any

import torch
from torchmetrics import Metric


class RecallAtK(Metric):
    """Recall@K.

    Args:
        top_k (int): k for @k metric, calculate top k predictions relevance.
    """

    def __init__(self, top_k: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.top_k = top_k
        self.add_state(
            "num_true_pos", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_total", torch.zeros(1, dtype=torch.long), dist_reduce_fx="sum"
        )

    # pyre-ignore [14]
    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric.

        Args:
            preds (Tensor): a float 2d-tensor of predictions.
            target (Tensor): a bool 2d-tensor of recall target.
        """
        pred_idx = torch.argsort(preds, dim=-1, descending=True)[..., : self.top_k]
        true_pos = torch.gather(target, -1, pred_idx)
        self.num_true_pos += torch.sum(true_pos)
        self.num_total += torch.sum(target)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        return self.num_true_pos / self.num_total
