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
from torchmetrics.functional.classification.auroc import _binary_auroc_compute
from torchmetrics.utilities.data import dim_zero_cat


class GroupedAUC(Metric):
    """Grouped AUC.

    Args:
        top_k (int): k for @k metric, calculate top k predictions relevance.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("grouping_key", default=[], dist_reduce_fx="cat")

    # pyre-ignore [14]
    def update(
        self, preds: torch.Tensor, target: torch.Tensor, grouping_key: torch.Tensor
    ) -> None:
        """Update the metric.

        Args:
            preds (Tensor): a float 1d-tensor of predictions.
            target (Tensor): a integer 1d-tensor of target.
            grouping_key (Tensor): a integer 1d-tensor with group id.
        """
        self.preds.append(preds)
        self.target.append(target)
        self.grouping_key.append(grouping_key)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        # pyre-ignore [6]
        grouping_key = dim_zero_cat(self.grouping_key)
        # pyre-ignore [6]
        preds = dim_zero_cat(self.preds)
        # pyre-ignore [6]
        target = dim_zero_cat(self.target)

        sorted_grouping_key, indices = torch.sort(grouping_key)
        sorted_preds = preds[indices]
        sorted_target = target[indices]

        _, counts = torch.unique_consecutive(sorted_grouping_key, return_counts=True)
        counts = counts.tolist()

        grouped_preds = torch.split(sorted_preds, counts)
        grouped_target = torch.split(sorted_target, counts)

        aucs = []
        for preds, target in zip(grouped_preds, grouped_target):
            mean_target = torch.mean(target.to(torch.float32)).item()
            if mean_target > 0 and mean_target < 1:
                aucs.append(_binary_auroc_compute((preds, target), None))

        return torch.mean(torch.Tensor(aucs))
