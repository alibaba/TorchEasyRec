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


import os
from typing import Any, List

import torch
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import _binary_auroc_compute


def custom_reduce_fx(data_list: List[torch.Tensor]) -> torch.Tensor:
    """Custom reduce func for distributed training. Distribute data to different GPUs.

    Args:
            data_list (list): list of tensors,
                    each tensor is a 2d-tensor of shape (3, num_samples).

    Returns:
            Tensor: a 2d-tensor of shape (3, num_samples_on_one_gpu).
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("RANK", 0))

    data_list_reduce = []
    for data in data_list:
        key_mask = data[2, :] % world_size == local_rank
        pred_selected = torch.masked_select(data[0, :], key_mask)
        target_selected = torch.masked_select(data[1, :], key_mask)
        key_selected = torch.masked_select(data[2, :], key_mask)
        data_list_reduce.append(
            torch.stack([pred_selected, target_selected, key_selected])
        )
    return torch.cat(data_list_reduce, dim=1)


class GroupedAUC(Metric):
    """Grouped AUC."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.add_state("eval_data", default=[], dist_reduce_fx=custom_reduce_fx)

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
        self.eval_data.append(torch.stack([preds, target, grouping_key]))

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if isinstance(self.eval_data, list):  # compatible with cpu mode
            self.eval_data = torch.cat(self.eval_data, dim=1)  # pyre-ignore [16][6]

        preds, target, grouping_key = (
            self.eval_data[0, :],  # pyre-ignore [29]
            self.eval_data[1, :],  # pyre-ignore [29]
            self.eval_data[2, :],  # pyre-ignore [29]
        )

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
        mean_gauc = torch.mean(torch.Tensor(aucs))
        return mean_gauc
