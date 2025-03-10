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
from typing import Any, List, Tuple

import torch
from torch import distributed as dist
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import _binary_auroc_compute


def custom_reduce_fx(data_list: List[torch.Tensor]) -> Tuple[torch.Tensor]:
    """Custom reduce func for distributed training. Distribute data to different GPUs.

    Args:
            data_list (list): list of tensors,
                    each tensor is a 2d-tensor of shape (3, num_samples).

    Returns:
            Tensor: a 2d-tensor of shape (3, num_samples_on_one_gpu).
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("RANK", 0))

    pred_reduce = []
    target_reduce = []
    key_reduce = []
    for data in data_list:
        for i in range(world_size):
            key_mask = (
                data["grouping_key"][i] % world_size == local_rank
            )  # pyre-ignore [6]
            pred_selected = torch.masked_select(
                data["preds"][i], key_mask
            )  # pyre-ignore [6]
            target_selected = torch.masked_select(
                data["target"][i], key_mask
            )  # pyre-ignore [6]
            key_selected = torch.masked_select(
                data["grouping_key"][i], key_mask
            )  # pyre-ignore [6]

            pred_reduce.append(pred_selected)
            target_reduce.append(target_selected)
            key_reduce.append(key_selected)
    return torch.cat(pred_reduce), torch.cat(target_reduce), torch.cat(key_reduce)


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
        self.eval_data.append(
            {"preds": preds, "target": target, "grouping_key": grouping_key}
        )

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if not dist.is_initialized():
            preds = torch.cat(
                [data["preds"] for data in self.eval_data]
            )  # pyre-ignore [29]
            target = torch.cat(
                [data["target"] for data in self.eval_data]
            )  # pyre-ignore [29]
            grouping_key = torch.cat(
                [data["grouping_key"] for data in self.eval_data]
            )  # pyre-ignore [29]
        else:
            preds, target, grouping_key = (
                self.eval_data[0],  # pyre-ignore [29]
                self.eval_data[1],  # pyre-ignore [29]
                self.eval_data[2],  # pyre-ignore [29]
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
        sum_gauc = torch.sum(torch.Tensor(aucs))

        # gather metric data across processes
        if dist.is_initialized() and dist.get_world_size() > 1:
            group_cnt = len(aucs)
            gather_metric_list = [
                torch.empty_like(sum_gauc.cuda())
                if dist.get_backend() == "nccl"
                else torch.empty_like(sum_gauc)
                for _ in range(dist.get_world_size())
            ]
            gather_group_count = [
                torch.empty_like(torch.Tensor([group_cnt]).cuda())
                if dist.get_backend() == "nccl"
                else torch.empty_like(torch.Tensor([group_cnt]))
                for _ in range(dist.get_world_size())
            ]

            dist.all_gather(
                gather_metric_list,
                sum_gauc.cuda() if dist.get_backend() == "nccl" else sum_gauc,
            )
            dist.all_gather(
                gather_group_count,
                torch.Tensor([group_cnt]).cuda()
                if dist.get_backend() == "nccl"
                else torch.Tensor([group_cnt]),
            )

            total_sum_gauc = torch.sum(torch.stack(gather_metric_list))
            total_group_cnt = torch.sum(torch.stack(gather_group_count))

            mean_gauc = total_sum_gauc / total_group_cnt
        else:
            mean_gauc = sum_gauc / len(aucs)
        return mean_gauc
