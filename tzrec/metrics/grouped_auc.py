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
from typing import Any

import torch
from torch import distributed as dist
from torchmetrics import Metric
from torchmetrics.functional.classification.auroc import _binary_auroc_compute


class GroupedAUC(Metric):
    """Grouped AUC."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(sync_on_compute=False, **kwargs)
        self.add_state("preds", default=[], dist_reduce_fx=None)
        self.add_state("target", default=[], dist_reduce_fx=None)
        self.add_state("grouping_key", default=[], dist_reduce_fx=None)
        self._world_size = int(os.environ.get("WORLD_SIZE", 1))
        self._rank = int(os.environ.get("RANK", 0))

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
        if dist.is_initialized() and self._world_size > 1:
            dest_ranks = grouping_key % self._world_size

            preds_list = []
            target_list = []
            grouping_key_list = []
            for i in range(self._world_size):
                mask = dest_ranks == i
                preds_list.append(preds[mask])
                target_list.append(target[mask])
                grouping_key_list.append(grouping_key[mask])

            input_splits = torch.tensor(
                [t.size(0) for t in preds_list], dtype=torch.int64, device=preds.device
            )
            output_splits = torch.empty_like(input_splits)
            work = dist.all_to_all_single(output_splits, input_splits, async_op=True)

            input_preds_t = torch.cat(preds_list)
            input_target_t = torch.cat(target_list)
            input_grouping_key_t = torch.cat(grouping_key_list)
            work.wait()

            outputs = []
            works = []
            for input_t in [input_preds_t, input_target_t, input_grouping_key_t]:
                output_t = torch.empty(
                    output_splits.sum().item(),
                    dtype=input_t.dtype,
                    device=input_t.device,
                )
                works.append(
                    dist.all_to_all_single(
                        output_t,
                        input_t,
                        output_split_sizes=output_splits.tolist(),
                        input_split_sizes=input_splits.tolist(),
                        async_op=True,
                    )
                )
                outputs.append(output_t)
            for work in works:
                work.wait()

            self.preds.append(outputs[0])
            self.target.append(outputs[1])
            self.grouping_key.append(outputs[2])
        else:
            self.preds.append(preds)
            self.target.append(target)
            self.grouping_key.append(grouping_key)

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        preds = torch.cat(self.preds)
        target = torch.cat(self.target)
        grouping_key = torch.cat(self.grouping_key)

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
        sum_gauc = torch.sum(torch.tensor(aucs, device=preds.device))
        group_cnt = torch.tensor(len(aucs), device=preds.device)

        if dist.is_initialized() and self._world_size > 1:
            dist.all_reduce(sum_gauc)
            dist.all_reduce(group_cnt)

        mean_gauc = sum_gauc / group_cnt
        return mean_gauc
