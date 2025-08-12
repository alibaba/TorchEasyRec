# Copyright (c) 2025, Alibaba Group;
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


class XAUC(Metric):
    """XAUC metric for short video rec.

    Ref:
        https://arxiv.org/pdf/2206.06003
        https://arxiv.org/pdf/2408.07759
        https://arxiv.org/pdf/2401.07521

    Args:
        sample_ratio(float): The ratio of sample pairs. Maximum number of
            sample pairs is n*(n-1)/2, where n is the number of samples.
            Actual number of sample pairs is n*(n-1)/2 * ratio. Reduce the
            ratio when eval set is huge and GPU memory is limited. Default
            value 1.0.
        memory_saving(bool): Use memory saving mode or not. When True,
            sampling is done per batch, xauc is calculated batch-wise and
            finally averaged, sample_ratio is ignored. Otherwise, xauc is
            calculated on the whole eval set. Default value False.

    """

    def __init__(
        self, sample_ratio: float = 1.0, memory_saving: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert sample_ratio > 0 and sample_ratio <= 1.0, (
            "sample_ratio must be between (0, 1]"
        )

        self.sample_ratio = sample_ratio
        self.memory_saving = memory_saving

        if memory_saving:
            self.add_state("batch_xauc", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("eval_data", default=[], dist_reduce_fx="cat")
            self.add_state(
                "total_sample_count", default=torch.tensor(0), dist_reduce_fx="sum"
            )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update the metric.

        Args:
            preds (Tensor): a float 1d-tensor of predictions.
            targets (Tensor): a integer 1d-tensor of target.
        """
        if self.memory_saving:
            n = targets.shape[0]
            batch_xauc = self.sampling_auc(preds, targets, n, n)
            self.batch_xauc.append(batch_xauc)
        else:
            self.eval_data.append({"preds": preds, "targets": targets})
            self.total_sample_count += preds.shape[0]

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.memory_saving:
            return self.batch_xauc.mean()
        else:
            n = self.total_sample_count
            max_pairs = self.total_sample_count * (self.total_sample_count - 1) // 2
            n_sample_pairs = int(max_pairs * self.sample_ratio)

            xauc = self.sampling_auc(
                self.eval_data["preds"], self.eval_data["targets"], n, n_sample_pairs
            )
            return xauc

            # all_pairs = torch.triu_indices(n, n, offset=1).t()
            # sampled_indices = torch.randperm(max_pairs)[:n_sample_pairs]
            # sampled_pairs = all_pairs[sampled_indices]

            # idx_i = sampled_pairs[:, 0]
            # idx_j = sampled_pairs[:, 1]

            # preds_i = self.eval_data['preds'][idx_i]
            # preds_j = self.eval_data['preds'][idx_j]
            # targets_i = self.eval_data['targets'][idx_i]
            # targets_j = self.eval_data['targets'][idx_j]

            # correct = ((targets_i - targets_j) * (preds_i - preds_j) > 0).float()
            # xauc = correct.mean()
            # return xauc

    def sampling_auc(
        self, preds: torch.Tensor, targets: torch.Tensor, n: int, n_sample_pairs: int
    ):
        """Sample pairs."""
        max_pairs = n * (n - 1) // 2
        all_pairs = torch.triu_indices(n, n, offset=1).t()
        sampled_indices = torch.randperm(max_pairs)[:n_sample_pairs]
        sampled_pairs = all_pairs[sampled_indices]
        idx_i = sampled_pairs[:, 0]
        idx_j = sampled_pairs[:, 1]

        preds_i = preds[idx_i]
        preds_j = preds[idx_j]
        targets_i = targets[idx_i]
        targets_j = targets[idx_j]

        correct = ((targets_i - targets_j) * (preds_i - preds_j) > 0).float()
        xauc = correct.mean()
        return xauc
