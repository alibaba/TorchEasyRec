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
        in_batch(bool): Get sample pair within a batch. When True,
            sampling is done per batch, xauc is calculated batch-wise and
            finally averaged, sample_ratio is ignored. Otherwise, xauc is
            calculated on the whole eval set. Pls note that in_batch
            is typically memory efficient and faster, but may have bias.
            Better to shuffle your eval dataset before doing in_batch xauc.
            Default value False.

    """

    def __init__(
        self, sample_ratio: float = 1.0, in_batch: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert sample_ratio > 0 and sample_ratio <= 1.0, (
            "sample_ratio must be between (0, 1]"
        )
        self.sample_ratio = sample_ratio
        self.in_batch = in_batch

        if in_batch:
            self.add_state("batch_xauc", default=[], dist_reduce_fx="cat")
        else:
            self.add_state("eval_preds", default=[], dist_reduce_fx="cat")
            self.add_state("eval_targets", default=[], dist_reduce_fx="cat")
            self.add_state(
                "total_sample_count", default=torch.tensor(0), dist_reduce_fx="sum"
            )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update the metric.

        Args:
            preds (Tensor): a float 1d-tensor of predictions.
            targets (Tensor): a integer 1d-tensor of target.
        """
        if self.in_batch:
            batch_xauc = self.in_batch_xauc(preds, targets)
            self.batch_xauc.append(batch_xauc)
        else:
            self.eval_preds.append(preds)
            self.eval_targets.append(targets)
            self.total_sample_count += torch.tensor(preds.shape[0])

        self.batch_size = int(preds.shape[0])

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        if self.in_batch:
            return self.batch_xauc.mean()
        else:
            preds = self.eval_preds
            target = self.eval_targets

            n = self.total_sample_count
            n_sample_pairs = int(n * (n - 1) // 2 * self.sample_ratio)

            xauc = self.sampling_xauc(preds, target, n, n_sample_pairs, self.batch_size)
            return xauc

    def sampling_xauc(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        n: int,
        n_sample_pairs: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Sample pairs."""
        n = int(n)
        total_pairs = n * (n - 1) // 2
        n_sample_pairs = int(n_sample_pairs)

        idx_i = []
        idx_j = []

        n_batch = n_sample_pairs // batch_size

        for _ in range(n_batch):
            sampled_flat = torch.randint(0, total_pairs, (batch_size,))

            i = torch.floor((-1 + torch.sqrt(1 + 8 * sampled_flat.float())) / 2).long()
            base = i * (i + 1) // 2
            batch_idx_i = sampled_flat - base
            batch_idx_j = i + 1

            idx_i.append(batch_idx_i)
            idx_j.append(batch_idx_j)

        idx_i = torch.concat(idx_i)
        idx_j = torch.concat(idx_j)

        idx_i = idx_i.to(preds.device)
        idx_j = idx_j.to(preds.device)

        preds_i = preds[idx_i]
        preds_j = preds[idx_j]
        targets_i = targets[idx_i]
        targets_j = targets[idx_j]

        correct = ((targets_i - targets_j) * (preds_i - preds_j) > 0).float()
        xauc = correct.mean()
        return xauc

    def in_batch_xauc(
        self, batch_preds: torch.Tensor, batch_targets: torch.Tensor
    ) -> torch.Tensor:
        """Xauc in a batch."""
        b = batch_preds.shape[0]
        all_pairs = torch.triu_indices(b, b, offset=1).t()
        idx_i = all_pairs[:, 0]
        idx_j = all_pairs[:, 1]

        preds_i = batch_preds[idx_i]
        preds_j = batch_preds[idx_j]
        targets_i = batch_targets[idx_i]
        targets_j = batch_targets[idx_j]

        correct = ((targets_i - targets_j) * (preds_i - preds_j) > 0).float()
        xauc = correct.mean()
        return xauc
