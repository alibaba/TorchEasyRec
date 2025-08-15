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

from typing import Any, Optional

import torch
from torchmetrics import Metric

from tzrec.utils.logging_util import logger


class XAUC(Metric):
    """XAUC metric for short video rec.

    XAUC is O(n^2) complexity, downsampling is necessary in practice. Set
    proper sample_ratio or max_pairs args when use.

    Ref:
        https://arxiv.org/pdf/2206.06003
        https://arxiv.org/pdf/2408.07759
        https://arxiv.org/pdf/2401.07521

    Args:
        sample_ratio(float): The ratio of downsampling pairs. Maximum number
            of pairs is n*(n-1)/2, where n is the number of eval samples.
            Actual number of pairs is n*(n-1)/2 * ratio. Reduce the
            ratio when eval set is large(which is common) and memory is
            limited. Default value 1e-3.
        max_pairs(optional int): The maximum number of pairs to sample. If
            specified, sample_ratio is ignored. Default None.
        in_batch(bool): Get sample pairs within a batch. When True,
            sampling is done per batch, xauc is calculated batch-wise and
            finally averaged, sample_ratio is ignored. Otherwise, xauc is
            calculated on the whole eval set. Pls note that in_batch
            is typically memory efficient and faster, but may have bias.
            Better to shuffle your eval dataset before doing in_batch xauc.
            Default value False.

    """

    def __init__(
        self,
        sample_ratio: float = 1e-3,
        max_pairs: Optional[int] = None,
        in_batch: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        assert sample_ratio > 0 and sample_ratio <= 1.0, (
            "sample_ratio must be between (0, 1]"
        )
        self.sample_ratio = sample_ratio
        self.max_pairs = int(max_pairs) if max_pairs else None
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

    def compute(self) -> torch.Tensor:
        """Compute the metric."""
        logger.info("xauc computing...")
        if self.in_batch:
            return torch.mean(torch.stack(self.batch_xauc))
        else:
            preds = (
                torch.concat(self.eval_preds)
                if isinstance(self.eval_preds, list)
                else self.eval_preds
            )
            target = (
                torch.concat(self.eval_targets)
                if isinstance(self.eval_targets, list)
                else self.eval_targets
            )

            n = int(self.total_sample_count)
            if self.max_pairs:
                assert self.max_pairs < n * (n - 1) // 2, "max_pairs is larger"
                "than maximum possible pairs, please check your setting."
                n_sample_pairs = self.max_pairs
            else:
                n_sample_pairs = int(n * (n - 1) // 2 * self.sample_ratio)

            xauc = self.sampling_xauc(preds, target, n, n_sample_pairs)
            return xauc

    def sampling_xauc(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        n: int,
        n_sample_pairs: int,
    ) -> torch.Tensor:
        """Sample pairs and calc xauc."""
        total_pairs = n * (n - 1) // 2

        # caution: may cost extreme high memory when n_sample_pairs is huge
        sampled_flat = torch.randint(0, total_pairs, (n_sample_pairs,))
        i = torch.floor((-1 + torch.sqrt(1 + 8 * sampled_flat.float())) / 2).long()
        base = i * (i + 1) // 2
        idx_i = sampled_flat - base
        idx_j = i + 1

        # # duplicates removal, caution: slow when n_sample_pairs is large,
        # but unnecessary when the sample pool is big.
        # pairs = torch.stack([idx_i, idx_j], dim=1)
        # unique_pairs = torch.unique(pairs, dim=0)
        # idx_i = unique_pairs[:, 0]
        # idx_j = unique_pairs[:, 1]

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
