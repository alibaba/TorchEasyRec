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


class UniqueRatio(Metric):
    """Codebook-coverage metric: mean of per-batch (unique rows / batch size).

    Each ``update`` counts the unique rows of a ``(B, n_layers)`` semantic-ID
    code tensor and accumulates the per-batch ratio; ``compute`` returns the
    running mean. Empty batches (``B == 0``, e.g. an empty final DDP/TorchRec
    shard) are skipped. States reduce by ``sum`` across ranks.
    """

    higher_is_better = True
    is_differentiable = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.add_state("ratio_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, codes: torch.Tensor) -> None:
        """Accumulate the unique-ratio of one batch of codes.

        Args:
            codes (Tensor): semantic-ID codes, shape (B, n_layers).
        """
        batch_size = codes.shape[0]
        if batch_size == 0:
            return
        unique = torch.unique(codes, dim=0).shape[0]
        self.ratio_sum += unique / batch_size
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Mean per-batch unique ratio (NaN before any non-empty update)."""
        return self.ratio_sum / self.count
