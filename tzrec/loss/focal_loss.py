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

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.loss import _Loss


class BinaryFocalLoss(_Loss):
    """Focal loss for binary classification task.

    Args:
        gamma(float, optional): the power ratio
        alpha(float, optional): the balance parameter for positive and negative classes.
        reduction (str, optional): Specifies the reduction to apply to the
            output: `none` | `mean`. `none`: no reduction will be applied
            , `mean`: the weighted mean of the output is taken.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self._gamma = gamma
        self._alpha = alpha
        self._reduction = reduction

        assert gamma >= 0, "Value of gamma should be greater than or equal to zero."
        assert alpha > 0 and alpha < 1, "Value of alpha should be in (0, 1)."
        assert reduction in ("none", "mean", "sum"), (
            "reduction should be one of ('none', 'mean', 'sum')."
        )

    def forward(
        self,
        preds: Tensor,
        labels: Tensor,
    ) -> Tensor:
        """Binary focal loss.

        Args:
            preds: a `Tensor` with shape [batch_size,].
            labels: a `Tensor` with shape [batch_size,].

        Returns:
            loss: a `Tensor` with shape [batch_size] if reduction is 'none',
                    otherwise with shape ().
        """
        torch._assert(preds.dim() == 1, "preds must be 1-D")
        torch._assert(labels.dim() == 1, "labels must be 1-D")

        p = F.sigmoid(preds)
        weight = self._alpha * labels * torch.pow(1 - p, self._gamma) + (
            1 - self._alpha
        ) * (1 - labels) * torch.pow(p, self._gamma)
        weight = weight.detach()
        loss = F.binary_cross_entropy_with_logits(
            preds, labels, weight=weight, reduction=self._reduction
        )
        return loss
