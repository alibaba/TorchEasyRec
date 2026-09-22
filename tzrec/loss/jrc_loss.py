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


from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss

from tzrec.ops.scatter_ops import lengths_to_index, scatter_logsumexp

# Logit assigned to samples that must not compete: exp underflows to exactly
# 0 after the segment max shift, so they drop out of the log-sum-exp.
_MASKED_LOGIT = -1e9


class JRCLoss(_Loss):
    """Positive sample probability competes in session.

    https://arxiv.org/abs/2208.06164

    The session term of the paper is a softmax of each positive against the
    negatives of its session (on the positive logit) and of each negative
    against the positives of its session (on the negative logit).  Both
    reduce to ``softplus(logsumexp(competitors) - own logit)``, where the
    log-sum-exp is one value per session, so the term costs two segment
    reductions instead of a batch-by-batch session mask.

    Args:
        alpha (float): cross entropy loss weight.
        reduction (str, optional): Specifies the reduction to apply to the
            output: `none` | `mean`. `none`: no reduction will be applied
            , `mean`: the weighted mean of the output is taken.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self._alpha = alpha
        self._reduction = reduction
        self._ce_loss = CrossEntropyLoss(reduction=reduction)

    def forward(
        self,
        logits: Tensor,
        labels: Tensor,
        lengths: Tensor,
        index: Optional[Tensor] = None,
    ) -> Tensor:
        """JRC loss.

        Without ``index``, ``logits`` and ``labels`` are laid out
        session by session in ``lengths`` order, each session contiguous.

        Args:
            logits: a `Tensor` with shape [batch_size, 2].
            labels: a `Tensor` with shape [batch_size].
            lengths: a `Tensor` with shape [num_sessions], samples per
                session, summing to batch_size.
            index: a `Tensor` with shape [batch_size], session of each
                sample (torch_scatter's ``index``), for samples in arbitrary
                order.

        Return:
            loss: a `Tensor` with shape [batch_size] if reduction is 'none',
                    otherwise with shape ().
        """
        ce_loss = self._ce_loss(logits, labels)

        if index is None:
            index = lengths_to_index(lengths, output_size=logits.size(0))
        logits_neg, logits_pos = logits[:, 0], logits[:, 1]
        is_pos = labels == 1
        # Competitors of a positive: the session's negatives on the positive
        # logit; of a negative: the session's positives on the negative logit.
        neg_competitors = torch.where(is_pos, _MASKED_LOGIT, logits_pos)
        pos_competitors = torch.where(is_pos, logits_neg, _MASKED_LOGIT)
        num_sessions = lengths.size(0)
        lse_neg = scatter_logsumexp(
            neg_competitors.unsqueeze(-1), index, num_sessions
        ).squeeze(-1)
        lse_pos = scatter_logsumexp(
            pos_competitors.unsqueeze(-1), index, num_sessions
        ).squeeze(-1)
        ge_loss = torch.where(
            is_pos,
            F.softplus(lse_neg.index_select(0, index) - logits_pos),
            F.softplus(lse_pos.index_select(0, index) - logits_neg),
        )
        if self._reduction != "none":
            ge_loss = ge_loss.mean()

        return self._alpha * ce_loss + (1 - self._alpha) * ge_loss
