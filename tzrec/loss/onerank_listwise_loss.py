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

"""Per-request list-wise InfoNCE over a jagged candidate list (paper 2.5)."""

import math
from typing import Optional

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from tzrec.modules.gr.onerank_jagged import (
    jagged_segment_ids,
    jagged_segment_max,
    jagged_segment_sum,
)

# CLIP temperature cap applied before ``exp`` (reference CLIP clamps to
# ln(100)): an unbounded temperature would overflow to +Inf -> NaN grad ->
# corrupt param.
_LOGIT_SCALE_MAX = math.log(100)


class OneRankListwiseLoss(_Loss):
    """Softmax cross-entropy over the candidates of one request.

    The negatives of a candidate are the other candidates of the *same*
    request, so this stays inside the jagged layout and needs no cross-rank
    all-gather -- unlike an in-batch contrastive loss such as
    :class:`~tzrec.loss.sid_contrastive_loss.SidContrastiveLoss`, gathering
    other ranks here would add candidates that are not negatives of this
    list at all.

    Two kinds of request contribute nothing and are masked out rather than
    branched on, so the shapes stay data-independent (``torch.compile``
    friendly):

    * no positive -- the target distribution is undefined;
    * no negative -- the objective degenerates to "make all scores equal",
      which is a gradient with no ranking information in it.

    A request with several positives uses the multi-positive form, i.e. the
    mean of the positives' log-probabilities.

    Args:
        temperature_init (float): initial softmax temperature; the logits
            are multiplied by ``1 / temperature``.
        learnable_temperature (bool): learn the temperature. Stored as
            ``log(1 / temperature)`` and clamped before ``exp``.
    """

    def __init__(
        self,
        temperature_init: float = 0.07,
        learnable_temperature: bool = True,
    ) -> None:
        super().__init__()
        if not temperature_init > 0:
            raise ValueError(f"temperature_init must be > 0, got {temperature_init}.")
        logit_scale = torch.ones([]) * math.log(1.0 / temperature_init)
        if learnable_temperature:
            self.logit_scale = nn.Parameter(logit_scale)
        else:
            self.register_buffer("logit_scale", logit_scale)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lengths: torch.Tensor,
        loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the list-wise InfoNCE term.

        ``logits``, ``labels`` and ``lengths`` must all be in the same
        request order, with the candidates of a request laid out
        contiguously.

        Args:
            logits (torch.Tensor): ``(total,)`` per-candidate score.
            labels (torch.Tensor): ``(total,)`` per-candidate label; any
                non-zero value counts as a positive.
            lengths (torch.Tensor): ``(B,)`` candidates per request,
                summing to ``total``.
            loss_weight (torch.Tensor, optional): scalar multiplier, used
                to turn the local-batch mean into a global-batch mean.

        Returns:
            torch.Tensor: scalar loss.
        """
        segment_ids = jagged_segment_ids(lengths)
        # Clamp before exp so a large temperature can't overflow to +Inf.
        scale = self.logit_scale.clamp(max=_LOGIT_SCALE_MAX).exp()
        scaled = (logits * scale).unsqueeze(-1)

        maxes = jagged_segment_max(scaled.detach(), lengths, segment_ids)
        shifted = scaled - maxes.index_select(0, segment_ids)
        # A non-empty segment always sums to >= 1, because the row holding
        # the segment max contributes exp(0) == 1.  So the clamp is exact
        # where it matters and only rewrites empty segments, where log(0)
        # would otherwise leak -inf into the tensor.
        denom = jagged_segment_sum(torch.exp(shifted), lengths, segment_ids).clamp(
            min=1.0
        )
        log_probs = shifted - torch.log(denom).index_select(0, segment_ids)

        positives = (labels != 0).to(log_probs.dtype).unsqueeze(-1)
        num_pos = jagged_segment_sum(positives, lengths, segment_ids)
        pos_log_prob = jagged_segment_sum(log_probs * positives, lengths, segment_ids)

        num_candidates = lengths.to(num_pos.dtype).unsqueeze(-1)
        valid = ((num_pos > 0) & (num_pos < num_candidates)).to(log_probs.dtype)
        per_request = -pos_log_prob / num_pos.clamp(min=1.0)
        # Backstop against a non-finite upstream logit; masked-out requests
        # must not turn the sum into NaN via 0 * NaN.
        per_request = torch.nan_to_num(per_request, nan=0.0)

        loss = (per_request * valid).sum() / valid.sum().clamp(min=1.0)
        if loss_weight is not None:
            loss = loss * loss_weight
        return loss
