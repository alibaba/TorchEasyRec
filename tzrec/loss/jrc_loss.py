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


import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss


@torch.fx.wrap
def _label_mask(labels: torch.Tensor) -> torch.Tensor:
    return torch.eye(labels.size(0), dtype=torch.int64, device=labels.device)


@torch.fx.wrap
def _diag_index(labels: torch.Tensor) -> torch.Tensor:
    return torch.arange(0, labels.size(0), dtype=torch.int64, device=labels.device)


class JRCLoss(_Loss):
    """Positive sample probability competes in session.

    https://arxiv.org/abs/2208.06164

    Args:
        alpha (float): cross entropy loss weight.
        same_label_loss (bool): whether use same label jrc loss.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        self._alpha = alpha
        self._ce_loss = CrossEntropyLoss()

    def forward(self, logits: Tensor, labels: Tensor, session_ids: Tensor) -> Tensor:
        """JRC loss.

        Args:
            logits: a `Tensor` with shape [batch_size, 2].
            labels: a `Tensor` with shape [batch_size].
            session_ids: a `Tensor` with shape [batch_size].

        Return:
            loss: a `Tensor`.
        """
        ce_loss = self._ce_loss(logits, labels)
        batch_size = labels.shape[0]
        mask = torch.eq(session_ids.unsqueeze(1), session_ids.unsqueeze(0)).float()
        diag_index = _diag_index(labels)
        logits_neg, logits_pos = logits[:, 0], logits[:, 1]
        diag = _label_mask(labels)
        pos_num = torch.sum(labels)
        neg_num = batch_size - pos_num

        # first, we calculate pos sample loss in during the session.
        pos_mask_index = torch.where(labels == 1.0)[0]
        pos_diag_label = torch.index_select(diag_index, 0, pos_mask_index)
        # pyre-ignore [6]
        logits_pos = logits_pos.unsqueeze(0).tile([pos_num, 1])
        pos_session_mask = torch.index_select(mask, 0, pos_mask_index)
        # pyre-ignore [6]
        y_pos = labels.unsqueeze(0).tile([pos_num, 1])
        diag_pos = torch.index_select(diag, 0, pos_mask_index)
        # we mask not in the same session, is diagonal and is positive.
        logits_pos = (
            logits_pos + ((1 - pos_session_mask) + (1 - diag_pos) * y_pos) * -1e9
        )
        loss_pos = self._ce_loss(logits_pos, pos_diag_label)

        # next, we calculate neg sample loss in during the session.
        neg_mask_index = torch.where(labels == 0.0)[0]
        neg_diag_label = torch.index_select(diag_index, 0, neg_mask_index)
        logits_neg = logits_neg.unsqueeze(0).tile([neg_num, 1])
        neg_session_mask = torch.index_select(mask, 0, neg_mask_index)
        y_neg = (1 - labels).unsqueeze(0).tile([neg_num, 1])
        diag_neg = torch.index_select(diag, 0, neg_mask_index)
        # we mask not in the same session, is diagonal and is negative.
        logits_neg = (
            logits_neg + ((1 - neg_session_mask) + (1 - diag_neg) * y_neg) * -1e9
        )
        loss_neg = self._ce_loss(logits_neg, neg_diag_label)

        ge_loss = (loss_pos * pos_num + loss_neg * neg_num) / batch_size
        loss = self._alpha * ce_loss + (1 - self._alpha) * ge_loss
        # pyre-ignore [7]
        return loss
