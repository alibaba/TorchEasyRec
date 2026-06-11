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

"""CLIP contrastive learning loss with distributed all-gather support."""

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

# Finite large-negative fill for masked-out (recon) logit columns. ``exp()`` of
# it underflows to 0 (same masking effect as ``-inf``), but unlike ``-inf`` it
# keeps cross-entropy and its gradient finite when a row has no valid column
# (a batch/rank with zero clip rows). With ``-inf`` such a row yields NaN and,
# crucially, a NaN gradient (``0 * NaN``) that survives the row mask and poisons
# the encoder. Finite, so a no-clip batch contributes exactly 0 with 0 gradient.
# Kept well within fp16 range (-65504) so it is safe under FP16/BF16 autocast.
_MASKED_LOGIT_FILL = -1e4


class MaskedCLIPLoss(_Loss):
    """Masked CLIP loss for mixed recon+clip batches.

    In a mixed batch, recon rows (clip_mask=False) should not
    contribute to CLIP loss, and recon columns should not serve as
    negatives.  This module applies row and column masks to achieve
    selective contrastive learning without data-dependent branching,
    ensuring ``torch.compile`` compatibility.

    Input dict keys:
        'image_embed':      (B, D)  quantized output of first feature
        'text_embed':       (B, D)  quantized output of second feature
        'image_embed_ori':  (B, D)  original embedding of first feature
        'text_embed_ori':   (B, D)  original embedding of second feature
        'logit_scale_self': scalar  self-contrast temperature
        'logit_scale_cl':   scalar  cross-modal contrast temperature
        'logit_scale':      scalar  original feature contrast temperature

    Output dict keys:
        'clip_loss':  scalar  mean of three losses (self/ori/cl)
        'clip_acc':   scalar  contrast accuracy (%); 0 during training
        'loss_self':  scalar  quantized vs quantized
        'loss_ori':   scalar  quantized vs original
        'loss_cl':    scalar  quantized vs counterpart original
    """

    def __init__(self) -> None:
        super().__init__()
        self.labels: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def _all_gather_with_grad(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-gather tensors across workers with gradient support.

        In single-process mode, returns the input tensors unchanged. In
        multi-process mode, uses ``torch.distributed.nn.functional
        .all_gather`` — the built-in differentiable collective (its backward
        sum-reduces the per-rank grads and returns this rank's slice), so no
        custom ``autograd.Function`` is needed.

        Args:
            tensors (List[Tensor]): list of tensors to gather.

        Returns:
            List[Tensor]: gathered tensors, each (world_size * B, ...).
        """
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return tensors
        gathered: List[torch.Tensor] = []
        for tensor in tensors:
            tensor_all = dist_nn.all_gather(tensor)  # differentiable, per rank
            gathered.append(torch.cat(tensor_all, dim=0))
        return gathered

    @staticmethod
    def _gather_bool_mask(mask: torch.Tensor) -> torch.Tensor:
        """All-gather bool mask across distributed workers."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return mask
        mask_list = [torch.zeros_like(mask) for _ in range(dist.get_world_size())]
        dist.all_gather(mask_list, mask)
        return torch.cat(mask_list, dim=0)

    def _masked_cross_entropy(
        self,
        logits_i: torch.Tensor,
        logits_t: torch.Tensor,
        safe_labels: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy on column-masked logits, row-masked average.

        Args:
            logits_i: (B, B_global) column-masked logits (image branch).
            logits_t: (B, B_global) column-masked logits (text branch).
            safe_labels: (B,) labels with recon rows fallback to safe col.
            clip_mask: (B,) bool, True = clip row.
        """
        ce_i = F.cross_entropy(logits_i, safe_labels, reduction="none")
        ce_t = F.cross_entropy(logits_t, safe_labels, reduction="none")
        # Backstop only: the finite _MASKED_LOGIT_FILL already keeps the
        # all-recon row finite, so this guards solely against a non-finite
        # logit arriving from upstream (e.g. an overflowed logit_scale).
        ce_i = torch.nan_to_num(ce_i, nan=0.0)
        ce_t = torch.nan_to_num(ce_t, nan=0.0)

        # Row mask: only clip rows contribute; clamp(min=1) keeps a no-clip
        # batch at 0 (not 0/0). Combined with the finite fill, a batch with no
        # clip rows yields exactly 0 loss and 0 gradient.
        n_valid = clip_mask.float().sum().clamp(min=1)
        return ((ce_i + ce_t) * clip_mask.float()).sum() / (2 * n_valid)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clip_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward with mask.

        Args:
            outputs: feature dict, see class docstring.
            clip_mask: (B,) bool, True = clip sample.
        """
        image_embed = outputs["image_embed"]
        text_embed = outputs["text_embed"]
        image_embed_ori = outputs["image_embed_ori"]
        text_embed_ori = outputs["text_embed_ori"]
        logit_scale = outputs["logit_scale"]
        logit_scale_self = outputs["logit_scale_self"]
        logit_scale_cl = outputs["logit_scale_cl"]

        local_batch_size = image_embed.size(0)

        # Update labels when batch size changes (multi-GPU offset)
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * self._rank + torch.arange(
                local_batch_size, device=image_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # L2 normalize quantized features
        image_embed = F.normalize(image_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # All-gather across GPUs (with gradient support)
        image_embed_all, text_embed_all = self._all_gather_with_grad(
            [image_embed, text_embed]
        )
        image_embed_all_ori, text_embed_all_ori = self._all_gather_with_grad(
            [image_embed_ori, text_embed_ori]
        )

        # --- Compute six groups of logits (image/text × self/ori/cl) ---
        logits_img_self = logit_scale_self * image_embed @ text_embed_all.t()
        logits_txt_self = logit_scale_self * text_embed @ image_embed_all.t()

        logits_img_ori = logit_scale * image_embed @ text_embed_all_ori.t()
        logits_txt_ori = logit_scale * text_embed @ image_embed_all_ori.t()

        logits_img_cl = logit_scale_cl * image_embed @ image_embed_all_ori.t()
        logits_txt_cl = logit_scale_cl * text_embed @ text_embed_all_ori.t()

        # --- Column mask: recon columns -> large-negative (not as negatives) ---
        # Finite fill (not -inf) so an all-recon row keeps a finite, non-NaN
        # gradient; see _MASKED_LOGIT_FILL.
        clip_mask_all = self._gather_bool_mask(clip_mask)
        col_mask = (~clip_mask_all).unsqueeze(0)  # (1, B_global)

        logits_img_self = logits_img_self.masked_fill(col_mask, _MASKED_LOGIT_FILL)
        logits_txt_self = logits_txt_self.masked_fill(col_mask, _MASKED_LOGIT_FILL)
        logits_img_ori = logits_img_ori.masked_fill(col_mask, _MASKED_LOGIT_FILL)
        logits_txt_ori = logits_txt_ori.masked_fill(col_mask, _MASKED_LOGIT_FILL)
        logits_img_cl = logits_img_cl.masked_fill(col_mask, _MASKED_LOGIT_FILL)
        logits_txt_cl = logits_txt_cl.masked_fill(col_mask, _MASKED_LOGIT_FILL)

        # --- Safe labels: recon rows fallback to first clip column ---
        labels = self.labels
        fallback = clip_mask.long().argmax()  # first clip sample index
        safe_labels = torch.where(clip_mask, labels, fallback.expand_as(labels))

        # --- Masked CE for three loss groups ---
        loss_self = self._masked_cross_entropy(
            logits_img_self, logits_txt_self, safe_labels, clip_mask
        )
        loss_ori = self._masked_cross_entropy(
            logits_img_ori, logits_txt_ori, safe_labels, clip_mask
        )
        loss_cl = self._masked_cross_entropy(
            logits_img_cl, logits_txt_cl, safe_labels, clip_mask
        )

        clip_loss = (loss_self + loss_ori + loss_cl) / 3

        # Retrieval accuracy is diagnostic-only; skip the four argmax+eq+sum
        # reductions during training (recover via the eval pass).
        if self.training:
            acc = torch.zeros((), device=clip_loss.device)
        else:
            with torch.no_grad():
                n_valid = clip_mask.float().sum().clamp(min=1)
                correct = (
                    (logits_img_self.argmax(-1).eq(safe_labels) & clip_mask).sum()
                    + (logits_txt_self.argmax(-1).eq(safe_labels) & clip_mask).sum()
                    + (logits_img_ori.argmax(-1).eq(safe_labels) & clip_mask).sum()
                    + (logits_txt_ori.argmax(-1).eq(safe_labels) & clip_mask).sum()
                )
                acc = 100 * correct / (n_valid * 4)

        return {
            "clip_loss": clip_loss,
            "clip_acc": acc,
            "loss_self": loss_self,
            "loss_ori": loss_ori,
            "loss_cl": loss_cl,
        }
