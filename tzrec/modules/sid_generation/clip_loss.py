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

"""CLIP contrastive learning loss with distributed all-gather support."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all workers with gradient support.

    Standard ``dist.all_gather`` detaches gradients; this custom
    ``autograd.Function`` keeps the computation graph connected so
    that contrastive losses can backpropagate through gathered tensors.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """All-gather ``x`` across ranks, returning one tensor per rank."""
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads: torch.Tensor) -> torch.Tensor:
        """Sum-reduce the per-rank grads and return this rank's slice."""
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def _all_gather_with_grad(
    tensors: List[torch.Tensor],
) -> List[torch.Tensor]:
    """All-gather tensors across distributed workers with gradient support.

    In single-process mode, returns input tensors unchanged.
    In multi-process mode, uses GatherLayer for backward-compatible
    all_gather.

    Args:
        tensors (List[Tensor]): list of tensors to gather.

    Returns:
        List[Tensor]: gathered tensors, each (world_size * B, ...).
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensors

    gathered: List[torch.Tensor] = []
    for tensor in tensors:
        tensor_all = GatherLayer.apply(tensor)
        gathered.append(torch.cat(tensor_all, dim=0))
    return gathered


class CLIPLoss(nn.Module):
    """Multi-level CLIP contrastive learning loss.

    Computes three InfoNCE contrastive losses and returns their mean:
    - loss_self:  quantized features vs quantized features
                  (paired items remain similar after quantization)
    - loss_ori:   quantized features vs original features
                  (quantization preserves original semantics)
    - loss_cl:    quantized features vs counterpart original features
                  (cross-modal alignment)

    Supports distributed all_gather to aggregate global batch.

    Input dict keys:
        'image_embed':      (B, D)  quantized output of first feature
        'text_embed':       (B, D)  quantized output of second feature
        'image_embed_ori':  (B, D)  original embedding of first feature
        'text_embed_ori':   (B, D)  original embedding of second feature
        'logit_scale_self': scalar  self-contrast temperature
        'logit_scale_cl':   scalar  cross-modal contrast temperature
        'logit_scale':      scalar  original feature contrast temperature

    Output dict keys:
        'clip_loss':  scalar  mean of three losses
        'loss_self':  scalar
        'loss_ori':   scalar
        'loss_cl':    scalar
        'clip_acc':   scalar  contrast accuracy (%)
    """

    def __init__(self) -> None:
        super().__init__()
        self.labels: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-level CLIP contrastive loss.

        Args:
            outputs (Dict[str, Tensor]): feature dict, see class docstring.

        Returns:
            Dict[str, Tensor]: losses and accuracy.
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
        image_embed_all, text_embed_all = _all_gather_with_grad(
            [image_embed, text_embed]
        )
        image_embed_all_ori, text_embed_all_ori = _all_gather_with_grad(
            [image_embed_ori, text_embed_ori]
        )

        # --- loss_self: quantized vs quantized ---
        logits_img_self = logit_scale_self * image_embed @ text_embed_all.t()
        logits_txt_self = logit_scale_self * text_embed @ image_embed_all.t()

        # --- loss_ori: quantized vs original ---
        logits_img_ori = logit_scale * image_embed @ text_embed_all_ori.t()
        logits_txt_ori = logit_scale * text_embed @ image_embed_all_ori.t()

        # --- loss_cl: quantized vs counterpart original ---
        logits_img_cl = logit_scale_cl * image_embed @ image_embed_all_ori.t()
        logits_txt_cl = logit_scale_cl * text_embed @ text_embed_all_ori.t()

        loss_self = (
            F.cross_entropy(logits_img_self, self.labels)
            + F.cross_entropy(logits_txt_self, self.labels)
        ) / 2
        loss_ori = (
            F.cross_entropy(logits_img_ori, self.labels)
            + F.cross_entropy(logits_txt_ori, self.labels)
        ) / 2
        loss_cl = (
            F.cross_entropy(logits_img_cl, self.labels)
            + F.cross_entropy(logits_txt_cl, self.labels)
        ) / 2

        loss = (loss_self + loss_ori + loss_cl) / 3

        # Retrieval accuracy is a diagnostic, not a training signal — only
        # spend the four argmax + eq + sum reductions in eval (training-loop
        # accuracy can be reconstructed from the eval pass).
        if self.training:
            acc = torch.zeros((), device=loss.device)
        else:
            with torch.no_grad():
                correct = (
                    logits_img_self.argmax(-1).eq(self.labels).sum()
                    + logits_txt_self.argmax(-1).eq(self.labels).sum()
                    + logits_img_ori.argmax(-1).eq(self.labels).sum()
                    + logits_txt_ori.argmax(-1).eq(self.labels).sum()
                )
                acc = 100 * correct / (local_batch_size * 4)

        return {
            "clip_loss": loss,
            "loss_self": loss_self,
            "loss_ori": loss_ori,
            "loss_cl": loss_cl,
            "clip_acc": acc,
        }


class MaskedCLIPLoss(nn.Module):
    """Masked CLIP loss for mixed recon+clip batches.

    In a mixed batch, recon rows (clip_mask=False) should not
    contribute to CLIP loss, and recon columns should not serve as
    negatives.  This module applies row and column masks to achieve
    selective contrastive learning without data-dependent branching,
    ensuring ``torch.compile`` compatibility.
    """

    def __init__(self) -> None:
        super().__init__()
        self.labels: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0

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
        # NaN can occur when all logits are -inf (all-recon edge case)
        ce_i = torch.nan_to_num(ce_i, nan=0.0)
        ce_t = torch.nan_to_num(ce_t, nan=0.0)

        n_valid = clip_mask.float().sum().clamp(min=1)
        return ((ce_i + ce_t) * clip_mask.float()).sum() / (2 * n_valid)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        clip_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward with mask.

        Args:
            outputs: same format as CLIPLoss input dict.
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
        image_embed_all, text_embed_all = _all_gather_with_grad(
            [image_embed, text_embed]
        )
        image_embed_all_ori, text_embed_all_ori = _all_gather_with_grad(
            [image_embed_ori, text_embed_ori]
        )

        # --- Compute six groups of logits (same as CLIPLoss) ---
        logits_img_self = logit_scale_self * image_embed @ text_embed_all.t()
        logits_txt_self = logit_scale_self * text_embed @ image_embed_all.t()

        logits_img_ori = logit_scale * image_embed @ text_embed_all_ori.t()
        logits_txt_ori = logit_scale * text_embed @ image_embed_all_ori.t()

        logits_img_cl = logit_scale_cl * image_embed @ image_embed_all_ori.t()
        logits_txt_cl = logit_scale_cl * text_embed @ text_embed_all_ori.t()

        # --- Column mask: recon columns -> -inf (not as negatives) ---
        clip_mask_all = self._gather_bool_mask(clip_mask)
        col_mask = (~clip_mask_all).unsqueeze(0)  # (1, B_global)

        logits_img_self = logits_img_self.masked_fill(col_mask, float("-inf"))
        logits_txt_self = logits_txt_self.masked_fill(col_mask, float("-inf"))
        logits_img_ori = logits_img_ori.masked_fill(col_mask, float("-inf"))
        logits_txt_ori = logits_txt_ori.masked_fill(col_mask, float("-inf"))
        logits_img_cl = logits_img_cl.masked_fill(col_mask, float("-inf"))
        logits_txt_cl = logits_txt_cl.masked_fill(col_mask, float("-inf"))

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
