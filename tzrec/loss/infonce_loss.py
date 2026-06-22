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

"""Masked InfoNCE contrastive loss with distributed all-gather support."""

from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class MaskedInfoNCELoss(_Loss):
    """Masked InfoNCE contrastive loss for mixed (paired + non-paired) batches.

    Modality-agnostic: aligns two reconstructed "views" (``embed_a``/``embed_b``)
    against each other and against their originals (``embed_a_ori``/
    ``embed_b_ori``) with three symmetric InfoNCE terms (self/ori/cl). In a mixed
    batch, non-pair rows (``pair_mask=False``) must not contribute and must not
    serve as negatives; row/column masks achieve this without data-dependent
    branching (``torch.compile``-friendly).

    Input dict keys (all embeddings shape (B, dim)):
        'embed_a':          reconstructed (decoder) output of view a
        'embed_b':          reconstructed (decoder) output of view b
        'embed_a_ori':      original embedding of view a
        'embed_b_ori':      original embedding of view b
        'logit_scale_self': scalar  temperature: recon-a vs recon-b
        'logit_scale_cl':   scalar  temperature: recon vs same-view original
        'logit_scale':      scalar  temperature: recon vs counterpart original

    Output dict keys:
        'loss':  scalar  mean of the three contrastive losses (self/ori/cl)
    """

    def __init__(self) -> None:
        super().__init__()
        self.labels: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0

    @staticmethod
    def _all_gather_with_grad(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
        """All-gather tensors across workers with gradient support.

        Single-process: returns the inputs unchanged. Multi-process: uses the
        built-in differentiable ``torch.distributed.nn.functional.all_gather``,
        so no custom ``autograd.Function`` is needed.

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
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        safe_labels: torch.Tensor,
        pair_mask_f: torch.Tensor,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Masked cross-entropy on column-masked logits, row-masked average.

        Args:
            logits_a: (B, B_global) column-masked logits (view-a branch).
            logits_b: (B, B_global) column-masked logits (view-b branch).
            safe_labels: (B,) labels with non-pair rows fallback to a safe col.
            pair_mask_f: (B,) float pair mask (1.0 = pair row).
            n_valid: scalar pair-row count, clamped to >= 1.
        """
        ce_a = F.cross_entropy(logits_a, safe_labels, reduction="none")
        ce_b = F.cross_entropy(logits_b, safe_labels, reduction="none")
        # Backstop against a non-finite upstream logit (e.g. overflowed scale).
        ce_a = torch.nan_to_num(ce_a, nan=0.0)
        ce_b = torch.nan_to_num(ce_b, nan=0.0)

        return ((ce_a + ce_b) * pair_mask_f).sum() / (2 * n_valid)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        pair_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward with the pair mask.

        Args:
            outputs: feature dict, see class docstring.
            pair_mask: (B,) bool, True = contrastive-pair sample.
        """
        embed_a = outputs["embed_a"]
        embed_b = outputs["embed_b"]
        embed_a_ori = outputs["embed_a_ori"]
        embed_b_ori = outputs["embed_b_ori"]
        logit_scale = outputs["logit_scale"]
        logit_scale_self = outputs["logit_scale_self"]
        logit_scale_cl = outputs["logit_scale_cl"]

        local_batch_size = embed_a.size(0)

        # Update labels when batch size changes (multi-GPU offset)
        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * self._rank + torch.arange(
                local_batch_size, device=embed_a.device
            )
            self.last_local_batch_size = local_batch_size

        # L2 normalize the reconstructed features
        embed_a = F.normalize(embed_a, dim=-1, p=2)
        embed_b = F.normalize(embed_b, dim=-1, p=2)

        # All-gather across GPUs (with gradient support)
        embed_a_all, embed_b_all = self._all_gather_with_grad([embed_a, embed_b])
        embed_a_all_ori, embed_b_all_ori = self._all_gather_with_grad(
            [embed_a_ori, embed_b_ori]
        )

        # --- Compute six groups of logits (a/b × self/ori/cl) ---
        logits_a_self = logit_scale_self * embed_a @ embed_b_all.t()
        logits_b_self = logit_scale_self * embed_b @ embed_a_all.t()

        logits_a_ori = logit_scale * embed_a @ embed_b_all_ori.t()
        logits_b_ori = logit_scale * embed_b @ embed_a_all_ori.t()

        logits_a_cl = logit_scale_cl * embed_a @ embed_a_all_ori.t()
        logits_b_cl = logit_scale_cl * embed_b @ embed_b_all_ori.t()

        # Mask non-pair columns out of the negatives with the dtype's most negative
        # finite value: below any real logit (masks like -inf), but finite so an
        # all-non-pair row gives a finite CE/grad instead of 0*NaN.
        pair_mask_all = self._gather_bool_mask(pair_mask)
        col_mask = (~pair_mask_all).unsqueeze(0)  # (1, B_global)
        neg_fill = torch.finfo(logits_a_self.dtype).min

        logits_a_self = logits_a_self.masked_fill(col_mask, neg_fill)
        logits_b_self = logits_b_self.masked_fill(col_mask, neg_fill)
        logits_a_ori = logits_a_ori.masked_fill(col_mask, neg_fill)
        logits_b_ori = logits_b_ori.masked_fill(col_mask, neg_fill)
        logits_a_cl = logits_a_cl.masked_fill(col_mask, neg_fill)
        logits_b_cl = logits_b_cl.masked_fill(col_mask, neg_fill)

        # --- Safe labels: non-pair rows fallback to the first pair column ---
        labels = self.labels
        fallback = pair_mask.long().argmax()  # first pair sample index
        safe_labels = torch.where(pair_mask, labels, fallback.expand_as(labels))

        # --- Masked CE for three loss groups (shared row mask + valid count) ---
        pair_mask_f = pair_mask.float()
        n_valid = pair_mask_f.sum().clamp(min=1)
        loss_self = self._masked_cross_entropy(
            logits_a_self, logits_b_self, safe_labels, pair_mask_f, n_valid
        )
        loss_ori = self._masked_cross_entropy(
            logits_a_ori, logits_b_ori, safe_labels, pair_mask_f, n_valid
        )
        loss_cl = self._masked_cross_entropy(
            logits_a_cl, logits_b_cl, safe_labels, pair_mask_f, n_valid
        )

        loss = (loss_self + loss_ori + loss_cl) / 3

        return {"loss": loss}
