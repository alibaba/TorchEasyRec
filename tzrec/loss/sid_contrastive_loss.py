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

import math
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.distributed.nn as dist_nn
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

# CLIP temperature init (reference CLIP: log(1 / 0.07)) and the cap applied
# before ``exp`` (reference CLIP clamps to ln(100)): an unbounded temperature
# would overflow to +Inf -> NaN grad -> corrupt param.
_LOGIT_SCALE_INIT = math.log(1 / 0.07)
_LOGIT_SCALE_MAX = math.log(100)


class SidContrastiveLoss(_Loss):
    """Masked InfoNCE pair-contrastive loss for mixed (paired + non-paired) batches.

    Modality-agnostic: aligns two reconstructed "views" (``embed_a`` / ``embed_b``)
    against each other and against their originals (``embed_a_ori`` /
    ``embed_b_ori``) with three symmetric InfoNCE terms (self/ori/cl). In a mixed
    batch, non-pair rows (``pair_mask=False``) must not contribute and must not
    serve as negatives. Cross-entropy ignores non-pair queries, and column masks
    exclude non-pair candidates and false negatives identified by equal original
    embeddings while preserving each query's designated positive.

    ``forward`` takes the four ``(B, dim)`` view embeddings plus the ``(B,)`` pair
    mask and returns the scalar mean of the three contrastive terms. The three
    temperatures (self/ori/cl) are learnable parameters owned by this module;
    ``forward`` clamps (to <= ln(100)) and ``exp``s them.
    """

    def __init__(self) -> None:
        super().__init__()
        self.label_index: Optional[torch.Tensor] = None
        self.last_local_batch_size: Optional[int] = None
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        # Learnable contrastive temperatures, one per group (self / ori / cl);
        # registered here so the loss module is self-contained.
        self.logit_scale_self = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)
        self.logit_scale_cl = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)
        self.logit_scale_ori = nn.Parameter(torch.ones([]) * _LOGIT_SCALE_INIT)

    @staticmethod
    def _scaled(logit_scale: torch.Tensor) -> torch.Tensor:
        # Clamp before exp so a large temperature can't overflow to +Inf -> NaN.
        return logit_scale.clamp(max=_LOGIT_SCALE_MAX).exp()

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

    @staticmethod
    @torch.no_grad()
    def _duplicate_masks(
        embed_a_all_ori: torch.Tensor,
        embed_b_all_ori: torch.Tensor,
        label_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mask false negatives using exact equality of original embeddings.

        Exclude three kinds of false negatives:
        1. Duplicate positives: for A -> B, mask other copies of B.
        2. The query itself: when A is the query, mask copies of A.
        3. Other explicit pairs: if A -> B and A -> C both occur, mask C
           when computing A -> B. Apply the same rule in the reverse direction.
        Always preserve each row's designated positive.

        Args:
            embed_a_all_ori: (B_global, dim) original view-a embeddings.
            embed_b_all_ori: (B_global, dim) original view-b embeddings.
            label_index: (B,) local rows' positions in the global batch.

        Returns:
            Four (B, B_global) masks for cross-view a/b and own-view a/b.
        """
        _, unique_emb_ids = torch.unique(
            torch.cat([embed_a_all_ori, embed_b_all_ori]),
            dim=0,
            return_inverse=True,
        )
        all_a_ids, all_b_ids = unique_emb_ids.chunk(2)
        local_a_ids = all_a_ids[label_index].unsqueeze(1)
        local_b_ids = all_b_ids[label_index].unsqueeze(1)
        a_matches_a = local_a_ids == all_a_ids
        b_matches_b = local_b_ids == all_b_ids
        a_matches_b = local_a_ids == all_b_ids
        b_matches_a = local_b_ids == all_a_ids

        cross_a_mask = a_matches_a | b_matches_b | a_matches_b  # A queries global B.
        cross_b_mask = a_matches_a | b_matches_b | b_matches_a  # B queries global A.
        own_a_mask = a_matches_a | b_matches_a  # A queries global A.
        own_b_mask = b_matches_b | a_matches_b  # B queries global B.
        masks = (cross_a_mask, cross_b_mask, own_a_mask, own_b_mask)
        for mask in masks:
            mask.scatter_(1, label_index.unsqueeze(1), False)
        return masks

    def _masked_cross_entropy(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        label_index: torch.Tensor,
        n_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Average bidirectional cross-entropy over valid pair rows.

        Args:
            logits_a: (B, B_global) column-masked logits (view-a branch).
            logits_b: (B, B_global) column-masked logits (view-b branch).
            label_index: (B,) positive column indices; -100 ignores non-pair rows.
            n_valid: scalar pair-row count, clamped to >= 1.

        Returns:
            Scalar mean loss over both directions and valid pair rows.
        """
        ce_a = F.cross_entropy(logits_a, label_index, reduction="none")
        ce_b = F.cross_entropy(logits_b, label_index, reduction="none")
        return (ce_a + ce_b).sum() / (2 * n_valid)

    def forward(
        self,
        embed_a: torch.Tensor,
        embed_b: torch.Tensor,
        embed_a_ori: torch.Tensor,
        embed_b_ori: torch.Tensor,
        pair_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the masked pair-contrastive loss.

        Args:
            embed_a: (B, dim) reconstructed (decoder) output of view a.
            embed_b: (B, dim) reconstructed (decoder) output of view b.
            embed_a_ori: (B, dim) original embedding of view a.
            embed_b_ori: (B, dim) original embedding of view b.
            pair_mask: (B,) bool, True = contrastive-pair sample.

        Returns:
            Tensor: scalar mean of the three contrastive terms (self/ori/cl).
        """
        logit_scale_self = self._scaled(self.logit_scale_self)
        logit_scale_ori = self._scaled(self.logit_scale_ori)
        logit_scale_cl = self._scaled(self.logit_scale_cl)

        local_batch_size = embed_a.size(0)

        # Labels carry the cross-rank offset, so refresh them on batch-size change.
        if local_batch_size != self.last_local_batch_size:
            self.label_index = local_batch_size * self._rank + torch.arange(
                local_batch_size, device=embed_a.device
            )
            self.last_local_batch_size = local_batch_size

        embed_a = F.normalize(embed_a, dim=-1, p=2)
        embed_b = F.normalize(embed_b, dim=-1, p=2)

        # One batched all-gather for all four operands (gradient-preserving).
        embed_a_all, embed_b_all, embed_a_all_ori, embed_b_all_ori = (
            self._all_gather_with_grad([embed_a, embed_b, embed_a_ori, embed_b_ori])
        )

        pair_mask_all = self._gather_bool_mask(pair_mask)
        col_mask = (~pair_mask_all).unsqueeze(0)  # (1, B_global)

        label_index = self.label_index
        assert label_index is not None
        cross_a, cross_b, own_a, own_b = self._duplicate_masks(
            embed_a_all_ori, embed_b_all_ori, label_index
        )
        for mask in (cross_a, cross_b, own_a, own_b):
            mask |= col_mask
        label_index = label_index.masked_fill(~pair_mask, -100)
        n_valid = pair_mask.sum().clamp(min=1)

        # Three symmetric contrastive groups, each (scale, a-target, b-target):
        #   self: recon-a vs recon-b              (vs the other recon view)
        #   ori:  recon vs the counterpart original
        #   cl:   recon vs its own-view original
        groups = (
            (logit_scale_self, embed_b_all, embed_a_all, cross_a, cross_b),
            (logit_scale_ori, embed_b_all_ori, embed_a_all_ori, cross_a, cross_b),
            (logit_scale_cl, embed_a_all_ori, embed_b_all_ori, own_a, own_b),
        )
        loss = embed_a.new_zeros(())
        for scale, a_target, b_target, mask_a, mask_b in groups:
            logits_a = scale * embed_a @ a_target.t()
            logits_b = scale * embed_b @ b_target.t()
            # Keep fully masked rows' backward finite; use the logits dtype to
            # avoid fill-value overflow under autocast.
            neg_fill = torch.finfo(logits_a.dtype).min
            logits_a = logits_a.masked_fill(mask_a, neg_fill)
            logits_b = logits_b.masked_fill(mask_b, neg_fill)
            loss = loss + self._masked_cross_entropy(
                logits_a, logits_b, label_index, n_valid
            )
        return loss / 3
