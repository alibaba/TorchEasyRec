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

"""SidReconLoss: mask-aware RQ-VAE reconstruction loss (input vs. decoder)."""

from typing import Optional

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss

from tzrec.modules.utils import div_no_nan


class SidReconLoss(_Loss):
    """Reconstruction loss for RQ-VAE: per-row distance reduced to a scalar.

    ``forward(x_hat, x, mask)`` computes the per-row distance for the configured
    ``recon_type`` and averages it over the masked-in rows (all rows if ``mask``
    is None; the mixed recon+contrastive path passes ``recon_mask`` to score the
    reconstruction-only rows). Registered as a ``_loss_modules`` entry alongside
    the commitment / contrastive losses and, like them, returns a scalar.

    Args:
        recon_type (str): the distance, ``"l2"`` (mse), ``"l1"`` or ``"cos"``.
            Default: ``"l2"``.
    """

    def __init__(self, recon_type: str = "l2") -> None:
        super().__init__()
        if recon_type not in ("l2", "l1", "cos"):
            raise ValueError(
                f"recon_type must be 'l2', 'l1' or 'cos', got {recon_type!r}"
            )
        self.recon_type = recon_type

    def _per_row(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-row reconstruction distance, shape (B,)."""
        if self.recon_type == "l1":
            return F.l1_loss(x_hat, x, reduction="none").mean(dim=-1)
        if self.recon_type == "cos":
            return 1 - F.cosine_similarity(x_hat, x, dim=-1)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)  # "l2"

    @staticmethod
    def _masked_mean(
        per_sample: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Mean over the masked-in rows (all rows if ``mask`` is None).

        The masked mean divides by the valid-row count (``div_no_nan`` keeps an
        empty mask at 0). No data-dependent branching -> ``torch.compile``-friendly.
        """
        if mask is None:
            return per_sample.mean()
        mask = mask.float()
        return div_no_nan((per_sample * mask).sum(), mask.sum())

    def forward(
        self,
        x_hat: torch.Tensor,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mask-aware reconstruction loss.

        Args:
            x_hat (Tensor): reconstruction (decoder output), shape (B, D).
            x (Tensor): the input it reconstructs, shape (B, D).
            mask (Tensor, optional): per-row bool; rows to include (all if None).

        Returns:
            Tensor: scalar reconstruction loss.
        """
        return self._masked_mean(self._per_row(x_hat, x), mask)
