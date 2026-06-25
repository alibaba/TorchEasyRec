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

"""SidReconLoss: per-row RQ-VAE reconstruction distance (input vs. decoder)."""

import torch
from torch.nn import functional as F
from torch.nn.modules.loss import _Loss


class SidReconLoss(_Loss):
    """Per-row reconstruction distance for the configured ``recon_type``.

    ``forward(x_hat, x)`` returns the per-row distance ``(B,)``
    (``reduction="none"``); the model reduces it (a masked mean over the
    reconstruction rows). Registered as a ``_loss_modules`` entry alongside the
    commitment / contrastive losses.

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

    def forward(self, x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Per-row reconstruction distance.

        Args:
            x_hat (Tensor): reconstruction (decoder output), shape (B, D).
            x (Tensor): the input it reconstructs, shape (B, D).

        Returns:
            Tensor: per-row distance, shape (B,).
        """
        if self.recon_type == "l1":
            return F.l1_loss(x_hat, x, reduction="none").mean(dim=-1)
        if self.recon_type == "cos":
            return 1 - F.cosine_similarity(x_hat, x, dim=-1)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=-1)  # "l2"
