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

"""CommitmentLoss: VQ-VAE commitment loss for residual quantizers."""

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CommitmentLoss(nn.Module):
    """Commitment loss between the encoder output and the quantized vectors.

    Operates on a residual quantizer's per-layer cumulative quantized vectors
    (the ``latents`` of :class:`~tzrec.modules.sid.types.ResidualQuantizerOutput`).
    Both VQ-VAE directions are summed and averaged over the residual layers:

      - ``loss1`` = encoder-toward-quant (gradient flows into the encoder), w1
      - ``loss2`` = quant-toward-encoder (gradient flows into the codebook), w2

    Args:
        latent_weight (Sequence[float]): commitment weights ``[w1, w2]``.
            Default: ``(1.0, 0.5)``.
        commitment_type (str): distance, ``"l2"``, ``"l1"`` or ``"cos"``.
            Default: ``"l2"``.
    """

    def __init__(
        self,
        latent_weight: Sequence[float] = (1.0, 0.5),
        commitment_type: str = "l2",
    ) -> None:
        super().__init__()
        if len(latent_weight) != 2:
            raise ValueError(
                f"latent_weight must have exactly 2 values [w1, w2], got "
                f"{list(latent_weight)}"
            )
        assert commitment_type in ("l2", "l1", "cos"), (
            f"commitment_type must be 'l2', 'l1' or 'cos', got {commitment_type!r}"
        )
        self.commitment_w1, self.commitment_w2 = latent_weight
        self.commitment_type = commitment_type

    def forward(self, encoder_out: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """Compute the commitment loss.

        Args:
            encoder_out (Tensor): encoder output (the quantizer input), shape
                (B, D).
            latents (Tensor): per-layer cumulative quantized vectors, shape
                (B, n_layers, D).

        Returns:
            Tensor: scalar commitment loss (averaged over layers).
        """
        x = encoder_out.unsqueeze(1)  # (B, 1, D) -> broadcasts over layers
        if self.commitment_type == "cos":
            loss1 = (1 - F.cosine_similarity(x, latents.detach(), dim=-1)).mean()
            loss2 = (1 - F.cosine_similarity(x.detach(), latents, dim=-1)).mean()
        elif self.commitment_type == "l1":
            loss1 = (x - latents.detach()).abs().mean()
            loss2 = (x.detach() - latents).abs().mean()
        else:  # "l2"
            loss1 = (x - latents.detach()).pow(2.0).mean()
            loss2 = (x.detach() - latents).pow(2.0).mean()
        return self.commitment_w1 * loss1 + self.commitment_w2 * loss2
