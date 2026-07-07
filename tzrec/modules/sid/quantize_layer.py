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

"""QuantizeLayer: the per-layer quantizer interface shared by SID backends."""

from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn

from tzrec.modules.sid.types import QuantizeOutput


class QuantizeLayer(nn.Module):
    """One quantize layer: assign inputs to a codebook and look codes up.

    Shared interface for the K-Means backend
    (:class:`~tzrec.modules.sid.kmeans_quantize.KMeansQuantizeLayer`) and the RQ-VAE
    backend's vector-quantize layer, so the residual quantizer can drive either
    uniformly. Owns the codebook shape; subclasses build the backend-specific
    codebook (a buffer, an ``nn.Embedding``, …) from it.

    Args:
        n_embed (int): number of codebook entries.
        embed_dim (int): feature dimension.
    """

    def __init__(self, n_embed: int, embed_dim: int) -> None:
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim

    @abstractmethod
    def quantize(self, x: torch.Tensor, topk: int = 1) -> QuantizeOutput:
        """Assign ``x`` (B, D) to the codebook, returning codes + embeddings."""
        raise NotImplementedError

    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        """Gather codebook embeddings for ``ids`` (indexes the codebook)."""
        return self.get_codebook_embeddings()[ids]

    def nearest_neighbors(
        self,
        distances: torch.Tensor,
        topk: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top-k nearest ids and scores from a distance matrix."""
        self._check_topk(topk)
        return torch.topk(distances, k=topk, dim=-1, largest=False)

    def _check_topk(self, topk: int) -> None:
        """Validate a top-k request against this codebook."""
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if topk > self.n_embed:
            raise ValueError(f"topk must be <= n_embed ({self.n_embed}), got {topk}")

    def _topk_output(self, distances: torch.Tensor, topk: int) -> QuantizeOutput:
        """Assemble the eval/inference output (greedy pick + top-k) from distances.

        Shared by both backends' eval paths: slot 0 of the ascending top-k is the
        nearest (greedy) code, and the full top-k rides along for candidate SIDs.
        The codebook read goes through :meth:`lookup`, so it stays backend-agnostic
        (``centroids`` for K-Means, the ``nn.Embedding`` table for VQ).
        """
        topk_scores, topk_ids = self.nearest_neighbors(distances, topk)
        ids = topk_ids[:, 0]
        return QuantizeOutput(
            embeddings=self.lookup(ids),
            ids=ids,
            scores=topk_scores[:, 0],
            topk_ids=topk_ids,
            topk_scores=topk_scores,
        )

    @abstractmethod
    def get_codebook_embeddings(self) -> torch.Tensor:
        """Return the full codebook, shape (n_embed, embed_dim).

        The codebook lives in a backend-specific attribute (a ``centroids``
        buffer for K-Means, an ``nn.Embedding`` for RQ-VAE), so this stays
        abstract; :meth:`lookup` is then concrete in terms of it.
        """
        raise NotImplementedError
