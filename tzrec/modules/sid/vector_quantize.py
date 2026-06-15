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

"""Single codebook vector quantization with Sinkhorn uniform assignment."""

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F

from tzrec.modules.sid.quantize_layer import QuantizeLayer
from tzrec.modules.sid.types import (
    QuantizeForwardMode,
    QuantizeOutput,
)


def _squared_euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Squared L2 distance between rows of ``x`` and ``y``.

    Args:
        x (Tensor): data points, shape (N, D).
        y (Tensor): centroids, shape (K, D).

    Returns:
        Tensor: squared distances, shape (N, K).

    Grad-enabled and branch-free (Gumbel needs grad; STE/Sinkhorn callers add
    their own ``no_grad``).
    """
    x_sq = x.pow(2).sum(dim=1, keepdim=True)  # (N, 1)
    y_sq = y.pow(2).sum(dim=1, keepdim=True).t()  # (1, K)
    return (x_sq + y_sq - 2.0 * x @ y.t()).clamp(min=0.0)


@torch.no_grad()
def _sinkhorn(
    cost: torch.Tensor,
    n_iters: int = 5,
    epsilon: float = 10.0,
    is_distributed: bool = True,
) -> torch.Tensor:
    """Sinkhorn-Knopp algorithm for optimal-transport based uniform assignment.

    Transforms a distance matrix into a soft assignment matrix via exponential
    kernel and alternating row-column normalization, approximating a doubly
    stochastic matrix to ensure uniform codebook utilization.

    Args:
        cost (Tensor): distance matrix, shape (B, K) where K is codebook size.
            IMPORTANT: must be z-score normalized and shifted to non-negative
            before calling this function to avoid numerical overflow.
        n_iters (int): number of Sinkhorn iterations. Default: 5.
        epsilon (float): sharpness parameter for exp(-cost * epsilon).
            Larger values produce sharper assignments. Default: 10.0.
        is_distributed (bool): whether running in distributed mode.
            If True, row sums are all_reduced across GPUs. Default: True.

    Returns:
        Tensor: assignment matrix, shape (B, K).
            Use Q.argmax(dim=-1) externally to get codebook indices.
    """
    # Step 1: exponential kernel transform  (B, K) -> (K, B)
    Q = torch.exp(-cost * epsilon).t()

    # Global batch size for distributed training
    if is_distributed and dist.is_initialized():
        B = Q.size(1) * dist.get_world_size()
    else:
        B = Q.size(1)
    K = Q.size(0)

    # Step 2: global normalization — make matrix sum to 1
    sum_Q = torch.sum(Q)
    if is_distributed and dist.is_initialized():
        dist.all_reduce(sum_Q)
    Q /= sum_Q + 1e-8

    # Step 3: alternating row-column normalization
    for _ in range(n_iters):
        # Row normalization: each prototype's total weight = 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if is_distributed and dist.is_initialized():
            dist.all_reduce(sum_of_rows)
        Q /= sum_of_rows + 1e-8
        Q /= K

        # Column normalization: each sample's total weight = 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True) + 1e-8
        Q /= B

    # Step 4: scale back so columns sum to 1 (assignment)
    Q *= B
    return Q.t()  # (B, K)


class VectorQuantize(QuantizeLayer):
    """Single codebook vector quantization layer (RQ-VAE backend).

    A gradient-trained ``nn.Embedding`` codebook (the VQ ``QuantizeLayer``),
    sibling of the K-Means backend's ``KMeansQuantizeLayer``. Maps inputs to a
    codebook entry via :meth:`quantize`. Loss-free: the commitment loss lives in
    :meth:`ResidualVectorQuantizer._single_commitment_loss`. Sinkhorn
    optimal-transport assignment optionally balances codebook usage in training.

    Args:
        embed_dim (int): dimension of each codebook embedding.
        n_embed (int): number of codebook entries.
        forward_mode (QuantizeForwardMode): quantization forward mode,
            either GUMBEL_SOFTMAX or STE. Default: STE.
        distance_type (str): distance metric, 'l2' or 'cosine'.
            Default: 'l2'.
        use_sinkhorn (bool): whether to use Sinkhorn uniform assignment
            during training. Default: True.
        sinkhorn_iters (int): number of Sinkhorn iterations. Default: 5.
        sinkhorn_epsilon (float): Sinkhorn sharpness parameter for
            exp(-cost * epsilon). Default: 10.0.
    """

    def __init__(
        self,
        embed_dim: int,
        n_embed: int,
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        distance_type: str = "l2",
        use_sinkhorn: bool = True,
        sinkhorn_iters: int = 5,
        sinkhorn_epsilon: float = 10.0,
    ) -> None:
        super().__init__(n_embed=n_embed, embed_dim=embed_dim)
        # Sinkhorn drives `ids` (balanced assignment), Gumbel drives `emb`
        # (nearest code); combining them makes the saved id and embedding
        # diverge, so reject the combo (see the assert message).
        _is_gumbel = forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX
        assert not (use_sinkhorn and _is_gumbel), (
            "use_sinkhorn=True is incompatible with forward_mode=GUMBEL_SOFTMAX: "
            "Sinkhorn drives `ids` (balanced assignment) while Gumbel drives "
            "`emb` (nearest code), so the returned id and embedding diverge. "
            "Use STE with Sinkhorn, or Gumbel-Softmax without Sinkhorn."
        )
        # ``n_embed`` / ``embed_dim`` are owned by the QuantizeLayer base.
        self.forward_mode = forward_mode
        self.distance_type = distance_type
        self.use_sinkhorn = use_sinkhorn
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_epsilon = sinkhorn_epsilon

        self.embedding = nn.Embedding(n_embed, embed_dim)
        nn.init.kaiming_uniform_(self.embedding.weight)

    def _compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L2/cosine distances between inputs and codebook entries.

        Not ``no_grad``: Gumbel calls this directly for the encoder gradient;
        the STE/Sinkhorn path calls it inside ``no_grad`` in
        :meth:`_find_nearest_embedding`.

        Args:
            x (Tensor): input vectors, shape (B, D).

        Returns:
            Tensor: pairwise distances, shape (B, n_embed).
        """
        codebook = self.embedding.weight  # (n_embed, D)

        if self.distance_type == "l2":
            distances = _squared_euclidean_distance(x, codebook)
        elif self.distance_type == "cosine":
            x_norm = F.normalize(x, p=2, dim=1)
            codebook_norm = F.normalize(codebook, p=2, dim=1)
            distances = -torch.matmul(x_norm, codebook_norm.t())
        else:
            raise ValueError(
                f"Unsupported distance_type '{self.distance_type}', "
                f"choose from ('l2', 'cosine')"
            )
        return distances

    @torch.no_grad()
    def _find_nearest_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Find the nearest codebook id for each input vector.

        During training with use_sinkhorn=True, applies z-score
        normalization + non-negative shift before Sinkhorn assignment.
        Otherwise falls back to argmin.

        Args:
            x (Tensor): input vectors, shape (B, D).

        Returns:
            Tensor: codebook indices, shape (B,).
        """
        distances = self._compute_distances(x)  # (B, n_embed)

        if self.training and self.use_sinkhorn:
            # Sinkhorn requires non-negative cost; z-score then shift.
            std, mean = torch.std_mean(distances, unbiased=False)
            distances = (distances - mean) / std.add(1e-12)
            distances = distances - distances.min()

            # Sinkhorn optimal-transport assignment
            Q = _sinkhorn(
                distances,
                n_iters=self.sinkhorn_iters,
                epsilon=self.sinkhorn_epsilon,
                is_distributed=dist.is_initialized(),
            )
            ids = Q.argmax(dim=-1)
        else:
            ids = distances.argmin(dim=-1)

        return ids

    def quantize(self, x: torch.Tensor, temperature: float = 1.0) -> QuantizeOutput:
        """Assign ``x`` to the codebook (the :class:`QuantizeLayer` interface).

        Commitment loss is computed by the caller
        (:meth:`ResidualVectorQuantizer._single_commitment_loss`); device follows
        ``x``, so this runs on CPU or GPU unchanged.

        Args:
            x (Tensor): input vectors, shape (B, D).
            temperature (float): temperature for Gumbel-Softmax.

        Returns:
            QuantizeOutput: named tuple of (embeddings, ids).
        """
        # Gumbel: grad-enabled distances feed the encoder; the hard sample drives
        # both emb and ids, so the saved code matches the vector used.
        if self.training and self.forward_mode == QuantizeForwardMode.GUMBEL_SOFTMAX:
            logits = -self._compute_distances(x)  # (B, n_embed), differentiable
            weights = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
            emb = weights @ self.embedding.weight
            ids = weights.argmax(dim=-1)
            return QuantizeOutput(embeddings=emb, ids=ids)

        # STE / eval: nearest-neighbour assignment under no_grad.
        ids = self._find_nearest_embedding(x)
        if self.training:
            quantized = self.embedding(ids)  # straight-through: grad passes to x
            emb = x + (quantized - x).detach()
        else:
            emb = self.embedding(ids)

        return QuantizeOutput(embeddings=emb, ids=ids)

    def get_codebook_embeddings(self) -> torch.Tensor:
        """Return the codebook table, shape (n_embed, embed_dim)."""
        return self.embedding.weight
