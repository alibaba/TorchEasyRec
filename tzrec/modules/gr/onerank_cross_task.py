# Copyright (c) 2025, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cross-task attention over the per-task request vectors (paper 2.4).

The ``K`` request-level vectors ``h_k`` produced by Situation Discernment
attend to each other under a task-order mask ``A``::

    h~_k = sum_j A[k][j] * softmax_j(q_k . k_j) * v_j
    h^_k = LN(h~_k) + h_k
    z_k  = LN(FFN(h^_k)) + h^_k

``A`` defaults to Cascade (``A[k][j] = 1`` iff ``j <= k``) because the
duration bitmask in this dataset only takes the values ``{0,1,3,7,15,31}``:
the labels form one strictly cumulative chain, so a task may read the tasks
it is nested inside and nothing else.

Strategic Gradient Detachment (on by default) makes every off-diagonal
key/value a constant: task ``k``'s loss can read task ``j``'s
representation but cannot reshape it.  Without it the cross-task path turns
into a second, uncontrolled route for one task's gradient to rewrite
another's channel, which the paper reports as strong but unstable (V4).
"""

from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from tzrec.modules.norm import LayerNorm
from tzrec.modules.utils import BaseModule

CASCADE = "ONERANK_MASK_CASCADE"
PARALLEL = "ONERANK_MASK_PARALLEL"
FULL = "ONERANK_MASK_FULL"
HYBRID = "ONERANK_MASK_HYBRID"


def build_cross_task_mask(
    mask_type: str,
    task_names: Sequence[str],
    hybrid_chain_task_names: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Build the ``(K, K)`` boolean cross-task attention mask ``A``.

    Args:
        mask_type (str): one of the ``ONERANK_MASK_*`` enum names.
        task_names (Sequence[str]): task names in ``task_configs`` order;
            this order *is* the cascade order.
        hybrid_chain_task_names (Sequence[str], optional): subset forming
            the chain when ``mask_type`` is ``ONERANK_MASK_HYBRID``.

    Returns:
        torch.Tensor: ``(K, K)`` bool; ``A[k][j]`` is True when task ``k``
            may read task ``j``.  The diagonal is always True, so no
            softmax row can be fully masked.
    """
    num_tasks = len(task_names)
    names = list(task_names)
    if mask_type == CASCADE:
        mask = torch.ones((num_tasks, num_tasks), dtype=torch.bool).tril()
    elif mask_type == PARALLEL:
        mask = torch.eye(num_tasks, dtype=torch.bool)
    elif mask_type == FULL:
        mask = torch.ones((num_tasks, num_tasks), dtype=torch.bool)
    elif mask_type == HYBRID:
        chain = list(hybrid_chain_task_names or [])
        if len(chain) < 2:
            raise ValueError(
                "ONERANK_MASK_HYBRID needs at least two entries in "
                f"hybrid_chain_task_names; got {chain}."
            )
        unknown = [name for name in chain if name not in names]
        if unknown:
            raise ValueError(
                f"hybrid_chain_task_names entries {unknown} are not task "
                f"names; known tasks are {names}."
            )
        if len(set(chain)) != len(chain):
            raise ValueError(f"hybrid_chain_task_names has duplicates: {chain}.")
        mask = torch.eye(num_tasks, dtype=torch.bool)
        indices = [names.index(name) for name in chain]
        for position, row in enumerate(indices):
            for col in indices[: position + 1]:
                mask[row, col] = True
    else:
        raise ValueError(f"unknown cross-task mask type: {mask_type}")
    return mask


class OneRankCrossTaskAttention(BaseModule):
    """Masked cross-task attention with residual FFN (paper 2.4).

    Attention runs over the task axis, which is ``K`` wide (7 here), so it
    is expressed densely -- the ``(B, K, K, D)`` pairwise expansion needed
    to detach off-diagonal keys/values costs a few tens of MB at this size
    and keeps the detachment exact instead of approximated.

    Args:
        embedding_dim (int): request-vector dim ``D``.
        task_names (Sequence[str]): task names in ``task_configs`` order.
        mask_type (str): ``ONERANK_MASK_*`` enum name.
        gradient_detachment (bool): keep gradients only on the diagonal.
        num_heads (int): attention heads; must divide ``embedding_dim``.
        ffn_hidden_dim (int): FFN hidden width; ``0`` means ``4 * D``.
        dropout_ratio (float): dropout on attention weights and inside the
            FFN.
        hybrid_chain_task_names (Sequence[str], optional): see
            :func:`build_cross_task_mask`.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        task_names: Sequence[str],
        mask_type: str = CASCADE,
        gradient_detachment: bool = True,
        num_heads: int = 4,
        ffn_hidden_dim: int = 0,
        dropout_ratio: float = 0.0,
        hybrid_chain_task_names: Optional[Sequence[str]] = None,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}")
        if embedding_dim % num_heads != 0:
            raise ValueError(
                f"embedding_dim ({embedding_dim}) must be divisible by "
                f"num_heads ({num_heads})."
            )
        self._num_tasks: int = len(task_names)
        self._num_heads: int = num_heads
        self._head_dim: int = embedding_dim // num_heads
        self._attn_scale: float = self._head_dim**-0.5
        self._dropout_ratio: float = dropout_ratio
        self._gradient_detachment: bool = gradient_detachment
        self._q_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._k_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._v_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._out_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._attn_norm = LayerNorm(dim=embedding_dim)
        hidden_dim = ffn_hidden_dim if ffn_hidden_dim > 0 else 4 * embedding_dim
        self._ffn = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, hidden_dim),
            # SiLU to match the STU trunk, which is silu-gated throughout.
            torch.nn.SiLU(),
            torch.nn.Dropout(p=dropout_ratio),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )
        self._ffn_norm = LayerNorm(dim=embedding_dim)
        self.register_buffer(
            "_attn_mask",
            build_cross_task_mask(mask_type, task_names, hybrid_chain_task_names),
            persistent=False,
        )

    def _pairwise_source(self, request_vectors: torch.Tensor) -> torch.Tensor:
        """Expand ``(B, K, D)`` to the ``(B, K, K, D)`` key/value source.

        Index ``[b, k, j]`` is the vector task ``k`` reads from task ``j``.
        Under detachment everything off the diagonal is a constant, so the
        projection weights still learn from every pair while ``h_j`` itself
        only ever hears from task ``j``'s own loss.
        """
        num_tasks = self._num_tasks
        pairwise = request_vectors.unsqueeze(1).expand(-1, num_tasks, -1, -1)
        if not self._gradient_detachment:
            return pairwise
        diagonal = torch.eye(
            num_tasks, dtype=pairwise.dtype, device=pairwise.device
        ).view(1, num_tasks, num_tasks, 1)
        return pairwise * diagonal + pairwise.detach() * (1.0 - diagonal)

    def forward(self, request_vectors: torch.Tensor) -> torch.Tensor:
        """Mix the per-task request vectors.

        Args:
            request_vectors (torch.Tensor): ``(B, K, D)`` per-task
                request-level vectors ``h_k``.

        Returns:
            torch.Tensor: ``(B, K, D)`` mixed vectors ``z_k``.
        """
        num_tasks = self._num_tasks
        num_heads = self._num_heads
        head_dim = self._head_dim
        batch_size = request_vectors.size(0)
        pairwise = self._pairwise_source(request_vectors)

        query = self._q_proj(request_vectors).view(
            batch_size, num_tasks, 1, num_heads, head_dim
        )
        keys = self._k_proj(pairwise).view(
            batch_size, num_tasks, num_tasks, num_heads, head_dim
        )
        values = self._v_proj(pairwise).view(
            batch_size, num_tasks, num_tasks, num_heads, head_dim
        )
        logits = (query * keys).sum(dim=-1) * self._attn_scale
        mask = self._attn_mask.view(1, num_tasks, num_tasks, 1)
        logits = logits.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(logits, dim=2)
        weights = F.dropout(weights, p=self._dropout_ratio, training=self.training)
        attended = (weights.unsqueeze(-1) * values).sum(dim=2)
        attended = self._out_proj(attended.reshape(batch_size, num_tasks, -1))

        # LN on the sublayer output, then the residual -- as in the paper,
        # which normalizes what the sublayer produced rather than what it
        # reads.  Flattened to 2D because the Triton layer_norm kernel
        # unpacks `x.shape` as `(N, D)`.
        flat_dim = attended.size(-1)
        residual = request_vectors.reshape(-1, flat_dim)
        hidden = self._attn_norm(attended.reshape(-1, flat_dim)) + residual
        output = self._ffn_norm(self._ffn(hidden)) + hidden
        return output.reshape(batch_size, num_tasks, flat_dim)
