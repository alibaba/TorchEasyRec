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

"""Situation Discernment (paper 2.3).

Each task builds its own query from the request's contextual token and uses
it to attend over that request's candidates::

    q_k = LN(f_k(s))                          f_k private per task
    h_k = MHCA_k(q_k, {r^i_k}, {r^i_k})       pool = own request only

``h_k`` is a request-level summary of "what this request looks like through
task k's eyes", which the scorer then dots against each candidate.  Dropping
this module is the most damaging ablation in the paper (V5), so the pooling
fallback in :mod:`tzrec.modules.gr.onerank_head` is a floor, not a peer.

The attention here is *not* HSTU attention.  ``hstu_mha`` computes
``silu(qk) / scaling * mask`` point-wise with no normalization over keys; a
weighted average over a candidate pool needs weights that sum to one, so
this is a plain softmax MHCA over the jagged pool.
"""

from typing import List

import torch
import torch.nn.functional as F

from tzrec.modules.gr.onerank_jagged import (
    jagged_segment_ids,
    jagged_segment_sum,
    jagged_softmax,
)
from tzrec.modules.norm import LayerNorm
from tzrec.modules.utils import BaseModule


class JaggedCrossAttention(BaseModule):
    """Softmax multi-head cross attention from one query per segment.

    Query is ``(B, D)`` -- exactly one per request -- and the keys/values
    are the request's own jagged rows, so this is the single-query special
    case of MHCA and needs no attention mask: segment membership already
    restricts what each query can see.

    Args:
        embedding_dim (int): query / key / value dim ``D``.
        num_heads (int): number of attention heads; must divide
            ``embedding_dim``.
        dropout_ratio (float): dropout on the attention weights.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout_ratio: float = 0.0,
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
        self._num_heads: int = num_heads
        self._head_dim: int = embedding_dim // num_heads
        self._attn_scale: float = self._head_dim**-0.5
        self._dropout_ratio: float = dropout_ratio
        self._q_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._k_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._v_proj = torch.nn.Linear(embedding_dim, embedding_dim)
        self._out_proj = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        query: torch.Tensor,
        pool: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Attend from each segment's query over that segment's rows.

        Args:
            query (torch.Tensor): ``(B, D)`` one query per segment.
            pool (torch.Tensor): ``(total, D)`` keys and values, jagged.
            lengths (torch.Tensor): ``(B,)`` rows per segment, summing to
                ``total``.

        Returns:
            torch.Tensor: ``(B, D)`` context vectors; empty segments get
                only ``_out_proj``'s bias.
        """
        num_heads = self._num_heads
        head_dim = self._head_dim
        segment_ids = jagged_segment_ids(lengths)
        q = self._q_proj(query).view(-1, num_heads, head_dim)
        k = self._k_proj(pool).view(-1, num_heads, head_dim)
        v = self._v_proj(pool).view(-1, num_heads, head_dim)
        # Broadcast the query onto its own rows instead of padding the pool
        # to a dense (B, N_max, D) block.
        q_rows = q.index_select(0, segment_ids)
        logits = (q_rows * k).sum(dim=-1) * self._attn_scale
        attn = jagged_softmax(logits, lengths)
        attn = F.dropout(attn, p=self._dropout_ratio, training=self.training)
        weighted = (attn.unsqueeze(-1) * v).reshape(-1, num_heads * head_dim)
        context = jagged_segment_sum(weighted, lengths, segment_ids)
        return self._out_proj(context)


class OneRankSituationDiscernment(BaseModule):
    """Per-task contextual query + cross-candidate aggregation (paper 2.3).

    The ``f_k`` are ``K`` independent linear maps rather than one shared map
    with per-task heads: the whole point is that each task reads a different
    view of the same request context, and a shared trunk would put that
    back into one bottleneck.

    Args:
        embedding_dim (int): STU embedding dim ``D``.
        num_tasks (int): number of tasks ``K``.
        contextual_feature_dim (int): width of the flattened contextual
            token ``s``, i.e. ``max_contextual_seq_len *
            contextual_feature_dim`` of the input preprocessor.
        num_heads (int): heads of each per-task MHCA.
        dropout_ratio (float): dropout on the attention weights.
        is_inference (bool): whether to run in inference mode.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_tasks: int,
        contextual_feature_dim: int,
        num_heads: int = 4,
        dropout_ratio: float = 0.0,
        is_inference: bool = False,
    ) -> None:
        super().__init__(is_inference=is_inference)
        if num_tasks <= 0:
            raise ValueError(f"num_tasks must be positive; got {num_tasks}")
        if contextual_feature_dim <= 0:
            raise ValueError(
                "situation_discernment needs a non-empty contextual feature "
                "group: the per-task query is a projection of the contextual "
                f"token, but its dim is {contextual_feature_dim}."
            )
        self._num_tasks: int = num_tasks
        self._contextual_feature_dim: int = contextual_feature_dim
        self._query_projs = torch.nn.ModuleList(
            [
                torch.nn.Linear(contextual_feature_dim, embedding_dim)
                for _ in range(num_tasks)
            ]
        )
        # LN on the query keeps its scale independent of how wide the
        # contextual group happens to be (1440 dims in the reference setup).
        self._query_norms = torch.nn.ModuleList(
            [LayerNorm(dim=embedding_dim) for _ in range(num_tasks)]
        )
        self._attentions = torch.nn.ModuleList(
            [
                JaggedCrossAttention(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    dropout_ratio=dropout_ratio,
                    is_inference=is_inference,
                )
                for _ in range(num_tasks)
            ]
        )

    def forward(
        self,
        contextual_embeddings: torch.Tensor,
        task_embeddings: torch.Tensor,
        num_candidates: torch.Tensor,
    ) -> torch.Tensor:
        """Summarize each request once per task.

        Args:
            contextual_embeddings (torch.Tensor): ``(B, contextual_dim)``
                flattened contextual token ``s``, in the same request order
                as ``num_candidates``.
            task_embeddings (torch.Tensor): ``(total_candidates, K, D)``
                per-candidate per-task representations ``r^i_k``.
            num_candidates (torch.Tensor): ``(B,)`` candidates per request.

        Returns:
            torch.Tensor: ``(B, K, D)`` request-level per-task vectors.
        """
        contextual = contextual_embeddings.reshape(-1, self._contextual_feature_dim).to(
            task_embeddings.dtype
        )
        outputs: List[torch.Tensor] = []
        for task_idx in range(self._num_tasks):
            query = self._query_norms[task_idx](self._query_projs[task_idx](contextual))
            outputs.append(
                self._attentions[task_idx](
                    query=query,
                    pool=task_embeddings[:, task_idx, :].contiguous(),
                    lengths=num_candidates,
                )
            )
        return torch.stack(outputs, dim=1)
