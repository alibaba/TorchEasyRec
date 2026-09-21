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

"""Situation Discernment (paper 2.3).

Each task builds its own query from the request's contextual token and uses
it to attend over that request's candidates::

    q_k = LN(f_k(s))                          f_k private per task
    h_k = MHCA_k(q_k, {r^i_k}, {r^i_k})       pool = own request only

``h_k`` is a request-level summary of "what this request looks like through
task k's eyes", which the scorer then dots against each candidate.  Dropping
this module is the most damaging ablation in the paper (V5), so the pooling
fallback in :mod:`tzrec.modules.task_tower` is a floor, not a peer.

The attention here is *not* HSTU attention.  ``hstu_mha`` computes
``silu(qk) / scaling * mask`` point-wise with no normalization over keys; a
weighted average over a candidate pool needs weights that sum to one.  The
single-query-per-segment special case maps directly onto
``torch.nn.attention.varlen.varlen_attn`` (packed Q/K/V plus cumulative
sequence bounds), so eligible CUDA batches take that kernel; everything
else -- CPU or fp32 tensors, attention dropout, a request with an empty
candidate list -- falls back to the reference math below, which is the
historical implementation verbatim.
"""

from typing import List

import torch
import torch.nn.functional as F

from tzrec.modules.norm import LayerNorm
from tzrec.modules.utils import BaseModule
from tzrec.ops.jagged_tensors import (
    jagged_segment_ids,
    jagged_segment_max,
    jagged_segment_sum,
)


def _jagged_softmax(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Softmax a jagged ``(total, C)`` tensor within each segment.

    The max shift is per segment (a global max would underflow a
    far-below-max segment to an all-zero row and a 0/0 denominator) and
    detached, because it is a constant of the softmax identity.
    """
    segment_ids = jagged_segment_ids(lengths, output_size=logits.size(0))
    maxes = jagged_segment_max(logits.detach(), lengths, segment_ids)
    exp = torch.exp(logits - maxes.index_select(0, segment_ids))
    denom = jagged_segment_sum(exp, lengths, segment_ids)
    return exp / torch.repeat_interleave(denom, lengths, dim=0, output_size=exp.size(0))


def _varlen_single_query_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    attn_scale: float,
) -> torch.Tensor:
    """Packed-varlen attention, one query per segment.  fx leaf.

    Maps the jagged layout onto
    ``torch.nn.attention.varlen.varlen_attn``: the queries are already
    packed (``B`` segments of exactly one query each), the keys/values are
    the jagged pool rows, and the segment lengths become the cumulative
    sequence bounds.  ``max_k`` is only a kernel-tile hint, so the static
    total ``k.size(0)`` is a safe upper bound that avoids a
    ``lengths.max()`` device->host sync.
    """
    from torch.nn.attention.varlen import varlen_attn

    batch_size = q.size(0)
    device = q.device
    cu_seq_q = torch.arange(batch_size + 1, device=device, dtype=torch.int32)
    cu_seq_k = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
    cu_seq_k[1:] = torch.cumsum(lengths.to(torch.int32), dim=0)
    out = varlen_attn(
        q,
        k,
        v,
        cu_seq_q,
        cu_seq_k,
        max_q=1,
        max_k=int(k.size(0)),
        scale=attn_scale,
    )
    return out.reshape(batch_size, q.size(1) * q.size(2))


torch.fx.wrap(_varlen_single_query_attn)


def _jagged_single_query_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    lengths: torch.Tensor,
    dropout_ratio: float,
    training: bool,
    attn_scale: float,
) -> torch.Tensor:
    """Single-query-per-segment softmax attention.  fx leaf.

    Dispatches to the varlen kernel only when the batch is eligible:
    CUDA with fp16/bf16 tensors (the flash kernel's dtype contract), no
    attention dropout (``varlen_attn`` takes none), and no empty segment
    (an empty candidate list must keep the documented bias-only output
    rather than depend on kernel-specific empty-KV behaviour).  The
    emptiness probe is the one device->host sync this path pays;
    everything else falls through to the reference math, which is the
    historical implementation verbatim.
    """
    if (
        q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and dropout_ratio == 0.0
        and bool((lengths > 0).all())
    ):
        return _varlen_single_query_attn(q, k, v, lengths, attn_scale)
    num_heads = q.size(1)
    head_dim = q.size(2)
    segment_ids = jagged_segment_ids(lengths, output_size=k.size(0))
    # Broadcast the query onto its own rows instead of padding the pool
    # to a dense (B, N_max, D) block.
    q_rows = q.index_select(0, segment_ids)
    logits = (q_rows * k).sum(dim=-1) * attn_scale
    attn = _jagged_softmax(logits, lengths)
    attn = F.dropout(attn, p=dropout_ratio, training=training)
    weighted = (attn.unsqueeze(-1) * v).reshape(-1, num_heads * head_dim)
    return jagged_segment_sum(weighted, lengths, segment_ids)


torch.fx.wrap(_jagged_single_query_attn)


class JaggedCrossAttention(BaseModule):
    """Softmax multi-head cross attention from one query per segment.

    Query is ``(B, D)`` -- exactly one per request -- and the keys/values
    are the request's own jagged rows, so this is the single-query special
    case of MHCA and needs no attention mask: segment membership already
    restricts what each query can see.  Eligible CUDA batches run through
    ``torch.nn.attention.varlen.varlen_attn`` (see
    :func:`_jagged_single_query_attn` for the eligibility rules); all
    other batches run the reference math, and the two agree to kernel
    rounding.

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
        q = self._q_proj(query).view(-1, num_heads, head_dim)
        k = self._k_proj(pool).view(-1, num_heads, head_dim)
        v = self._v_proj(pool).view(-1, num_heads, head_dim)
        context = _jagged_single_query_attn(
            q,
            k,
            v,
            lengths,
            dropout_ratio=self._dropout_ratio,
            training=self.training,
            attn_scale=self._attn_scale,
        )
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
        # The K per-task passes are kept as-is on purpose: the per-task
        # modules (query norm / projection and each JaggedCrossAttention's
        # weights) are part of the checkpoint layout, and the attention
        # leaf pays at most one device->host sync per pass (the varlen
        # path's empty-segment probe; the reference path has none thanks
        # to `output_size=`).  Batching the K passes into one grouped GEMM
        # would change that layout for a modest kernel-count win.
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
