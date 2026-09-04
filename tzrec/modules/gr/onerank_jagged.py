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

"""Per-request segment reductions over jagged candidate tensors.

OneRank aggregates and normalizes across the candidates of one request
(paper 2.3 and 2.5).  The pool is tiny -- candidate counts are single
digits -- so these helpers stay in the jagged layout instead of padding to
a dense ``(B, N_max, ...)`` block: no host sync for ``N_max``, no wasted
compute on padding, and no way to silently drop candidates past a
configured bound.

All of them take ``lengths`` (candidates per request) and assume the
segments are laid out contiguously in ``lengths`` order, which is how every
jagged tensor in the HSTU pipeline is shaped.  Empty segments are legal and
reduce to zeros rather than NaN.
"""

import torch


def jagged_segment_ids(lengths: torch.Tensor) -> torch.Tensor:
    """Map each jagged row to its segment index.

    Args:
        lengths (torch.Tensor): ``(B,)`` segment lengths.

    Returns:
        torch.Tensor: ``(total,)`` int64 segment index per row.
    """
    return torch.repeat_interleave(
        torch.arange(lengths.size(0), device=lengths.device), lengths
    )


def jagged_segment_sum(
    values: torch.Tensor,
    lengths: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Sum a jagged ``(total, C)`` tensor within each segment.

    Args:
        values (torch.Tensor): ``(total, C)`` jagged values.
        lengths (torch.Tensor): ``(B,)`` segment lengths.
        segment_ids (torch.Tensor): ``(total,)`` from
            :func:`jagged_segment_ids`; passed in so a caller doing several
            reductions builds it once.

    Returns:
        torch.Tensor: ``(B, C)``; empty segments sum to zeros.
    """
    return torch.zeros(
        (lengths.size(0), values.size(-1)),
        dtype=values.dtype,
        device=values.device,
    ).index_add_(0, segment_ids, values)


def jagged_segment_max(
    values: torch.Tensor,
    lengths: torch.Tensor,
    segment_ids: torch.Tensor,
) -> torch.Tensor:
    """Max-reduce a jagged ``(total, C)`` tensor within each segment.

    Args:
        values (torch.Tensor): ``(total, C)`` jagged values.
        lengths (torch.Tensor): ``(B,)`` segment lengths.
        segment_ids (torch.Tensor): ``(total,)`` from
            :func:`jagged_segment_ids`.

    Returns:
        torch.Tensor: ``(B, C)``; empty segments reduce to zeros.
    """
    # scatter_reduce_ rather than index_reduce_: the latter is still flagged
    # beta and warns on every call.  It has no amax backward either, so
    # callers that use the result as a shift constant must detach `values`.
    maxes = torch.full(
        (lengths.size(0), values.size(-1)),
        float("-inf"),
        dtype=values.dtype,
        device=values.device,
    ).scatter_reduce_(
        0,
        segment_ids.unsqueeze(-1).expand_as(values),
        values,
        "amax",
        include_self=False,
    )
    # Empty segments keep -inf; nothing reads them back, but leaving -inf in
    # the tensor makes downstream broadcasts produce NaN under torch.compile.
    return torch.nan_to_num(maxes, neginf=0.0)


def jagged_mean(values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Mean-pool the segments of a jagged ``(total, K, D)`` tensor.

    Args:
        values (torch.Tensor): ``(total, K, D)`` jagged values.
        lengths (torch.Tensor): ``(B,)`` segment lengths summing to
            ``total``.

    Returns:
        torch.Tensor: ``(B, K, D)``; empty segments pool to zeros.
    """
    batch_size = lengths.size(0)
    num_tasks = values.size(1)
    dim = values.size(2)
    flat = values.reshape(values.size(0), num_tasks * dim)
    sums = jagged_segment_sum(flat, lengths, jagged_segment_ids(lengths))
    denom = lengths.clamp(min=1).to(flat.dtype).unsqueeze(-1)
    return (sums / denom).reshape(batch_size, num_tasks, dim)


def jagged_softmax(logits: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Softmax a jagged ``(total, C)`` tensor within each segment.

    Args:
        logits (torch.Tensor): ``(total, C)`` jagged logits; each column is
            normalized independently.
        lengths (torch.Tensor): ``(B,)`` segment lengths summing to
            ``total``.

    Returns:
        torch.Tensor: ``(total, C)`` weights summing to 1 per
            ``(segment, column)``.
    """
    segment_ids = jagged_segment_ids(lengths)
    # Per-segment max shift.  A global max would be numerically wrong here,
    # not just loose: one segment whose logits all sit far below the global
    # max underflows to an all-zero row and a 0/0 denominator.
    # Detached because the shift is a constant of the softmax identity.
    maxes = jagged_segment_max(logits.detach(), lengths, segment_ids)
    exp = torch.exp(logits - maxes.index_select(0, segment_ids))
    denom = jagged_segment_sum(exp, lengths, segment_ids)
    return exp / torch.repeat_interleave(denom, lengths, dim=0)
