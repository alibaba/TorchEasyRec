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

from typing import Optional

import torch


def lengths_to_index(
    lengths: torch.Tensor, output_size: Optional[int] = None
) -> torch.Tensor:
    """Per-row list index (torch_scatter's ``index``) of a jagged ``lengths`` layout.

    ``output_size`` -- the statically known ``sum(lengths)`` -- avoids the
    hidden device->host sync (and data-dependent graph break) that
    ``repeat_interleave`` with tensor repeats otherwise performs.
    """
    return torch.repeat_interleave(
        torch.arange(lengths.size(0), device=lengths.device),
        lengths,
        output_size=output_size,
    )


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Sum the rows of ``src`` ``(N, C)`` into ``dim_size`` groups by ``index``.

    Groups without a row are 0.  Reduced-precision inputs (fp16/bf16)
    accumulate in fp32 and cast back: ``index_add_`` is not on autocast's
    promote list, so bf16 inputs would otherwise add through bf16 atomics
    whose reorder noise sits at bf16 rounding scale.  ``promote_types``
    keeps the dtype choice a graph node rather than Python control flow,
    so fx tracing still inlines this.
    """
    acc_dtype = torch.promote_types(src.dtype, torch.float32)
    out = torch.zeros((dim_size, src.size(-1)), dtype=acc_dtype, device=src.device)
    out.index_add_(0, index, src.to(acc_dtype))
    return out.to(src.dtype)


def scatter_max(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Max-reduce the rows of ``src`` ``(N, C)`` into ``dim_size`` groups.

    ``scatter_reduce_`` rather than the still-beta ``index_reduce_`` (which
    warns on every call); it has no amax backward either, so callers using
    the result as a shift constant must detach ``src``.  Groups without a
    row keep ``-inf``, which would make downstream broadcasts NaN under
    torch.compile, so they are rewritten to 0 -- nothing reads them back.
    """
    out = torch.full(
        (dim_size, src.size(-1)), float("-inf"), dtype=src.dtype, device=src.device
    ).scatter_reduce_(
        0, index.unsqueeze(-1).expand_as(src), src, "amax", include_self=False
    )
    return torch.nan_to_num(out, neginf=0.0)


def scatter_logsumexp(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Log-sum-exp of the rows of ``src`` ``(N, C)`` within each group.

    Shifts by the detached group max, which the gradient does not depend
    on, so the result is exact while ``exp`` cannot overflow.  A group
    without a row yields ``-inf``; nothing that has rows reads it back.
    """
    maxes = scatter_max(src.detach(), index, dim_size)
    sums = scatter_sum(torch.exp(src - maxes.index_select(0, index)), index, dim_size)
    return maxes + torch.log(sums)
