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

# We use the hstu_attention ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.fx.wrap
def _get_valid_attn_mask(
    device: torch.device,
    causal: bool,
    N: int,
    seq_lengths: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    ids = torch.arange(0, N, device=device).view(1, N)
    max_ids = seq_lengths.view(-1, 1, 1)
    if contextual_seq_len > 0:
        ids = ids - contextual_seq_len + 1
        ids = torch.clamp(ids, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets.view(-1, 1, 1)
        ids = torch.clamp(
            ids,
            max=max_ids,
        )
        row_ids = ids.view(-1, N, 1).expand(-1, N, N)
        col_ids = ids.view(-1, 1, N).expand(-1, N, N)
    else:
        row_ids = ids.view(N, 1).expand(N, N)
        col_ids = row_ids.t()
        row_ids = row_ids.view(1, N, N)
        col_ids = col_ids.view(1, N, N)
    row_col_dist = row_ids - col_ids
    valid_attn_mask = torch.eye(N, device=device, dtype=torch.bool).view(1, N, N)
    if not causal:
        row_col_dist = torch.where(row_col_dist > 0, row_col_dist, -row_col_dist)
    valid_attn_mask = torch.logical_or(valid_attn_mask, row_col_dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask,
                torch.logical_or(
                    row_col_dist <= max_attn_len,
                    row_ids >= max_ids - min_full_attn_seq_len,
                ),
            )
        else:
            valid_attn_mask = torch.logical_and(
                valid_attn_mask, row_col_dist <= max_attn_len
            )
    if contextual_seq_len > 0:
        valid_attn_mask = torch.logical_or(
            valid_attn_mask, torch.logical_and(row_ids == 0, col_ids < max_ids)
        )
    return valid_attn_mask


def _pad_qkv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    L, H, D = q.shape
    V = v.shape[2]
    padded_q = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=q.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(L, H * D),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, D)
        .transpose(1, 2)
    )  # [B, H, N, A]
    padded_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(L, H * V),
            offsets=[seq_offsets],
            max_lengths=[N],
            padding_value=0.0,
        )
        .view(-1, N, H, V)
        .transpose(1, 2)
    )  # [B, H, N, D]
    return padded_q, padded_k, padded_v


@torch.fx.wrap
def pytorch_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    causal: bool = True,
    dropout_pr: float = 0.0,
    training: bool = True,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
    min_full_attn_seq_len: int = 0,
) -> torch.Tensor:
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_valid_attn_mask(
        device=q.device,
        causal=causal,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
        min_full_attn_seq_len=min_full_attn_seq_len,
    )
    # raise NotImplementedError(valid_attn_mask[0, :, :].to(torch.int32))
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    if dropout_pr > 0.0:
        qk_attn = F.dropout(qk_attn, p=dropout_pr, training=training)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)  # [B, H, N, V]
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),  # [B, N, H, V]->[B, N, H * V]
        [seq_offsets],
        L,
    )[0].view(L, H, V)


@torch.fx.wrap
def pytorch_cached_hstu_mha(
    max_seq_len: int,
    alpha: float,
    delta_q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    num_targets: Optional[torch.Tensor] = None,
    max_attn_len: int = 0,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    L, H, D = delta_q.shape
    _, _, V = v.shape
    B = seq_offsets.size(0) - 1
    delta_size = L // B
    delta_q = delta_q.view(B, -1, H, D).transpose(1, 2)
    full_k = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=k.reshape(-1, H * D),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, D)
        .transpose(1, 2)
    )
    full_v = (
        torch.ops.fbgemm.jagged_to_padded_dense(
            values=v.reshape(-1, H * V),
            offsets=[seq_offsets],
            max_lengths=[max_seq_len],
            padding_value=0.0,
        )
        .view(B, -1, H, V)
        .transpose(1, 2)
    )
    qk_attn = torch.einsum("bhxa,bhya->bhxy", delta_q, full_k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    full_valid_attn_mask = _get_valid_attn_mask(
        device=delta_q.device,
        causal=True,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        num_targets=num_targets,
        max_attn_len=max_attn_len,
        contextual_seq_len=contextual_seq_len,
    )
    seq_lengths = seq_offsets[1:] - seq_offsets[:-1]
    mask = torch.arange(max_seq_len, device=delta_q.device).view(1, -1)
    mask = torch.logical_and(
        mask >= (seq_lengths - delta_size).view(-1, 1),
        mask < seq_lengths.view(-1, 1),
    )
    valid_attn_mask = (
        full_valid_attn_mask.expand(B, -1, -1)
        .flatten(0, 1)[mask.view(-1), :]
        .view(-1, delta_size, max_seq_len)
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_output = torch.einsum("bhxd,bhdv->bhxv", qk_attn, full_v)
    return attn_output.transpose(1, 2).reshape(-1, H, V)


@torch.fx.wrap
def _get_sla_attn_mask(
    device: torch.device,
    N: int,
    seq_lengths: torch.Tensor,
    sla_k1: int,
    sla_k2: int,
    num_targets: Optional[torch.Tensor] = None,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    """Build SLA mask with contextual prefix and target isolation.

    History tokens: M[i,j] = 1 iff j<=i AND ((i-j)<K1 OR j<effective_k2).
    Target tokens:  M[i,j] = 1 iff j < (seq_len - num_targets).
    effective_k2 = max(sla_k2, contextual_seq_len).
    """
    effective_k2 = max(sla_k2, contextual_seq_len)
    B = seq_lengths.size(0)
    row_ids = torch.arange(N, device=device).view(1, N, 1)  # (1, N, 1)
    col_ids = torch.arange(N, device=device).view(1, 1, N)  # (1, 1, N)

    causal = col_ids <= row_ids
    local_window = (row_ids - col_ids) < sla_k1
    global_prefix = col_ids < effective_k2
    sla_mask = causal & (local_window | global_prefix)

    # Sequence-length mask: zero out cols beyond each sequence's length.
    col_valid = col_ids < seq_lengths.view(B, 1, 1)

    if num_targets is not None:
        history_boundary = (seq_lengths - num_targets).view(B, 1, 1)  # (B, 1, 1)
        is_target_row = row_ids >= history_boundary  # (B, N, 1)
        # Target rows see [0, history_boundary) only.
        target_mask = col_ids < history_boundary
        mask = torch.where(is_target_row, target_mask, sla_mask)
    else:
        mask = sla_mask

    return mask & col_valid  # (B, N, N)


@torch.fx.wrap
def pytorch_sla_hstu_mha(
    max_seq_len: int,
    alpha: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_offsets: torch.Tensor,
    sla_k1: int,
    sla_k2: int,
    num_targets: Optional[torch.Tensor] = None,
    contextual_seq_len: int = 0,
) -> torch.Tensor:
    """PyTorch reference impl of HSTU attention with SLA + target isolation."""
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(q, k, v, seq_offsets, max_seq_len)
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / max_seq_len
    valid_attn_mask = _get_sla_attn_mask(
        device=q.device,
        N=max_seq_len,
        seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
        sla_k1=sla_k1,
        sla_k2=sla_k2,
        num_targets=num_targets,
        contextual_seq_len=contextual_seq_len,
    )
    qk_attn = qk_attn * valid_attn_mask.unsqueeze(1)
    attn_dense = torch.einsum("bhxd,bhdv->bhxv", qk_attn, v)
    return torch.ops.fbgemm.dense_to_jagged(
        attn_dense.transpose(1, 2).flatten(2, 3),
        [seq_offsets],
        L,
    )[0].view(L, H, V)
