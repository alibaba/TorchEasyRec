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
def _decode_attn_func_to_mask(
    attn_func: torch.Tensor,
    seq_offsets: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Decode a CUTLASS-style NFUNC=3 mask tensor into a dense bool mask.

    The NFUNC encoding (see ``build_sla_func_tensor``) has shape
    ``(nheads, 3, total_q)`` where for each query position q the three
    values are ``[col_max0, col_min0, col_max1]``.  Query q attends to key
    positions in ``[0, col_max0) ∪ [col_min0, col_max1)``.

    This helper produces a per-sample dense ``(B, H, N, N)`` boolean mask
    consumable by the jagged-to-padded PyTorch reference path.  Columns
    past each sample's jagged length are forced to False so padding keys
    never contribute to attention.

    Args:
        attn_func: shape ``(H, 3, total_q)`` int32, jagged along ``total_q``.
        seq_offsets: shape ``(B + 1,)`` int32 cumulative offsets matching
            ``attn_func``'s jagged layout.
        N: padded maximum sequence length.

    Returns:
        bool tensor of shape ``(B, H, N, N)``.
    """
    H, three, total_q = attn_func.shape
    torch._assert(three == 3, "attn_func must have shape (H, 3, total_q)")
    # Fold (H, 3) into channels so we can use jagged_to_padded_dense with
    # the (total_q, C) 2D layout; unfold after padding.
    padded_flat = attn_func.permute(2, 0, 1).reshape(total_q, H * 3)
    padded = torch.ops.fbgemm.jagged_to_padded_dense(
        values=padded_flat,
        offsets=[seq_offsets],
        max_lengths=[N],
        padding_value=0,
    )  # (B, N, H * 3)
    B = padded.size(0)
    padded = padded.view(B, N, H, 3).permute(0, 2, 1, 3)  # (B, H, N, 3)
    col_max0 = padded[..., 0:1]  # (B, H, N, 1)
    col_min0 = padded[..., 1:2]
    col_max1 = padded[..., 2:3]
    col_ids = torch.arange(N, device=attn_func.device, dtype=torch.int32).view(
        1, 1, 1, N
    )
    in_0 = col_ids < col_max0
    in_1 = (col_ids >= col_min0) & (col_ids < col_max1)
    mask = in_0 | in_1  # (B, H, N, N) bool
    seq_lengths = (seq_offsets[1:] - seq_offsets[:-1]).to(torch.int32).view(B, 1, 1, 1)
    col_valid = col_ids < seq_lengths
    return mask & col_valid


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
    attn_func: Optional[torch.Tensor] = None,
    scaling_seqlen: int = -1,
) -> torch.Tensor:
    """PyTorch reference HSTU attention.

    When ``attn_func`` is provided the mask is decoded from the NFUNC=3
    tensor (matching the CUTLASS kernel's arbitrary-mask path).  When
    ``attn_func`` is ``None`` the fixed-mask path uses ``causal`` /
    ``max_attn_len`` / ``contextual_seq_len`` / ``num_targets``.
    """
    if scaling_seqlen == -1:
        scaling_seqlen = max_seq_len
    L, H, _ = q.shape
    V = v.shape[2]
    q, k, v = _pad_qkv(
        q, k, v, seq_offsets, max_seq_len
    )  # [B, H, N, D) and [B, H, N, V]
    qk_attn = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
    qk_attn = F.silu(qk_attn) / scaling_seqlen
    if attn_func is not None:
        # NFUNC arbitrary-mask path; the mask already encodes causality,
        # local window, contextual prefix and target isolation.
        valid_attn_mask = _decode_attn_func_to_mask(
            attn_func=attn_func, seq_offsets=seq_offsets, N=max_seq_len
        )  # (B, H, N, N)
        qk_attn = qk_attn * valid_attn_mask
    else:
        valid_attn_mask = _get_valid_attn_mask(
            device=q.device,
            causal=causal,
            N=max_seq_len,
            seq_lengths=seq_offsets[1:] - seq_offsets[:-1],
            num_targets=num_targets,
            max_attn_len=max_attn_len,
            contextual_seq_len=contextual_seq_len,
            min_full_attn_seq_len=min_full_attn_seq_len,
        )  # (B, N, N) -- broadcasts over H
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
    scaling_seqlen: int = -1,
) -> torch.Tensor:
    if scaling_seqlen == -1:
        scaling_seqlen = max_seq_len
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
    qk_attn = F.silu(qk_attn) / scaling_seqlen
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
