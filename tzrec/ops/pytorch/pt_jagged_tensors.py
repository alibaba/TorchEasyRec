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

# We use the jagged_tensors ops from generative-recommenders a starting point.
# https://github.com/facebookresearch/generative-recommenders
# thanks to their public work.


from typing import Optional, Tuple

import torch

from tzrec.utils.fx_util import fx_arange

torch.fx.wrap(fx_arange)


def _concat_2D_jagged_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> torch.Tensor:
    max_seq_len = max_len_left + max_len_right
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    padded_left = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_left,
        offsets=[offsets_left],
        max_lengths=[max_len_left],
        padding_value=0.0,
    )
    padded_right = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values_right,
        offsets=[offsets_right],
        max_lengths=[max_len_right],
        padding_value=0.0,
    )
    concatted_dense = torch.cat([padded_left, padded_right], dim=1)
    mask = fx_arange(max_seq_len, device=offsets_left.device).view(1, -1)
    mask = torch.logical_or(
        mask < lengths_left.view(-1, 1),
        torch.logical_and(
            mask >= max_len_left,
            mask < max_len_left + lengths_right.view(-1, 1),
        ),
    )
    return concatted_dense.flatten(0, 1)[mask.view(-1), :]


@torch.fx.wrap
def pytorch_concat_2D_jagged(
    values_left: torch.Tensor,
    values_right: torch.Tensor,
    max_len_left: int,
    max_len_right: int,
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> torch.Tensor:
    if offsets_left is None:
        B = values_left.shape[0] // max_len_left
        offsets_left_non_optional = max_len_left * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        B = values_right.shape[0] // max_len_right
        offsets_right_non_optional = max_len_right * torch.arange(
            B + 1, device=values_left.device
        )
    else:
        offsets_right_non_optional = offsets_right
    return _concat_2D_jagged_jagged(
        values_left=values_left,
        values_right=values_right,
        max_len_left=max_len_left,
        max_len_right=max_len_right,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def _split_2D_jagged_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    offsets_left: torch.Tensor,
    offsets_right: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    offsets = offsets_left + offsets_right
    padded_values = torch.ops.fbgemm.jagged_to_padded_dense(
        values=values,
        offsets=[offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    ).flatten(0, 1)
    lengths_left = offsets_left[1:] - offsets_left[:-1]
    lengths_right = offsets_right[1:] - offsets_right[:-1]
    mask = fx_arange(max_seq_len, device=values.device).view(1, -1)
    mask_left = mask < lengths_left.view(-1, 1)
    mask_right = torch.logical_and(
        mask >= lengths_left.view(-1, 1),
        mask < (lengths_left + lengths_right).view(-1, 1),
    )
    return padded_values[mask_left.view(-1), :], padded_values[mask_right.view(-1), :]


@torch.fx.wrap
def pytorch_split_2D_jagged(
    max_seq_len: int,
    values: torch.Tensor,
    max_len_left: Optional[int],
    max_len_right: Optional[int],
    offsets_left: Optional[torch.Tensor],
    offsets_right: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if offsets_left is None:
        assert max_len_left is not None
        assert offsets_right is not None
        offsets_left_non_optional = max_len_left * torch.arange(
            offsets_right.shape[0], device=values.device
        )
    else:
        offsets_left_non_optional = offsets_left
    if offsets_right is None:
        assert max_len_right is not None
        assert offsets_left is not None
        offsets_right_non_optional = max_len_right * torch.arange(
            offsets_left.shape[0], device=values.device
        )
    else:
        offsets_right_non_optional = offsets_right
    return _split_2D_jagged_jagged(
        max_seq_len=max_seq_len,
        values=values,
        offsets_left=offsets_left_non_optional,
        offsets_right=offsets_right_non_optional,
    )


def pytorch_hstu_split_l2_embeddings(
    max_seq_len: int,
    x: torch.Tensor,
    minus_l2_offsets: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_offsets = minus_l2_offsets + l2_offsets
    x_lengths = x_offsets[1:] - x_offsets[:-1]
    padded_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=x,
        offsets=[x_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    ).flatten(0, 1)
    minus_l2_lengths = minus_l2_offsets[1:] - minus_l2_offsets[:-1]
    mask = fx_arange(max_seq_len, device=x_offsets.device).view(1, -1)
    mask_minus_l2 = torch.logical_and(
        mask >= contextual_seq_len,
        mask < minus_l2_lengths.view(-1, 1) + contextual_seq_len,
    )
    mask_l2 = torch.logical_or(
        mask < contextual_seq_len,
        torch.logical_and(
            mask >= minus_l2_lengths.view(-1, 1) + contextual_seq_len,
            mask < x_lengths.view(-1, 1),
        ),
    )
    return padded_x[mask_minus_l2.view(-1), :], padded_x[mask_l2.view(-1), :]


def pytorch_hstu_concat_l2_embeddings(
    max_minus_l2_len: int,
    minus_l2_x: torch.Tensor,
    minus_l2_offsets: torch.Tensor,
    max_l2_len: int,
    l2_x: torch.Tensor,
    l2_offsets: torch.Tensor,
    contextual_seq_len: int,
) -> torch.Tensor:
    padded_minus_l2_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=minus_l2_x,
        offsets=[minus_l2_offsets],
        max_lengths=[max_minus_l2_len],
        padding_value=0.0,
    )
    padded_l2_x = torch.ops.fbgemm.jagged_to_padded_dense(
        values=l2_x,
        offsets=[l2_offsets],
        max_lengths=[max_l2_len],
        padding_value=0.0,
    )
    padded_x = torch.cat(
        [
            padded_l2_x[:, 0:contextual_seq_len, :],
            padded_minus_l2_x,
            padded_l2_x[:, contextual_seq_len:, :],
        ],
        dim=1,
    )
    mask = fx_arange(max_minus_l2_len + max_l2_len, device=minus_l2_x.device).view(
        1, -1
    )
    minus_l2_lengths = minus_l2_offsets[1:] - minus_l2_offsets[:-1]
    l2_lengths = l2_offsets[1:] - l2_offsets[:-1]
    mask = torch.logical_or(
        mask < minus_l2_lengths.view(-1, 1) + contextual_seq_len,
        torch.logical_and(
            mask >= max_minus_l2_len + contextual_seq_len,
            mask < max_minus_l2_len + l2_lengths.view(-1, 1),
        ),
    )
    return padded_x.flatten(0, 1)[mask.view(-1), :]


def pytorch_jagged_dense_bmm_broadcast_add(
    max_seq_len: int,
    seq_offsets: torch.Tensor,
    jagged: torch.Tensor,
    dense: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    dtype = jagged.dtype
    jagged = jagged.to(torch.float32)
    dense = dense.to(torch.float32)
    padded_jagged = torch.ops.fbgemm.jagged_to_padded_dense(
        values=jagged,
        offsets=[seq_offsets],
        max_lengths=[max_seq_len],
        padding_value=0.0,
    )
    bmm_out = torch.bmm(padded_jagged, dense)
    jagged_out = torch.ops.fbgemm.dense_to_jagged(
        bmm_out + bias.unsqueeze(1), [seq_offsets], total_L=jagged.shape[0]
    )[0]
    jagged_out = jagged_out.to(dtype)
    return jagged_out
