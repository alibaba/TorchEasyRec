# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple

import torch
from torch import distributed as dist
from torch.autograd.profiler import record_function
from torchrec.distributed import embeddingbag
from torchrec.distributed.utils import none_throws
from torchrec.modules.embedding_configs import PoolingType
from torchrec.sparse.jagged_tensor import _to_offsets


def broadcast_string(s: str, src: int = 0) -> str:
    """Broadcasts a string from the source rank to all other ranks."""
    if dist.get_rank() == src:
        s_tensor = torch.ByteTensor(bytearray(s, "utf-8"))
        length = torch.tensor([len(s_tensor)])
    else:
        length = torch.tensor([0], dtype=torch.long)

    if dist.get_backend() == dist.Backend.NCCL:
        length = length.cuda()
    dist.broadcast(length, src)

    if dist.get_rank() != src:
        s_tensor = torch.ByteTensor(length.item())

    if dist.get_backend() == dist.Backend.NCCL:
        s_tensor = s_tensor.cuda()
    # pyre-ignore [61]
    dist.broadcast(s_tensor, src)

    s_recv = s_tensor.cpu().numpy().tobytes().decode("utf-8")
    return s_recv


def gather_strings(s: str, dst: int = 0) -> List[str]:
    """Gather strings from all ranks to the destination rank."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    s_tensor = torch.ByteTensor(bytearray(s, "utf-8"))

    max_len = torch.tensor([len(s_tensor)], dtype=torch.long)
    max_len_list = [torch.tensor([0], dtype=torch.long) for _ in range(world_size)]
    if dist.get_backend() == dist.Backend.NCCL:
        max_len = max_len.cuda()
        max_len_list = [x.cuda() for x in max_len_list]
    dist.all_gather(max_len_list, max_len)

    # pyre-ignore [6]
    max_len = max(max_len_list).item()
    padded_s_tensor = torch.cat(
        (s_tensor, torch.zeros(max_len - len(s_tensor), dtype=torch.uint8))
    )
    if rank == dst:
        gather_list = [
            torch.zeros(max_len, dtype=torch.uint8) for _ in range(world_size)
        ]
    else:
        gather_list = []
    if dist.get_backend() == dist.Backend.NCCL:
        padded_s_tensor = padded_s_tensor.cuda()
        gather_list = [x.cuda() for x in gather_list]
    dist.gather(padded_s_tensor, gather_list, dst)

    gathered_strings = []
    if rank == dst:
        for tensor in gather_list:
            string = tensor.cpu().numpy().tobytes().decode("utf-8").rstrip("\x00")
            gathered_strings.append(string)

    return gathered_strings


# lengths of kjt will be modified by create_mean_pooling_divisor, we fit it temporarily.
def _create_mean_pooling_divisor(
    lengths: torch.Tensor,
    keys: List[str],
    offsets: torch.Tensor,
    stride: int,
    stride_per_key: List[int],
    dim_per_key: torch.Tensor,
    pooling_type_to_rs_features: Dict[str, List[str]],
    embedding_names: List[str],
    embedding_dims: List[int],
    variable_batch_per_feature: bool,
    kjt_inverse_order: torch.Tensor,
    kjt_key_indices: Dict[str, int],
    kt_key_ordering: torch.Tensor,
    inverse_indices: Optional[Tuple[List[str], torch.Tensor]] = None,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    with record_function("## ebc create mean pooling callback ##"):
        batch_size = (
            none_throws(inverse_indices)[1].size(dim=1)
            if variable_batch_per_feature
            else stride
        )

        if weights is not None:
            # if we have weights, lengths is the sum of weights by offsets for feature
            lengths = torch.ops.fbgemm.segment_sum_csr(1, offsets.int(), weights)

        if variable_batch_per_feature:
            inverse_indices = none_throws(inverse_indices)
            device = inverse_indices[1].device
            inverse_indices_t = inverse_indices[1]
            if len(keys) != len(inverse_indices[0]):
                inverse_indices_t = torch.index_select(
                    inverse_indices[1], 0, kjt_inverse_order
                )
            offsets = _to_offsets(torch.tensor(stride_per_key, device=device))[
                :-1
            ].unsqueeze(-1)
            indices = (inverse_indices_t + offsets).flatten()
            lengths = torch.index_select(input=lengths, dim=0, index=indices)

        # only convert the sum pooling features to be 1 lengths
        lengths = lengths.clone()
        for feature in pooling_type_to_rs_features[PoolingType.SUM.value]:
            feature_index = kjt_key_indices[feature]
            feature_index = feature_index * batch_size
            lengths[feature_index : feature_index + batch_size] = 1

        if len(embedding_names) != len(keys):
            lengths = torch.index_select(
                lengths.reshape(-1, batch_size),
                0,
                kt_key_ordering,
            ).reshape(-1)

        # transpose to align features with keyed tensor dim_per_key
        lengths = lengths.reshape(-1, batch_size).T  # [batch_size, num_features]
        output_size = sum(embedding_dims)

        divisor = torch.repeat_interleave(
            input=lengths,
            repeats=dim_per_key,
            dim=1,
            output_size=output_size,
        )
        eps = 1e-6  # used to safe guard against 0 division
        divisor = divisor + eps
        return divisor.detach()


# pyre-ignore [9]
embeddingbag._create_mean_pooling_divisor = _create_mean_pooling_divisor
