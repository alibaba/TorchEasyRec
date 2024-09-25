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

from collections import OrderedDict
from typing import List

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.types import ShardingPlan, ShardingType


def sync_dp_emb_table(model: nn.Module, plan: ShardingPlan) -> None:
    """Sync data parallel embedding table params."""
    dp_param_names = []
    for _, module_plan in plan.plan.items():
        # pyre-ignore [16]
        for param_name, param_sharding in module_plan.items():
            if param_sharding.sharding_type == ShardingType.DATA_PARALLEL.value:
                dp_param_names.append(param_name)
    dp_params = OrderedDict()
    for name, param in model.named_parameters():
        name_parts = name.split(".")
        if (
            len(name_parts) > 2
            and name_parts[-1] == "weight"
            and name_parts[-2] in dp_param_names
        ):
            # pyre-ignore [16]
            ori_t = param._original_tensor
            if ori_t not in dp_params:
                dp_params[ori_t] = 1
    broadcast_works = []
    for t in dp_params:
        broadcast_works.append(dist.broadcast(t.detach(), src=0, async_op=True))
    for w in broadcast_works:
        w.wait()


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
