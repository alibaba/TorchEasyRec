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

from typing import Dict, Union

import torch
from torch import distributed as dist
from torch.distributed._shard.sharded_tensor import ShardedTensor


def state_dict_gather(
    src: Dict[str, Union[torch.Tensor, ShardedTensor]],
    dst: Dict[str, torch.Tensor],
) -> None:
    """Gathers the values of the src state_dict into the dst state_dict.

    Gathers the values of the src state_dict of the keys present in the dst state_dict.
    Can handle ShardedTensors in the src state_dict.

    Args:
        src (Dict[str, Union[torch.Tensor, ShardedTensor]]): source's state_dict for
            this rank.
        dst (Dict[str, torch.Tensor]): destination's state_dict
    """
    for key, dst_tensor in dst.items():
        src_tensor = src[key]
        if isinstance(src_tensor, ShardedTensor):
            src_tensor.gather(
                out=dst_tensor if (dist.get_rank() == 0) else None,
                dtype=dst_tensor.dtype,
            )
        elif isinstance(src_tensor, torch.Tensor):
            dst_tensor.copy_(src_tensor)
        else:
            raise ValueError(f"Unsupported tensor {key} type {type(src_tensor)}")
