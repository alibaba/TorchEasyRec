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

from typing import List, Optional

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.embedding_types import (
    KJTList,
)
from torchrec.distributed.embeddingbag import (
    ShardedEmbeddingBagCollection,
    _create_mean_pooling_divisor,
)
from torchrec.distributed.mc_embedding_modules import (
    BaseShardedManagedCollisionEmbeddingCollection,
    ShrdCtx,
)
from torchrec.distributed.model_parallel import DataParallelWrapper
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as _DistributedModelParallel,
)
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


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


# fix missing create_mean_pooling_callback of mc-ebc input_dist
def _mc_input_dist(
    # pyre-ignore [2]
    self,
    ctx: ShrdCtx,
    features: KeyedJaggedTensor,
) -> Awaitable[Awaitable[KJTList]]:
    if self._embedding_module._has_uninitialized_input_dist:
        if isinstance(self._embedding_module, ShardedEmbeddingBagCollection):
            self._features_order = []
            # disable feature permutation in mc, because we should
            # permute features in mc-ebc before mean pooling callback.
            if self._managed_collision_collection._has_uninitialized_input_dists:
                self._managed_collision_collection._create_input_dists(
                    input_feature_names=features.keys()
                )
                self._managed_collision_collection._has_uninitialized_input_dists = (
                    False
                )
                if self._managed_collision_collection._features_order:
                    self._features_order = (
                        self._managed_collision_collection._features_order
                    )
                    self._managed_collision_collection._features_order = []
            if self._embedding_module._has_mean_pooling_callback:
                self._embedding_module._init_mean_pooling_callback(
                    features.keys(),
                    # pyre-ignore [16]
                    ctx.inverse_indices,
                )
        self._embedding_module._has_uninitialized_input_dist = False
    if isinstance(self._embedding_module, ShardedEmbeddingBagCollection):
        with torch.no_grad():
            if self._features_order:
                features = features.permute(
                    self._features_order,
                    self._managed_collision_collection._features_order_tensor,
                )
            if self._embedding_module._has_mean_pooling_callback:
                ctx.divisor = _create_mean_pooling_divisor(
                    lengths=features.lengths(),
                    stride=features.stride(),
                    keys=features.keys(),
                    offsets=features.offsets(),
                    pooling_type_to_rs_features=self._embedding_module._pooling_type_to_rs_features,
                    stride_per_key=features.stride_per_key(),
                    dim_per_key=self._embedding_module._dim_per_key,
                    embedding_names=self._embedding_module._embedding_names,
                    embedding_dims=self._embedding_module._embedding_dims,
                    # pyre-ignore [16]
                    variable_batch_per_feature=ctx.variable_batch_per_feature,
                    kjt_inverse_order=self._embedding_module._kjt_inverse_order,
                    kjt_key_indices=self._embedding_module._kjt_key_indices,
                    kt_key_ordering=self._embedding_module._kt_key_ordering,
                    inverse_indices=ctx.inverse_indices,
                    weights=features.weights_or_none(),
                )
    # TODO: resolve incompatibility with different contexts
    return self._managed_collision_collection.input_dist(
        ctx,
        features,
    )


BaseShardedManagedCollisionEmbeddingCollection.input_dist = _mc_input_dist


def DistributedModelParallel(
    module: nn.Module,
    env: Optional[ShardingEnv] = None,
    device: Optional[torch.device] = None,
    plan: Optional[ShardingPlan] = None,
    sharders: Optional[List[ModuleSharder[torch.nn.Module]]] = None,
    init_data_parallel: bool = True,
    init_parameters: bool = True,
    data_parallel_wrapper: Optional[DataParallelWrapper] = None,
) -> _DistributedModelParallel:
    """Entry point to model parallelism.

    we custom ddp to make input_dist of ShardModel uninitialized.
    mc-ebc now make _has_uninitialized_input_dist = True in init.
    TODO: use torchrec DistributedModelParallel when torchrec fix it.
    """
    model = _DistributedModelParallel(
        module,
        env,
        device,
        plan,
        sharders,
        init_data_parallel,
        init_parameters,
        data_parallel_wrapper,
    )
    for _, m in model.named_modules():
        if hasattr(m, "_has_uninitialized_input_dist"):
            m._has_uninitialized_input_dist = True
    return model
