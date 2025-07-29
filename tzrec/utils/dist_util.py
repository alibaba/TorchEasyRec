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

import os
from typing import List, Optional

import torch
from torch import distributed as dist
from torch import nn
from torch.autograd.profiler import record_function
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
from torchrec.distributed.train_pipeline import TrainPipeline
from torchrec.distributed.train_pipeline import TrainPipelineBase as _TrainPipelineBase
from torchrec.distributed.train_pipeline import (
    TrainPipelineSparseDist as _TrainPipelineSparseDist,
)
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def get_dist_object_pg(world_size: Optional[int] = None) -> Optional[dist.ProcessGroup]:
    """New ProcessGroup used for broadcast_object or gather_object."""
    pg = None
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if dist.is_initialized() and dist.GroupMember.WORLD.size() == world_size:
            pg = dist.GroupMember.WORLD
        else:
            pg = dist.new_group(ranks=list(range(world_size)), backend="gloo")
    return pg


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


def _pipeline_backward(losses: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
    with record_function("## backward ##"):
        loss = torch.sum(losses, dim=0)
        if (
            hasattr(optimizer, "_gradient_accumulation_steps")
            and optimizer._gradient_accumulation_steps > 1
        ):
            loss = loss / optimizer._gradient_accumulation_steps
        if hasattr(optimizer, "_grad_scaler") and optimizer._grad_scaler is not None:
            optimizer._grad_scaler.scale(loss).backward()
        else:
            loss.backward()


class TrainPipelineBase(_TrainPipelineBase):
    """TorchEasyRec's TrainPipelineBase, make backward support grad scaler."""

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)


class TrainPipelineSparseDist(_TrainPipelineSparseDist):
    """TorchEasyRec's TrainPipelineSparseDist, make backward support grad scaler."""

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)


def create_train_pipeline(
    model: _DistributedModelParallel, optimizer: Optional[torch.optim.Optimizer] = None
) -> TrainPipeline:
    """Create TrainPipeline."""
    trainable_params, frozen_params = model.module.model.sparse_parameters()
    if len(trainable_params) == 0 and len(frozen_params) == 0:
        # use TrainPipelineBase when model do not have sparse parameters.
        return TrainPipelineBase(model, optimizer, model.device)
    else:
        return TrainPipelineSparseDist(
            model, optimizer, model.device, execute_all_batches=True
        )
