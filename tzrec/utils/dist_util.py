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

import copy
import os
from datetime import timedelta
from queue import Queue
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Type

import numpy as np
import torch
from scipy.optimize import minimize, nnls
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
from torchrec.distributed.train_pipeline import TrainPipeline, TrainPipelineContext
from torchrec.distributed.train_pipeline import TrainPipelineBase as _TrainPipelineBase
from torchrec.distributed.train_pipeline import (
    TrainPipelineSparseDist as _TrainPipelineSparseDist,
)
from torchrec.distributed.train_pipeline.pipeline_context import In, Out
from torchrec.distributed.train_pipeline.types import PipelineState
from torchrec.distributed.types import (
    Awaitable,
    ModuleSharder,
    ShardedModule,
    ShardingEnv,
    ShardingPlan,
)
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor


def init_process_group() -> Tuple[torch.device, str]:
    """Init process_group, device, rank, backend."""
    rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device: torch.device = torch.device(f"cuda:{rank}")
        backend = "nccl"
        torch.cuda.set_device(device)
    else:
        device: torch.device = torch.device("cpu")
        backend = "gloo"

    pg_timeout = None
    if "PROCESS_GROUP_TIMEOUT_SECONDS" in os.environ:
        pg_timeout = timedelta(
            seconds=(int(os.environ["PROCESS_GROUP_TIMEOUT_SECONDS"]))
        )
    dist.init_process_group(backend=backend, timeout=pg_timeout)

    return device, backend


def get_dist_object_pg(world_size: Optional[int] = None) -> Optional[dist.ProcessGroup]:
    """New ProcessGroup used for broadcast_object or gather_object."""
    pg = None
    world_size = world_size or int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        # pyre-ignore [16]
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
            # pyre-ignore [16]
            and optimizer._gradient_accumulation_steps > 1
        ):
            loss = loss / optimizer._gradient_accumulation_steps
        # pyre-ignore [16]
        if hasattr(optimizer, "_grad_scaler") and optimizer._grad_scaler is not None:
            optimizer._grad_scaler.scale(loss).backward()
        else:
            loss.backward()


def _pareto_step(W: torch.Tensor, C: torch.Tensor, G: torch.Tensor) -> np.array:
    """Ref:http://ofey.me/papers/Pareto.pdf.

    Args:
        W(Tensor): dimension (K,1)
        C(Tensor): dimension (K,1)
        G(Tensor): dimension (K,M)
    """
    GGT = torch.matmul(G, G.T)  # (K, K)
    e = torch.ones(W.shape, device=W.device)  # (K, 1)
    m_up = torch.hstack((GGT, e))  # (K, K+1)
    m_down = torch.hstack(
        (e.T, torch.zeros([1, 1], dtype=torch.float32, device=W.device))
    )  # (1, K+1)
    M = torch.vstack((m_up, m_down))  # (K+1, K+1)
    z = torch.vstack((-torch.matmul(GGT, C), 1 - torch.sum(C)))  # (K+1, 1)
    hat_w = torch.matmul(
        torch.matmul(torch.linalg.inv(torch.matmul(M.T, M)), M), z
    )  # (K+1, 1)
    hat_w.detach()
    hat_w = hat_w[:-1].cpu().numpy()  # (K, 1)
    hat_w = hat_w.reshape(-1)
    c = C.data.clone().cpu().numpy().reshape(-1)  # (K,)
    new_w = _ASM(hat_w, c)
    return new_w


def _ASM(hat_w: np.array, c: np.array) -> np.array:
    """Ref: http://ofey.me/papers/Pareto.pdf.

    Args:
        hat_w: dimension (K)
        c: dimension (K)
    """
    A = np.array([[0 if i != j else 1 for i in range(len(c))] for j in range(len(c))])
    b = hat_w
    x0, _ = nnls(A, b)

    def _fn(x, A, b):
        return np.linalg.norm(A.dot(x) - b)

    cons = {"type": "eq", "fun": lambda x: np.sum(x) + np.sum(c) - 1}
    bounds = [[0.0, None] for _ in range(len(hat_w))]
    min_out = minimize(
        _fn, x0, args=(A, b), method="SLSQP", bounds=bounds, constraints=cons
    )
    new_w = min_out.x + c
    return new_w


def _pareto_w(
    losses: Dict[str, torch.Tensor], model: torch.nn.Module, init_c: float
) -> np.array:
    """Compute pareto front of losses for weight."""
    grads = []

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    for loss in losses.values():
        loss_sum = torch.sum(loss, dim=0)
        gradients = torch.autograd.grad(
            loss_sum,
            trainable_params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )
        grad_flattened = []
        for grad, param in zip(gradients, trainable_params):
            if grad is not None:
                grad_flattened.append(grad.view(-1))
            else:
                grad_flattened.append(torch.zeros_like(param).view(-1))

        all_grads = torch.cat(grad_flattened)
        grads.append(all_grads)
        # grads.append(all_grads.cpu().numpy())
    grads = torch.stack(grads)
    init_weight = 1 / len(losses)
    # init_c = init_weight * 0.5
    assert len(losses) * init_c <= 1.0 and len(losses) * init_c >= 0.0, (
        f"all init_c should be in [0, 1], we found loss number {len(losses)}"
    )
    w = torch.fill(
        torch.empty((len(losses), 1), dtype=torch.float32, device=grads.device),
        init_weight,
    )
    c = torch.fill(
        torch.empty((len(losses), 1), dtype=torch.float32, device=grads.device), init_c
    )
    return _pareto_step(w, c, grads)


class TrainPipelineBase(_TrainPipelineBase):
    """TorchEasyRec's TrainPipelineBase, make backward support grad scaler."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        custom_model_fwd: Optional[
            Callable[[In], Tuple[torch.Tensor, List[torch.Tensor]]]
        ] = None,
    ) -> None:
        super().__init__(model, optimizer, device, custom_model_fwd)
        self.pareto_init_weight_c = model.module.model._pareto_init_weight_c

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """For TrainPipelineBase progress."""
        if not self._connected:
            self._connect(dataloader_iter)
        if self._data_iter_stopped:
            raise StopIteration()

        # Fetch next batch, if depleted, raise at start of next progress
        next_batch = self._next_batch(dataloader_iter)
        cur_batch = self._cur_batch

        # for exhaustive data iter, some ranks will first depletes data,
        # but we still need progress the train pipeline for other ranks;
        # cur_batch could be None

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        if cur_batch is not None:
            self._wait_for_batch(cur_batch)

        # model will need to handle if cur_batch is empty; this is needed if there's
        # communicative ops
        with record_function("## forward ##"):
            (losses, output) = self._model(cur_batch)

        if self._model.training:
            if self.pareto_init_weight_c > 0.0:
                losses_dict, predictions, batch = output
                w = _pareto_w(losses_dict, self._model, self.pareto_init_weight_c)
                new_loss = []
                for i, loss in enumerate(losses_dict.values()):
                    new_loss.append(loss * w[i])
                losses = torch.stack(new_loss).sum()
            self._backward(losses)

        # Copy the next batch to GPU
        self._cur_batch = cur_batch = next_batch
        if cur_batch is not None:
            self._copy_batch_to_gpu(cur_batch)

        # Update
        if self._model.training:
            with record_function("## optimizer ##"):
                self._optimizer.step()

        return output


class TrainPipelineSparseDist(_TrainPipelineSparseDist):
    """TorchEasyRec's TrainPipelineSparseDist, make backward support grad scaler."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        execute_all_batches: bool = True,
        apply_jit: bool = False,
        context_type: Type[TrainPipelineContext] = TrainPipelineContext,
        # keep for backward compatibility
        pipeline_postproc: bool = False,
        custom_model_fwd: Optional[
            Callable[[Optional[In]], Tuple[torch.Tensor, Out]]
        ] = None,
        dmp_collection_sync_interval_batches: Optional[int] = 1,
        enqueue_batch_after_forward: bool = False,
        check_all_workers_data_status: bool = False,
    ) -> None:
        super().__init__(
            model,
            optimizer,
            device,
            execute_all_batches,
            apply_jit,
            context_type,
            pipeline_postproc,
            custom_model_fwd,
            dmp_collection_sync_interval_batches,
            enqueue_batch_after_forward,
        )
        self._check_all_workers_data_status = check_all_workers_data_status
        self.pareto_init_weight_c = model.module.model._pareto_init_weight_c

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)

            if self._check_all_workers_data_status:
                # Check if all workers either have or do not have a batch available.
                has_batch = torch.tensor(
                    0 if batch is None else 1, dtype=torch.float, device=self._device
                )
                dist.all_reduce(has_batch, dist.ReduceOp.AVG)
                if has_batch.item() < 1:
                    # We drop remainder batches on all workers,
                    # if one worker does not have a batch
                    self._dataloader_exhausted = True
                    batch = None
            else:
                if batch is None:
                    self._dataloader_exhausted = True
        return batch

    def _backward(self, losses: torch.Tensor) -> None:
        _pipeline_backward(losses, self._optimizer)

    def progress(self, dataloader_iter: Iterator[In]) -> Out:
        """For TrainPipelineSparseDist progress."""
        self._state = PipelineState.IDLE
        if not self._model_attached:
            self.attach(self._model)

        # fill the pipeline is only needed for the beginning when
        # the pipeline (batches) is empty
        self.fill_pipeline(dataloader_iter)

        # here is the expected stop after exhausting all batches
        if not self.batches:
            raise StopIteration

        # TODO: Remove once Bulk Eval migrated (needed for bwd compat, this class only)
        self._set_module_context(self.contexts[0])

        if self._model.training:
            with record_function("## zero_grad ##"):
                self._optimizer.zero_grad()

        self._wait_for_batch()

        if len(self.batches) >= 2:
            # invoke splits all_to_all comms (first part of input_dist)
            self.start_sparse_data_dist(self.batches[1], self.contexts[1])

        if not self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        # forward
        with record_function(f"## forward {self.contexts[0].index} ##"):
            self._state = PipelineState.CALL_FWD
            losses, output = self._model_fwd(self.batches[0])

        if self._enqueue_batch_after_forward:
            self.enqueue_batch(dataloader_iter)

        if len(self.batches) >= 2:
            self.wait_sparse_data_dist(self.contexts[1])

        if self._model.training:
            # backward
            if self.pareto_init_weight_c > 0.0:
                losses_dict, predictions, batch = output
                w = _pareto_w(losses_dict, self._model, self.pareto_init_weight_c)
                new_loss = []
                for i, loss in enumerate(losses_dict.values()):
                    new_loss.append(loss * w[i])
                losses = torch.stack(new_loss).sum()

            self._state = PipelineState.CALL_BWD
            self._backward(losses)

            self.sync_embeddings(
                self._model,
                self._dmp_collection_sync_interval_batches,
                self.contexts[0],
            )

            # update
            with record_function(f"## optimizer {self.contexts[0].index} ##"):
                self._optimizer.step()

        self.dequeue_batch()
        return output


class PredictPipelineSparseDist(_TrainPipelineSparseDist):
    """TorchEasyRec's PredictPipelineSparseDist, make predict do not hang."""

    def _next_batch(self, dataloader_iter: Iterator[In]) -> Optional[In]:
        if dataloader_iter is not self._dataloader_iter:
            self._dataloader_iter = dataloader_iter
            self._dataloader_exhausted = False

        if self._dataloader_exhausted:
            batch = None
        else:
            with record_function("## next_batch ##"):
                batch = next(dataloader_iter, None)

            # Check if all workers either have or do not have a batch available.
            has_batch = torch.tensor(
                0 if batch is None else 1, dtype=torch.float, device=self._device
            )
            dist.all_reduce(has_batch, dist.ReduceOp.AVG)
            if batch is None:
                if has_batch.item() > 0:
                    # If some workers still have a batch, create a dummy batch
                    # to avoid potential hang.
                    batch = copy.copy(self.batches[0])
                    batch.dummy = True
                else:
                    self._dataloader_exhausted = True
        return batch


def create_train_pipeline(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    check_all_workers_data_status: bool = False,
) -> TrainPipeline:
    """Create TrainPipeline.

    Args:
        model (nn.Module): a DMP model.
        optimizer (torch.optim.Optimizer): a KeyedOptimizer.
        check_all_workers_data_status (bool): check data on all workers
            is available or not.

    Return:
        a TrainPipeline.
    """
    has_sparse_module = False

    q = Queue()
    q.put(model.module)
    while not q.empty():
        m = q.get()
        if isinstance(m, ShardedModule):
            has_sparse_module = True
            break
        else:
            for child in m.children():
                q.put(child)

    if not has_sparse_module:
        # use TrainPipelineBase when model do not have sparse parameters.
        # pyre-ignore [6]
        return TrainPipelineBase(model, optimizer, model.device)
    else:
        return TrainPipelineSparseDist(
            model,
            # pyre-ignore [6]
            optimizer,
            model.device,
            execute_all_batches=True,
            check_all_workers_data_status=check_all_workers_data_status,
        )
