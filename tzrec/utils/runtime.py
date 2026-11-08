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

"""The per-job runtime shared by entries that plan, restore and use a sharded model."""

import os
from contextlib import closing, contextmanager
from typing import Iterator, List, Literal, Optional, Tuple

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as _DistributedModelParallel,
)
from torchrec.distributed.types import ModuleSharder, ShardingPlan

from tzrec.protos.feature_pb2 import ParameterConstraints
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils.checkpoint_util import CheckpointManager
from tzrec.utils.dist_util import DistributedModelParallel
from tzrec.utils.logging_util import logger
from tzrec.utils.plan_util import create_planner, get_default_sharders


class ShardedModelRuntime:
    """Own the checkpoint manager and the sharding setup of one job.

    Train, standalone eval and checkpoint prediction repeat the same
    sharded-model sequence: create the planner with the effective global
    embedding constraints, run collective planning, log the plan on rank
    zero and construct the DMP model. This runtime shares that sequence and
    owns the ``CheckpointManager`` for the job's whole lifetime, so setup
    failures and early returns inside :meth:`open` still close the manager
    and drain its queued prune work. What differs between the entries --
    when and what to restore -- stays with the caller through
    ``self.checkpoints``.
    """

    def __init__(
        self,
        checkpoints: CheckpointManager,
        global_constraints_cfg: Optional[ParameterConstraints],
    ) -> None:
        """Initialize around an owned manager and job-wide constraints.

        Args:
            checkpoints: the checkpoint manager this runtime owns.
            global_constraints_cfg: the pipeline's global embedding
                constraints used for planning, or None when unset.
        """
        self._global_constraints_cfg = global_constraints_cfg
        self.checkpoints = checkpoints

    @classmethod
    @contextmanager
    def open(
        cls,
        pipeline_config: EasyRecConfig,
        *,
        mode: Literal["train", "eval", "predict_checkpoint", "export"],
    ) -> Iterator["ShardedModelRuntime"]:
        """Open the runtime for one entry point of the job.

        Constructs and owns the real ``CheckpointManager``: training keeps
        its retention and export settings, while the other entries keep
        the read-only defaults. The manager is closed on every exit path,
        including setup failures and early returns inside the ``with``
        body; without a prune worker, close is a no-op.

        Args:
            pipeline_config: the job's pipeline configuration.
            mode: the entry point opening the runtime, one of "train",
                "eval", "predict_checkpoint" or "export".

        Yields:
            The runtime owning the checkpoint manager.
        """
        if mode == "train":
            manager = CheckpointManager(
                pipeline_config.model_dir,
                keep_checkpoint_max=pipeline_config.train_config.keep_checkpoint_max,
                export_config=pipeline_config.export_config,
            )
        else:
            # Existing standalone eval/predict/export manager defaults.
            manager = CheckpointManager(pipeline_config.model_dir)
        with closing(manager):
            train = pipeline_config.train_config
            constraints = (
                train.global_embedding_constraints
                if train.HasField("global_embedding_constraints")
                else None
            )
            yield cls(manager, constraints)

    def plan(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        batch_size: int,
        checkpoint_path: Optional[str],
    ) -> Tuple[ShardingPlan, List[ModuleSharder[nn.Module]]]:
        """Run collective planning once, returning the plan and its sharders.

        Uses the caller's device, sampled batch size and optional saved
        plan, with the constraints captured at open time. The returned
        plan and sharders are passed explicitly to :meth:`apply_plan`;
        the runtime does not retain them, so one plan can be applied to
        several graphs (RTP traces the full model, then applies the same
        plan to the derived sparse graph).

        Args:
            model: the wrapped model to plan for.
            device: the local device to plan onto.
            batch_size: the dataloader's sampled batch size.
            checkpoint_path: the checkpoint whose saved plan should be
                reused, or None to plan from scratch.

        Returns:
            The collective sharding plan and the sharders that apply it.
        """
        planner = create_planner(
            device=device,
            batch_size=batch_size,
            ckpt_plan_path=os.path.join(checkpoint_path, "plan")
            if checkpoint_path
            else None,
            global_constraints_cfg=self._global_constraints_cfg,
            model=model,
        )
        sharders = get_default_sharders()
        plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(str(plan))
        return plan, sharders

    def apply_plan(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        plan: ShardingPlan,
        sharders: List[ModuleSharder[nn.Module]],
        init_parameters: bool = True,
        init_data_parallel: bool = True,
    ) -> _DistributedModelParallel:
        """Construct the DMP model from an existing plan, without replanning.

        Keeps the DMP defaults; RTP's FX tracing passes
        ``init_parameters=False, init_data_parallel=False`` for the
        unsharded trace view and reuses the same plan with the defaults
        on the derived sparse graph.

        Args:
            model: the module to shard with the plan.
            device: the local device to construct the DMP model on.
            plan: an existing collective sharding plan.
            sharders: the sharders that produced the plan.
            init_parameters: whether DMP initializes module parameters.
            init_data_parallel: whether DMP wraps the module in data
                parallel.

        Returns:
            The ``DistributedModelParallel`` sharded model.
        """
        return DistributedModelParallel(
            module=model,
            sharders=sharders,
            device=device,
            plan=plan,
            init_parameters=init_parameters,
            init_data_parallel=init_data_parallel,
        )

    def shard(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        batch_size: int,
        checkpoint_path: Optional[str],
    ) -> _DistributedModelParallel:
        """Collectively plan and shard one model, returning the DMP model.

        Composes :meth:`plan` and :meth:`apply_plan`, passing the plan and
        sharders explicitly between them. Uses the caller's device,
        sampled batch size and optional saved plan, with the constraints
        captured at open time. Does not initialize a process group,
        create features, change the train/eval mode or restore weights;
        restore belongs to the caller via ``self.checkpoints``.

        Args:
            model: the wrapped model to shard.
            device: the local device to shard onto.
            batch_size: the dataloader's sampled batch size.
            checkpoint_path: the checkpoint whose saved plan should be
                reused, or None to plan from scratch.

        Returns:
            The ``DistributedModelParallel`` sharded model.
        """
        plan, sharders = self.plan(
            model,
            device=device,
            batch_size=batch_size,
            checkpoint_path=checkpoint_path,
        )
        return self.apply_plan(model, device=device, plan=plan, sharders=sharders)
