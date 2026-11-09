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
from typing import Any, Iterator, Literal, Optional

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.model_parallel import (
    DistributedModelParallel as _DistributedModelParallel,
)

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

    With ``train_config.rocksdb_config`` the runtime also owns this rank's
    optional RocksDB backend: it acquires it at open time, binds the
    planned external storages to the rank-local session inside
    :meth:`shard`, attaches the built caches to their storage, and closes
    it after the checkpoint manager on every exit path. Ordinary
    configurations run without the backend and without importing it.
    """

    def __init__(
        self,
        checkpoints: CheckpointManager,
        global_constraints_cfg: Optional[ParameterConstraints],
        rocksdb: Optional[Any] = None,
    ) -> None:
        """Initialize around an owned manager and job-wide constraints.

        Args:
            checkpoints: the checkpoint manager this runtime owns.
            global_constraints_cfg: the pipeline's global embedding
                constraints used for planning, or None when unset.
            rocksdb: this rank's RocksDB runtime
                (``rocksdb_util.RocksDBRuntime``), or None when the job
                does not offload embeddings.
        """
        self._global_constraints_cfg = global_constraints_cfg
        self._rocksdb = rocksdb
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
        body; without a prune worker, close is a no-op. With
        ``rocksdb_config`` the optional rank-local backend is acquired
        here and closed after the manager, so its working directory never
        outlives queued checkpoint work.

        Args:
            pipeline_config: the job's pipeline configuration.
            mode: the entry point opening the runtime, one of "train",
                "eval", "predict_checkpoint" or "export".

        Yields:
            The runtime owning the checkpoint manager and, when
            configured, the RocksDB backend.
        """
        train = pipeline_config.train_config
        constraints = (
            train.global_embedding_constraints
            if train.HasField("global_embedding_constraints")
            else None
        )
        backend = None
        if train.HasField("rocksdb_config"):
            from tzrec.utils.rocksdb_util import create_rocksdb_runtime

            backend = create_rocksdb_runtime(
                train.rocksdb_config, pipeline_config.model_dir, mode
            )
        manager_kwargs = (
            dict(
                keep_checkpoint_max=train.keep_checkpoint_max,
                export_config=pipeline_config.export_config,
            )
            if mode == "train"
            else {}
        )
        try:
            with closing(
                CheckpointManager(
                    pipeline_config.model_dir,
                    before_dynamicemb_load=(
                        backend.reset_for_import if backend is not None else None
                    ),
                    **manager_kwargs,
                )
            ) as checkpoints:
                yield cls(checkpoints, constraints, rocksdb=backend)
        finally:
            if backend is not None:
                backend.close()

    def shard(
        self,
        model: nn.Module,
        *,
        device: torch.device,
        batch_size: int,
        checkpoint_path: Optional[str],
        init_parameters: bool = True,
        init_data_parallel: bool = True,
    ) -> _DistributedModelParallel:
        """Collectively plan and shard one model, returning the DMP model.

        Uses the caller's device, sampled batch size and optional saved
        plan, with the constraints captured at open time. The broadcast
        plan is bound to this rank's backend session before the DMP model
        is constructed from it, and the built caches are attached to their
        storage afterwards; the backend stays None for ordinary
        configurations. Keeps the DMP defaults; RTP's FX tracing passes
        False for both flags to trace the unsharded view. Does not
        initialize a process group, create features, change the train/eval
        mode or restore weights; restore belongs to the caller via
        ``self.checkpoints``.

        Args:
            model: the wrapped model to shard.
            device: the local device to shard onto.
            batch_size: the dataloader's sampled batch size.
            checkpoint_path: the checkpoint whose saved plan should be
                reused, or None to plan from scratch.
            init_parameters: whether DMP initializes module parameters.
            init_data_parallel: whether DMP wraps the module in data
                parallel.

        Returns:
            The ``DistributedModelParallel`` sharded model.
        """
        planner = create_planner(
            device=device,
            batch_size=batch_size,
            ckpt_plan_path=os.path.join(checkpoint_path, "plan")
            if checkpoint_path
            else None,
            global_constraints_cfg=self._global_constraints_cfg,
            model=model,
            rocksdb=self._rocksdb,
        )
        sharders = get_default_sharders()
        plan = planner.collective_plan(model, sharders, dist.GroupMember.WORLD)
        if int(os.environ.get("RANK", 0)) == 0:
            logger.info(str(plan))
        if self._rocksdb is not None:
            self._rocksdb.bind(plan)
        dmp_model = DistributedModelParallel(
            module=model,
            sharders=sharders,
            device=device,
            plan=plan,
            init_parameters=init_parameters,
            init_data_parallel=init_data_parallel,
        )
        if self._rocksdb is not None:
            self._rocksdb.attach(dmp_model)
        return dmp_model

    def enforce_embedding_capacity(self) -> None:
        """Evict offloaded dynamic tables above their planned capacity.

        Delegates to the attached backend when the job has one; without
        one there is nothing to enforce. Call only with no lookup in
        flight: after a completed update, and at the drained final-batch
        boundary, before the step is published or saved.
        """
        if self._rocksdb is not None:
            self._rocksdb.evict()
