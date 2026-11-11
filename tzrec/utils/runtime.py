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

    Shares collective planning and DMP construction across job entry points.
    Owns the checkpoint manager and optional rank-local RocksDB backend,
    closing the manager first on every exit path. Callers restore weights
    through ``self.checkpoints``.
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

        Applies training retention and export settings in train mode and
        manager defaults otherwise. On exit or setup failure, closes the
        checkpoint manager before the optional RocksDB backend.

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

        Uses the constraints captured at open time and an optional saved plan.
        Binds the RocksDB backend before DMP construction and attaches caches
        afterwards, when configured. RTP tracing disables both initialization
        flags. Callers restore weights through ``self.checkpoints``.

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

    def maintain(self, *, global_step: int) -> None:
        """Run synchronous maintenance at a safe training boundary.

        Call with no lookup in flight, before publication or saving.

        Args:
            global_step: Current training step. Maintenance currently runs
                on every call, independent of the step number.
        """
        if self._rocksdb is not None:
            self._rocksdb.evict()
