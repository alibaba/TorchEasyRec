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
import json
import os
from queue import Queue
from typing import List, Optional, cast

import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.enumerators import ParameterConstraints
from torchrec.distributed.planner.planners import EmbeddingPlannerBase
from torchrec.distributed.planner.proposers import UniformProposer
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.sharding_plan import (
    get_default_sharders as _get_default_sharders,
)
from torchrec.distributed.types import ModuleSharder

from tzrec.protos import feature_pb2
from tzrec.utils import env_util, resource_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.logging_util import logger
from tzrec.utils.planner.enumerators import EmbeddingEnumerator
from tzrec.utils.planner.proposers import DynamicProgrammingProposer


def create_planner(
    device: torch.device,
    batch_size: int,
    ckpt_plan_path: Optional[str] = None,
    global_constraints_cfg: Optional[feature_pb2.ParameterConstraints] = None,
    model: Optional[nn.Module] = None,
) -> EmbeddingPlannerBase:
    """Create an embedding sharding planner.

    Builds the hardware topology and storage reservation, resolves
    per-table and global constraints, applies applicable sharding
    constraints from a saved checkpoint plan, and configures the
    enumerator and the selected planning algorithm.
    """
    if global_constraints_cfg is not None and (
        global_constraints_cfg.HasField("pooling_factor")
        or global_constraints_cfg.HasField("unique_id_factor")
    ):
        raise ValueError(
            "global_embedding_constraints cannot set pooling_factor or "
            "unique_id_factor; the ids a sample looks up differ per feature, "
            "set them on each feature's embedding_constraints."
        )
    local_world_size = get_local_size()
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0

    # build topo
    topo_kwargs = {}
    if torch.cuda.is_available():
        topo_kwargs["hbm_cap"] = torch.cuda.get_device_properties(device).total_memory

    host_reserve_percent = float(os.environ.get("HOST_MEMORY_RESERVE_PERCENT", 0.0))
    topo_kwargs["ddr_cap"] = resource_util.host_memory_capacity_per_rank(
        local_world_size=local_world_size,
        reserve_percent=host_reserve_percent,
    )
    if "INTRA_NODE_BANDWIDTH" in os.environ:
        topo_kwargs["intra_host_bw"] = float(os.environ["INTRA_NODE_BANDWIDTH"])
    if "CROSS_NODE_BANDWIDTH" in os.environ:
        topo_kwargs["inter_host_bw"] = float(os.environ["CROSS_NODE_BANDWIDTH"])
    topology = Topology(
        local_world_size=local_world_size,
        world_size=dist.get_world_size(),
        compute_device=device.type,
        **topo_kwargs,
    )

    # build storage reservation
    # If experience OOM, increase the percentage. see
    # https://pytorch.org/torchrec/torchrec.distributed.planner.html#torchrec.distributed.planner.storage_reservations.HeuristicalStorageReservation    # NOQA
    storage_reserve_percent = 0.15
    if "STORAGE_RESERVE_PERCENT" in os.environ:
        storage_reserve_percent = float(os.environ["STORAGE_RESERVE_PERCENT"])
    storage_reservation = HeuristicalStorageReservation(
        percentage=storage_reserve_percent
    )

    fqn_constraints = {}

    # add parameter constraints for each embedding parameter.
    q = Queue()
    q.put(("", model))
    while not q.empty():
        path, m = q.get()
        if hasattr(m, "parameter_constraints"):
            fqn_constraints.update(m.parameter_constraints(path))
        else:
            for name, child in m.named_children():
                q.put((f"{path}{name}.", child))

    if ckpt_plan_path is not None and os.path.exists(ckpt_plan_path):
        force_load_sharding_plan = env_util.force_load_sharding_plan()
        with open(ckpt_plan_path, "r") as f:
            ckpt_plan = json.load(f)
            for module_path, module_plan in ckpt_plan.items():
                for param_name, param_sharding in module_plan.items():
                    if (
                        param_sharding["sharding_type"] == "data_parallel"
                        or force_load_sharding_plan
                    ):
                        # the optimizer state key names differ when using data_parallel
                        # for embedding sharding compared to when using row_wise and
                        # table_wise https://github.com/pytorch/torchrec/issues/2394.
                        # So that, we add constraints for params with data_parallel
                        # plan in ckpt.
                        fqn = f"{module_path}.{param_name}"
                        # a table's own constraint entry outranks the saved
                        # plan; entries that do not restrict sharding (such as
                        # measured pooling factors) let the pin through
                        const = fqn_constraints.get(fqn)
                        if const is None or const.sharding_types is None:
                            if is_rank_zero:
                                logger.info(
                                    f"add ParameterConstraints[sharding_types=['{param_sharding['sharding_type']}']] for param[{fqn}] from checkpoint plan."  # NOQA
                                )
                            if const is None:
                                fqn_constraints[fqn] = ParameterConstraints(
                                    sharding_types=[param_sharding["sharding_type"]]
                                )
                            else:
                                # the entries are the model's own constraint
                                # objects; pin a copy so a later replan of the
                                # same model does not inherit this pin
                                pinned = copy.copy(const)
                                pinned.sharding_types = [
                                    param_sharding["sharding_type"]
                                ]
                                fqn_constraints[fqn] = pinned

    global_constraints = None
    if global_constraints_cfg is not None:
        global_constraints = ParameterConstraints(
            sharding_types=list(global_constraints_cfg.sharding_types)
            if len(global_constraints_cfg.sharding_types) > 0
            else None,
            compute_kernels=list(global_constraints_cfg.compute_kernels)
            if len(global_constraints_cfg.compute_kernels) > 0
            else None,
        )

    enumerator = EmbeddingEnumerator(
        topology=topology,
        batch_size=batch_size,
        fqn_constraints=fqn_constraints,
        global_constraints=global_constraints,
    )
    planner_type = os.environ.get("TZREC_PLANNER", "dp")
    if planner_type == "milp":
        from tzrec.utils.planner.planners import MILPEmbeddingShardingPlanner

        return MILPEmbeddingShardingPlanner(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            relative_gap=float(os.environ.get("TZREC_MILP_RELATIVE_GAP", "0.001")),
            time_limit=float(os.environ.get("TZREC_MILP_TIME_LIMIT_SECONDS", "60.0")),
        )
    # build planner
    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=enumerator,
        storage_reservation=storage_reservation,
        # DP bin counts are env-tunable; defaults match the proposer signature.
        proposer=[
            DynamicProgrammingProposer(
                hbm_bins_per_device=int(
                    os.environ.get("TZREC_DP_HBM_BINS_PER_DEVICE", "100")
                ),
                ddr_bins_per_device=int(
                    os.environ.get("TZREC_DP_DDR_BINS_PER_DEVICE", "25")
                ),
            ),
            UniformProposer(),
        ],
        debug=True,
    )
    return planner


def get_default_sharders() -> List[ModuleSharder[nn.Module]]:
    """Get embedding module default sharder."""
    if torch.cuda.is_available():
        sharders = _get_default_sharders()
        if has_dynamicemb:
            from dynamicemb.shard import (
                DynamicEmbeddingBagCollectionSharder,
                DynamicEmbeddingCollectionSharder,
            )

            sharders.extend(
                [
                    cast(
                        ModuleSharder[nn.Module], DynamicEmbeddingBagCollectionSharder()
                    ),
                    cast(ModuleSharder[nn.Module], DynamicEmbeddingCollectionSharder()),
                ]
            )
        return sharders
    else:
        # ShardedEmbeddingCollection is not supported yet.
        sharders = []
        for sharder in _get_default_sharders():
            if "EmbeddingCollection" not in str(sharder):
                sharders.append(sharder)
        return sharders
