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

import copy
import math
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from torch import nn
from torchrec.distributed.planner import planners
from torchrec.distributed.planner.planners import (
    EmbeddingPlannerBase,
    validate_compute_kernels,
    validate_modules_inclusion_in_sharding_plan,
    validate_rank_assignment,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    ParameterConstraints,
    PlannerError,
    PlannerErrorType,
    ShardingOption,
    Stats,
    Storage,
    StorageReservation,
    Topology,
)
from torchrec.distributed.types import ModuleSharder, ShardingPlan, ShardingType

from tzrec.utils.logging_util import logger


def _solve_placement(
    options: List[ShardingOption],
    topology: Topology,
    relative_gap: float,
) -> List[ShardingOption]:
    """Choose one option per table and its ranks in a single mixed-integer model.

    Binary x variables select options; y variables place movable shards. Uniform
    shards use x directly. Host selectors couple table-row/column-wise shards and
    dependent tables. Each rank has separate HBM, DDR and SSD constraints. The
    continuous objective bounds every rank's summed estimated execution cost.
    Resource coefficients are normalized by capacity without discretizing bytes.
    """
    started = time.perf_counter()
    world_size = topology.world_size
    host_size = topology.intra_group_size
    num_hosts, remainder = divmod(world_size, host_size)
    if remainder:
        raise ValueError("MILP requires complete hosts in the topology")
    supported = {
        ShardingType.ROW_WISE.value,
        ShardingType.DATA_PARALLEL.value,
        ShardingType.TABLE_WISE.value,
        ShardingType.COLUMN_WISE.value,
        ShardingType.TABLE_ROW_WISE.value,
        ShardingType.TABLE_COLUMN_WISE.value,
    }
    for option in options:
        if option.sharding_type not in supported:
            raise NotImplementedError(
                f"MILP does not support {option.sharding_type} for {option.fqn}; "
                "use TZREC_PLANNER=dp for this sharding type"
            )

    groups: Dict[str, List[int]] = defaultdict(list)
    for i, option in enumerate(options):
        groups[option.fqn].append(i)
    row_indices: List[int] = []
    col_indices: List[int] = []
    coefficients: List[float] = []
    lower: List[float] = []
    upper: List[float] = []
    variable_count = len(options)

    def binary_variables(count: int) -> List[int]:
        """Allocate indices for binary decision variables."""
        nonlocal variable_count
        indices = list(range(variable_count, variable_count + count))
        variable_count += count
        return indices

    def constraint(
        terms: List[tuple[int, float]], lo: float = 0, hi: float = 0
    ) -> None:
        """Append a sparse linear constraint."""
        row = len(lower)
        for col, value in terms:
            if value:
                row_indices.append(row)
                col_indices.append(col)
                coefficients.append(value)
        lower.append(lo)
        upper.append(hi)

    for indices in groups.values():
        constraint([(i, 1) for i in indices], 1, 1)

    placements: List[List[Dict[int, int]]] = []
    dependency_hosts: Dict[str, List[int]] = {}
    for i, option in enumerate(options):
        shard_placements: List[Dict[int, int]] = []
        kind = option.sharding_type
        if kind in (ShardingType.ROW_WISE.value, ShardingType.DATA_PARALLEL.value):
            if len(option.shards) != world_size:
                raise ValueError(f"{option.fqn}: uniform shards must match world_size")
            shard_placements = [{rank: i} for rank in range(world_size)]
        elif kind in (
            ShardingType.TABLE_ROW_WISE.value,
            ShardingType.TABLE_COLUMN_WISE.value,
        ):
            hosts = binary_variables(num_hosts) if num_hosts > 1 else [i]
            if num_hosts > 1:
                constraint([(h, 1) for h in hosts] + [(i, -1)])
                if option.dependency is not None:
                    if option.dependency not in dependency_hosts:
                        selectors = binary_variables(num_hosts)
                        dependency_hosts[option.dependency] = selectors
                        constraint([(h, 1) for h in selectors], 1, 1)
                    for h, group_h in zip(hosts, dependency_hosts[option.dependency]):
                        constraint([(h, 1), (group_h, -1)], -np.inf, 0)
            if kind == ShardingType.TABLE_ROW_WISE.value:
                if len(option.shards) != host_size:
                    raise ValueError(
                        f"{option.fqn}: table-row-wise shards must fit a host"
                    )
                shard_placements = [
                    {host * host_size + local_rank: h for host, h in enumerate(hosts)}
                    for local_rank in range(host_size)
                ]
            else:
                for _ in option.shards:
                    ranks = binary_variables(world_size)
                    for host, h in enumerate(hosts):
                        members = ranks[host * host_size : (host + 1) * host_size]
                        constraint([(r, 1) for r in members] + [(h, -1)])
                    shard_placements.append(dict(enumerate(ranks)))
        else:
            for _ in option.shards:
                ranks = binary_variables(world_size)
                constraint([(r, 1) for r in ranks] + [(i, -1)])
                shard_placements.append(dict(enumerate(ranks)))
            if (
                kind == ShardingType.COLUMN_WISE.value
                and option.compute_kernel == "key_value"
            ):
                for rank in range(world_size):
                    constraint(
                        [(p[rank], 1) for p in shard_placements] + [(i, -1)],
                        -np.inf,
                        0,
                    )
        placements.append(shard_placements)

    capacities = [
        [getattr(device.storage, name) or 0 for name in ("hbm", "ddr", "ssd")]
        for device in topology.devices
    ]
    resources: List[List[List[tuple[int, float]]]] = [
        [[] for _ in range(3)] for _ in range(world_size)
    ]
    perfs: List[List[tuple[int, float]]] = [[] for _ in range(world_size)]
    perf_scale = max(
        float(np.median([o.total_perf / world_size for o in options])), 1e-12
    )
    for option, mappings in zip(options, placements):
        for shard, mapping in zip(option.shards, mappings):
            assert shard.storage is not None and shard.perf is not None
            for rank, variable in mapping.items():
                for k, name in enumerate(("hbm", "ddr", "ssd")):
                    cost = getattr(shard.storage, name) or 0
                    resources[rank][k].append(
                        (variable, cost / max(capacities[rank][k], 1))
                    )
                perfs[rank].append((variable, shard.perf.total / perf_scale))

    target = variable_count
    for rank in range(world_size):
        for k in range(3):
            constraint(resources[rank][k], -np.inf, 1 if capacities[rank][k] > 0 else 0)
        constraint(perfs[rank] + [(target, -1)], -np.inf, 0)
    matrix = coo_matrix(
        (coefficients, (row_indices, col_indices)),
        shape=(len(lower), variable_count + 1),
    ).tocsc()
    objective = np.zeros(variable_count + 1)
    objective[target] = 1
    bounds_upper = np.ones(variable_count + 1)
    bounds_upper[target] = np.inf
    integrality = np.ones(variable_count + 1)
    integrality[target] = 0
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(0, bounds_upper),
        constraints=LinearConstraint(matrix, lower, upper),
        options={"time_limit": 60.0, "mip_rel_gap": relative_gap},
    )
    if result.status == 2:
        raise PlannerError(
            "MILP found no feasible selection and placement under the per-rank "
            "HBM, DDR, SSD and sharding constraints",
            PlannerErrorType.STRICT_CONSTRAINTS,
        )
    if result.status == 1 and result.x is None:
        raise PlannerError(
            "MILP reached the 60s solver limit without a feasible placement. "
            "This does not prove insufficient capacity; use TZREC_PLANNER=dp "
            "to try the DP planner."
        )
    if result.status not in (0, 1) or result.x is None:
        raise PlannerError(f"MILP placement solver failed: {result.message}")
    if not np.allclose(
        result.x[:target], np.rint(result.x[:target]), atol=1e-5, rtol=0
    ):
        raise PlannerError("MILP returned a non-integral placement")

    plan: List[ShardingOption] = []
    used = [Storage(hbm=0, ddr=0, ssd=0) for _ in range(world_size)]
    for indices in groups.values():
        selected = [i for i in indices if result.x[i] > 0.5]
        if len(selected) != 1:
            raise PlannerError("MILP did not select exactly one option per table")
        i = selected[0]
        option = copy.deepcopy(options[i])
        for shard, mapping in zip(option.shards, placements[i]):
            ranks = [
                rank for rank, variable in mapping.items() if result.x[variable] > 0.5
            ]
            if len(ranks) != 1:
                raise PlannerError(
                    f"MILP did not assign one rank to a shard of {option.fqn}"
                )
            shard.rank = ranks[0]
            assert shard.storage is not None
            used[shard.rank] += shard.storage
        plan.append(option)
    for rank, (storage, device) in enumerate(zip(used, topology.devices)):
        if not storage.fits_in(device.storage):
            raise PlannerError(
                f"MILP placement exceeds exact byte capacity on rank {rank}: "
                f"required={storage}, available={device.storage}",
                PlannerErrorType.PARTITION,
            )
    elapsed = time.perf_counter() - started
    logger.info(
        f"MILP selected {len(plan)} tables from {len(options)} options in "
        f"{elapsed:.3f}s; gap={result.mip_gap:.6g}"
    )
    if result.status == 1:
        logger.warning(
            "MILP reached the 60s solver limit; using the validated feasible "
            f"plan with gap={result.mip_gap:.6g}"
        )
    return plan


class MILPPlanner(EmbeddingPlannerBase):
    """Jointly select sharding options and rank assignments without resource bins.

    Supports row-wise, data-parallel, table-wise, column-wise, table-row-wise
    and table-column-wise sharding. Host-local shards and dependent tables stay
    on one host. GRID_SHARD is explicitly unsupported. The solver minimizes the
    maximum per-rank estimated performance cost, matching NoopPerfModel.
    The solver has a 60-second limit. A feasible incumbent is accepted after
    exact capacity validation even if the relative gap has not been reached.

    Args:
        topology: Hardware capacities and host layout.
        batch_size: Batch size used by the enumerator and storage reservation.
        enumerator: Supplies costed sharding options, including DynamicEmb options.
        storage_reservation: Reserves non-embedding memory before optimization.
        relative_gap: Relative optimality tolerance for the predicted cost.
        constraints: Constraints used by storage reservation and statistics.
        stats: Optional statistics reporters; an empty list disables reporting.
        callbacks: Callbacks applied to the selected, assigned options.
        debug: Whether to report detailed planning statistics.
    """

    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        enumerator: Enumerator,
        storage_reservation: StorageReservation,
        relative_gap: float = 0.001,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        stats: Optional[List[Stats]] = None,
        callbacks: Optional[
            List[Callable[[List[ShardingOption]], List[ShardingOption]]]
        ] = None,
        debug: bool = True,
    ) -> None:
        if not math.isfinite(relative_gap) or relative_gap < 0:
            raise ValueError("MILP relative_gap must be nonnegative and finite")
        super().__init__(
            topology=topology,
            batch_size=batch_size,
            enumerator=enumerator,
            storage_reservation=storage_reservation,
            constraints=constraints,
            stats=stats,
            callbacks=callbacks,
            debug=debug,
        )
        self._relative_gap = relative_gap
        self._selected_options: List[ShardingOption] = []
        self._search_space: Optional[List[ShardingOption]] = None

    def plan(
        self, module: nn.Module, sharders: List[ModuleSharder[nn.Module]]
    ) -> ShardingPlan:
        """Reserve memory and return a validated plan with solver-assigned ranks."""
        started = time.perf_counter()
        self._selected_options = []
        self._search_space = None
        capacity = self._storage_reservation.reserve(
            topology=self._topology,
            batch_size=self._batch_size,
            module=module,
            sharders=sharders,
            constraints=self._constraints,
        )
        self._search_space = self._enumerator.enumerate(module, sharders)
        if not self._search_space:
            return ShardingPlan({})
        selected = _solve_placement(self._search_space, capacity, self._relative_gap)
        for callback in self._callbacks:
            selected = callback(selected)
        plan = planners.to_sharding_plan(selected, self._topology)
        validate_modules_inclusion_in_sharding_plan(plan, module, sharders)
        validate_rank_assignment(plan, self._topology)
        validate_compute_kernels(selected)
        self._selected_options = selected
        for stats in self._stats:
            stats.log(
                sharding_plan=plan,
                topology=self._topology,
                batch_size=self._batch_size,
                storage_reservation=self._storage_reservation,
                num_proposals=1,
                num_plans=1,
                run_time=time.perf_counter() - started,
                best_plan=selected,
                constraints=self._constraints,
                enumerator=self._enumerator,
                sharders=sharders,
                debug=self._debug,
            )
        return plan

    def get_selected_options(self) -> List[ShardingOption]:
        """Return the assigned options from the last successful plan."""
        return self._selected_options

    def get_search_space(self) -> Optional[List[ShardingOption]]:
        """Return the last enumerated candidates, or None before enumeration."""
        return self._search_space
