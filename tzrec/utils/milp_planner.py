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


class _PlacementModel:
    """Select table options and place shards under per-rank HBM/DDR constraints.

    Each option's index is its binary selection variable. Shard-to-rank mappings
    reference selection, host or shard-placement variables, depending on the
    sharding type. The objective minimizes the maximum estimated rank cost.
    """

    def __init__(self, options: List[ShardingOption], topology: Topology) -> None:
        self._options = options
        self._topology = topology
        self._world_size = topology.world_size
        self._host_size = topology.intra_group_size
        self._num_hosts, remainder = divmod(self._world_size, self._host_size)
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

        self._option_indices_by_table: Dict[str, List[int]] = defaultdict(list)
        for option_index, option in enumerate(options):
            self._option_indices_by_table[option.fqn].append(option_index)
        self._shard_rank_variables: List[List[Dict[int, int]]] = [[] for _ in options]
        self._dependency_host_variables: Dict[str, List[int]] = {}
        self._variable_count = len(options)
        self._row_indices: List[int] = []
        self._col_indices: List[int] = []
        self._coefficients: List[float] = []
        self._constraint_lower: List[float] = []
        self._constraint_upper: List[float] = []

        self._add_selection_and_placement_constraints()
        self._max_rank_cost_variable = self._variable_count
        self._add_rank_cost_constraints()

        variable_count = self._variable_count + 1
        matrix = coo_matrix(
            (self._coefficients, (self._row_indices, self._col_indices)),
            shape=(len(self._constraint_lower), variable_count),
        ).tocsc()
        self.constraints = LinearConstraint(
            matrix, self._constraint_lower, self._constraint_upper
        )
        self.objective = np.zeros(variable_count)
        self.objective[self._max_rank_cost_variable] = 1
        bounds_upper = np.ones(variable_count)
        bounds_upper[self._max_rank_cost_variable] = np.inf
        self.bounds = Bounds(0, bounds_upper)
        self.integrality = np.ones(variable_count)
        self.integrality[self._max_rank_cost_variable] = 0

    def _add_binary_variables(self, count: int) -> List[int]:
        """Allocate indices for binary decision variables."""
        variables = list(range(self._variable_count, self._variable_count + count))
        self._variable_count += count
        return variables

    def _add_constraint(
        self, terms: List[tuple[int, float]], *, lower: float, upper: float
    ) -> None:
        """Append a bounded linear expression to the sparse constraint matrix."""
        row = len(self._constraint_lower)
        for col, coefficient in terms:
            if coefficient:
                self._row_indices.append(row)
                self._col_indices.append(col)
                self._coefficients.append(coefficient)
        self._constraint_lower.append(lower)
        self._constraint_upper.append(upper)

    def _add_selection_and_placement_constraints(self) -> None:
        """Select one option per table and place all and only its selected shards."""
        for option_indices in self._option_indices_by_table.values():
            for option_variable in option_indices:
                option = self._options[option_variable]
                shard_mappings = self._shard_rank_variables[option_variable]
                kind = option.sharding_type
                if kind in (
                    ShardingType.ROW_WISE.value,
                    ShardingType.DATA_PARALLEL.value,
                ):
                    if len(option.shards) != self._world_size:
                        raise ValueError(
                            f"{option.fqn}: uniform shards must match world_size"
                        )
                    # Fixed ranks reuse the option's selection variable.
                    shard_mappings.extend(
                        {rank: option_variable} for rank in range(self._world_size)
                    )
                elif kind in (
                    ShardingType.TABLE_WISE.value,
                    ShardingType.COLUMN_WISE.value,
                ):
                    for _ in option.shards:
                        rank_variables = self._add_binary_variables(self._world_size)
                        # Sum of shard placements equals the option selection.
                        self._add_constraint(
                            [(variable, 1) for variable in rank_variables]
                            + [(option_variable, -1)],
                            lower=0,
                            upper=0,
                        )
                        shard_mappings.append(dict(enumerate(rank_variables)))
                    if (
                        kind == ShardingType.COLUMN_WISE.value
                        and option.compute_kernel == "key_value"
                    ):
                        # Key-value column shards require distinct ranks.
                        for rank in range(self._world_size):
                            self._add_constraint(
                                [(mapping[rank], 1) for mapping in shard_mappings]
                                + [(option_variable, -1)],
                                lower=-np.inf,
                                upper=0,
                            )
                else:
                    host_variables = (
                        self._add_binary_variables(self._num_hosts)
                        if self._num_hosts > 1
                        else [option_variable]
                    )
                    if self._num_hosts > 1:
                        self._add_constraint(
                            [(variable, 1) for variable in host_variables]
                            + [(option_variable, -1)],
                            lower=0,
                            upper=0,
                        )
                        if option.dependency is not None:
                            if option.dependency not in self._dependency_host_variables:
                                group_variables = self._add_binary_variables(
                                    self._num_hosts
                                )
                                self._dependency_host_variables[option.dependency] = (
                                    group_variables
                                )
                                self._add_constraint(
                                    [(variable, 1) for variable in group_variables],
                                    lower=1,
                                    upper=1,
                                )
                            group_variables = self._dependency_host_variables[
                                option.dependency
                            ]
                            # A selected option must use its dependency group's host.
                            for host_variable, group_variable in zip(
                                host_variables, group_variables
                            ):
                                self._add_constraint(
                                    [(host_variable, 1), (group_variable, -1)],
                                    lower=-np.inf,
                                    upper=0,
                                )
                    if kind == ShardingType.TABLE_ROW_WISE.value:
                        if len(option.shards) != self._host_size:
                            raise ValueError(
                                f"{option.fqn}: table-row-wise shards must fit a host"
                            )
                        shard_mappings.extend(
                            {
                                host * self._host_size + local_rank: host_variable
                                for host, host_variable in enumerate(host_variables)
                            }
                            for local_rank in range(self._host_size)
                        )
                    else:
                        for _ in option.shards:
                            rank_variables = self._add_binary_variables(
                                self._world_size
                            )
                            for host, host_variable in enumerate(host_variables):
                                host_rank_variables = rank_variables[
                                    host * self._host_size : (host + 1)
                                    * self._host_size
                                ]
                                # Each shard has one rank within the selected host.
                                self._add_constraint(
                                    [(variable, 1) for variable in host_rank_variables]
                                    + [(host_variable, -1)],
                                    lower=0,
                                    upper=0,
                                )
                            shard_mappings.append(dict(enumerate(rank_variables)))
            self._add_constraint(
                [(option_variable, 1) for option_variable in option_indices],
                lower=1,
                upper=1,
            )

    def _add_rank_cost_constraints(self) -> None:
        """Aggregate existing shard storage and perf estimates by placement.

        Normalize resource costs by capacity and performance by a common scale.
        This changes neither the byte budgets nor the estimated execution costs;
        no resource bins or new performance estimates are introduced here.
        """
        capacities = [
            (device.storage.hbm, device.storage.ddr)
            for device in self._topology.devices
        ]
        rank_storage_terms: List[List[List[tuple[int, float]]]] = [
            [[] for _ in range(2)] for _ in range(self._world_size)
        ]
        rank_perf_terms: List[List[tuple[int, float]]] = [
            [] for _ in range(self._world_size)
        ]
        perf_scale = max(
            float(
                np.median(
                    [option.total_perf / self._world_size for option in self._options]
                )
            ),
            1e-12,
        )
        for option, shard_mappings in zip(self._options, self._shard_rank_variables):
            for shard, rank_variables in zip(option.shards, shard_mappings):
                assert shard.storage is not None and shard.perf is not None
                storage_costs = (shard.storage.hbm, shard.storage.ddr)
                scaled_perf_cost = shard.perf.total / perf_scale
                for rank, placement_variable in rank_variables.items():
                    for resource_index, cost in enumerate(storage_costs):
                        capacity_fraction = cost / max(
                            capacities[rank][resource_index], 1
                        )
                        rank_storage_terms[rank][resource_index].append(
                            (placement_variable, capacity_fraction)
                        )
                    rank_perf_terms[rank].append((placement_variable, scaled_perf_cost))
        for rank in range(self._world_size):
            for resource_index, capacity in enumerate(capacities[rank]):
                self._add_constraint(
                    rank_storage_terms[rank][resource_index],
                    lower=-np.inf,
                    upper=1 if capacity > 0 else 0,
                )
            self._add_constraint(
                rank_perf_terms[rank] + [(self._max_rank_cost_variable, -1)],
                lower=-np.inf,
                upper=0,
            )

    def decode_plan(self, solution: np.ndarray) -> List[ShardingOption]:
        """Recover selected options, assign ranks and check exact byte capacities."""
        binary_solution = solution[: self._max_rank_cost_variable]
        if not np.allclose(
            binary_solution, np.rint(binary_solution), atol=1e-5, rtol=0
        ):
            raise PlannerError("MILP returned a non-integral placement")
        plan: List[ShardingOption] = []
        used_storage = [Storage(hbm=0, ddr=0) for _ in range(self._world_size)]
        for option_indices in self._option_indices_by_table.values():
            selected = [index for index in option_indices if solution[index] > 0.5]
            if len(selected) != 1:
                raise PlannerError("MILP did not select exactly one option per table")
            option_index = selected[0]
            option = copy.deepcopy(self._options[option_index])
            for shard, rank_variables in zip(
                option.shards, self._shard_rank_variables[option_index]
            ):
                selected_ranks = [
                    rank
                    for rank, variable in rank_variables.items()
                    if solution[variable] > 0.5
                ]
                if len(selected_ranks) != 1:
                    raise PlannerError(
                        f"MILP did not assign one rank to a shard of {option.fqn}"
                    )
                shard.rank = selected_ranks[0]
                assert shard.storage is not None
                used_storage[shard.rank].hbm += shard.storage.hbm
                used_storage[shard.rank].ddr += shard.storage.ddr
            plan.append(option)
        for rank, (storage, device) in enumerate(
            zip(used_storage, self._topology.devices)
        ):
            if storage.hbm > device.storage.hbm or storage.ddr > device.storage.ddr:
                raise PlannerError(
                    f"MILP placement exceeds exact byte capacity on rank {rank}: "
                    f"required HBM={storage.hbm}, DDR={storage.ddr}; "
                    f"available HBM={device.storage.hbm}, DDR={device.storage.ddr}",
                    PlannerErrorType.PARTITION,
                )
        return plan


def _solve_placement(
    options: List[ShardingOption],
    topology: Topology,
    relative_gap: float,
) -> List[ShardingOption]:
    """Build the joint placement model, solve it and return a validated plan."""
    started = time.perf_counter()
    model = _PlacementModel(options, topology)
    result = milp(
        c=model.objective,
        integrality=model.integrality,
        bounds=model.bounds,
        constraints=model.constraints,
        options={"time_limit": 60.0, "mip_rel_gap": relative_gap},
    )
    if result.status == 2:
        raise PlannerError(
            "MILP found no feasible selection and placement under the per-rank "
            "HBM, DDR and sharding constraints",
            PlannerErrorType.STRICT_CONSTRAINTS,
        )
    solution = result.x
    if result.status == 1 and solution is None:
        raise PlannerError(
            "MILP reached the 60s solver limit without a feasible placement. "
            "This does not prove insufficient capacity; use TZREC_PLANNER=dp "
            "to try the DP planner."
        )
    if result.status not in (0, 1) or solution is None:
        raise PlannerError(f"MILP placement solver failed: {result.message}")
    plan = model.decode_plan(solution)
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
