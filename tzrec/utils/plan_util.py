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
from collections import OrderedDict
from typing import List, Optional, Union

import psutil
import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.proposers import UniformProposer
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    Proposer,
    ShardingOption,
    Topology,
)
from torchrec.distributed.sharding_plan import ModuleSharder
from torchrec.distributed.sharding_plan import (
    get_default_sharders as _get_default_sharders,
)


def _bytes_to_float_bin(num_bytes: Union[float, int], bin_size: float) -> float:
    return float(num_bytes) / bin_size


def create_planner(device: torch.device, batch_size: int) -> EmbeddingShardingPlanner:
    """Create EmbeddingShardingPlanner."""
    local_world_size = get_local_size()

    # build topo
    topo_kwargs = {}
    if torch.cuda.is_available():
        topo_kwargs["hbm_cap"] = torch.cuda.get_device_properties(device).total_memory

    ddr_cap_per_rank = int(float(psutil.virtual_memory().total) / local_world_size)
    topo_kwargs["ddr_cap"] = ddr_cap_per_rank
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

    # build planner
    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        storage_reservation=storage_reservation,
        proposer=[DynamicProgrammingProposer(), UniformProposer()],
        debug=True,
    )
    return planner


def get_default_sharders() -> List[ModuleSharder[nn.Module]]:
    """Get embedding module default sharder."""
    if torch.cuda.is_available():
        return _get_default_sharders()
    else:
        # ShardedEmbeddingCollection is not supported yet.
        sharders = []
        for sharder in _get_default_sharders():
            if "EmbeddingCollection" not in str(sharder):
                sharders.append(sharder)
        return sharders


class DynamicProgrammingProposer(Proposer):
    r"""Proposes sharding plans in dynamic programming fashion.

        The problem of the Embedding Sharding Plan can be framed as follows: Given
    :math:`M` tables and their corresponding :math:`N` Sharding Options, we need to
    select one sharding option for each table such that the total performance is
    minimized, while keeping the overall memory constraint :math:`K` in check. This can
    be abstracted into the following mathematical formulation:

    Given a matrix :math:`A` of dimensions :math:`(M, N)` and another matrix :math:`B`
    of the same dimensions, let the elements of matrix :math:`A` be denoted as
    :math:`a_{i,j}` and the elements of matrix :math:`B` as :math:`b_{i,j}`. We aim
    to find a set of column indices :math:`\{ j_0, j_1, \ldots, j_{M-1} \}` such that
    the following conditions are satisfied:

    1. :math:`\sum_{i=0}^{M-1} a_{i,j_i} \leq K`, where :math:`K` is a float.
    2. :math:`\sum_{i=0}^{M-1} b_{i,j_i}` is minimized.

    This problem can be tackled using dynamic programming. First, discretize :math:`K`
    into :math:`K_i`, and denote the discretization function as :math:`f`.

    Define the state :math:`dp[i][f(k)]` to represent the minimum value of :math:`B`
    when considering the first :math:`i` rows and the total sum of :math:`A` is equal to
    the discretized value :math:`k`.

    The state transition can then be represented as:

    .. math::
        dp[i][f(k)] = \min_{j=0}^{N-1} \left( dp[i-1][f(k - A[i][j])] + B[i][j] \right)

    Since :math:`K` is the sum allocated across all memory, simply satisfying that the
    total memory in the plan equals :math:`K` does not guarantee that the allocation
    will fit on all cards. Therefore, it is essential to maintain all the states of the
    last layer of :math:`dp`. This allows us to propose different plans under varying
    total memory constraints.

    Args:
        mem_bins_per_device (int): memory bins for dynamic programming precision.
    """

    def __init__(self, mem_bins_per_device: int = 100) -> None:
        self._inited: bool = False
        self._mem_bins_per_device: int = max(mem_bins_per_device, 1)
        self._sharding_options_by_fqn: OrderedDict[str, List[ShardingOption]] = (
            OrderedDict()
        )
        # list of proposals with different total_mem, a proposal is a list of
        # indices of sharding_options
        self._proposal_list: List[List[int]] = []
        self._current_proposal: int = -1
        self._plan_by_hbm = True
        if not torch.cuda.is_available():
            self._plan_by_hbm = False

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        """Load search space."""
        self._reset()
        # order the sharding_option by total_storage.hbm from low to high
        for sharding_option in sorted(
            search_space,
            key=lambda x: x.total_storage.hbm
            if self._plan_by_hbm
            else x.total_storage.ddr,
        ):
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

    def _reset(self) -> None:
        self._sharding_options_by_fqn = OrderedDict()
        self._proposal_list = []
        self._current_proposal = -1

    def propose(self) -> Optional[List[ShardingOption]]:
        """Propose a sharding plan."""
        if not self._inited:
            return [
                sharding_options[0]
                for sharding_options in self._sharding_options_by_fqn.values()
            ]
        elif self._current_proposal >= 0:
            proposal_index = self._proposal_list[self._current_proposal]
            return [
                self._sharding_options_by_fqn[fqn][index]
                for fqn, index in zip(
                    self._sharding_options_by_fqn.keys(), proposal_index
                )
            ]
        else:
            return None

    def feedback(
        self,
        partitionable: bool,
        plan: Optional[List[ShardingOption]] = None,
        perf_rating: Optional[float] = None,
        storage_constraint: Optional[Topology] = None,
    ) -> None:
        """Feedback last proposed plan."""
        if not self._inited:
            self._inited = True
            table_count = len(self._sharding_options_by_fqn)
            option_count = max([len(x) for x in self._sharding_options_by_fqn.values()])

            assert storage_constraint is not None
            # are we assuming the table will be evenly sharded on all devices?
            mem_total = sum(
                [
                    x.storage.hbm if self._plan_by_hbm else x.storage.ddr
                    for x in storage_constraint.devices
                ]
            )

            bin_count = self._mem_bins_per_device * len(storage_constraint.devices)
            bin_size = float(mem_total) / bin_count

            dp = [
                [(float("inf"), float("inf"))] * bin_count for _ in range(table_count)
            ]  # [table_id][mem_bin][perf, mem]

            backtrack = [
                [(-1, -1)] * bin_count for _ in range(table_count)
            ]  # [table_id][mem_bin][opt_id, prev_mem_bin]

            mem_by_fqn = [
                [float("inf") for _ in range(option_count)] for _ in range(table_count)
            ]  # memory constraint lookup table: [table_id][sharding_option_id]
            perf_by_fqn = [
                [float("inf") for _ in range(option_count)] for _ in range(table_count)
            ]  # performance metrics lookup table: [table_id][sharding_option_id]

            # populate mem and perf for each sharding option and table:
            # A[table_id][sharding_option_id]
            for table_id, sharding_options in enumerate(
                self._sharding_options_by_fqn.values()
            ):
                for opt_id, sharding_option in enumerate(sharding_options):
                    mem_by_fqn[table_id][opt_id] = _bytes_to_float_bin(
                        sharding_option.total_storage.hbm
                        if self._plan_by_hbm
                        else sharding_option.total_storage.ddr,
                        bin_size,
                    )
                    perf_by_fqn[table_id][opt_id] = sharding_option.total_perf

            table_0 = 0
            for opt_j in range(option_count):
                if mem_by_fqn[0][opt_j] < bin_count:
                    mem_i = int(mem_by_fqn[0][opt_j])
                    # options are ordered in increasing order of mem, we only want to
                    # consider a sharding option that has higher mem and better perf
                    # (the smaller the better)
                    if dp[table_0][mem_i][0] > perf_by_fqn[table_0][opt_j]:
                        dp[table_0][mem_i] = (
                            perf_by_fqn[table_0][opt_j],
                            mem_by_fqn[table_0][opt_j],
                        )
                        backtrack[table_0][mem_i] = (opt_j, -1)

            # dp: table_count x option_count x bin_count
            for table_i in range(1, table_count):
                for opt_j in range(option_count):
                    for mem in range(bin_count):
                        prev_perf, perv_mem = dp[table_i - 1][mem]
                        if prev_perf < float("inf"):
                            new_mem = perv_mem + mem_by_fqn[table_i][opt_j]
                            if new_mem < bin_count:
                                new_mem_i = int(new_mem)
                                new_perf = prev_perf + perf_by_fqn[table_i][opt_j]
                                if dp[table_i][new_mem_i][0] > new_perf:
                                    dp[table_i][new_mem_i] = (new_perf, new_mem)
                                    backtrack[table_i][new_mem_i] = (opt_j, mem)
            self._proposal_list = []
            # fill in all the proposals, starting from highest mem to lowest mem
            for c in range(bin_count - 1, -1, -1):
                cur_opt_idx, cur_mem_idx = backtrack[table_count - 1][c]
                if cur_opt_idx >= 0:
                    proposal_indices = [-1] * table_count
                    proposal_indices[table_count - 1] = cur_opt_idx
                    for i in range(table_count - 2, -1, -1):
                        proposal_indices[i], cur_mem_idx = backtrack[i][cur_mem_idx]
                    self._proposal_list.append(proposal_indices)
            if len(self._proposal_list) > 0:
                self._current_proposal = 0
        else:
            self._current_proposal += 1
            if self._current_proposal >= len(self._proposal_list):
                self._current_proposal = -1
