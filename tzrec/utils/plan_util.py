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


class DynamicProgrammingProposer(Proposer):
    r"""Proposes sharding plans in dynamic programming fashion.

    The problem of the Embedding Sharding Plan can be framed as follows: Given
    :math:`M` tables and their corresponding :math:`N` Sharding Options, we need to
    select one sharding option for each table such that the total performance is
    minimized, while keeping the overall HBM constraint :math:`K` in check. This can
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

    Since :math:`K` is the sum allocated across all HBM, simply satisfying that the
    total HBM in the plan equals :math:`K` does not guarantee that the allocation will
    fit on all cards. Therefore, it is essential to maintain all the states of the last
    layer of :math:`dp`. This allows us to propose different plans under varying total
    HBM constraints.

    Args:
        hbm_bins_per_device (int): hdm bins for dynamic programming precision.
    """

    def __init__(self, hbm_bins_per_device: int = 100) -> None:
        self._inited: bool = False
        self._hbm_bins_per_device: int = max(hbm_bins_per_device, 1)
        self._sharding_options_by_fqn: OrderedDict[str, List[ShardingOption]] = (
            OrderedDict()
        )
        self._proposal_indices: List[List[int]] = []
        self._current_proposal: int = -1

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        """Load search space."""
        self._reset()
        for sharding_option in sorted(search_space, key=lambda x: x.total_storage.hbm):
            fqn = sharding_option.fqn
            if fqn not in self._sharding_options_by_fqn:
                self._sharding_options_by_fqn[fqn] = []
            self._sharding_options_by_fqn[fqn].append(sharding_option)

    def _reset(self) -> None:
        self._sharding_options_by_fqn = OrderedDict()
        self._proposal_indices = []
        self._current_proposal = -1

    def propose(self) -> Optional[List[ShardingOption]]:
        """Propose a sharding plan."""
        if not self._inited:
            return [
                sharding_options[0]
                for sharding_options in self._sharding_options_by_fqn.values()
            ]
        elif self._current_proposal >= 0:
            proposal_index = self._proposal_indices[self._current_proposal]
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
            M = len(self._sharding_options_by_fqn)
            N = max([len(x) for x in self._sharding_options_by_fqn.values()])

            assert storage_constraint is not None
            hbm_total = sum([x.storage.hbm for x in storage_constraint.devices])
            K = self._hbm_bins_per_device * len(storage_constraint.devices)
            bin_size = float(hbm_total) / K

            dp = [[(float("inf"), float("inf"))] * K for _ in range(M)]
            backtrack = [[(-1, -1)] * K for _ in range(M)]

            hbm_by_fqn = [[float("inf") for _ in range(N)] for _ in range(M)]
            perf_by_fqn = [[float("inf") for _ in range(N)] for _ in range(M)]
            for m, sharding_options in enumerate(
                self._sharding_options_by_fqn.values()
            ):
                for n, sharding_option in enumerate(sharding_options):
                    hbm_by_fqn[m][n] = _bytes_to_float_bin(
                        sharding_option.total_storage.hbm, bin_size
                    )
                    perf_by_fqn[m][n] = sharding_option.total_perf

            for j in range(N):
                if hbm_by_fqn[0][j] < K:
                    hbm_i = int(hbm_by_fqn[0][j])
                    if dp[0][hbm_i][0] > perf_by_fqn[0][j]:
                        dp[0][hbm_i] = (perf_by_fqn[0][j], hbm_by_fqn[0][j])
                        backtrack[0][hbm_i] = (j, -1)

            for i in range(1, M):
                for j in range(N):
                    for c in range(K):
                        prev_perf, perv_hbm = dp[i - 1][c]
                        if prev_perf < float("inf"):
                            new_hbm = perv_hbm + hbm_by_fqn[i][j]
                            if new_hbm < K:
                                new_hbm_i = int(new_hbm)
                                new_perf = prev_perf + perf_by_fqn[i][j]
                                if dp[i][new_hbm_i][0] > new_perf:
                                    dp[i][new_hbm_i] = (new_perf, new_hbm)
                                    backtrack[i][new_hbm_i] = (j, c)

            self._proposal_indices = []
            for c in range(K - 1, -1, -1):
                cur_col_idx, cur_hbm_idx = backtrack[M - 1][c]
                if cur_col_idx >= 0:
                    column_indices = [-1] * M
                    column_indices[M - 1] = cur_col_idx
                    for i in range(M - 2, -1, -1):
                        column_indices[i], cur_hbm_idx = backtrack[i][cur_hbm_idx]
                    self._proposal_indices.append(column_indices)
            if len(self._proposal_indices) > 0:
                self._current_proposal = 0
        else:
            self._current_proposal += 1
            if self._current_proposal >= len(self._proposal_indices):
                self._current_proposal = -1
