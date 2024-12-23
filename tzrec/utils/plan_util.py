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
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner import EmbeddingShardingPlanner, shard_estimators
from torchrec.distributed.planner.proposers import UniformProposer
from torchrec.distributed.planner.shard_estimators import (
    _calculate_shard_io_sizes,
    _calculate_storage_specific_sizes,
    calculate_pipeline_io_cost,
)
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    Enumerator,
    Proposer,
    ShardingOption,
    Storage,
    Topology,
)
from torchrec.distributed.sharding_plan import (
    get_default_sharders as _get_default_sharders,
)
from torchrec.distributed.types import (
    ModuleSharder,
    PipelineType,
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


def _calculate_shard_storages(  # NOQA
    sharder: ModuleSharder[nn.Module],
    sharding_type: str,
    tensor: torch.Tensor,
    compute_device: str,
    compute_kernel: str,
    shard_sizes: List[List[int]],
    batch_sizes: List[int],
    world_size: int,
    local_world_size: int,
    input_lengths: List[float],
    num_poolings: List[float],
    caching_ratio: float,
    is_pooled: bool,
    input_data_type_size: float,
    output_data_type_size: float,
    pipeline_type: PipelineType = PipelineType.NONE,
    count_ephemeral_storage_cost: bool = False,
    is_inference: bool = False,
    multipass_prefetch_max_pass: Optional[int] = None,
) -> List[Storage]:
    """Calculates estimated storage sizes for each sharded tensor, comprised of input,
    output, tensor, gradient, and optimizer sizes.

    Args:
        sharder (ModuleSharder[nn.Module]): sharder for module that supports sharding.
        sharding_type (str): provided ShardingType value.
        tensor (torch.Tensor): tensor to be sharded.
        compute_device (str): compute device to be used.
        compute_kernel (str): compute kernel to be used.
        shard_sizes (List[List[int]]): list of dimensions of each sharded tensor.
        batch_sizes (List[int]): batch size for each input feature.
        world_size (int): total number of devices in topology.
        local_world_size (int): total number of devices in host group topology.
        input_lengths (List[float]): average input lengths synonymous with pooling
            factors.
        num_poolings (List[float]): average number of poolings per sample
            (typically 1.0).
        caching_ratio (float): ratio of HBM to DDR memory for UVM caching.
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`), False
            if unpooled/sequential (ie. `Embedding`).
        input_data_type_size (int): number of bytes of input data type.
        output_data_type_size (int): number of bytes of output data type.
        pipeline_type: PipelineType: pipeline type if for training.
        is_inference: bool, whether the model is for inference.

    Returns:
        List[Storage]: storage object for each device in topology.
    """  # NOQA
    input_sizes, output_sizes = _calculate_shard_io_sizes(
        sharding_type=sharding_type,
        batch_sizes=batch_sizes,
        world_size=world_size,
        local_world_size=local_world_size,
        input_lengths=input_lengths,
        emb_dim=tensor.shape[1],
        shard_sizes=shard_sizes,
        input_data_type_size=input_data_type_size,
        output_data_type_size=output_data_type_size,
        num_poolings=num_poolings,
        is_pooled=is_pooled,
    )

    tensor_storage = sharder.storage_usage(tensor, compute_device, compute_kernel)
    hbm_storage: int = tensor_storage.get("hbm", 0)
    ddr_storage: int = tensor_storage.get("ddr", 0)

    table_cached: bool = False
    if compute_kernel in {
        EmbeddingComputeKernel.FUSED_UVM_CACHING.value,
        EmbeddingComputeKernel.QUANT_UVM_CACHING.value,
        EmbeddingComputeKernel.KEY_VALUE.value,
    }:
        hbm_storage = round(ddr_storage * caching_ratio)
        table_cached = True
    if compute_kernel in {EmbeddingComputeKernel.KEY_VALUE.value}:
        ddr_storage = 0

    optimizer_class = getattr(tensor, "_optimizer_classes", [None])[0]

    hbm_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=hbm_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
        is_inference=is_inference,
        clf=caching_ratio if table_cached else None,
    )
    ddr_specific_sizes: List[int] = _calculate_storage_specific_sizes(
        storage=ddr_storage,
        shape=tensor.shape,
        shard_sizes=shard_sizes,
        sharding_type=sharding_type,
        optimizer_class=optimizer_class,
        is_inference=is_inference,
    )

    hbm_sizes: List[int] = [
        (
            hbm_specific_size
            + calculate_pipeline_io_cost(
                input_size=input_size,
                output_size=output_size,
                prefetch_size=input_size if table_cached else 0,
                pipeline_type=pipeline_type,
                multipass_prefetch_max_pass=multipass_prefetch_max_pass,
                count_ephemeral_storage_cost=count_ephemeral_storage_cost,
                is_inference=is_inference,
            )
            if compute_device == "cuda"
            else 0
        )
        for input_size, output_size, hbm_specific_size in zip(
            input_sizes,
            output_sizes,
            hbm_specific_sizes,
        )
    ]
    ddr_sizes: List[int] = [
        (
            input_size + output_size + ddr_specific_size
            if compute_device in {"cpu", "mtia"} and not is_inference
            else ddr_specific_size
        )
        for input_size, output_size, ddr_specific_size in zip(
            input_sizes,
            output_sizes,
            ddr_specific_sizes,
        )
    ]

    return [
        Storage(
            hbm=hbm_size,
            ddr=ddr_size,
        )
        for hbm_size, ddr_size in zip(hbm_sizes, ddr_sizes)
    ]


# temporarily fix optimizer storage of trec calculate_shard_storages
shard_estimators.calculate_shard_storages = _calculate_shard_storages
