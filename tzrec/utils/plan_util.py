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
from collections import OrderedDict
from queue import Queue
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import psutil
import torch
from torch import distributed as dist
from torch import nn
from torchrec.distributed.comm import get_local_size
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.constants import (
    BIGINT_DTYPE,
    KV_CACHING_RATIO,
    POOLING_FACTOR,
    UVM_CACHING_RATIO,
)
from torchrec.distributed.planner.enumerators import (
    GUARDED_COMPUTE_KERNELS,
    GUARDED_SHARDING_TYPES_FOR_FP_MODULES,
    EmbeddingComputeKernel,
    EmbeddingTower,
    EmbeddingTowerCollection,
    ParameterConstraints,
    Shard,
    ShardEstimator,
    _get_tower_index,
    calculate_shard_sizes_and_offsets,
    get_partition_by_type,
    sharder_name,
)
from torchrec.distributed.planner.enumerators import (
    EmbeddingEnumerator as _EmbeddingEnumerator,
)
from torchrec.distributed.planner.proposers import UniformProposer
from torchrec.distributed.planner.shard_estimators import (
    EmbeddingPerfEstimator,
    get_num_poolings,
)
from torchrec.distributed.planner.shard_estimators import (
    calculate_shard_storages as tzrec_calculate_shard_storages,
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
from torchrec.distributed.planner.utils import build_sharder_data_map
from torchrec.distributed.sharding_plan import (
    get_default_sharders as _get_default_sharders,
)
from torchrec.distributed.types import (
    BoundsCheckMode,
    CacheParams,
    KeyValueParams,
    ModuleSharder,
    PipelineType,
    ShardingType,
)
from torchrec.modules.embedding_configs import DATA_TYPE_NUM_BITS, DataType
from torchrec.modules.embedding_modules import (
    EmbeddingBagCollection,
    EmbeddingCollection,
)

from tzrec.protos import feature_pb2
from tzrec.utils import env_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.logging_util import logger


def _bytes_to_float_bin(num_bytes: Union[float, int], bin_size: float) -> float:
    return float(num_bytes) / bin_size


def create_planner(
    device: torch.device,
    batch_size: int,
    ckpt_plan_path: Optional[str] = None,
    global_constraints_cfg: Optional[feature_pb2.ParameterConstraints] = None,
    model: Optional[nn.Module] = None,
) -> EmbeddingShardingPlanner:
    """Create EmbeddingShardingPlanner."""
    local_world_size = get_local_size()
    is_rank_zero = int(os.environ.get("RANK", 0)) == 0

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
                        if fqn not in fqn_constraints:
                            if is_rank_zero:
                                logger.info(
                                    f"add ParameterConstraints[sharding_types=['{param_sharding['sharding_type']}']] for param[{fqn}] from checkpoint plan."  # NOQA
                                )
                            fqn_constraints[fqn] = ParameterConstraints(
                                sharding_types=[param_sharding["sharding_type"]]
                            )

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

    # build planner
    planner = EmbeddingShardingPlanner(
        topology=topology,
        batch_size=batch_size,
        enumerator=EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            fqn_constraints=fqn_constraints,
            global_constraints=global_constraints,
        ),
        storage_reservation=storage_reservation,
        proposer=[DynamicProgrammingProposer(), UniformProposer()],
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


_INF = float("inf")
# Cap per-axis bin count for the 2D DP. Without this, `bins_per_device *
# num_devices` blows up on multi-host worlds (e.g. 16 GPUs at 100 hbm
# bins/dev = 1600 bins, * 800 ddr bins = 1.28M cells per table).
_DP_AXIS_BIN_CAP = 1024


class DynamicProgrammingProposer(Proposer):
    r"""Proposes sharding plans via 2D (HBM × DDR) dynamic programming.

    Given :math:`M` tables each with up to :math:`N` ShardingOptions, pick one
    option per table to minimize total perf while respecting both per-rank
    HBM and per-machine DDR budgets from the topology. (HBM is GPU-local, so
    each device gets its own quota; DDR is host-shared across ranks
    co-located on the same machine, so the prune threshold is the sum over
    a ``local_world_size``-sized window.)

    Each axis (HBM, DDR) is discretized into bins; ``dp[table][h][d]`` holds
    the minimum perf over the first ``table`` tables using ``≈ h`` HBM bins
    and ``≈ d`` DDR bins. The transition is:

    .. math::
        dp[i][h][d] = \min_{j} dp[i-1][h - A_h[i][j]][d - A_d[i][j]]
                              + B[i][j]

    Backtracking emits one proposal per HBM bin -- the perf-best plan
    across all DDR bins at that HBM level -- in decreasing HBM order.
    Plans at the same HBM bin with worse perf are strictly dominated
    (same HBM cost, higher perf), so they are skipped. The host axis is
    load-bearing for dynamicemb CACHING mode (where DDR = full table) vs
    HYBRID (DDR = ``(1 - load_factor) · table``).

    Args:
        hbm_bins_per_device: HBM discretization bins per device. Default 100.
        ddr_bins_per_device: DDR discretization bins per device. Default 50 —
            DDR budgets dominate embedding demand, so a coarser axis suffices.
    """

    def __init__(
        self,
        hbm_bins_per_device: int = 100,
        ddr_bins_per_device: int = 50,
    ) -> None:
        self._inited: bool = False
        self._hbm_bins_per_device: int = max(hbm_bins_per_device, 1)
        self._ddr_bins_per_device: int = max(ddr_bins_per_device, 1)
        self._sharding_options_by_fqn: OrderedDict[str, List[ShardingOption]] = (
            OrderedDict()
        )
        # list of proposals with different total_mem; each proposal is a list
        # of indices into self._sharding_options_by_fqn[fqn].
        self._proposal_list: List[List[int]] = []
        self._current_proposal: int = -1

    def load(
        self,
        search_space: List[ShardingOption],
        enumerator: Optional[Enumerator] = None,
    ) -> None:
        """Load search space, sorted by total (hbm + ddr) ascending."""
        self._reset()
        for sharding_option in sorted(
            search_space,
            key=lambda x: (x.total_storage.hbm or 0) + (x.total_storage.ddr or 0),
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
        """Run 2D DP on first feedback; otherwise advance the proposal cursor."""
        if self._inited:
            self._current_proposal += 1
            if self._current_proposal >= len(self._proposal_list):
                self._current_proposal = -1
            return

        self._inited = True
        assert storage_constraint is not None
        table_count = len(self._sharding_options_by_fqn)
        if table_count == 0:
            return
        option_count = max(len(x) for x in self._sharding_options_by_fqn.values())

        num_devices = len(storage_constraint.devices)
        max_device_hbm = 0
        hbm_total = 0
        ddr_total = 0
        for device in storage_constraint.devices:
            max_device_hbm = max(max_device_hbm, device.storage.hbm or 0)
            hbm_total += device.storage.hbm or 0
            ddr_total += device.storage.ddr or 0
        # DDR is host-shared across ranks co-located on one machine, so the
        # per-option fit check compares against the largest machine's DDR pool
        # -- not per-device. HBM is GPU-local, so its prune stays per-device.
        per_host = getattr(storage_constraint, "local_world_size", None) or num_devices
        per_host = max(per_host, 1)
        max_machine_ddr = 0
        for host_start in range(0, num_devices, per_host):
            host_end = min(host_start + per_host, num_devices)
            machine_ddr = sum(
                (storage_constraint.devices[i].storage.ddr or 0)
                for i in range(host_start, host_end)
            )
            max_machine_ddr = max(max_machine_ddr, machine_ddr)

        hbm_bins = max(
            min(self._hbm_bins_per_device * num_devices, _DP_AXIS_BIN_CAP), 1
        )
        ddr_bins = max(
            min(self._ddr_bins_per_device * num_devices, _DP_AXIS_BIN_CAP), 1
        )
        # Collapse a degenerate axis to a single bin so we don't waste states
        # on (e.g.) CPU-only topologies that have hbm == 0 everywhere.
        if hbm_total == 0:
            hbm_bins = 1
        if ddr_total == 0:
            ddr_bins = 1
        hbm_bin_size = float(hbm_total) / hbm_bins if hbm_bins > 0 else 1.0
        ddr_bin_size = float(ddr_total) / ddr_bins if ddr_bins > 0 else 1.0

        hbm_by_fqn = [[_INF] * option_count for _ in range(table_count)]
        ddr_by_fqn = [[_INF] * option_count for _ in range(table_count)]
        perf_by_fqn = [[_INF] * option_count for _ in range(table_count)]

        for table_id, sharding_options in enumerate(
            self._sharding_options_by_fqn.values()
        ):
            for opt_id, sharding_option in enumerate(sharding_options):
                max_shard_hbm = max(
                    (shard.storage.hbm or 0) for shard in sharding_option.shards
                )
                max_shard_ddr = max(
                    (shard.storage.ddr or 0) for shard in sharding_option.shards
                )
                # HBM is per-device, DDR is per-machine: see comment above.
                if hbm_total > 0 and max_shard_hbm > max_device_hbm:
                    continue
                if ddr_total > 0 and max_shard_ddr > max_machine_ddr:
                    continue
                hbm_by_fqn[table_id][opt_id] = (
                    _bytes_to_float_bin(
                        sharding_option.total_storage.hbm or 0, hbm_bin_size
                    )
                    if hbm_total > 0
                    else 0.0
                )
                ddr_by_fqn[table_id][opt_id] = (
                    _bytes_to_float_bin(
                        sharding_option.total_storage.ddr or 0, ddr_bin_size
                    )
                    if ddr_total > 0
                    else 0.0
                )
                perf_by_fqn[table_id][opt_id] = sharding_option.total_perf

        # dp[table][hbm_bin][ddr_bin] = (perf, hbm_actual, ddr_actual)
        # backtrack[table][hbm_bin][ddr_bin] = (opt_id, prev_hbm_bin, prev_ddr_bin)
        empty_state = (_INF, _INF, _INF)
        empty_back = (-1, -1, -1)
        dp = [
            [[empty_state] * ddr_bins for _ in range(hbm_bins)]
            for _ in range(table_count)
        ]
        backtrack = [
            [[empty_back] * ddr_bins for _ in range(hbm_bins)]
            for _ in range(table_count)
        ]

        # Seed the first table.
        for opt_j in range(option_count):
            h = hbm_by_fqn[0][opt_j]
            d = ddr_by_fqn[0][opt_j]
            if h >= hbm_bins or d >= ddr_bins:
                continue
            h_i = int(h)
            d_i = int(d)
            if dp[0][h_i][d_i][0] > perf_by_fqn[0][opt_j]:
                dp[0][h_i][d_i] = (perf_by_fqn[0][opt_j], h, d)
                backtrack[0][h_i][d_i] = (opt_j, -1, -1)

        for table_i in range(1, table_count):
            prev_dp = dp[table_i - 1]
            cur_dp = dp[table_i]
            cur_back = backtrack[table_i]
            hbm_t = hbm_by_fqn[table_i]
            ddr_t = ddr_by_fqn[table_i]
            perf_t = perf_by_fqn[table_i]
            for h in range(hbm_bins):
                prev_dp_h = prev_dp[h]
                for d in range(ddr_bins):
                    prev_state = prev_dp_h[d]
                    prev_perf = prev_state[0]
                    if prev_perf == _INF:
                        continue
                    prev_h_val = prev_state[1]
                    prev_d_val = prev_state[2]
                    for opt_j in range(option_count):
                        delta_perf = perf_t[opt_j]
                        if delta_perf == _INF:
                            continue
                        new_h = prev_h_val + hbm_t[opt_j]
                        new_d = prev_d_val + ddr_t[opt_j]
                        if new_h >= hbm_bins or new_d >= ddr_bins:
                            continue
                        new_h_i = int(new_h)
                        new_d_i = int(new_d)
                        new_perf = prev_perf + delta_perf
                        if cur_dp[new_h_i][new_d_i][0] > new_perf:
                            cur_dp[new_h_i][new_d_i] = (new_perf, new_h, new_d)
                            cur_back[new_h_i][new_d_i] = (opt_j, h, d)

        # For each HBM bin, emit only the lowest-perf DDR cell. Any other
        # (h, d') at the same h has the same HBM cost and worse-or-equal
        # perf -- strictly dominated, no point proposing. Caps the proposal
        # count at hbm_bins (<= _DP_AXIS_BIN_CAP), down from
        # hbm_bins x ddr_bins.
        self._proposal_list = []
        last_dp = dp[table_count - 1]
        last_back = backtrack[table_count - 1]
        for h in range(hbm_bins - 1, -1, -1):  # high HBM first
            best_perf = _INF
            best_d = -1
            for d in range(ddr_bins):
                if last_back[h][d][0] >= 0 and last_dp[h][d][0] < best_perf:
                    best_perf = last_dp[h][d][0]
                    best_d = d
            if best_d < 0:
                continue
            cur_opt_idx, prev_h, prev_d = last_back[h][best_d]
            proposal_indices = [-1] * table_count
            proposal_indices[table_count - 1] = cur_opt_idx
            for i in range(table_count - 2, -1, -1):
                proposal_indices[i], prev_h, prev_d = backtrack[i][prev_h][prev_d]
            self._proposal_list.append(proposal_indices)
        if self._proposal_list:
            self._current_proposal = 0


def _extract_constraints_for_param(
    constraints: Optional[Dict[str, ParameterConstraints]], name: str
) -> Tuple[
    List[float],
    Optional[int],
    Optional[CacheParams],
    Optional[bool],
    Optional[bool],
    Optional[BoundsCheckMode],
    Optional[List[str]],
    Optional[DataType],
    Optional[str],
    Optional[KeyValueParams],
    bool,
    Any,
]:
    input_lengths = [POOLING_FACTOR]
    col_wise_shard_dim = None
    cache_params = None
    enforce_hbm = None
    stochastic_rounding = None
    bounds_check_mode = None
    feature_names = None
    output_dtype = None
    device_group = None
    key_value_params = None
    use_dynamicemb = False
    dynamicemb_options = None

    if constraints and constraints.get(name):
        input_lengths = constraints[name].pooling_factors
        col_wise_shard_dim = constraints[name].min_partition
        cache_params = constraints[name].cache_params
        enforce_hbm = constraints[name].enforce_hbm
        stochastic_rounding = constraints[name].stochastic_rounding
        bounds_check_mode = constraints[name].bounds_check_mode
        feature_names = constraints[name].feature_names
        output_dtype = constraints[name].output_dtype
        device_group = constraints[name].device_group
        key_value_params = constraints[name].key_value_params
        if hasattr(constraints[name], "use_dynamicemb"):
            # pyre-ignore [16]
            use_dynamicemb = constraints[name].use_dynamicemb
        if hasattr(constraints[name], "dynamicemb_options"):
            # pyre-ignore [16]
            dynamicemb_options = constraints[name].dynamicemb_options

    return (
        input_lengths,
        col_wise_shard_dim,
        cache_params,
        enforce_hbm,
        stochastic_rounding,
        bounds_check_mode,
        feature_names,
        output_dtype,
        device_group,
        key_value_params,
        use_dynamicemb,
        dynamicemb_options,
    )


class EmbeddingStorageEstimator(ShardEstimator):
    """Embedding Storage Usage Estimator.

    Args:
        topology (Topology): device topology.
        constraints (Optional[Dict[str, ParameterConstraints]]): parameter constraints.
        pipeline_type (PipelineType): The type of pipeline, if any. Will determine the
            input replication factor during memory estimation.
        run_embedding_at_peak_memory (bool): If the embedding fwd/bwd will be execute
            when HBM usage is at peak. When set to TRUE, any temporary memory allocation
            during embedding forward/backward, as long as output sizes before
            output_dist will be counted towards HBM storage cost. Otherwise they won't
            since they'll be "hidden" by the real memory peak.

            Only take effect if pipeline_type is set for backward compatibility (not
            affecting models using old pipeline-agnostic formula)

            Default to false because this is typically false for RecSys since memory
            peak happens at the end of dense forwrad / beginning of dense backward
            instead.
        is_inference (bool): If the model is inference model. Default to False.
    """

    def __init__(
        self,
        topology: Topology,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        pipeline_type: PipelineType = PipelineType.NONE,
        run_embedding_at_peak_memory: bool = False,
        is_inference: bool = False,
    ) -> None:
        self._topology = topology
        self._constraints = constraints
        self._pipeline_type = pipeline_type
        self._run_embedding_at_peak_memory = run_embedding_at_peak_memory
        self._is_inference = is_inference

    def estimate(
        self,
        sharding_options: List[ShardingOption],
        sharder_map: Optional[Dict[str, ModuleSharder[nn.Module]]] = None,
    ) -> None:
        """Estimate the storage cost of each sharding option.

        Args:
            sharding_options (List[ShardingOption]): list of sharding options.
            sharder_map (Optional[Dict[str, ModuleSharder[nn.Module]]]): map from module
                type to sharder.
        """
        if not sharder_map:
            assert not sharding_options, "sharder_map not provided for sharding_options"
            return

        for sharding_option in sharding_options:
            sharder_key = sharding_option.module_type_key
            sharder = sharder_map[sharder_key]

            caching_ratio = sharding_option.cache_load_factor
            # TODO: remove after deprecating fused_params in sharder
            if caching_ratio is None:
                caching_ratio = (
                    sharder.fused_params.get("cache_load_factor")  # pyre-ignore[16]
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )
            constraints: Optional[ParameterConstraints] = (
                self._constraints.get(sharding_option.name, None)
                if self._constraints
                else None
            )
            num_poolings = get_num_poolings(self._constraints, sharding_option)
            assert len(num_poolings) == sharding_option.num_inputs
            batch_sizes = (
                constraints.batch_sizes
                if constraints and constraints.batch_sizes
                else [sharding_option.batch_size] * sharding_option.num_inputs
            )

            key_value_params: Optional[KeyValueParams] = (
                constraints.key_value_params
                if constraints and constraints.key_value_params
                else None
            )
            kv_cache_load_factor: float = (
                sharder.fused_params.get("cache_load_factor", KV_CACHING_RATIO)
                if sharder.fused_params
                else KV_CACHING_RATIO
            )
            use_virtual_table: bool = (
                constraints.use_virtual_table if constraints else False
            )

            # hardcoded as 8 bytes
            # input indices can be of int32,
            # but in TBE they get converted to int64 anyway
            input_data_type_size = BIGINT_DTYPE

            output_data_type_size: float = (
                DATA_TYPE_NUM_BITS[sharding_option.output_dtype] / 8
                if sharding_option.output_dtype
                else sharding_option.tensor.element_size()
            )

            mpp_conf = (
                sharding_option.cache_params.multipass_prefetch_config
                if sharding_option.cache_params
                else None
            )
            # TODO: remove after deprecating fused_params in sharder
            if mpp_conf is None:
                mpp_conf = (
                    sharder.fused_params.get("multipass_prefetch_config", None)
                    if hasattr(sharder, "fused_params") and sharder.fused_params
                    else None
                )

            use_dynamicemb = (
                hasattr(sharding_option, "use_dynamicemb")
                and sharding_option.use_dynamicemb
            )
            dynamicemb_options = None
            if use_dynamicemb:
                dynamicemb_options = sharding_option.dynamicemb_options

            shard_storages = calculate_shard_storages(
                sharder=sharder,
                sharding_type=sharding_option.sharding_type,
                tensor=sharding_option.tensor,
                compute_device=self._topology.compute_device,
                compute_kernel=sharding_option.compute_kernel,
                shard_sizes=[shard.size for shard in sharding_option.shards],
                batch_sizes=batch_sizes,
                world_size=self._topology.world_size,
                local_world_size=self._topology.local_world_size,
                input_lengths=sharding_option.input_lengths,
                num_poolings=num_poolings,
                caching_ratio=caching_ratio if caching_ratio else UVM_CACHING_RATIO,
                is_pooled=sharding_option.is_pooled,
                input_data_type_size=input_data_type_size,
                output_data_type_size=output_data_type_size,
                pipeline_type=self._pipeline_type,
                count_ephemeral_storage_cost=self._run_embedding_at_peak_memory,
                is_inference=self._is_inference,
                multipass_prefetch_max_pass=mpp_conf.num_passes if mpp_conf else None,
                key_value_params=key_value_params,
                kv_cache_load_factor=kv_cache_load_factor,
                use_virtual_table=use_virtual_table,
                dynamicemb_options=dynamicemb_options,
            )
            for shard, storage in zip(sharding_option.shards, shard_storages):
                shard.storage = storage


def calculate_shard_storages(
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
    key_value_params: Optional[KeyValueParams] = None,
    kv_cache_load_factor: float = KV_CACHING_RATIO,
    use_virtual_table: bool = False,
    dynamicemb_options=None,
) -> List[Storage]:
    """Calculates estimated storage sizes.

    for each sharded tensor, comprised of input, output, tensor, gradient,
    and optimizer sizes.

    Args:
        sharder (ModuleSharder[nn.Module]): sharder for module that supports
            sharding.
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
        is_pooled (bool): True if embedding output is pooled (ie. `EmbeddingBag`),
            False if unpooled/sequential (ie. `Embedding`).
        input_data_type_size (int): number of bytes of input data type.
        output_data_type_size (int): number of bytes of output data type.
        pipeline_type: PipelineType: pipeline type if for training.
        count_ephemeral_storage_cost (bool): count ephemeral storage cost
        is_inference: bool, whether the model is for inference.
        multipass_prefetch_max_pass (int): multipass prefetch max_pass
        key_value_params (Optional[KeyValueParams]): fused params for SSD/DRAM KV
            cache.
        kv_cache_load_factor (float): ratio of kv caching.
        use_virtual_table (bool): use virtual table or not.
        dynamicemb_options (DynamicEmbTableOptions): dynamice embedding options
    Returns:
        List[Storage]: storage object for each device in topology.
    """
    if dynamicemb_options is not None:
        from tzrec.utils.dynamicemb_util import dynamicemb_calculate_shard_storages

        return dynamicemb_calculate_shard_storages(
            sharder=sharder,
            sharding_type=sharding_type,
            tensor=tensor,
            compute_device=compute_device,
            compute_kernel=compute_kernel,
            shard_sizes=shard_sizes,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            num_poolings=num_poolings,
            caching_ratio=caching_ratio,
            is_pooled=is_pooled,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            pipeline_type=pipeline_type,
            count_ephemeral_storage_cost=count_ephemeral_storage_cost,
            is_inference=is_inference,
            multipass_prefetch_max_pass=multipass_prefetch_max_pass,
            dynamicemb_options=dynamicemb_options,
        )
    else:
        return tzrec_calculate_shard_storages(
            sharder=sharder,
            sharding_type=sharding_type,
            tensor=tensor,
            compute_device=compute_device,
            compute_kernel=compute_kernel,
            shard_sizes=shard_sizes,
            batch_sizes=batch_sizes,
            world_size=world_size,
            local_world_size=local_world_size,
            input_lengths=input_lengths,
            num_poolings=num_poolings,
            caching_ratio=caching_ratio,
            is_pooled=is_pooled,
            input_data_type_size=input_data_type_size,
            output_data_type_size=output_data_type_size,
            pipeline_type=pipeline_type,
            count_ephemeral_storage_cost=count_ephemeral_storage_cost,
            is_inference=is_inference,
            multipass_prefetch_max_pass=multipass_prefetch_max_pass,
            key_value_params=key_value_params,
            kv_cache_load_factor=kv_cache_load_factor,
            use_virtual_table=use_virtual_table,
        )


def _emit_dynamicemb_variants(
    base_option: ShardingOption,
) -> List[ShardingOption]:
    """Expand a dynamicemb ShardingOption into HYBRID + CACHING variants.

    Sweeps both placement modes (``caching=False`` and ``caching=True``) and,
    when ``base_option.cache_params`` is unset, ten cache_load_factor values
    (0.1, 0.2, ..., 1.0). The downstream 2D DP proposer picks per table the
    best (mode, ratio) that fits both HBM and host topology budgets.

    ``base_option.dynamicemb_options`` must already be attached by the
    caller; each returned ShardingOption owns a freshly deep-copied
    ``dynamicemb_options`` instance so per-option ``caching`` mutations do
    not bleed across variants.
    """
    if base_option.cache_params is None:
        load_factors = [(i + 1) / 10 for i in range(10)]
        stats = None
    else:
        load_factors = [base_option.cache_params.load_factor]
        stats = base_option.cache_params.stats
    variants: List[ShardingOption] = []
    for caching_mode in (False, True):
        for load_factor in load_factors:
            opt = copy.deepcopy(base_option)
            opt.cache_params = CacheParams(load_factor=load_factor, stats=stats)
            # deepcopy(base_option) already produced a fresh dynamicemb_options.
            opt.dynamicemb_options.caching = caching_mode  # pyre-ignore [16]
            variants.append(opt)
    return variants


class EmbeddingEnumerator(_EmbeddingEnumerator):
    """Generates embedding sharding options for given `nn.Module` with constraints.

    Args:
        topology (Topology): device topology.
        batch_size (int): batch size.
        constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter names
            to provided ParameterConstraints.
        estimator (Optional[Union[ShardEstimator, List[ShardEstimator]]]): shard
            performance estimators.
        use_exact_enumerate_order (bool): whether to enumerate shardable parameters in
            the exact name_children enumeration order
        fqn_constraints (Optional[Dict[str, ParameterConstraints]]): dict of parameter
            fqns to provided ParameterConstraints.
        global_constraints (Optional[ParameterConstraints]): all parameters provided
            ParameterConstraints.
    """

    def __init__(
        self,
        topology: Topology,
        batch_size: int,
        constraints: Optional[Dict[str, ParameterConstraints]] = None,
        estimator: Optional[Union[ShardEstimator, List[ShardEstimator]]] = None,
        use_exact_enumerate_order: Optional[bool] = False,
        fqn_constraints: Optional[Dict[str, ParameterConstraints]] = None,
        global_constraints: Optional[ParameterConstraints] = None,
    ) -> None:
        if not estimator:
            estimator: List[ShardEstimator] = [
                EmbeddingPerfEstimator(topology=topology, constraints=constraints),
                EmbeddingStorageEstimator(topology=topology, constraints=constraints),
            ]
        super().__init__(
            topology, batch_size, constraints, estimator, use_exact_enumerate_order
        )
        self._fqn_constraints = fqn_constraints
        self._global_constraints = global_constraints

    def _get_constraints(
        self, child_path: str, name: str
    ) -> Tuple[Optional[Dict[str, ParameterConstraints]], str]:
        if self._fqn_constraints is not None:
            constraint_key = child_path + "." + name
            # pyre-ignore [58]
            if constraint_key in self._fqn_constraints:
                return self._fqn_constraints, constraint_key
        if self._global_constraints is not None:
            return {"__GLOBAL__": self._global_constraints}, "__GLOBAL__"
        return self._constraints, name

    def enumerate(
        self,
        module: nn.Module,
        sharders: List[ModuleSharder[nn.Module]],
    ) -> List[ShardingOption]:
        """Generates relevant sharding options given module and sharders.

        Args:
            module (nn.Module): module to be sharded.
            sharders (List[ModuleSharder[nn.Module]]): provided sharders for module.

        Returns:
            List[ShardingOption]: valid sharding options with values populated.
        """
        self._sharder_map = {
            sharder_name(sharder.module_type): sharder for sharder in sharders
        }
        self._sharder_data_map = build_sharder_data_map(self._sharder_map)
        sharding_options: List[ShardingOption] = []

        named_modules_queue = [("", module)]
        while named_modules_queue:
            if not self._use_exact_enumerate_order:
                child_path, child_module = named_modules_queue.pop()
            else:
                child_path, child_module = named_modules_queue.pop(0)
            sharder_key = sharder_name(type(child_module))
            sharder = self._sharder_map.get(sharder_key, None)
            if not sharder:
                for n, m in child_module.named_children():
                    if child_path != "":
                        named_modules_queue.append((child_path + "." + n, m))
                    else:
                        named_modules_queue.append((n, m))
                continue

            # Determine the pooling state for all sharding_options using this
            # (child_module, child_path). With this optimization, we change enumerate()
            # from being O(N^2) with respect to the number of tables to O(N). The
            # previous quadratic behavior is because in populate_estimates() invoked
            # below, each sharding_option needs to determine its pooling state, which
            # is does via an expensive O(N) walk through the list of embedding tables.
            # With this change sharding_option.is_pooled becomes O(1).
            is_pooled = ShardingOption.module_pooled(child_module, child_path)

            for name, param in sharder.shardable_parameters(child_module).items():
                _constraints, key = self._get_constraints(child_path, name)
                (
                    input_lengths,
                    col_wise_shard_dim,
                    cache_params,
                    enforce_hbm,
                    stochastic_rounding,
                    bounds_check_mode,
                    feature_names,
                    output_dtype,
                    device_group,
                    key_value_params,
                    use_dynamicemb,
                    dynamicemb_options,
                ) = _extract_constraints_for_param(_constraints, key)

                # skip for other device groups
                if device_group and device_group != self._compute_device:
                    continue

                num_buckets = self._get_num_buckets(name, child_module)
                sharding_options_per_table: List[ShardingOption] = []

                for sharding_type in self._filter_sharding_types(
                    name,
                    sharder.sharding_types(self._compute_device),
                    child_path,
                    sharder_key=sharder_key,
                ):
                    for compute_kernel in self._filter_compute_kernels(
                        name,
                        sharder.compute_kernels(sharding_type, self._compute_device),
                        sharding_type,
                        child_path,
                    ):
                        (
                            shard_sizes,
                            shard_offsets,
                        ) = calculate_shard_sizes_and_offsets(
                            tensor=param,
                            world_size=self._world_size,
                            local_world_size=self._local_world_size,
                            sharding_type=sharding_type,
                            col_wise_shard_dim=col_wise_shard_dim,
                            device_memory_sizes=self._device_memory_sizes,
                            num_buckets=num_buckets,
                        )
                        dependency = None
                        if isinstance(child_module, EmbeddingTower):
                            dependency = child_path
                        elif isinstance(child_module, EmbeddingTowerCollection):
                            tower_index = _get_tower_index(name, child_module)
                            dependency = child_path + ".tower_" + str(tower_index)
                        sharding_option = ShardingOption(
                            name=name,
                            tensor=param,
                            module=(child_path, child_module),
                            input_lengths=input_lengths,
                            batch_size=self._batch_size,
                            compute_kernel=compute_kernel,
                            sharding_type=sharding_type,
                            partition_by=get_partition_by_type(sharding_type),
                            shards=[
                                Shard(size=size, offset=offset)
                                for size, offset in zip(shard_sizes, shard_offsets)
                            ],
                            cache_params=cache_params,
                            enforce_hbm=enforce_hbm,
                            stochastic_rounding=stochastic_rounding,
                            bounds_check_mode=bounds_check_mode,
                            dependency=dependency,
                            is_pooled=is_pooled,
                            feature_names=feature_names,
                            output_dtype=output_dtype,
                            key_value_params=key_value_params,
                        )
                        # hack sharding option for dynamicemb
                        if use_dynamicemb:
                            # pyre-ignore [16]
                            sharding_option.use_dynamicemb = use_dynamicemb
                            # pyre-ignore [16]
                            sharding_option.dynamicemb_options = dynamicemb_options
                            sharding_options_per_table.extend(
                                _emit_dynamicemb_variants(sharding_option)
                            )
                        else:
                            sharding_options_per_table.append(sharding_option)

                if not sharding_options_per_table:
                    raise RuntimeError(
                        "No available sharding type and compute kernel combination "
                        f"after applying user provided constraints for {name}. "
                        f"Module: {sharder_key}, sharder: {sharder.__class__.__name__},"
                        f" compute device: {self._compute_device}. "
                        "To debug, search above for warning logs about no available "
                        f"sharding types/compute kernels for table: {name}"
                    )

                sharding_options.extend(sharding_options_per_table)

        self.populate_estimates(sharding_options)

        return sharding_options

    def _get_num_buckets(self, parameter: str, module: nn.Module) -> Optional[int]:
        """Get total_num_buckets for virtual-table sharding, or None."""
        if isinstance(module, EmbeddingBagCollection):
            embedding_configs = module.embedding_bag_configs()
        elif isinstance(module, EmbeddingCollection):
            embedding_configs = module.embedding_configs()
        else:
            return None
        for config in embedding_configs:
            if config.name == parameter and getattr(config, "use_virtual_table", False):
                return getattr(config, "total_num_buckets", None)
        return None

    # pyre-ignore [14]
    def _filter_sharding_types(
        self,
        name: str,
        allowed_sharding_types: List[str],
        child_path: str,
        sharder_key: str = "",
    ) -> List[str]:
        _constraints, key = self._get_constraints(child_path, name)
        # GRID_SHARD and row-wise on FP modules require explicit opt-in.
        is_fp_module = "FeatureProcessedEmbeddingBagCollection" in sharder_key
        if not _constraints or not _constraints.get(key):
            filtered = [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
            if is_fp_module:
                filtered = [
                    t
                    for t in filtered
                    if t not in GUARDED_SHARDING_TYPES_FOR_FP_MODULES
                ]
            return filtered
        constraints: ParameterConstraints = _constraints[key]
        if not constraints.sharding_types:
            filtered = [
                t for t in allowed_sharding_types if t != ShardingType.GRID_SHARD.value
            ]
            if is_fp_module:
                filtered = [
                    t
                    for t in filtered
                    if t not in GUARDED_SHARDING_TYPES_FOR_FP_MODULES
                ]
            return filtered
        constrained_sharding_types: List[str] = constraints.sharding_types

        filtered_sharding_types = list(
            set(constrained_sharding_types) & set(allowed_sharding_types)
        )
        if not filtered_sharding_types:
            logger.warn(
                "No available sharding types after applying user provided "
                f"constraints for {key}. Constrained sharding types: "
                f"{constrained_sharding_types}, allowed sharding types: "
                f"{allowed_sharding_types}, filtered sharding types: "
                f"{filtered_sharding_types}. Please check if the constrained "
                "sharding types are too restrictive, if the sharder allows the "
                "sharding types, or if non-strings are passed in."
            )
        return filtered_sharding_types

    # pyre-ignore [14]
    def _filter_compute_kernels(
        self,
        name: str,
        allowed_compute_kernels: List[str],
        sharding_type: str,
        child_path: str,
    ) -> List[str]:
        _constraints, key = self._get_constraints(child_path, name)
        # setup constrained_compute_kernels
        if _constraints and _constraints.get(key) and _constraints[key].compute_kernels:
            # pyre-ignore
            constrained_compute_kernels: List[str] = _constraints[key].compute_kernels
        else:
            constrained_compute_kernels: List[str] = [
                compute_kernel.value
                for compute_kernel in EmbeddingComputeKernel
                if compute_kernel not in GUARDED_COMPUTE_KERNELS
            ]

        # setup filtered_compute_kernels
        filtered_compute_kernels = list(
            set(constrained_compute_kernels) & set(allowed_compute_kernels)
        )

        # special rules
        if EmbeddingComputeKernel.DENSE.value in filtered_compute_kernels:
            if (
                EmbeddingComputeKernel.FUSED.value in filtered_compute_kernels
            ):  # always false for data_parallel
                filtered_compute_kernels.remove(EmbeddingComputeKernel.DENSE.value)

        if not filtered_compute_kernels:
            logger.warn(
                "No available compute kernels after applying user provided "
                f"constraints for {key}. Constrained compute kernels: "
                f"{constrained_compute_kernels}, allowed compute kernels: "
                f"{allowed_compute_kernels}, filtered compute kernels: "
                f"{filtered_compute_kernels}, sharding type: {sharding_type}. "
                "Please check if the constrained "
                "compute kernels are too restrictive, if the sharder allows the "
                "compute kernels, or if non-strings are passed in."
            )
        return filtered_compute_kernels
