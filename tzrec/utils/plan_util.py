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

import numpy as np
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
    SharderData,
    SharderDataMap,
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


_INF = float("inf")


def _argmin_per_group(group_keys: np.ndarray, perf_keys: np.ndarray) -> np.ndarray:
    """Index of the min-``perf_keys`` entry per distinct ``group_keys`` value.

    Both inputs must be 1-D and the same length. Ties on ``perf_keys`` are
    broken by input order. Output indexes into the original (pre-sort)
    arrays, ordered by ascending ``group_keys``.
    """
    if group_keys.size == 0:
        return np.empty(0, dtype=np.int64)
    lexsort_order = np.lexsort((perf_keys, group_keys))
    sorted_groups = group_keys[lexsort_order]
    is_first_of_run = np.empty(sorted_groups.shape, dtype=bool)
    is_first_of_run[0] = True
    is_first_of_run[1:] = sorted_groups[1:] != sorted_groups[:-1]
    return lexsort_order[is_first_of_run]


def _sparse_dp_proposor_numpy(
    table_opts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hbm_bins: int,
    ddr_bins: int,
) -> List[List[int]]:
    """Sparse-K NumPy 2D DP over (hbm_bin, ddr_bin) reachable cells.

    Per table, broadcasts (K_prev, N) candidates and uses :func:`_argmin_per_group`
    (lexsort + first-of-run) for groupby-argmin on the destination cell.
    Reachable cells K_t saturate at the actual reachable count, not
    ``hbm_bins * ddr_bins``. Backtracking walks T x K_t int32 arrays in
    O(T) Python steps.

    ``table_opts`` entries are ``(opt_hbm, opt_ddr, opt_perf, opt_global_id)``
    per table, all 1-D NumPy arrays (the float ones in bin-units; the index
    one in opt-id space, used to reconstruct proposals).

    Returns a list of proposals; each proposal is a list of opt-id integers,
    one per table, emitted in decreasing-HBM order with at most one entry per
    HBM bin (the perf-best plan across all DDR bins at that HBM level).
    """
    table_count = len(table_opts)
    if table_count == 0:
        return []

    seed_hbm, seed_ddr, seed_perf, seed_opt_id = table_opts[0]
    valid_mask = (seed_hbm < hbm_bins) & (seed_ddr < ddr_bins)
    if not valid_mask.any():
        return []
    seed_hbm = seed_hbm[valid_mask]
    seed_ddr = seed_ddr[valid_mask]
    seed_perf = seed_perf[valid_mask]
    seed_opt_id = seed_opt_id[valid_mask]
    seed_hbm_i = seed_hbm.astype(np.int32)
    seed_ddr_i = seed_ddr.astype(np.int32)
    flat_cell_i = seed_hbm_i.astype(np.int64) * ddr_bins + seed_ddr_i
    winners = _argmin_per_group(group_keys=flat_cell_i, perf_keys=seed_perf)

    dp_perf = seed_perf[winners]
    dp_hbm = seed_hbm[winners]
    dp_ddr = seed_ddr[winners]
    dp_hbm_i = seed_hbm_i[winners]
    back_opt_j: List[np.ndarray] = [seed_opt_id[winners]]
    back_prev_cell_i: List[np.ndarray] = [np.full(winners.size, -1, dtype=np.int32)]

    for table_i in range(1, table_count):
        if dp_perf.size == 0:
            break
        cur_table_hbm, cur_table_ddr, cur_table_perf, cur_table_opt_id = table_opts[
            table_i
        ]
        if cur_table_perf.size == 0:
            dp_perf = np.zeros(0)
            break

        new_hbm = dp_hbm[:, None] + cur_table_hbm[None, :]
        new_ddr = dp_ddr[:, None] + cur_table_ddr[None, :]
        new_perf = dp_perf[:, None] + cur_table_perf[None, :]
        valid_mask = (new_hbm < hbm_bins) & (new_ddr < ddr_bins)
        if not valid_mask.any():
            dp_perf = np.zeros(0)
            break

        new_hbm_i = new_hbm.astype(np.int32)
        new_ddr_i = new_ddr.astype(np.int32)
        flat_cell_i = new_hbm_i.astype(np.int64) * ddr_bins + new_ddr_i

        valid_flat_cell_i = flat_cell_i[valid_mask]
        valid_new_perf = new_perf[valid_mask]
        valid_new_hbm = new_hbm[valid_mask]
        valid_new_ddr = new_ddr[valid_mask]
        valid_new_hbm_i = new_hbm_i[valid_mask]
        valid_prev_cell_i, valid_opt_j = np.where(valid_mask)

        winners = _argmin_per_group(
            group_keys=valid_flat_cell_i, perf_keys=valid_new_perf
        )

        dp_perf = valid_new_perf[winners]
        dp_hbm = valid_new_hbm[winners]
        dp_ddr = valid_new_ddr[winners]
        dp_hbm_i = valid_new_hbm_i[winners]
        back_opt_j.append(cur_table_opt_id[valid_opt_j[winners]])
        back_prev_cell_i.append(valid_prev_cell_i[winners].astype(np.int32))

    if dp_perf.size == 0:
        return []

    # Per-HBM-bin best, decreasing HBM order.
    chosen_cell_i = _argmin_per_group(group_keys=dp_hbm_i, perf_keys=dp_perf)[::-1]

    proposals: List[List[int]] = []
    for last_cell_i in chosen_cell_i:
        proposal_indices = [-1] * table_count
        cur_cell_i = int(last_cell_i)
        for table_i in range(table_count - 1, -1, -1):
            proposal_indices[table_i] = int(back_opt_j[table_i][cur_cell_i])
            cur_cell_i = int(back_prev_cell_i[table_i][cur_cell_i])
        proposals.append(proposal_indices)
    return proposals


class DynamicProgrammingProposer(Proposer):
    r"""Proposes sharding plans in 2D (HBM × DDR) dynamic programming fashion.

        The problem of the Embedding Sharding Plan can be framed as follows: Given
    :math:`M` tables and their corresponding :math:`N` Sharding Options, we need to
    select one sharding option for each table such that the total performance is
    minimized, while keeping both an HBM constraint :math:`K_h` and a host DDR
    constraint :math:`K_d` in check. This can be abstracted into the following
    mathematical formulation:

    Given matrices :math:`A^h`, :math:`A^d`, and :math:`B` of dimensions
    :math:`(M, N)`, let :math:`a^h_{i,j}` and :math:`a^d_{i,j}` be the per-option
    HBM and DDR storage costs, and :math:`b_{i,j}` the perf cost. We aim to find a
    set of column indices :math:`\{ j_0, j_1, \ldots, j_{M-1} \}` such that the
    following conditions are satisfied:

    1. :math:`\sum_{i=0}^{M-1} a^h_{i,j_i} \leq K_h`.
    2. :math:`\sum_{i=0}^{M-1} a^d_{i,j_i} \leq K_d`.
    3. :math:`\sum_{i=0}^{M-1} b_{i,j_i}` is minimized.

    This problem can be tackled using 2D dynamic programming. First, discretize
    :math:`K_h` and :math:`K_d` into bins, and denote the discretization functions
    as :math:`f_h` and :math:`f_d`.

    Define the state :math:`dp[i][f_h(k_h)][f_d(k_d)]` to represent the minimum
    value of :math:`B` when considering the first :math:`i` rows and the totals of
    :math:`A^h` and :math:`A^d` equal the discretized values :math:`k_h` and
    :math:`k_d` respectively.

    The state transition can then be represented as:

    .. math::
        dp[i][f_h(k_h)][f_d(k_d)] = \min_{j=0}^{N-1} \left(
            dp[i-1][f_h(k_h - a^h_{i,j})][f_d(k_d - a^d_{i,j})] + b_{i,j} \right)

    Since :math:`K_h` and :math:`K_d` are sums allocated across all memory, simply
    satisfying that the totals in the plan equal them does not guarantee that the
    allocation will fit on all cards / hosts. Therefore, it is essential to
    maintain all the states of the last layer of :math:`dp`. For each HBM bin we
    emit one proposal -- the lowest-:math:`B` plan across all DDR bins at that
    HBM level -- in decreasing HBM order; plans at the same HBM bin with worse
    perf are strictly dominated and skipped.

    Args:
        hbm_bins_per_device (int): per-device HBM bins for DP precision.
        ddr_bins_per_device (int): per-device DDR bins for DP precision.
    """

    def __init__(
        self,
        hbm_bins_per_device: int = 100,
        ddr_bins_per_device: int = 25,
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
        if not self._sharding_options_by_fqn:
            return

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

        hbm_bins = max(self._hbm_bins_per_device * num_devices, 1)
        ddr_bins = max(self._ddr_bins_per_device * num_devices, 1)
        # Collapse a degenerate axis to a single bin so we don't waste states
        # on (e.g.) CPU-only topologies that have hbm == 0 everywhere.
        if hbm_total == 0:
            hbm_bins = 1
        if ddr_total == 0:
            ddr_bins = 1
        hbm_bin_size = float(hbm_total) / hbm_bins if hbm_bins > 0 else 1.0
        ddr_bin_size = float(ddr_total) / ddr_bins if ddr_bins > 0 else 1.0

        # Per-table option arrays in bin-units, with infeasible options pruned.
        table_opts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for sharding_options in self._sharding_options_by_fqn.values():
            hbm_list: List[float] = []
            ddr_list: List[float] = []
            perf_list: List[float] = []
            opt_id_list: List[int] = []
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
                hbm_list.append(
                    (sharding_option.total_storage.hbm or 0) / hbm_bin_size
                    if hbm_total > 0
                    else 0.0
                )
                ddr_list.append(
                    (sharding_option.total_storage.ddr or 0) / ddr_bin_size
                    if ddr_total > 0
                    else 0.0
                )
                perf_list.append(sharding_option.total_perf)
                opt_id_list.append(opt_id)
            table_opts.append(
                (
                    np.asarray(hbm_list, dtype=np.float32),
                    np.asarray(ddr_list, dtype=np.float32),
                    np.asarray(perf_list, dtype=np.float32),
                    np.asarray(opt_id_list, dtype=np.int32),
                )
            )

        self._proposal_list = _sparse_dp_proposor_numpy(table_opts, hbm_bins, ddr_bins)
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
        sharder_data_map: SharderDataMap,
    ) -> None:
        """Estimate the storage cost of each sharding option.

        Args:
            sharding_options (List[ShardingOption]): list of sharding options.
            sharder_data_map (SharderDataMap): map from module type to sharder data.
        """
        for sharding_option in sharding_options:
            sharder_key = sharding_option.module_type_key
            sharder_data = sharder_data_map[sharder_key]

            caching_ratio = sharding_option.cache_load_factor
            # TODO: remove after deprecating fused_params in sharder
            if caching_ratio is None:
                caching_ratio = (
                    sharder_data.fused_params.get("cache_load_factor")
                    if hasattr(sharder_data, "fused_params")
                    and sharder_data.fused_params
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
                sharder_data.fused_params.get("cache_load_factor", KV_CACHING_RATIO)
                if sharder_data.fused_params
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
                    sharder_data.fused_params.get("multipass_prefetch_config", None)
                    if hasattr(sharder_data, "fused_params")
                    and sharder_data.fused_params
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
                sharder_data=sharder_data,
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
    sharder_data: SharderData,
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
        sharder_data (SharderData): precomputed sharder data for the module.
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
            sharder_data=sharder_data,
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
            sharder_data=sharder_data,
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
