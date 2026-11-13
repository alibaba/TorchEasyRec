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

from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
from torchrec.distributed.planner.types import (
    Enumerator,
    Proposer,
    ShardingOption,
    Topology,
)


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
        """Reset the proposal list and traversal state."""
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
