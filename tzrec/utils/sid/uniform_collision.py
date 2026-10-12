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

"""Candidate-free least-loaded collision resolution for semantic IDs."""

from typing import Optional

import numpy as np

from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
    concat_ranges,
    lookup_sorted,
    ranks_within_runs,
    run_starts,
    stable_order_hash,
)


class UniformCollisionResolver(CollisionResolver):
    """Spread overflow evenly over the least-loaded codes of each SID prefix.

    Each SID keeps the rows :func:`prepare_collision_plan` retains. The
    overflow rows of a prefix take, in stable item order, the prefix's
    least-loaded last-layer codes one at a time, so a prefix with room never
    leaves a row over capacity. Ties between equally loaded codes follow a
    per-prefix rotated code order, which keeps relocated rows from piling onto
    the same codes in every prefix. When a prefix has more rows than room, the
    filling continues past capacity, so its largest SID is as small as its
    published occupancy allows; rows left over capacity count as unresolved
    even when they changed code.

    Published items participate only through their occupancy. They are never
    overflow rows, so they cannot be moved.
    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows by least-loaded filling within each prefix.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_last_codes: Ignored; this strategy uses no candidates.
            collect_grouping: Whether to retain final SID metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.
        """
        in_overflow_prefix, initial_keys, initial_counts = self._initial_occupancy(plan)
        overflow_count = plan.overflow_rows.shape[0]
        if overflow_count == 0:
            return self._build_result(
                plan,
                plan.origin_last_codes,
                plan.initial_slot_indices,
                np.empty(0, dtype=np.int64),
                in_overflow_prefix,
                initial_keys,
                initial_counts,
                collect_grouping,
            )

        last_size = plan.config.codebook[-1]
        prefix_starts = run_starts(plan.overflow_prefix_ids)
        prefix_row_counts = np.diff(np.append(prefix_starts, overflow_count))
        prefix_ids = plan.overflow_prefix_ids[prefix_starts]
        base_keys = prefix_ids * last_size
        occupied_starts = np.searchsorted(initial_keys, base_keys)
        prefix_occupied_counts = (
            np.searchsorted(initial_keys, base_keys + last_size) - occupied_starts
        )
        # uint64 modulus: a Python-int modulus would promote a uint64 scalar to float.
        rotations = (stable_order_hash(prefix_ids) % np.uint64(last_size)).astype(
            np.int64
        )

        ordered_rows = plan.overflow_rows[
            np.lexsort(
                (
                    plan.overflow_rows,
                    stable_order_hash(plan.overflow_item_ids),
                    plan.overflow_prefix_ids,
                )
            )
        ]
        assigned_codes = np.empty(overflow_count, dtype=np.int64)
        assigned_slots = np.empty(overflow_count, dtype=np.int64)

        # Rows that all fit on empty codes need no load tracking.
        fits_empty = prefix_row_counts <= last_size - prefix_occupied_counts
        fast_prefixes = np.flatnonzero(fits_empty)
        fast_keys = np.empty(0, dtype=np.int64)
        if fast_prefixes.size:
            fast_rows = concat_ranges(
                prefix_starts[fast_prefixes], prefix_row_counts[fast_prefixes]
            )
            assigned_codes[fast_rows], fast_keys = self._fill_empty_codes(
                prefix_row_counts[fast_prefixes],
                prefix_occupied_counts[fast_prefixes],
                base_keys[fast_prefixes],
                rotations[fast_prefixes],
                initial_keys,
                last_size,
            )
            assigned_slots[fast_rows] = 1

        final_counts = initial_counts.copy()
        new_key_arrays = [fast_keys]
        new_count_arrays = [np.ones(fast_keys.shape[0], dtype=np.int64)]
        for prefix_index in np.flatnonzero(~fits_empty):
            row_start = prefix_starts[prefix_index]
            row_end = row_start + prefix_row_counts[prefix_index]
            occupied_start = occupied_starts[prefix_index]
            occupied_end = occupied_start + prefix_occupied_counts[prefix_index]
            base_key = base_keys[prefix_index]
            initial_codes = initial_keys[occupied_start:occupied_end] - base_key
            (
                assigned_codes[row_start:row_end],
                assigned_slots[row_start:row_end],
                target_codes,
                added_item_counts,
            ) = self._water_fill(
                int(row_end - row_start),
                initial_codes,
                initial_counts[occupied_start:occupied_end],
                last_size,
                int(rotations[prefix_index]),
            )
            positions, found = lookup_sorted(initial_codes, target_codes)
            final_counts[occupied_start + positions[found]] += added_item_counts[found]
            new_key_arrays.append(base_key + target_codes[~found])
            new_count_arrays.append(added_item_counts[~found])

        resolved_last_codes = plan.origin_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        resolved_last_codes[ordered_rows] = assigned_codes
        slot_indices[ordered_rows] = assigned_slots
        unresolved_rows = ordered_rows[assigned_slots > plan.config.capacity]

        return self._build_result(
            plan,
            resolved_last_codes,
            slot_indices,
            unresolved_rows,
            in_overflow_prefix,
            np.concatenate((initial_keys, *new_key_arrays)),
            np.concatenate((final_counts, *new_count_arrays)),
            collect_grouping,
        )

    @staticmethod
    def _fill_empty_codes(
        prefix_row_counts: np.ndarray,
        prefix_occupied_counts: np.ndarray,
        base_keys: np.ndarray,
        rotations: np.ndarray,
        initial_keys: np.ndarray,
        last_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Give every row of prefixes with enough empty codes one empty code.

        The first ``m`` empty codes in rotated order lie within the first
        ``m + occupied`` rotated positions, so only that window is scanned.

        Args:
            prefix_row_counts: Overflow row count of each selected prefix.
            prefix_occupied_counts: Occupied SID count of each selected prefix.
            base_keys: SID key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            initial_keys: Sorted keys of occupied SIDs in overflow prefixes.
            last_size: Cardinality of the last SID layer.

        Returns:
            Assigned last-layer codes in prefix order and their SID keys.
        """
        scan_widths = np.minimum(last_size, prefix_row_counts + prefix_occupied_counts)
        owners = np.repeat(
            np.arange(prefix_row_counts.shape[0], dtype=np.int64), scan_widths
        )
        codes = (rotations[owners] + ranks_within_runs(owners)) % last_size
        keys = base_keys[owners] + codes
        _, taken = lookup_sorted(initial_keys, keys)
        free = np.flatnonzero(~taken)
        free_owners = owners[free]
        free = free[ranks_within_runs(free_owners) < prefix_row_counts[free_owners]]
        return codes[free], keys[free]

    @staticmethod
    def _water_fill(
        overflow_count: int,
        initial_codes: np.ndarray,
        initial_counts: np.ndarray,
        last_size: int,
        rotation: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fill one prefix with fewer empty codes than overflow rows.

        Taking the least-loaded ``(load, rotated position)`` for each row in
        turn yields the smallest pairs ``(level, position)`` with ``level`` at
        or above the code's load, so the final fill level is found by bisection
        and the pairs are enumerated directly.

        Args:
            overflow_count: Number of overflow items to place in this prefix.
            initial_codes: Occupied last-layer codes in this prefix.
            initial_counts: Item counts aligned with ``initial_codes``.
            last_size: Cardinality of the last SID layer.
            rotation: Starting last-layer code for breaking equal-load ties.

        Returns:
            Assigned codes and slots in stable item order, followed by unique
            target last-layer codes and their added item counts.
        """
        loads = np.zeros(last_size, dtype=np.int64)
        loads[(initial_codes - rotation) % last_size] = initial_counts

        low = int(loads.min())
        high = int(loads.max()) + (overflow_count + last_size - 1) // last_size
        while low < high:
            middle = (low + high) // 2
            if np.maximum(middle + 1 - loads, 0).sum() >= overflow_count:
                high = middle
            else:
                low = middle + 1
        level = low
        below = np.maximum(level - loads, 0)
        remaining = overflow_count - int(below.sum())
        at_level_positions = np.flatnonzero(loads <= level)[:remaining]

        below_positions = np.flatnonzero(below)
        below_counts = below[below_positions]
        pair_levels = np.concatenate(
            (
                concat_ranges(loads[below_positions], below_counts),
                np.full(at_level_positions.size, level, dtype=np.int64),
            )
        )
        pair_positions = np.concatenate(
            (np.repeat(below_positions, below_counts), at_level_positions)
        )
        order = np.lexsort((pair_positions, pair_levels))
        added_item_counts = below
        added_item_counts[at_level_positions] += 1
        target_positions = np.flatnonzero(added_item_counts)
        return (
            (pair_positions[order] + rotation) % last_size,
            pair_levels[order] + 1,
            (target_positions + rotation) % last_size,
            added_item_counts[target_positions],
        )
