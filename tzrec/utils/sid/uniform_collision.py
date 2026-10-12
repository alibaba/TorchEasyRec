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

_CHUNK_CELLS = 1 << 22


class UniformCollisionResolver(CollisionResolver):
    """Spread overflow evenly over the least-loaded codes of each SID prefix.

    Each bucket keeps the rows :func:`prepare_collision_plan` retains. The
    overflow rows of a prefix take, in stable item order, the prefix's
    least-loaded last-layer codes one at a time, so a prefix with room never
    leaves a row over capacity. Ties between equally loaded codes follow a
    per-prefix rotated code order, which keeps relocated rows from piling onto
    the same codes in every prefix. When a prefix has more rows than room, the
    filling continues past capacity, so its largest bucket is as small as its
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
            candidate_last_codes: Must be ``None``; this strategy uses no
                candidates.
            collect_grouping: Whether to retain final bucket metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidate codes are supplied.
        """
        if candidate_last_codes is not None:
            raise ValueError(
                "UniformCollisionResolver does not accept candidate_last_codes."
            )
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

        last_size = plan.config.layer_sizes[-1]
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
        fast_key_parts = []
        if fast_prefixes.size:
            scan_widths = np.minimum(
                last_size,
                prefix_row_counts[fast_prefixes]
                + prefix_occupied_counts[fast_prefixes],
            )
            scan_ends = np.cumsum(scan_widths)
            first = 0
            while first < fast_prefixes.shape[0]:
                end = int(
                    np.searchsorted(
                        scan_ends,
                        int(scan_ends[first] - scan_widths[first]) + _CHUNK_CELLS,
                        side="right",
                    )
                )
                end = max(end, first + 1)
                fast_key_parts.append(
                    self._fill_empty_codes(
                        fast_prefixes[first:end],
                        scan_widths[first:end],
                        prefix_row_counts,
                        prefix_starts,
                        base_keys,
                        rotations,
                        initial_keys,
                        occupied_starts,
                        prefix_occupied_counts,
                        last_size,
                        assigned_codes,
                        assigned_slots,
                    )
                )
                first = end

        general_prefixes = np.flatnonzero(~fits_empty)
        general_key_parts = []
        chunk_prefixes = max(1, _CHUNK_CELLS // last_size)
        for first in range(0, general_prefixes.shape[0], chunk_prefixes):
            general_key_parts.append(
                self._water_fill(
                    general_prefixes[first : first + chunk_prefixes],
                    prefix_row_counts,
                    prefix_starts,
                    base_keys,
                    rotations,
                    initial_keys,
                    initial_counts,
                    occupied_starts,
                    prefix_occupied_counts,
                    last_size,
                    assigned_codes,
                    assigned_slots,
                )
            )

        resolved_last_codes = plan.origin_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        resolved_last_codes[ordered_rows] = assigned_codes
        slot_indices[ordered_rows] = assigned_slots
        unresolved_rows = ordered_rows[assigned_slots > plan.config.capacity]

        general_keys = np.concatenate((np.empty(0, dtype=np.int64), *general_key_parts))
        final_counts = initial_counts.copy()
        positions, found = lookup_sorted(initial_keys, general_keys)
        np.add.at(final_counts, positions[found], 1)
        new_keys, new_counts = np.unique(general_keys[~found], return_counts=True)
        fast_keys = np.concatenate((np.empty(0, dtype=np.int64), *fast_key_parts))
        return self._build_result(
            plan,
            resolved_last_codes,
            slot_indices,
            unresolved_rows,
            in_overflow_prefix,
            np.concatenate((initial_keys, fast_keys, new_keys)),
            np.concatenate(
                (final_counts, np.ones(fast_keys.shape[0], dtype=np.int64), new_counts)
            ),
            collect_grouping,
        )

    @staticmethod
    def _fill_empty_codes(
        prefix_indices: np.ndarray,
        scan_widths: np.ndarray,
        prefix_row_counts: np.ndarray,
        prefix_starts: np.ndarray,
        base_keys: np.ndarray,
        rotations: np.ndarray,
        initial_keys: np.ndarray,
        occupied_starts: np.ndarray,
        prefix_occupied_counts: np.ndarray,
        last_size: int,
        assigned_codes: np.ndarray,
        assigned_slots: np.ndarray,
    ) -> np.ndarray:
        """Give every row of prefixes with enough empty codes one empty code.

        The first ``m`` empty codes in rotated order lie within the first
        ``m + occupied`` rotated positions, so only that window is scanned.

        Args:
            prefix_indices: Indices of the prefixes to fill, ascending.
            scan_widths: Rotated positions to scan for each of ``prefix_indices``.
            prefix_row_counts: Overflow row count of every prefix.
            prefix_starts: Offset of every prefix's rows in stable item order.
            base_keys: Bucket key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            initial_keys: Sorted keys of occupied buckets in overflow prefixes.
            occupied_starts: First ``initial_keys`` position of every prefix.
            prefix_occupied_counts: Occupied bucket count of every prefix.
            last_size: Cardinality of the last SID layer.
            assigned_codes: Per-row last codes in stable item order, updated.
            assigned_slots: Per-row slots in stable item order, updated.

        Returns:
            Global keys of the buckets the rows now occupy.
        """
        owners = np.repeat(
            np.arange(prefix_indices.shape[0], dtype=np.int64), scan_widths
        )
        codes = (
            rotations[prefix_indices][owners] + ranks_within_runs(owners)
        ) % last_size
        keys = base_keys[prefix_indices][owners] + codes
        _, taken = lookup_sorted(
            initial_keys[
                int(occupied_starts[prefix_indices[0]]) : int(
                    occupied_starts[prefix_indices[-1]]
                    + prefix_occupied_counts[prefix_indices[-1]]
                )
            ],
            keys,
        )
        free = np.flatnonzero(~taken)
        free_owners = owners[free]
        free = free[
            ranks_within_runs(free_owners)
            < prefix_row_counts[prefix_indices][free_owners]
        ]
        order_positions = concat_ranges(
            prefix_starts[prefix_indices], prefix_row_counts[prefix_indices]
        )
        assigned_codes[order_positions] = codes[free]
        assigned_slots[order_positions] = 1
        return keys[free]

    @staticmethod
    def _water_fill(
        prefix_indices: np.ndarray,
        prefix_row_counts: np.ndarray,
        prefix_starts: np.ndarray,
        base_keys: np.ndarray,
        rotations: np.ndarray,
        initial_keys: np.ndarray,
        initial_counts: np.ndarray,
        occupied_starts: np.ndarray,
        prefix_occupied_counts: np.ndarray,
        last_size: int,
        assigned_codes: np.ndarray,
        assigned_slots: np.ndarray,
    ) -> np.ndarray:
        """Fill prefixes with fewer empty codes than overflow rows.

        Taking the least-loaded ``(load, rotated position)`` for each row in
        turn yields the smallest pairs ``(level, position)`` with ``level`` at
        or above the code's load, so the final fill level is found by bisection
        and the pairs are enumerated directly.

        Args:
            prefix_indices: Indices of the prefixes to fill.
            prefix_row_counts: Overflow row count of every prefix.
            prefix_starts: Offset of every prefix's rows in stable item order.
            base_keys: Bucket key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            initial_keys: Sorted keys of occupied buckets in overflow prefixes.
            initial_counts: Initial counts aligned with ``initial_keys``.
            occupied_starts: First ``initial_keys`` position of every prefix.
            prefix_occupied_counts: Occupied bucket count of every prefix.
            last_size: Cardinality of the last SID layer.
            assigned_codes: Per-row last codes in stable item order, updated.
            assigned_slots: Per-row slots in stable item order, updated.

        Returns:
            Global bucket key of every assigned row.
        """
        prefix_count = prefix_indices.shape[0]
        loads = np.zeros((prefix_count, last_size), dtype=np.int64)
        lengths = prefix_occupied_counts[prefix_indices]
        selected = concat_ranges(occupied_starts[prefix_indices], lengths)
        owners = np.repeat(np.arange(prefix_count, dtype=np.int64), lengths)
        loads[
            owners,
            (
                initial_keys[selected]
                - base_keys[prefix_indices][owners]
                - rotations[prefix_indices][owners]
            )
            % last_size,
        ] = initial_counts[selected]

        row_counts = prefix_row_counts[prefix_indices]
        low = loads.min(axis=1)
        high = loads.max(axis=1) + -(-row_counts // last_size)
        while np.any(low < high):
            middle = (low + high) // 2
            enough = (
                np.maximum(middle[:, None] + 1 - loads, 0).sum(axis=1) >= row_counts
            )
            high = np.where(enough, middle, high)
            low = np.where(enough, low, middle + 1)
        level = low
        below = np.maximum(level[:, None] - loads, 0)
        at_level = loads <= level[:, None]
        at_level &= (
            np.cumsum(at_level, axis=1) <= (row_counts - below.sum(axis=1))[:, None]
        )

        below_cells = np.flatnonzero(below)
        below_counts = below.ravel()[below_cells]
        pair_levels = np.concatenate(
            (
                concat_ranges(loads.ravel()[below_cells], below_counts),
                np.repeat(level, at_level.sum(axis=1)),
            )
        )
        pair_owners, pair_positions = np.divmod(
            np.concatenate(
                (np.repeat(below_cells, below_counts), np.flatnonzero(at_level))
            ),
            last_size,
        )
        order = np.lexsort((pair_positions, pair_levels, pair_owners))
        pair_owners = pair_owners[order]
        pair_codes = (
            pair_positions[order] + rotations[prefix_indices][pair_owners]
        ) % last_size
        order_positions = concat_ranges(prefix_starts[prefix_indices], row_counts)
        assigned_codes[order_positions] = pair_codes
        assigned_slots[order_positions] = pair_levels[order] + 1
        return base_keys[prefix_indices][pair_owners] + pair_codes
