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
    CandidateLastCodes,
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
    concat_ranges,
    find_in_sorted_array,
    get_same_value_positions,
    get_same_value_start_indices,
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

    def requires_candidates(self) -> bool:
        """Return False because uniform placement needs no candidates."""
        return False

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[CandidateLastCodes] = None,
        collect_sid_metadata: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows by least-loaded filling within each prefix.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_last_codes: Ignored; this strategy uses no candidates.
            collect_sid_metadata: Whether to retain final SID metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.
        """
        initial_state, result = self._prepare_collision(plan, collect_sid_metadata)
        if result is not None:
            return result

        overflow_item_count = plan.overflow_rows.shape[0]

        prefix_offsets = np.append(
            get_same_value_start_indices(plan.overflow_prefix_ids), overflow_item_count
        )
        prefix_ids = plan.overflow_prefix_ids[prefix_offsets[:-1]]
        last_codebook_size = plan.config.codebook[-1]
        prefix_base_keys = prefix_ids * last_codebook_size
        # Initial keys contain only these prefixes, so the next start is this one's end.
        used_code_offsets = np.append(
            np.searchsorted(initial_state.sid_keys, prefix_base_keys),
            initial_state.sid_keys.shape[0],
        )

        # Rows that all fit on empty codes need no load tracking.
        prefix_row_counts = np.diff(prefix_offsets)
        used_code_counts = np.diff(used_code_offsets)
        fits_empty = prefix_row_counts <= last_codebook_size - used_code_counts
        fast_prefixes = np.flatnonzero(fits_empty)
        # uint64 modulus: a Python-int modulus would promote a uint64 scalar to float.
        rotations = (
            stable_order_hash(prefix_ids) % np.uint64(last_codebook_size)
        ).astype(np.int32)
        target_last_codes = np.empty(overflow_item_count, dtype=np.int32)
        target_slot_indices = np.empty(overflow_item_count, dtype=np.int64)
        fast_keys = np.empty(0, dtype=np.int64)
        # All overflow items fit in empty codes; use _get_empty_last_codes.
        if fast_prefixes.size:
            fast_rows = concat_ranges(
                prefix_offsets[fast_prefixes], prefix_row_counts[fast_prefixes]
            )
            target_last_codes[fast_rows], fast_keys = self._get_empty_last_codes(
                prefix_row_counts[fast_prefixes],
                used_code_counts[fast_prefixes],
                prefix_base_keys[fast_prefixes],
                rotations[fast_prefixes],
                initial_state.sid_keys,
                last_codebook_size,
            )
            target_slot_indices[fast_rows] = 1

        final_counts = initial_state.item_counts.copy()
        new_key_arrays = [fast_keys]
        new_count_arrays = [np.ones(fast_keys.shape[0], dtype=np.int64)]
        for prefix_index in np.flatnonzero(~fits_empty):
            row_start, row_end = prefix_offsets[prefix_index : prefix_index + 2]
            prefix_sid_start, prefix_sid_end = used_code_offsets[
                prefix_index : prefix_index + 2
            ]
            prefix_base_key = prefix_base_keys[prefix_index]
            initial_last_codes = (
                initial_state.sid_keys[prefix_sid_start:prefix_sid_end]
                - prefix_base_key
            ).astype(np.int32, copy=False)
            (
                target_last_codes[row_start:row_end],
                target_slot_indices[row_start:row_end],
                target_codes,
                added_item_counts,
            ) = self._water_fill(
                int(row_end - row_start),
                initial_last_codes,
                initial_state.item_counts[prefix_sid_start:prefix_sid_end],
                last_codebook_size,
                int(rotations[prefix_index]),
            )
            insert_positions, is_existing = find_in_sorted_array(
                initial_last_codes, target_codes
            )
            final_counts[prefix_sid_start + insert_positions[is_existing]] += (
                added_item_counts[is_existing]
            )
            new_key_arrays.append(prefix_base_key + target_codes[~is_existing])
            new_count_arrays.append(added_item_counts[~is_existing])

        resolved_last_codes, final_slot_indices, unresolved_rows = (
            self._apply_overflow_results(plan, target_last_codes, target_slot_indices)
        )
        return self._build_result(
            plan,
            resolved_last_codes,
            final_slot_indices,
            unresolved_rows,
            initial_state,
            np.concatenate((initial_state.sid_keys, *new_key_arrays)),
            np.concatenate((final_counts, *new_count_arrays)),
            collect_sid_metadata,
        )

    def _apply_overflow_results(
        self,
        plan: CollisionPlan,
        target_last_codes: np.ndarray,
        target_slot_indices: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Write overflow placements back to the original input row order.

        Args:
            plan: Original last-layer codes, slots, and overflow row indices.
            target_last_codes: Overflow last codes in prefix and stable item order.
            target_slot_indices: Overflow slots in the same order as target codes.

        Returns:
            Last codes and slots for all input rows, followed by original row
            indices of overflow items whose target slots exceed SID capacity.
        """
        sorted_overflow_rows = self._get_sorted_overflow_rows(plan)
        resolved_last_codes = plan.origin_last_codes.copy()
        resolved_last_codes[sorted_overflow_rows] = target_last_codes
        final_slot_indices = plan.initial_slot_indices.copy()
        final_slot_indices[sorted_overflow_rows] = target_slot_indices
        unresolved_rows = sorted_overflow_rows[
            target_slot_indices > plan.config.capacity
        ]
        return resolved_last_codes, final_slot_indices, unresolved_rows

    @staticmethod
    def _get_sorted_overflow_rows(plan: CollisionPlan) -> np.ndarray:
        """Return overflow rows ordered by prefix and stable item hash.

        Args:
            plan: Overflow prefixes, item IDs, and original input row indices.

        Returns:
            Original input row indices sorted by prefix, item hash, and row
            index, aligned with the per-prefix placement results.
        """
        return plan.overflow_rows[
            np.lexsort(
                (
                    plan.overflow_rows,
                    stable_order_hash(plan.overflow_item_ids),
                    plan.overflow_prefix_ids,
                )
            )
        ]

    @staticmethod
    def _get_empty_last_codes(
        prefix_row_counts: np.ndarray,
        used_code_counts: np.ndarray,
        prefix_base_keys: np.ndarray,
        rotations: np.ndarray,
        initial_keys: np.ndarray,
        last_codebook_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get one empty last-layer code per overflow item, in rotated code order.

        The first ``m`` empty codes in rotated order lie within the first
        ``m + used_code_count`` rotated positions, so only that window is scanned.

        Args:
            prefix_row_counts: Overflow row count of each selected prefix.
            used_code_counts: Number of last-layer codes with items in each prefix.
            prefix_base_keys: SID key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            initial_keys: Sorted keys of occupied SIDs in overflow prefixes.
            last_codebook_size: Cardinality of the last SID layer.

        Returns:
            Assigned last-layer codes in prefix order and their SID keys.
        """
        scan_widths = np.minimum(
            last_codebook_size, prefix_row_counts + used_code_counts
        )
        prefix_indices = np.repeat(
            np.arange(prefix_row_counts.shape[0], dtype=np.int64), scan_widths
        )
        codes = (
            (rotations[prefix_indices] + get_same_value_positions(prefix_indices))
            % last_codebook_size
        ).astype(np.int32, copy=False)
        keys = prefix_base_keys[prefix_indices] + codes
        _, is_existing = find_in_sorted_array(initial_keys, keys)
        empty_code_positions = np.flatnonzero(~is_existing)
        empty_prefix_indices = prefix_indices[empty_code_positions]
        empty_code_positions = empty_code_positions[
            get_same_value_positions(empty_prefix_indices)
            < prefix_row_counts[empty_prefix_indices]
        ]
        return codes[empty_code_positions], keys[empty_code_positions]

    @staticmethod
    def _get_fill_level(sid_item_counts: np.ndarray, overflow_item_count: int) -> int:
        """Find the last load level needed to place the overflow items.

        Args:
            sid_item_counts: Current item count of each last-layer code in one prefix.
            overflow_item_count: Number of items to place in this prefix.

        Returns:
            The smallest level where filling each lower load through
            ``level + 1`` provides enough positions for all overflow items.
        """
        last_codebook_size = sid_item_counts.size
        low = int(sid_item_counts.min())
        high = (
            int(sid_item_counts.max())
            + (overflow_item_count + last_codebook_size - 1) // last_codebook_size
        )
        while low < high:
            middle = (low + high) // 2
            if np.maximum(middle + 1 - sid_item_counts, 0).sum() >= overflow_item_count:
                high = middle
            else:
                low = middle + 1
        return low

    @staticmethod
    def _water_fill(
        overflow_item_count: int,
        initial_last_codes: np.ndarray,
        initial_item_counts: np.ndarray,
        last_codebook_size: int,
        rotation: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Fill one prefix with fewer empty codes than overflow rows.

        Taking the least-loaded ``(load, rotated position)`` for each row in
        turn yields the smallest pairs ``(level, position)`` with ``level`` at
        or above the code's load, so the final fill level is found by bisection
        and the pairs are enumerated directly.

        Args:
            overflow_item_count: Number of overflow items to place in this prefix.
            initial_last_codes: Occupied last-layer codes in this prefix.
            initial_item_counts: Item counts aligned with ``initial_last_codes``.
            last_codebook_size: Cardinality of the last SID layer.
            rotation: Starting last-layer code for breaking equal-load ties.

        Returns:
            Assigned codes and slots in stable item order, followed by unique
            target last-layer codes and their added item counts.
        """
        sid_item_counts = np.zeros(last_codebook_size, dtype=np.int64)
        sid_item_counts[(initial_last_codes - rotation) % last_codebook_size] = (
            initial_item_counts
        )

        level = UniformCollisionResolver._get_fill_level(
            sid_item_counts, overflow_item_count
        )
        below = np.maximum(level - sid_item_counts, 0)
        below_positions = np.flatnonzero(below)
        below_counts = below[below_positions]
        remaining = overflow_item_count - int(below.sum())
        at_level_positions = np.flatnonzero(sid_item_counts <= level)[:remaining]
        pair_levels = np.concatenate(
            (
                concat_ranges(sid_item_counts[below_positions], below_counts),
                np.full(at_level_positions.size, level, dtype=np.int64),
            )
        )
        pair_positions = np.concatenate(
            (np.repeat(below_positions, below_counts), at_level_positions)
        )
        added_item_counts = below
        added_item_counts[at_level_positions] += 1
        target_positions = np.flatnonzero(added_item_counts)
        sort_indices = np.lexsort((pair_positions, pair_levels))
        return (
            ((pair_positions[sort_indices] + rotation) % last_codebook_size).astype(
                np.int32
            ),
            pair_levels[sort_indices] + 1,
            ((target_positions + rotation) % last_codebook_size).astype(np.int32),
            added_item_counts[target_positions],
        )
