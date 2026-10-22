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

"""Deterministic multi-round collision resolution for semantic IDs."""

from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger, logger
from tzrec.utils.sid.collision import (
    CandidateLastCodes,
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
    get_same_value_positions,
    get_same_value_start_indices,
    stable_order_hash,
)


class IterativeCollisionResolver(CollisionResolver):
    """Resolve overflow with deterministic synchronous batch arbitration.

    Each SID prefix is independent. In every round, each unplaced
    overflow row requests placement at candidates that still have room, except that
    within one origin SID only the first ``capacity`` rows (in stable item
    order) with room at a candidate position send a request at that position, which
    bounds the requests of large same-origin groups whose candidate lists are
    near-identical. A target accepts up to its remaining capacity, ordered by
    candidate position then stable item order, and each row commits only its
    earliest-position accepted request. Slots accepted for a row that
    committed elsewhere reopen in the next round. A row left with no candidate
    that has room keeps its origin SID over capacity.

    Acceptance favors candidate proximity: a SID goes to the row that ranks
    it earlier even when that row has other options, so this policy can leave
    more rows unresolved than first-fit placement. Within a SID, relocated
    rows are numbered after its initial occupancy by round, candidate position
    and stable item order.

    Published items participate only through their occupancy. They are never
    request rows, so they cannot be moved, and an already over-capacity
    published SID retains its complete count.

    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[CandidateLastCodes] = None,
        collect_sid_metadata: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows through synchronous request rounds.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_last_codes: Ordered last-layer candidates aligned with
                ``plan.overflow_rows``, or their loader. It may be omitted only
                without overflow.
                Candidates remain int32 last-layer codes throughout resolution.
            collect_sid_metadata: Whether to retain final SID metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidates are absent while overflow rows exist.
        """
        candidate_last_codes = self._validate_candidate_last_codes(
            plan, candidate_last_codes
        )
        initial_occupancy, result = self._prepare_collision(plan, collect_sid_metadata)
        if result is not None:
            return result

        prefix_ids, overflow_prefix_indices = np.unique(
            plan.overflow_prefix_ids, return_inverse=True
        )

        last_codebook_size = plan.config.codebook[-1]
        prefix_base_keys = prefix_ids * last_codebook_size
        used_code_offsets = np.append(
            np.searchsorted(initial_occupancy.sid_keys, prefix_base_keys),
            initial_occupancy.sid_keys.size,
        )
        used_code_counts = np.diff(used_code_offsets)
        prefix_count = prefix_ids.shape[0]
        logger.info(
            "Iterative collision: overflow_items=%d prefixes=%d candidates=%d",
            plan.overflow_rows.size,
            prefix_count,
            candidate_last_codes.shape[1],
        )
        used_sid_indices = self._get_used_sid_indices(
            initial_occupancy.sid_keys, used_code_counts, last_codebook_size
        )
        sid_item_counts = np.zeros(prefix_count * last_codebook_size, dtype=np.int64)
        sid_item_counts[used_sid_indices] = initial_occupancy.item_counts

        overflow_item_count = plan.overflow_rows.shape[0]
        item_priority = self._get_item_priority(
            plan.overflow_item_ids, plan.overflow_rows
        )

        prefix_sid_offsets = overflow_prefix_indices * last_codebook_size
        origin_sid_indices = (
            prefix_sid_offsets + plan.origin_last_codes[plan.overflow_rows]
        )
        target_sid_indices = origin_sid_indices.copy()
        is_resolved = np.zeros(overflow_item_count, dtype=bool)
        target_slot_indices = np.empty(overflow_item_count, dtype=np.int64)
        pending_item_indices = np.arange(overflow_item_count, dtype=np.int64)
        pending_candidate_last_codes = candidate_last_codes
        del candidate_last_codes
        capacity = plan.config.capacity
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )
        while pending_item_indices.size:
            (
                selected_item_indices,
                selected_sid_indices,
                has_available_candidate,
            ) = self._select_target_sids(
                pending_item_indices,
                pending_candidate_last_codes,
                prefix_sid_offsets,
                origin_sid_indices,
                item_priority,
                sid_item_counts,
                capacity,
            )
            if selected_item_indices.size == 0:
                break

            target_slot_indices[selected_item_indices] = (
                sid_item_counts[selected_sid_indices]
                + get_same_value_positions(selected_sid_indices)
                + 1
            )
            np.add.at(sid_item_counts, selected_sid_indices, 1)
            is_resolved[selected_item_indices] = True
            target_sid_indices[selected_item_indices] = selected_sid_indices
            del selected_item_indices, selected_sid_indices
            is_still_pending = (
                has_available_candidate & ~is_resolved[pending_item_indices]
            )
            pending_item_indices = pending_item_indices[is_still_pending]
            pending_candidate_last_codes = pending_candidate_last_codes[
                is_still_pending
            ]
            progress.log(overflow_item_count - pending_item_indices.size)

        del pending_candidate_last_codes
        is_unresolved = ~is_resolved
        unresolved_sid_indices = origin_sid_indices[is_unresolved]
        target_slot_indices[is_unresolved] = (
            sid_item_counts[unresolved_sid_indices]
            + get_same_value_positions(unresolved_sid_indices)
            + 1
        )
        np.add.at(sid_item_counts, unresolved_sid_indices, 1)

        final_sid_indices = np.unique(
            np.concatenate((used_sid_indices, target_sid_indices))
        )
        final_prefix_indices, final_last_codes = np.divmod(
            final_sid_indices, last_codebook_size
        )
        final_last_codes = final_last_codes.astype(np.int32)
        resolved_last_codes = plan.origin_last_codes.copy()
        resolved_last_codes[plan.overflow_rows] = (
            target_sid_indices - prefix_sid_offsets
        )
        final_slot_indices = plan.initial_slot_indices.copy()
        final_slot_indices[plan.overflow_rows] = target_slot_indices
        progress.log(overflow_item_count)
        final_item_counts = sid_item_counts[final_sid_indices]
        return self._build_result(
            plan,
            resolved_last_codes,
            final_slot_indices,
            plan.overflow_rows[is_unresolved],
            prefix_base_keys[final_prefix_indices] + final_last_codes,
            final_item_counts,
            collect_sid_metadata,
        )

    @staticmethod
    def _get_used_sid_indices(
        sid_keys: np.ndarray,
        used_code_counts: np.ndarray,
        last_codebook_size: int,
    ) -> np.ndarray:
        """Return indices of SIDs with items in the dense SID count array.

        Args:
            sid_keys: Full SID keys sorted by prefix and last code.
            used_code_counts: Number of occupied SIDs in each prefix, in the
                same prefix order as ``sid_keys``. These are SID counts, not
                item counts.
            last_codebook_size: Number of last-layer codes reserved per prefix.

        Returns:
            Int64 SID count array indices aligned with ``sid_keys``.

        Examples:
            Prefixes 10 and 20 with last-layer size 4 occupy array ranges
            [0, 4) and [4, 8). Their starts [0, 4] repeat as [0, 0, 4, 4, 4],
            then adding last codes [1, 3, 0, 2, 3] gives the final indices:

            >>> IterativeCollisionResolver._get_used_sid_indices(
            ...     np.array([41, 43, 80, 82, 83]), np.array([2, 3]), 4
            ... )
            array([1, 3, 4, 6, 7])
        """
        prefix_sid_start_indices = (
            np.arange(used_code_counts.size, dtype=np.int64) * last_codebook_size
        )
        initial_last_codes = (sid_keys % last_codebook_size).astype(np.int32)
        used_sid_indices = np.repeat(prefix_sid_start_indices, used_code_counts)
        used_sid_indices += initial_last_codes
        return used_sid_indices

    @staticmethod
    def _get_item_priority(
        overflow_item_ids: np.ndarray, overflow_rows: np.ndarray
    ) -> np.ndarray:
        """Compute stable arbitration priorities aligned with overflow items.

        Args:
            overflow_item_ids: Item IDs aligned with ``overflow_rows``.
            overflow_rows: Original input row indices used to break hash ties.

        Returns:
            Int64 priorities aligned with overflow items; smaller values win.

        Examples:
            These item IDs sort by stable hash as [30, 10, 20]. The output
            follows the input order: item 10 has priority 1, item 20 has
            priority 2, and item 30 has priority 0.

            >>> IterativeCollisionResolver._get_item_priority(
            ...     np.array([10, 20, 30]), np.array([0, 1, 2])
            ... )
            array([1, 2, 0])
        """
        item_priority = np.empty(overflow_rows.size, dtype=np.int64)
        sort_indices = np.lexsort((overflow_rows, stable_order_hash(overflow_item_ids)))
        item_priority[sort_indices] = np.arange(overflow_rows.size)
        return item_priority

    def _select_target_sids(
        self,
        pending_item_indices: np.ndarray,
        pending_candidate_last_codes: np.ndarray,
        prefix_sid_offsets: np.ndarray,
        origin_sid_indices: np.ndarray,
        item_priority: np.ndarray,
        sid_item_counts: np.ndarray,
        capacity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select one target SID per item from accepted requests.

        Targets accept requests by candidate position, then item priority,
        up to their remaining capacity. Each item selects its earliest accepted
        candidate. SID counts are updated by the caller after selection.

        Args:
            pending_item_indices: Overflow item indices still awaiting placement.
            pending_candidate_last_codes: Int32 candidates aligned with pending
                items.
            prefix_sid_offsets: Start index of each overflow item's prefix in
                the SID count array.
            origin_sid_indices: Original SID index of every overflow item.
            item_priority: Arbitration priority of every overflow item.
            sid_item_counts: Current item count of every SID.
            capacity: Maximum number of items accepted by one SID.

        Returns:
            Selected overflow item indices, their target SID indices, and a
            mask of pending items with any available candidate. Selected rows
            follow target SID, candidate position, and item priority order.
        """
        (
            request_item_indices,
            request_candidate_positions,
            request_sid_indices,
            has_available_candidate,
        ) = self._build_candidate_requests(
            pending_item_indices,
            pending_candidate_last_codes,
            prefix_sid_offsets,
            origin_sid_indices,
            sid_item_counts,
            capacity,
        )
        if not has_available_candidate.any():
            return request_item_indices, request_sid_indices, has_available_candidate

        request_sort_indices = np.lexsort(
            (
                item_priority[request_item_indices],
                request_candidate_positions,
                request_sid_indices,
            )
        )
        accepted_request_indices = self._get_accepted_request_indices(
            request_sid_indices, request_sort_indices, sid_item_counts, capacity
        )
        selected_request_indices = self._select_first_accepted_requests(
            accepted_request_indices, request_item_indices, request_candidate_positions
        )
        del accepted_request_indices, request_candidate_positions
        is_request_selected = np.zeros(request_item_indices.shape[0], dtype=bool)
        is_request_selected[selected_request_indices] = True
        selected_request_indices = request_sort_indices[
            is_request_selected[request_sort_indices]
        ]
        return (
            request_item_indices[selected_request_indices],
            request_sid_indices[selected_request_indices],
            has_available_candidate,
        )

    @staticmethod
    def _get_accepted_request_indices(
        request_sid_indices: np.ndarray,
        request_sort_indices: np.ndarray,
        sid_item_counts: np.ndarray,
        capacity: int,
    ) -> np.ndarray:
        """Return requests accepted within each target SID's remaining capacity.

        Args:
            request_sid_indices: Target SID index of every request.
            request_sort_indices: Request indices ordered by target SID,
                candidate position, and item priority.
            sid_item_counts: Current item count of every SID.
            capacity: Maximum number of items accepted by one SID.

        Returns:
            Accepted request indices in the supplied arbitration order.
        """
        sorted_request_sid_indices = request_sid_indices[request_sort_indices]
        sorted_request_positions = get_same_value_positions(sorted_request_sid_indices)
        remaining_sid_capacity = capacity - sid_item_counts[sorted_request_sid_indices]
        is_sorted_request_accepted = sorted_request_positions < remaining_sid_capacity
        del sorted_request_positions, remaining_sid_capacity
        return request_sort_indices[is_sorted_request_accepted]

    @staticmethod
    def _select_first_accepted_requests(
        accepted_request_indices: np.ndarray,
        request_item_indices: np.ndarray,
        request_candidate_positions: np.ndarray,
    ) -> np.ndarray:
        """Select each item's earliest accepted candidate request.

        Args:
            accepted_request_indices: Nonempty array of accepted request indices.
            request_item_indices: Overflow item index of every request.
            request_candidate_positions: Candidate position of every request.

        Returns:
            One selected request index per item, in item index order.
        """
        accepted_request_indices = accepted_request_indices[
            np.lexsort(
                (
                    request_candidate_positions[accepted_request_indices],
                    request_item_indices[accepted_request_indices],
                )
            )
        ]
        return accepted_request_indices[
            get_same_value_start_indices(request_item_indices[accepted_request_indices])
        ]

    @staticmethod
    def _build_candidate_requests(
        pending_item_indices: np.ndarray,
        pending_candidate_last_codes: np.ndarray,
        prefix_sid_offsets: np.ndarray,
        origin_sid_indices: np.ndarray,
        sid_item_counts: np.ndarray,
        capacity: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build one round's requests using one candidate column at a time.

        Args:
            pending_item_indices: Overflow item indices still awaiting placement.
            pending_candidate_last_codes: Candidate last codes aligned with pending
                items, stored as int32.
            prefix_sid_offsets: Int64 start index of each overflow item's prefix
                in the SID count array.
            origin_sid_indices: Original SID index of every overflow item.
            sid_item_counts: Current occupancy indexed by SID.
            capacity: Maximum number of items accepted by one SID.

        Returns:
            Request item indices, candidate positions, target SID indices, and
            a mask of pending items with capacity at any candidate.
        """
        origin_sid_start_indices = get_same_value_start_indices(
            origin_sid_indices[pending_item_indices]
        )
        origin_sid_item_counts = np.diff(
            np.append(origin_sid_start_indices, pending_item_indices.size)
        )
        has_available_candidate = np.zeros(pending_item_indices.size, dtype=bool)
        pending_prefix_sid_offsets = prefix_sid_offsets[pending_item_indices]
        request_batches = []
        request_counts = []
        for candidate_position in range(pending_candidate_last_codes.shape[1]):
            candidate_sid_indices = (
                pending_prefix_sid_offsets
                + pending_candidate_last_codes[:, candidate_position]
            )
            candidate_has_capacity = sid_item_counts[candidate_sid_indices] < capacity
            del candidate_sid_indices
            has_available_candidate |= candidate_has_capacity
            candidate_request_positions = np.cumsum(
                candidate_has_capacity, dtype=np.int64
            )
            candidate_request_positions -= np.repeat(
                candidate_request_positions[origin_sid_start_indices]
                - candidate_has_capacity[origin_sid_start_indices],
                origin_sid_item_counts,
            )
            requests = np.flatnonzero(
                candidate_has_capacity & (candidate_request_positions <= capacity)
            )
            request_batches.append(requests)
            request_counts.append(requests.size)

        if not has_available_candidate.any():
            empty = np.empty(0, dtype=np.int64)
            return empty, empty, empty, has_available_candidate
        request_pending_item_indices = np.concatenate(request_batches)
        del request_batches
        request_candidate_positions = np.repeat(
            np.arange(pending_candidate_last_codes.shape[1], dtype=np.int32),
            request_counts,
        )
        request_item_indices = pending_item_indices[request_pending_item_indices]
        request_sid_indices = (
            prefix_sid_offsets[request_item_indices]
            + pending_candidate_last_codes[
                request_pending_item_indices, request_candidate_positions
            ]
        )
        return (
            request_item_indices,
            request_candidate_positions,
            request_sid_indices,
            has_available_candidate,
        )
