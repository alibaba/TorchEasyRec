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

from tzrec.utils.logging_util import ProgressLogger
from tzrec.utils.sid.collision import (
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
    overflow row proposes to its candidates that still have room, except that
    within one origin SID only the first ``capacity`` rows (in stable item
    order) with room at a candidate position propose at that position, which
    bounds the proposals of large same-origin groups whose candidate lists are
    near-identical. A target accepts up to its remaining capacity, ordered by
    candidate position then stable item order, and each row commits only its
    earliest-position accepted proposal. Slots accepted for a row that
    committed elsewhere reopen in the next round. A row left with no candidate
    that has room keeps its origin SID over capacity.

    Acceptance favors candidate proximity: a SID goes to the row that ranks
    it earlier even when that row has other options, so this policy can leave
    more rows unresolved than first-fit placement. Within a SID, relocated
    rows are numbered after its initial occupancy by round, candidate position
    and stable item order.

    Published items participate only through their occupancy. They are never
    proposal rows, so they cannot be moved, and an already over-capacity
    published SID retains its complete count.

    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_sid_metadata: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows through synchronous proposal rounds.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_last_codes: Ordered last-layer candidates aligned with
                ``plan.overflow_rows``. It may be omitted only without overflow.
            collect_sid_metadata: Whether to retain final SID metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidates are absent while overflow rows exist.
        """
        candidate_last_codes = self._validate_candidate_last_codes(
            plan, candidate_last_codes
        )
        initial_state, result = self._prepare_collision(plan, collect_sid_metadata)
        if result is not None:
            return result

        prefix_ids, overflow_prefix_indices = np.unique(
            plan.overflow_prefix_ids, return_inverse=True
        )

        last_codebook_size = plan.config.codebook[-1]
        prefix_base_keys = prefix_ids * last_codebook_size
        used_code_offsets = np.append(
            np.searchsorted(initial_state.sid_keys, prefix_base_keys),
            initial_state.sid_keys.size,
        )
        used_code_counts = np.diff(used_code_offsets)
        prefix_count = prefix_ids.shape[0]
        initial_sid_indices = (
            np.repeat(
                np.arange(prefix_count, dtype=np.int64) * last_codebook_size,
                used_code_counts,
            )
            + initial_state.sid_keys % last_codebook_size
        )
        sid_item_counts = np.zeros(prefix_count * last_codebook_size, dtype=np.int64)
        sid_item_counts[initial_sid_indices] = initial_state.item_counts

        overflow_rows = plan.overflow_rows
        overflow_item_count = overflow_rows.shape[0]
        item_priority_indices = np.empty(overflow_item_count, dtype=np.int64)
        item_priority_indices[
            np.lexsort((overflow_rows, stable_order_hash(plan.overflow_item_ids)))
        ] = np.arange(overflow_item_count)

        prefix_sid_offsets = overflow_prefix_indices * last_codebook_size
        origin_sid_indices = prefix_sid_offsets + plan.origin_last_codes[overflow_rows]
        target_sid_indices = origin_sid_indices.copy()
        is_resolved = np.zeros(overflow_item_count, dtype=bool)
        target_slot_indices = np.empty(overflow_item_count, dtype=np.int64)
        pending_item_indices = np.arange(overflow_item_count, dtype=np.int64)
        pending_candidate_sid_indices = (
            candidate_last_codes + prefix_sid_offsets[:, None]
        )
        capacity = plan.config.capacity
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )
        while True:
            candidate_has_capacity = (
                sid_item_counts[pending_candidate_sid_indices] < capacity
            )
            # Occupancy only grows, so a row without room anywhere is final.
            has_available_candidate = candidate_has_capacity.any(axis=1)
            if not has_available_candidate.all():
                pending_item_indices = pending_item_indices[has_available_candidate]
                pending_candidate_sid_indices = pending_candidate_sid_indices[
                    has_available_candidate
                ]
                candidate_has_capacity = candidate_has_capacity[has_available_candidate]
            if pending_item_indices.size == 0:
                break

            origin_sid_start_indices = get_same_value_start_indices(
                origin_sid_indices[pending_item_indices]
            )
            candidate_proposal_positions = np.cumsum(
                candidate_has_capacity, axis=0, dtype=np.int32
            )
            candidate_proposal_positions -= np.repeat(
                candidate_proposal_positions[origin_sid_start_indices]
                - candidate_has_capacity[origin_sid_start_indices],
                np.diff(
                    np.append(origin_sid_start_indices, pending_item_indices.shape[0])
                ),
                axis=0,
            )
            proposal_pending_item_indices, proposal_candidate_positions = np.nonzero(
                candidate_has_capacity & (candidate_proposal_positions <= capacity)
            )
            proposal_item_indices = pending_item_indices[proposal_pending_item_indices]
            proposal_sid_indices = pending_candidate_sid_indices[
                proposal_pending_item_indices, proposal_candidate_positions
            ]

            proposal_sort_indices = np.lexsort(
                (
                    item_priority_indices[proposal_item_indices],
                    proposal_candidate_positions,
                    proposal_sid_indices,
                )
            )
            sorted_proposal_sid_indices = proposal_sid_indices[proposal_sort_indices]
            is_proposal_accepted = np.zeros(proposal_item_indices.shape[0], dtype=bool)
            is_proposal_accepted[
                proposal_sort_indices[
                    get_same_value_positions(sorted_proposal_sid_indices)
                    < capacity - sid_item_counts[sorted_proposal_sid_indices]
                ]
            ] = True
            # Proposals are row-major, so a row's first accepted one is its best.
            accepted_proposal_indices = np.flatnonzero(is_proposal_accepted)
            selected_proposal_indices = accepted_proposal_indices[
                get_same_value_start_indices(
                    proposal_item_indices[accepted_proposal_indices]
                )
            ]
            is_proposal_selected = np.zeros(proposal_item_indices.shape[0], dtype=bool)
            is_proposal_selected[selected_proposal_indices] = True
            selected_proposal_indices = proposal_sort_indices[
                is_proposal_selected[proposal_sort_indices]
            ]
            selected_item_indices = proposal_item_indices[selected_proposal_indices]
            selected_sid_indices = proposal_sid_indices[selected_proposal_indices]

            target_slot_indices[selected_item_indices] = (
                sid_item_counts[selected_sid_indices]
                + get_same_value_positions(selected_sid_indices)
                + 1
            )
            np.add.at(sid_item_counts, selected_sid_indices, 1)
            is_resolved[selected_item_indices] = True
            target_sid_indices[selected_item_indices] = selected_sid_indices
            is_still_pending = ~is_resolved[pending_item_indices]
            pending_item_indices = pending_item_indices[is_still_pending]
            pending_candidate_sid_indices = pending_candidate_sid_indices[
                is_still_pending
            ]
            progress.log(overflow_item_count - pending_item_indices.size)

        is_unresolved = ~is_resolved
        unresolved_sid_indices = origin_sid_indices[is_unresolved]
        target_slot_indices[is_unresolved] = (
            sid_item_counts[unresolved_sid_indices]
            + get_same_value_positions(unresolved_sid_indices)
            + 1
        )
        np.add.at(sid_item_counts, unresolved_sid_indices, 1)

        final_sid_indices = np.unique(
            np.concatenate((initial_sid_indices, target_sid_indices))
        )
        final_prefix_indices, final_last_codes = np.divmod(
            final_sid_indices, last_codebook_size
        )
        resolved_last_codes = plan.origin_last_codes.copy()
        resolved_last_codes[overflow_rows] = target_sid_indices - prefix_sid_offsets
        final_slot_indices = plan.initial_slot_indices.copy()
        final_slot_indices[overflow_rows] = target_slot_indices
        progress.log(overflow_item_count)
        final_item_counts = sid_item_counts[final_sid_indices]
        return self._build_result(
            plan,
            resolved_last_codes,
            final_slot_indices,
            overflow_rows[is_unresolved],
            initial_state,
            prefix_base_keys[final_prefix_indices] + final_last_codes,
            final_item_counts,
            collect_sid_metadata,
        )
