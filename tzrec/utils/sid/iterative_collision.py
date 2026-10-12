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
    _INT64_MAX,
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
    concat_ranges,
    ranks_within_runs,
    run_starts,
    stable_order_hash,
)

_CHUNK_CELLS = 1 << 22


class IterativeCollisionResolver(CollisionResolver):
    """Resolve overflow with deterministic synchronous batch arbitration.

    Each SID prefix is independent. In every round, each unplaced
    overflow row proposes to its candidates that still have room, except that
    within one origin bucket only the first ``capacity`` rows (in stable item
    order) with room at a candidate position propose at that position, which
    bounds the proposals of large same-origin groups whose candidate lists are
    near-identical. A target accepts up to its remaining capacity, ordered by
    candidate position then stable item order, and each row commits only its
    earliest-position accepted proposal. Slots accepted for a row that
    committed elsewhere reopen in the next round. A row left with no candidate
    that has room keeps its origin SID over capacity.

    Acceptance favors candidate proximity: a bucket goes to the row that ranks
    it earlier even when that row has other options, so this policy can leave
    more rows unresolved than first-fit placement. Within a bucket, relocated
    rows are numbered after its initial occupancy by round, candidate position
    and stable item order.

    Published items participate only through their occupancy. They are never
    proposal rows, so they cannot be moved, and an already over-capacity
    published bucket retains its complete count.

    Prefixes are resolved in chunks that share dense occupancy and run their
    rounds together; a prefix that stops proposing never changes again, so the
    result is the same as resolving each prefix alone.
    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows through synchronous proposal rounds.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_last_codes: Ordered last-layer candidates aligned with
                ``plan.overflow_rows``. It may be omitted only without overflow.
            collect_grouping: Whether to retain final bucket metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidates are absent while overflow rows exist.
        """
        candidates = self._validate_candidate_last_codes(plan, candidate_last_codes)
        in_overflow_prefix, affected_keys, affected_counts = self._initial_occupancy(
            plan
        )
        if plan.overflow_rows.size == 0:
            return self._build_result(
                plan,
                plan.origin_last_codes,
                plan.initial_slot_indices,
                np.empty(0, dtype=np.int64),
                in_overflow_prefix,
                affected_keys,
                affected_counts,
                collect_grouping,
            )

        last_size = plan.config.layer_sizes[-1]
        resolved_last_codes = plan.origin_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_parts = []
        final_key_parts = []
        final_count_parts = []
        prefix_starts = run_starts(plan.overflow_prefix_ids)
        prefix_stops = np.append(prefix_starts[1:], plan.overflow_rows.shape[0])
        prefix_count = prefix_starts.shape[0]
        chunk_prefixes = max(1, _CHUNK_CELLS // last_size)
        max_chunk_rows = max(1, _CHUNK_CELLS // max(1, candidates.shape[1]))
        occupancy = np.zeros(
            min(chunk_prefixes, prefix_count) * last_size, dtype=np.int64
        )
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        first_prefix = 0
        while first_prefix < prefix_count:
            start = int(prefix_starts[first_prefix])
            end_prefix = int(
                np.searchsorted(prefix_stops, start + max_chunk_rows, side="right")
            )
            end_prefix = min(
                max(end_prefix, first_prefix + 1), first_prefix + chunk_prefixes
            )
            stop = int(prefix_stops[end_prefix - 1])
            rows = plan.overflow_rows[start:stop]
            (
                resolved_last_codes[rows],
                slot_indices[rows],
                unresolved_rows,
                final_keys,
                final_counts,
            ) = self._resolve_chunk(
                plan,
                affected_keys,
                affected_counts,
                occupancy,
                slice(start, stop),
                candidates[start:stop],
            )
            unresolved_parts.append(unresolved_rows)
            final_key_parts.append(final_keys)
            final_count_parts.append(final_counts)
            progress.log(stop)
            first_prefix = end_prefix

        return self._build_result(
            plan,
            resolved_last_codes,
            slot_indices,
            np.concatenate(unresolved_parts),
            in_overflow_prefix,
            np.concatenate(final_key_parts),
            np.concatenate(final_count_parts),
            collect_grouping,
        )

    def _resolve_chunk(
        self,
        plan: CollisionPlan,
        affected_keys: np.ndarray,
        affected_counts: np.ndarray,
        occupancy: np.ndarray,
        span: slice,
        candidates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resolve a contiguous run of whole prefixes through shared rounds.

        Buckets are addressed by a chunk-local key, ``prefix position * last_size
        + last code``, into ``occupancy``, which must be all zero on entry and
        is zeroed again on return.

        Args:
            plan: Grouping and append-aware occupancy plan.
            affected_keys: Ascending keys of the occupied buckets in overflow
                prefixes.
            affected_counts: Initial occupancy aligned with ``affected_keys``.
            occupancy: Reusable dense occupancy buffer of at least ``last_size``
                cells per prefix in ``span``.
            span: Slice of the overflow-aligned plan arrays covering whole
                prefixes.
            candidates: Int64 candidate last codes of the rows in ``span``.

        Returns:
            Last codes and slots of the rows in ``span`` order, the unresolved
            rows, and the keys and final counts of the chunk's occupied buckets.

        Raises:
            ValueError: If the chunk is too large to pack arbitration keys.
        """
        capacity = plan.config.capacity
        last_size = plan.config.layer_sizes[-1]
        overflow_rows = plan.overflow_rows[span]
        row_count = overflow_rows.shape[0]
        prefix_ids, prefix_indices = np.unique(
            plan.overflow_prefix_ids[span], return_inverse=True
        )
        prefix_count = prefix_ids.shape[0]
        candidate_count = candidates.shape[1]
        base_keys = prefix_ids * last_size
        local_base_keys = prefix_indices * last_size
        origin_keys = local_base_keys + plan.origin_last_codes[overflow_rows]
        if prefix_count * last_size * candidate_count * row_count > _INT64_MAX:
            raise ValueError(
                f"{row_count} overflow rows with {candidate_count} candidates in "
                "one prefix chunk are too many to pack arbitration keys."
            )

        starts = np.searchsorted(affected_keys, base_keys)
        lengths = np.searchsorted(affected_keys, base_keys + last_size) - starts
        selected = concat_ranges(starts, lengths)
        seeded_keys = (
            np.repeat(np.arange(prefix_count, dtype=np.int64) * last_size, lengths)
            + affected_keys[selected] % last_size
        )
        occupancy[seeded_keys] = affected_counts[selected]

        row_ranks = np.empty(row_count, dtype=np.int64)
        row_ranks[
            np.lexsort((overflow_rows, stable_order_hash(plan.overflow_item_ids[span])))
        ] = np.arange(row_count)

        resolved_keys = origin_keys.copy()
        assigned = np.zeros(row_count, dtype=bool)
        slot_indices = np.empty(row_count, dtype=np.int64)
        pending = np.arange(row_count, dtype=np.int64)
        targets = candidates + local_base_keys[:, None]
        while True:
            has_room = occupancy[targets] < capacity
            # Occupancy only grows, so a row without room anywhere is final.
            reachable = has_room.any(axis=1)
            if not reachable.all():
                pending = pending[reachable]
                targets = targets[reachable]
                has_room = has_room[reachable]
            if pending.size == 0:
                break

            origin_starts = run_starts(origin_keys[pending])
            room_ranks = np.cumsum(has_room, axis=0, dtype=np.int32)
            room_ranks -= np.repeat(
                room_ranks[origin_starts] - has_room[origin_starts],
                np.diff(np.append(origin_starts, pending.shape[0])),
                axis=0,
            )
            proposals = np.flatnonzero(has_room & (room_ranks <= capacity))
            pending_idx, proposal_positions = np.divmod(proposals, candidate_count)
            proposal_rows = pending[pending_idx]
            proposal_targets = targets.ravel()[proposals]

            target_order = np.argsort(
                (proposal_targets * candidate_count + proposal_positions) * row_count
                + row_ranks[proposal_rows]
            )
            ordered_targets = proposal_targets[target_order]
            accepted = np.zeros(proposal_rows.shape[0], dtype=bool)
            accepted[
                target_order[
                    ranks_within_runs(ordered_targets)
                    < capacity - occupancy[ordered_targets]
                ]
            ] = True
            # Proposals are row-major, so a row's first accepted one is its best.
            accepted_proposals = np.flatnonzero(accepted)
            winners = accepted_proposals[run_starts(proposal_rows[accepted_proposals])]
            won = np.zeros(proposal_rows.shape[0], dtype=bool)
            won[winners] = True
            winners = target_order[won[target_order]]
            winner_rows = proposal_rows[winners]
            winner_targets = proposal_targets[winners]

            slot_indices[winner_rows] = (
                occupancy[winner_targets] + ranks_within_runs(winner_targets) + 1
            )
            np.add.at(occupancy, winner_targets, 1)
            assigned[winner_rows] = True
            resolved_keys[winner_rows] = winner_targets
            still_pending = ~assigned[pending]
            pending = pending[still_pending]
            targets = targets[still_pending]

        unresolved = ~assigned
        unresolved_keys = origin_keys[unresolved]
        slot_indices[unresolved] = (
            occupancy[unresolved_keys] + ranks_within_runs(unresolved_keys) + 1
        )
        np.add.at(occupancy, unresolved_keys, 1)

        final_keys = np.unique(np.concatenate((seeded_keys, resolved_keys)))
        final_counts = occupancy[final_keys]
        occupancy[final_keys] = 0
        owners, last_codes = np.divmod(final_keys, last_size)
        return (
            resolved_keys - local_base_keys,
            slot_indices,
            overflow_rows[unresolved],
            base_keys[owners] + last_codes,
            final_counts,
        )
