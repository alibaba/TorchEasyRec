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

"""First-fit candidate collision resolution for semantic IDs."""

from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger
from tzrec.utils.sid.collision import (
    CandidateLastCodes,
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
)


class GreedyCollisionResolver(CollisionResolver):
    """Resolve collisions first-fit from externally supplied candidate codes."""

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[CandidateLastCodes] = None,
        collect_sid_metadata: bool = True,
    ) -> CollisionResolutionResult:
        """Greedily relocate overflow rows to their first free candidate slot.

        An overflow row that finds no free candidate slot keeps its origin SID
        (over capacity), so every input row is preserved in the output.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.
            candidate_last_codes: Last-layer candidate matrix with shape
                ``(M, K)``, row-aligned with ``plan.overflow_rows``, or their
                loader. It may be omitted only without overflow.
            collect_sid_metadata: Whether to retain final SID metadata required
                by :func:`build_resolved_sid_item_row_index`.

        Returns:
            Resolved last-layer codes, slot indices, unresolved row indices,
            and diagnostics.

        Raises:
            TypeError: If candidate codes do not use an integer dtype.
            ValueError: If candidates are omitted for a plan with overflow, are
                not a row-aligned two-dimensional matrix, or contain
                out-of-range last-layer indices.
        """
        candidate_last_codes = self._validate_candidate_last_codes(
            plan, candidate_last_codes
        )
        initial_occupancy, result = self._prepare_collision(plan, collect_sid_metadata)
        if result is not None:
            return result

        resolved_last_codes = plan.origin_last_codes.copy()
        final_slot_indices = plan.initial_slot_indices.copy()
        unresolved_rows = []
        sid_item_counts = dict(
            zip(
                initial_occupancy.sid_keys.tolist(),
                initial_occupancy.item_counts.tolist(),
            )
        )
        capacity = plan.config.capacity
        last_codebook_size = plan.config.codebook[-1]
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        for processed_item_count, (
            row,
            prefix_base_key,
            origin_last_code,
            item_candidate_last_codes,
        ) in enumerate(
            zip(
                plan.overflow_rows,
                plan.overflow_prefix_ids * last_codebook_size,
                plan.origin_last_codes[plan.overflow_rows],
                candidate_last_codes,
            ),
            start=1,
        ):
            for candidate_last_code in item_candidate_last_codes:
                target_sid_key = prefix_base_key + candidate_last_code
                target_sid_item_count = sid_item_counts.get(target_sid_key, 0)
                if target_sid_item_count < capacity:
                    sid_item_counts[target_sid_key] = target_sid_item_count + 1
                    resolved_last_codes[row] = candidate_last_code
                    final_slot_indices[row] = target_sid_item_count + 1
                    break
            else:
                unresolved_rows.append(row)
                origin_sid_key = prefix_base_key + origin_last_code
                origin_sid_item_count = sid_item_counts.get(origin_sid_key, 0) + 1
                sid_item_counts[origin_sid_key] = origin_sid_item_count
                final_slot_indices[row] = origin_sid_item_count
            progress.log(processed_item_count)

        return self._build_result(
            plan,
            resolved_last_codes,
            final_slot_indices,
            np.asarray(unresolved_rows, dtype=np.int64),
            np.fromiter(sid_item_counts, dtype=np.int64, count=len(sid_item_counts)),
            np.fromiter(
                sid_item_counts.values(), dtype=np.int64, count=len(sid_item_counts)
            ),
            collect_sid_metadata,
        )
