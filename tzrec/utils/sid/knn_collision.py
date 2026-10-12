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

from __future__ import annotations

from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger
from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolver,
)


class KnnCollisionResolver(CollisionResolver):
    """Resolve collisions first-fit from externally supplied candidate codes."""

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Greedily relocate overflow rows to their first free candidate slot.

        An overflow row that finds no free candidate slot keeps its origin SID
        (over capacity), so every input row is preserved in the output.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.
            candidate_last_codes: Last-layer candidate matrix with shape
                ``(M, K)``, row-aligned with ``plan.overflow_rows``. It may be
                omitted only when the plan has no overflow rows.
            collect_grouping: Whether to retain final SID metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, unresolved row indices,
            and diagnostics.

        Raises:
            TypeError: If candidate codes do not use an integer dtype.
            ValueError: If candidates are omitted for a plan with overflow, are
                not a row-aligned two-dimensional matrix, or contain
                out-of-range last-layer indices.
        """
        candidates = self._validate_candidate_last_codes(plan, candidate_last_codes)
        in_overflow_prefix, affected_keys, affected_counts = self._initial_occupancy(
            plan
        )
        overflow_count = plan.overflow_rows.shape[0]
        if overflow_count == 0:
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

        last_size = plan.config.codebook[-1]
        capacity = plan.config.capacity
        occupancy = dict(zip(affected_keys.tolist(), affected_counts.tolist()))
        get_occupancy = occupancy.get
        resolved_last_codes = plan.origin_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_rows = []
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        for resolved_count, (
            row,
            base_key,
            origin_last_code,
            candidate_row,
        ) in enumerate(
            zip(
                plan.overflow_rows,
                plan.overflow_prefix_ids * last_size,
                plan.origin_last_codes[plan.overflow_rows],
                candidates,
            ),
            start=1,
        ):
            for candidate in candidate_row:
                destination_key = base_key + candidate
                destination_occupancy = get_occupancy(destination_key, 0)
                if destination_occupancy < capacity:
                    occupancy[destination_key] = destination_occupancy + 1
                    resolved_last_codes[row] = candidate
                    slot_indices[row] = destination_occupancy + 1
                    break
            else:
                unresolved_rows.append(row)
                origin_key = base_key + origin_last_code
                origin_occupancy = get_occupancy(origin_key, 0) + 1
                occupancy[origin_key] = origin_occupancy
                slot_indices[row] = origin_occupancy
            progress.log(resolved_count)

        return self._build_result(
            plan,
            resolved_last_codes,
            slot_indices,
            np.asarray(unresolved_rows, dtype=np.int64),
            in_overflow_prefix,
            np.fromiter(occupancy, dtype=np.int64, count=len(occupancy)),
            np.fromiter(occupancy.values(), dtype=np.int64, count=len(occupancy)),
            collect_grouping,
        )
