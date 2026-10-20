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

"""Deterministic random-candidate collision resolution for semantic IDs."""

from typing import Optional

import numpy as np

from tzrec.utils.sid.collision import (
    _SPLITMIX_INCREMENT,
    CandidateLastCodes,
    CollisionPlan,
    CollisionResolutionResult,
    _splitmix64,
    stable_order_hash,
)
from tzrec.utils.sid.knn_collision import KnnCollisionResolver


class RandomCollisionResolver(KnnCollisionResolver):
    """Resolve collisions with deterministic random last-code candidates.

    Args:
        num_candidates: Positive number of raw candidate draws per overflow
            row, capped at one less than the final-layer cardinality.
        progress_interval: Number of overflow rows between progress checks.
    """

    def __init__(
        self,
        num_candidates: int,
        progress_interval: int = 1_000_000,
    ) -> None:
        super().__init__(progress_interval)
        if num_candidates < 1:
            raise ValueError(f"num_candidates must be >= 1, got {num_candidates}.")
        self._num_candidates = num_candidates

    def requires_candidates(self) -> bool:
        """Return False because random candidates are generated internally."""
        return False

    def _generate_candidate_last_codes(
        self, item_ids: np.ndarray, last_size: int
    ) -> np.ndarray:
        """Generate deterministic full-space random candidate draws.

        Sampling is with replacement and includes each item's original code
        because that code is not an input to this method. The origin SID is
        already full, so an origin draw cannot place the item and is not replaced.

        Args:
            item_ids: One-dimensional IDs for the overflow rows.
            last_size: Cardinality of the last SID layer.

        Returns:
            An ``(len(item_ids), K)`` int64 matrix, where ``K`` is the smaller
            of ``num_candidates`` and ``last_size - 1``.

        Raises:
            ValueError: If ``last_size`` is smaller than two.
        """
        if last_size < 2:
            raise ValueError("random candidates require last_size >= 2.")
        candidate_count = min(self._num_candidates, last_size - 1)
        hashes = stable_order_hash(item_ids)
        draw_indices = np.arange(candidate_count, dtype=np.uint64)
        with np.errstate(over="ignore"):
            mixed = _splitmix64(
                hashes[:, None] + draw_indices[None, :] * np.uint64(_SPLITMIX_INCREMENT)
            )
        return (mixed % np.uint64(last_size)).astype(np.int64)

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[CandidateLastCodes] = None,
        collect_sid_metadata: bool = True,
    ) -> CollisionResolutionResult:
        """Generate deterministic candidates and resolve overflow rows.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.
            candidate_last_codes: Must be ``None`` because this strategy generates
                its own candidates.
            collect_sid_metadata: Whether to retain final SID metadata.

        Returns:
            Resolved last-layer codes, slot indices, and diagnostics.

        Raises:
            ValueError: If external candidates are supplied.
        """
        if candidate_last_codes is not None:
            raise ValueError(
                "RandomCollisionResolver does not accept candidate_last_codes."
            )
        if plan.overflow_rows.size:
            candidate_last_codes = self._generate_candidate_last_codes(
                plan.overflow_item_ids,
                plan.config.codebook[-1],
            )
        else:
            candidate_last_codes = np.empty((0, 0), dtype=np.int64)
        return super().resolve(
            plan, candidate_last_codes, collect_sid_metadata=collect_sid_metadata
        )
