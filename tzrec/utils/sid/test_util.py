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

from typing import Optional

import numpy as np

from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionConfig,
    ExistingSidState,
    prepare_collision_plan,
)


def create_collision_plan(
    codebook,
    capacity: int,
    item_ids,
    codes,
    existing_sid_state: Optional[ExistingSidState] = None,
) -> CollisionPlan:
    """Create a collision test plan from array-like input data.

    Args:
        codebook: Codebook size of each SID layer.
        capacity: Maximum number of items retained in one SID.
        item_ids: Item IDs in input row order; their dtype is preserved.
        codes: SID codes aligned with item IDs, converted to an int64 matrix.
        existing_sid_state: Published SID state, or None for a full resolve.

    Returns:
        A collision plan for the supplied input and existing SID state.
    """
    return prepare_collision_plan(
        np.asarray(item_ids),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(codebook, capacity),
        existing_sid_state=existing_sid_state,
    )


def create_existing_sid_state(sid_keys, item_counts) -> ExistingSidState:
    """Create published SID state from array-like keys and item counts.

    Args:
        sid_keys: Existing SID keys in ascending order.
        item_counts: Existing item counts aligned with the SID keys.

    Returns:
        Existing SID state with int64 key and count arrays.
    """
    return ExistingSidState(
        np.asarray(sid_keys, dtype=np.int64),
        np.asarray(item_counts, dtype=np.int64),
    )
