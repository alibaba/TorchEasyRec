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

import dataclasses
import unittest

import numpy as np

from tzrec.utils.sid.test_util import (
    create_collision_plan,
    create_existing_sid_state,
)
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver


class UniformCollisionResolverTest(unittest.TestCase):
    def _assert_resolution(
        self,
        plan,
        *,
        last_codes,
        slot_indices,
        sid_keys,
        item_counts,
        stats,
        unresolved_rows=(),
    ) -> None:
        result = UniformCollisionResolver().resolve(plan)
        np.testing.assert_array_equal(result.resolved_last_codes, last_codes)
        np.testing.assert_array_equal(result.slot_indices, slot_indices)
        np.testing.assert_array_equal(result.unresolved_rows, unresolved_rows)
        np.testing.assert_array_equal(result.final_sid_keys, sid_keys)
        np.testing.assert_array_equal(result.final_item_counts, item_counts)
        self.assertEqual(dataclasses.astuple(result.stats), stats)

    def test_no_overflow(self) -> None:
        plan = create_collision_plan((4,), 1, [2, 9], [[0], [1]])
        self._assert_resolution(
            plan,
            last_codes=[0, 1],
            slot_indices=[1, 1],
            sid_keys=[0, 1],
            item_counts=[1, 1],
            stats=(2, 0, 0, 0, 0, 1),
        )

    def test_empty_codes_follow_rotated_order(self) -> None:
        plan = create_collision_plan((4,), 1, [2, 9, 4], [[0]] * 3)
        self._assert_resolution(
            plan,
            last_codes=[0, 2, 3],
            slot_indices=[1, 1, 1],
            sid_keys=[0, 2, 3],
            item_counts=[1, 1, 1],
            stats=(3, 1, 0, 2, 0, 1),
        )

    def test_water_fill_uneven_counts(self) -> None:
        plan = create_collision_plan(
            (3,),
            4,
            [2, 9, 4, 11],
            [[0]] * 4,
            existing_sid_state=create_existing_sid_state([0, 1, 2], [4, 1, 2]),
        )
        self._assert_resolution(
            plan,
            last_codes=[1, 1, 2, 1],
            slot_indices=[2, 3, 3, 4],
            sid_keys=[0, 1, 2],
            item_counts=[4, 4, 3],
            stats=(4, 1, 0, 4, 0, 4),
        )

    def test_insufficient_prefix_capacity(self) -> None:
        plan = create_collision_plan((2,), 1, [2, 9, 4, 11, 0], [[0]] * 5)
        self._assert_resolution(
            plan,
            last_codes=[0, 1, 0, 1, 0],
            slot_indices=[1, 1, 2, 2, 3],
            unresolved_rows=[2, 3, 4],
            sid_keys=[0, 1],
            item_counts=[3, 2],
            stats=(5, 1, 2, 1, 3, 3),
        )

    def test_append_preserves_existing_over_capacity_sid(self) -> None:
        plan = create_collision_plan(
            (3,),
            2,
            [2, 9],
            [[0], [0]],
            existing_sid_state=create_existing_sid_state([0, 2], [5, 1]),
        )
        self._assert_resolution(
            plan,
            last_codes=[1, 1],
            slot_indices=[1, 2],
            sid_keys=[0, 1, 2],
            item_counts=[5, 2, 1],
            stats=(2, 1, 1, 2, 0, 5),
        )

    def test_multiple_prefixes_follow_item_priority(self) -> None:
        plan = create_collision_plan(
            (2, 4),
            1,
            [2, 9, 0, 11, 4, 7, 8, 1, 6],
            [[0, 0], [0, 1]] * 3 + [[1, 0]] * 3,
        )
        self._assert_resolution(
            plan,
            last_codes=[0, 1, 2, 3, 2, 3, 0, 1, 2],
            slot_indices=[1, 1, 2, 1, 1, 2, 1, 1, 1],
            unresolved_rows=[2, 5],
            sid_keys=[0, 1, 2, 3, 4, 5, 6],
            item_counts=[1, 1, 2, 2, 1, 1, 1],
            stats=(9, 3, 2, 4, 2, 2),
        )


if __name__ == "__main__":
    unittest.main()
