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
from parameterized import parameterized

from tzrec.utils.sid.collision import (
    build_origin_sid_item_row_index,
    build_resolved_sid_item_row_index,
)
from tzrec.utils.sid.greedy_collision import GreedyCollisionResolver
from tzrec.utils.sid.test_util import (
    create_collision_plan,
    create_existing_sid_occupancy,
)
from tzrec.utils.test_util import parameterized_name_func


class GreedyCollisionResolverTest(unittest.TestCase):
    def test_golden_greedy_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in SID (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        plan = create_collision_plan((2, 4), 2, item_ids, codes)
        candidates = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 2]], dtype=np.int64)
        result = GreedyCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(
            result.resolved_last_codes, [0, 1, 0, 3, 1, 2, 2, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices, [2, 2, 1, 1, 1, 2, 1, 2, 1, 1]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result.final_item_counts, [2, 2, 2, 1, 2, 1])
        self.assertTrue(result.sid_metadata_collected)
        self.assertEqual(result.stats.raw_colliding_sid_count, 2)
        self.assertEqual(result.stats.final_colliding_sid_count, 0)
        self.assertEqual(result.stats.relocated_count, 3)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_items_per_sid, 2)

    @parameterized.expand(
        [
            ("original", [0, 1, 2, 3]),
            ("reordered", [3, 1, 0, 2]),
        ],
        name_func=parameterized_name_func,
    )
    def test_resolution_is_deterministic_and_order_independent(
        self, _name, order
    ) -> None:
        item_ids = np.array([2, 9, 4, 11])[order]
        plan = create_collision_plan((3,), 1, item_ids, [[0]] * 4)
        result = GreedyCollisionResolver().resolve(
            plan, np.array([[1, 2]] * 3, dtype=np.int32)
        )

        original_order_indices = np.argsort(order)
        np.testing.assert_array_equal(
            result.resolved_last_codes[original_order_indices], [0, 1, 2, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices[original_order_indices], [1, 1, 1, 2]
        )
        np.testing.assert_array_equal(item_ids[result.unresolved_rows], [11])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 1, 2])
        np.testing.assert_array_equal(result.final_item_counts, [2, 1, 1])
        self.assertEqual(dataclasses.astuple(result.stats), (4, 1, 1, 2, 1, 2))

        origin_index = build_origin_sid_item_row_index(plan)
        resolved_index = build_resolved_sid_item_row_index(plan, result)
        np.testing.assert_array_equal(
            item_ids[origin_index.item_row_indices], [2, 9, 4, 11]
        )
        np.testing.assert_array_equal(
            item_ids[resolved_index.item_row_indices], [2, 11, 9, 4]
        )

    def test_keep_origin_over_capacity(self) -> None:
        plan = create_collision_plan((2,), 1, [0, 1], [[0], [0]])
        result = GreedyCollisionResolver().resolve(
            plan, np.asarray([[0]], dtype=np.int64)
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(result.slot_indices, [1, 2])
        np.testing.assert_array_equal(result.unresolved_rows, [1])
        self.assertEqual(result.stats.final_colliding_sid_count, 1)
        self.assertEqual(result.stats.max_final_items_per_sid, 2)
        self.assertEqual(result.stats.unresolved_count, 1)
        resolved_sid_item_row_index = build_resolved_sid_item_row_index(plan, result)
        np.testing.assert_array_equal(resolved_sid_item_row_index.sid_keys, [0])
        np.testing.assert_array_equal(resolved_sid_item_row_index.item_counts, [2])
        np.testing.assert_array_equal(
            resolved_sid_item_row_index.item_row_indices, [0, 1]
        )

    def test_three_layer_sparse_prefix_relocation(self) -> None:
        plan = create_collision_plan(
            (2, 3, 4),
            2,
            [0, 1, 2, 3, 4],
            [[0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0]],
        )
        np.testing.assert_array_equal(plan.sid_keys, [0, 8, 12])
        np.testing.assert_array_equal(plan.input_item_counts, [1, 3, 1])
        np.testing.assert_array_equal(plan.overflow_prefix_ids, [2])
        np.testing.assert_array_equal(plan.in_overflow_prefix, [False, True, False])

        result = GreedyCollisionResolver().resolve(
            plan, np.asarray([[1, 2, 3]], dtype=np.int64)
        )

        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(int(result.resolved_last_codes.sum()), 1)
        self.assertEqual(result.resolved_last_codes[3], 0)
        self.assertEqual(result.resolved_last_codes[4], 0)
        np.testing.assert_array_equal(result.final_sid_keys, [0, 8, 9, 12])
        np.testing.assert_array_equal(result.final_item_counts, [1, 2, 1, 1])
        sid_item_row_index = build_resolved_sid_item_row_index(plan, result)
        np.testing.assert_array_equal(
            sid_item_row_index.sid_keys, result.final_sid_keys
        )
        np.testing.assert_array_equal(
            sid_item_row_index.item_counts, result.final_item_counts
        )

    def test_prior_over_capacity_never_reissues_a_published_slot(self) -> None:
        # A previous run left three items on key 0 at capacity 1 (unresolved
        # overflow). The new row cannot relocate, so it must land at slot 4.
        plan = create_collision_plan(
            (2, 4),
            1,
            [10],
            [[0, 0]],
            existing_sid_occupancy=create_existing_sid_occupancy([0], [3]),
        )
        result = GreedyCollisionResolver().resolve(
            plan, np.asarray([[0]], dtype=np.int64)
        )
        self.assertEqual(result.stats.unresolved_count, 1)
        self.assertEqual(int(result.slot_indices[0]), 4)

    def test_relocation_sees_prior_only_sids_in_the_prefix(self) -> None:
        # Key 0 is full, and key 1 -- in the same prefix but holding no new row --
        # is already full too. The only free candidate is key 2.
        plan = create_collision_plan(
            (1, 4),
            1,
            [10],
            [[0, 0]],
            existing_sid_occupancy=create_existing_sid_occupancy([0, 1], [1, 1]),
        )
        result = GreedyCollisionResolver().resolve(
            plan, np.asarray([[1, 2]], dtype=np.int64)
        )
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(int(result.resolved_last_codes[0]), 2)
        self.assertEqual(int(result.slot_indices[0]), 1)
        np.testing.assert_array_equal(
            build_resolved_sid_item_row_index(plan, result).sid_keys, [2]
        )


if __name__ == "__main__":
    unittest.main()
