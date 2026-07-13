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

import unittest
from unittest import mock

import numpy as np

from tzrec.tools.sid import collision_resolution
from tzrec.tools.sid.collision_resolution import (
    CollisionResolutionConfig,
    build_original_item_grouping,
    build_resolved_item_grouping,
    generate_random_candidate_last_codes,
    prepare_collision_plan,
    resolve_sid_collisions,
)


class CollisionResolutionTest(unittest.TestCase):
    def test_golden_candidate_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        codes = np.asarray(
            [
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 1],
                [0, 2],
                [0, 2],
                [1, 0],
                [1, 0],
                [1, 0],
            ],
            dtype=np.int64,
        )
        config = CollisionResolutionConfig((2, 4), 2, "error")
        with mock.patch.object(collision_resolution, "_GROUPING_ROW_CHUNK", 2):
            plan = prepare_collision_plan(item_ids, codes, config)

        np.testing.assert_array_equal(plan.overflow_rows, [1, 3, 8])
        np.testing.assert_array_equal(plan.overflow_item_ids, [1, 3, 8])
        np.testing.assert_array_equal(
            plan.origin_bucket_ids, [0, 0, 0, 0, 1, 2, 2, 3, 3, 3]
        )
        np.testing.assert_array_equal(plan.occupied_sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(
            plan.origin_sid_keys, [0, 0, 0, 0, 1, 2, 2, 4, 4, 4]
        )
        candidates = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 2]], dtype=np.int64)
        result = resolve_sid_collisions(plan, candidates)

        np.testing.assert_array_equal(
            result.resolved_last_codes, [0, 1, 0, 3, 1, 2, 2, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices, [2, 2, 1, 1, 1, 2, 1, 2, 1, 1]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(
            result.final_occupied_sid_keys, [0, 1, 2, 3, 4, 5]
        )
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 2, 2, 1, 2, 1])
        self.assertTrue(result.grouping_collected)
        self.assertIsNone(result.retained_mask)
        self.assertEqual(result.stats.total_items, 10)
        self.assertEqual(result.stats.raw_collision_buckets, 2)
        self.assertEqual(result.stats.final_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 3)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_bucket_size, 2)
        self.assertEqual(result.stats.reassigned_count, 3)
        self.assertEqual(result.stats.unassigned_count, 0)
        self.assertEqual(
            result.stats.to_output_dict(),
            {
                "total_items": 10,
                "raw_collision_buckets": 2,
                "final_collision_buckets": 0,
                "reassigned_count": 3,
                "unassigned_count": 0,
                "max_final_bucket_size": 2,
            },
        )

        with mock.patch.object(collision_resolution, "_GROUPING_ROW_CHUNK", 2):
            original_grouping = build_original_item_grouping(plan)
        np.testing.assert_array_equal(original_grouping.sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(original_grouping.counts, [4, 1, 2, 3])
        np.testing.assert_array_equal(original_grouping.offsets, [0, 4, 5, 7, 10])
        np.testing.assert_array_equal(
            original_grouping.row_order, [2, 0, 1, 3, 4, 6, 5, 9, 7, 8]
        )
        np.testing.assert_array_equal(
            original_grouping.representative_rows, [2, 4, 6, 9]
        )

        with mock.patch.object(collision_resolution, "_GROUPING_ROW_CHUNK", 2):
            resolved_grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(resolved_grouping.counts, [2, 2, 2, 1, 2, 1])
        np.testing.assert_array_equal(
            resolved_grouping.row_order, [2, 0, 4, 1, 6, 5, 3, 9, 7, 8]
        )

    def test_error_fallback(self) -> None:
        config = CollisionResolutionConfig((2,), 1, "error")
        plan = prepare_collision_plan(
            np.asarray([0, 1]), np.asarray([[0], [0]]), config
        )

        with self.assertRaisesRegex(RuntimeError, "first unresolved row indices: 1"):
            resolve_sid_collisions(plan, np.asarray([[0]], dtype=np.int64))

    def test_drop_fallback(self) -> None:
        config = CollisionResolutionConfig((2,), 1, "drop")
        plan = prepare_collision_plan(
            np.asarray([0, 1]), np.asarray([[0], [0]]), config
        )
        result = resolve_sid_collisions(plan, np.asarray([[0]], dtype=np.int64))

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(result.slot_indices, [1, 2])
        np.testing.assert_array_equal(result.retained_mask, [True, False])
        np.testing.assert_array_equal(result.unresolved_rows, [1])
        self.assertEqual(result.stats.final_collision_buckets, 0)
        self.assertEqual(result.stats.unresolved_count, 1)
        with mock.patch.object(collision_resolution, "_GROUPING_ROW_CHUNK", 1):
            original_grouping = build_original_item_grouping(plan)
            resolved_grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(original_grouping.counts, [2])
        np.testing.assert_array_equal(original_grouping.row_order, [0, 1])
        np.testing.assert_array_equal(resolved_grouping.counts, [1])
        np.testing.assert_array_equal(resolved_grouping.row_order, [0])

    def test_keep_original_fallback(self) -> None:
        config = CollisionResolutionConfig((2,), 1, "keep_original")
        plan = prepare_collision_plan(
            np.asarray([0, 1]), np.asarray([[0], [0]]), config
        )
        result = resolve_sid_collisions(plan, np.asarray([[0]], dtype=np.int64))

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(result.slot_indices, [1, 2])
        self.assertIsNone(result.retained_mask)
        np.testing.assert_array_equal(result.unresolved_rows, [1])
        self.assertEqual(result.stats.final_collision_buckets, 1)
        self.assertEqual(result.stats.max_final_bucket_size, 2)
        self.assertEqual(result.stats.unresolved_count, 1)
        resolved_grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0])
        np.testing.assert_array_equal(resolved_grouping.counts, [2])
        np.testing.assert_array_equal(resolved_grouping.row_order, [0, 1])

    def test_no_overflow(self) -> None:
        config = CollisionResolutionConfig((2, 4), 2, "error")
        codes = np.asarray([[0, 0], [0, 0], [1, 2]], dtype=np.int64)
        plan = prepare_collision_plan(np.asarray([0, 1, 2]), codes, config)
        with (
            mock.patch.object(collision_resolution.np, "fromiter") as fromiter,
            mock.patch.object(collision_resolution.np, "argsort") as argsort,
        ):
            result = resolve_sid_collisions(plan, np.empty((0, 3), dtype=np.int64))

        fromiter.assert_not_called()
        argsort.assert_not_called()
        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_occupied_sid_keys, [0, 6])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1])
        self.assertEqual(result.stats.raw_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 0)

    def test_empty_groupings(self) -> None:
        config = CollisionResolutionConfig((2, 4), 2, "error")
        plan = prepare_collision_plan(
            np.empty(0, dtype=np.int64),
            np.empty((0, 2), dtype=np.int64),
            config,
        )
        result = resolve_sid_collisions(plan, np.empty((0, 0), dtype=np.int64))

        for grouping in (
            build_original_item_grouping(plan),
            build_resolved_item_grouping(plan, result),
        ):
            np.testing.assert_array_equal(grouping.sid_keys, [])
            np.testing.assert_array_equal(grouping.counts, [])
            np.testing.assert_array_equal(grouping.row_order, [])
            np.testing.assert_array_equal(grouping.offsets, [0])
            np.testing.assert_array_equal(grouping.representative_rows, [])

    def test_random_candidate_golden_draws(self) -> None:
        actual = generate_random_candidate_last_codes(
            np.asarray([0, 1], dtype=np.int64), num=3, last_size=4
        )

        np.testing.assert_array_equal(actual, [[1, 2, 0], [2, 0, 2]])

    def test_candidate_matrix_must_be_two_dimensional(self) -> None:
        config = CollisionResolutionConfig((2,), 1, "drop")
        plan = prepare_collision_plan(
            np.asarray([0, 1]), np.asarray([[0], [0]]), config
        )

        with self.assertRaisesRegex(ValueError, "must be 2-D"):
            resolve_sid_collisions(plan, np.asarray([1], dtype=np.int64))

    def test_candidate_matrix_must_align_with_overflow_rows(self) -> None:
        config = CollisionResolutionConfig((2,), 1, "drop")
        plan = prepare_collision_plan(
            np.asarray([0, 1]), np.asarray([[0], [0]]), config
        )

        with self.assertRaisesRegex(ValueError, "row-aligned"):
            resolve_sid_collisions(plan, np.empty((0, 1), dtype=np.int64))

    def test_fixed_width_candidates_can_contend_for_free_slots(self) -> None:
        config = CollisionResolutionConfig((6,), 1, "drop")
        item_ids = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
        codes = np.asarray([[0], [0], [1], [3], [4], [4]], dtype=np.int64)
        plan = prepare_collision_plan(item_ids, codes, config)
        np.testing.assert_array_equal(plan.overflow_origin_last_codes, [0, 4])
        candidates = np.asarray(
            [
                [0, 1, 2, 5],
                [4, 2, 1, 3],
            ],
            dtype=np.int64,
        )

        result = resolve_sid_collisions(plan, candidates)

        np.testing.assert_array_equal(result.unresolved_rows, [5])
        self.assertEqual(result.resolved_last_codes[1], 2)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 1)

    def test_rejects_out_of_range_candidate(self) -> None:
        config = CollisionResolutionConfig((4,), 1, "drop")
        item_ids = np.asarray([0, 1, 2], dtype=np.int64)
        codes = np.asarray([[0], [0], [1]], dtype=np.int64)
        plan = prepare_collision_plan(item_ids, codes, config)

        with self.assertRaisesRegex(ValueError, r"must be in \[0, 4\)"):
            resolve_sid_collisions(plan, np.asarray([[5]], dtype=np.int64))

    def test_resolution_can_skip_grouping_metadata(self) -> None:
        config = CollisionResolutionConfig((4,), 1, "drop")
        item_ids = np.asarray([0, 1, 2], dtype=np.int64)
        codes = np.asarray([[0], [0], [1]], dtype=np.int64)
        plan = prepare_collision_plan(item_ids, codes, config)
        candidates = np.asarray([[2]], dtype=np.int64)
        expected = resolve_sid_collisions(plan, candidates)

        with (
            mock.patch.object(collision_resolution.np, "fromiter") as fromiter,
            mock.patch.object(collision_resolution.np, "argsort") as argsort,
            mock.patch.object(collision_resolution.np, "unique") as unique,
        ):
            result = resolve_sid_collisions(plan, candidates, collect_grouping=False)

        fromiter.assert_not_called()
        argsort.assert_not_called()
        unique.assert_not_called()
        np.testing.assert_array_equal(
            result.resolved_last_codes, expected.resolved_last_codes
        )
        np.testing.assert_array_equal(result.slot_indices, expected.slot_indices)
        np.testing.assert_array_equal(result.retained_mask, expected.retained_mask)
        np.testing.assert_array_equal(result.unresolved_rows, expected.unresolved_rows)
        np.testing.assert_array_equal(result.final_occupied_sid_keys, [])
        np.testing.assert_array_equal(result.final_bucket_counts, [])
        self.assertFalse(result.grouping_collected)
        self.assertEqual(result.stats, expected.stats)

        with self.assertRaisesRegex(RuntimeError, "metadata was not collected"):
            build_resolved_item_grouping(plan, result)


if __name__ == "__main__":
    unittest.main()
