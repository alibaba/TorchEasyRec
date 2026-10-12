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

import numpy as np
from parameterized import parameterized

from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    PriorOccupancy,
    build_origin_item_grouping,
    build_resolved_item_grouping,
    prepare_collision_plan,
)
from tzrec.utils.sid.knn_collision import KnnCollisionResolver
from tzrec.utils.test_util import parameterized_name_func


def _plan(codebook, capacity, item_ids, codes, prior=None):
    return prepare_collision_plan(
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(codebook, capacity),
        prior=prior,
    )


def _prior(keys, counts):
    return PriorOccupancy(
        np.asarray(keys, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    )


class KnnCollisionResolverTest(unittest.TestCase):
    def test_golden_candidate_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in SID (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        plan = _plan((2, 4), 2, item_ids, codes)
        candidates = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 2]], dtype=np.int64)
        result = KnnCollisionResolver().resolve(plan, candidates)
        origin_grouping = build_origin_item_grouping(plan)
        resolved_grouping = build_resolved_item_grouping(plan, result)

        np.testing.assert_array_equal(plan.overflow_rows, [1, 3, 8])
        np.testing.assert_array_equal(plan.overflow_item_ids, [1, 3, 8])
        np.testing.assert_array_equal(
            plan.origin_sid_key_indices, [0, 0, 0, 0, 1, 2, 2, 3, 3, 3]
        )
        np.testing.assert_array_equal(plan.sid_keys, [0, 1, 2, 4])

        np.testing.assert_array_equal(
            result.resolved_last_codes, [0, 1, 0, 3, 1, 2, 2, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices, [2, 2, 1, 1, 1, 2, 1, 2, 1, 1]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result.final_item_counts, [2, 2, 2, 1, 2, 1])
        self.assertTrue(result.grouping_collected)
        self.assertEqual(result.stats.raw_colliding_sid_count, 2)
        self.assertEqual(result.stats.final_colliding_sid_count, 0)
        self.assertEqual(result.stats.relocated_count, 3)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_items_per_sid, 2)

        np.testing.assert_array_equal(origin_grouping.sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(origin_grouping.item_counts, [4, 1, 2, 3])
        np.testing.assert_array_equal(origin_grouping.offsets, [0, 4, 5, 7, 10])
        np.testing.assert_array_equal(
            origin_grouping.row_order, [2, 0, 1, 3, 4, 6, 5, 9, 7, 8]
        )

        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(resolved_grouping.item_counts, [2, 2, 2, 1, 2, 1])
        np.testing.assert_array_equal(
            resolved_grouping.row_order, [2, 0, 4, 1, 6, 5, 3, 9, 7, 8]
        )

    def test_resolution_is_deterministic_and_order_independent(self) -> None:
        rng = np.random.default_rng(0)
        item_ids = np.arange(200, dtype=np.int64)
        codes = np.column_stack(
            (rng.integers(0, 3, item_ids.size), rng.integers(0, 2, item_ids.size))
        )
        codebook = (8, 8)

        def decisions(order):
            ordered_item_ids = item_ids[order]
            ordered_codes = codes[order]
            plan = _plan(codebook, 2, ordered_item_ids, ordered_codes)
            candidates = np.tile(
                np.arange(codebook[-1], dtype=np.int64),
                (plan.overflow_rows.size, 1),
            )
            result = KnnCollisionResolver().resolve(plan, candidates)
            resolved_codes = ordered_codes.copy()
            resolved_codes[:, -1] = result.resolved_last_codes

            def grouped_item_ids(grouping):
                return {
                    int(sid_key): tuple(
                        ordered_item_ids[
                            grouping.row_order[
                                grouping.offsets[index] : grouping.offsets[index + 1]
                            ]
                        ].tolist()
                    )
                    for index, sid_key in enumerate(grouping.sid_keys)
                }

            return (
                dict(
                    zip(ordered_item_ids.tolist(), map(tuple, resolved_codes.tolist()))
                ),
                grouped_item_ids(build_origin_item_grouping(plan)),
                grouped_item_ids(build_resolved_item_grouping(plan, result)),
                result.stats,
            )

        base_order = np.arange(item_ids.size)
        expected = decisions(base_order)
        self.assertEqual(decisions(base_order), expected)
        self.assertEqual(decisions(rng.permutation(item_ids.size)), expected)

    def test_keep_origin_over_capacity(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])
        result = KnnCollisionResolver().resolve(plan, np.asarray([[0]], dtype=np.int64))

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(result.slot_indices, [1, 2])
        np.testing.assert_array_equal(result.unresolved_rows, [1])
        self.assertEqual(result.stats.final_colliding_sid_count, 1)
        self.assertEqual(result.stats.max_final_items_per_sid, 2)
        self.assertEqual(result.stats.unresolved_count, 1)
        resolved_grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0])
        np.testing.assert_array_equal(resolved_grouping.item_counts, [2])
        np.testing.assert_array_equal(resolved_grouping.row_order, [0, 1])

    def test_no_overflow(self) -> None:
        plan = _plan((2, 4), 2, [0, 1, 2], [[0, 0], [0, 0], [1, 2]])
        result = KnnCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 6])
        np.testing.assert_array_equal(result.final_item_counts, [2, 1])
        self.assertEqual(result.stats.raw_colliding_sid_count, 0)
        self.assertEqual(result.stats.relocated_count, 0)
        self.assertEqual(result.stats.max_final_items_per_sid, 2)

    @parameterized.expand(
        [
            (
                "one_dimensional",
                np.asarray([1], dtype=np.int64),
                ValueError,
                "must be 2-D",
            ),
            (
                "misaligned_rows",
                np.empty((0, 1), dtype=np.int64),
                ValueError,
                "row-aligned",
            ),
            (
                "below_range",
                np.asarray([[-1]], dtype=np.int64),
                ValueError,
                r"must be in \[0, 4\)",
            ),
            (
                "at_upper_bound",
                np.asarray([[4]], dtype=np.int64),
                ValueError,
                r"must be in \[0, 4\)",
            ),
            (
                "non_integer",
                np.asarray([[2.0]]),
                TypeError,
                "must use an integer dtype",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_candidates(
        self, _case_name, candidates, exception_type, message
    ) -> None:
        plan = _plan((4,), 1, [0, 1, 2], [[0], [0], [1]])

        with self.assertRaisesRegex(exception_type, message):
            KnnCollisionResolver().resolve(plan, candidates)

    def test_three_layer_sparse_prefix_relocation(self) -> None:
        plan = _plan(
            (2, 3, 4),
            2,
            [0, 1, 2, 3, 4],
            [[0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0]],
        )
        np.testing.assert_array_equal(plan.sid_keys, [0, 8, 12])
        np.testing.assert_array_equal(plan.new_item_counts, [1, 3, 1])
        np.testing.assert_array_equal(plan.overflow_prefix_ids, [2])

        result = KnnCollisionResolver().resolve(
            plan, np.asarray([[1, 2, 3]], dtype=np.int64)
        )

        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(int(result.resolved_last_codes.sum()), 1)
        self.assertEqual(result.resolved_last_codes[3], 0)
        self.assertEqual(result.resolved_last_codes[4], 0)
        np.testing.assert_array_equal(result.final_sid_keys, [0, 8, 9, 12])
        np.testing.assert_array_equal(result.final_item_counts, [1, 2, 1, 1])
        grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(grouping.sid_keys, result.final_sid_keys)
        np.testing.assert_array_equal(grouping.item_counts, result.final_item_counts)

    def test_prior_over_capacity_never_reissues_a_published_slot(self) -> None:
        # A previous run left three items on key 0 at capacity 1 (unresolved
        # overflow). The new row cannot relocate, so it must land at slot 4.
        plan = _plan((2, 4), 1, [10], [[0, 0]], prior=_prior([0], [3]))
        result = KnnCollisionResolver().resolve(plan, np.asarray([[0]], dtype=np.int64))
        self.assertEqual(result.stats.unresolved_count, 1)
        self.assertEqual(int(result.slot_indices[0]), 4)

    def test_relocation_sees_prior_only_sids_in_the_prefix(self) -> None:
        # Key 0 is full, and key 1 -- in the same prefix but holding no new row --
        # is already full too. The only free candidate is key 2.
        plan = _plan((1, 4), 1, [10], [[0, 0]], prior=_prior([0, 1], [1, 1]))
        result = KnnCollisionResolver().resolve(
            plan, np.asarray([[1, 2]], dtype=np.int64)
        )
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(int(result.resolved_last_codes[0]), 2)
        self.assertEqual(int(result.slot_indices[0]), 1)
        np.testing.assert_array_equal(
            build_resolved_item_grouping(plan, result).sid_keys, [2]
        )


if __name__ == "__main__":
    unittest.main()
