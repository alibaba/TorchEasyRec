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

from tzrec.utils.sid.collision import (
    build_resolved_sid_item_row_index,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.sid.knn_collision import KnnCollisionResolver
from tzrec.utils.sid.test_util import (
    create_collision_plan,
    create_existing_sid_occupancy,
)


class IterativeCollisionResolverTest(unittest.TestCase):
    def test_arbitration_prefers_earlier_candidate_positions(self) -> None:
        plan = create_collision_plan(
            (5,),
            1,
            [10, 11, 20, 21, 30],
            [[0], [0], [1], [1], [2]],
        )
        np.testing.assert_array_equal(
            plan.origin_last_codes[plan.overflow_rows], [0, 1]
        )
        candidates = np.asarray([[2, 3, 4], [3, 2, 2]], dtype=np.int64)

        first_fit = KnnCollisionResolver().resolve(plan, candidates)
        iterative = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(
            iterative.resolved_last_codes[plan.overflow_rows], [4, 3]
        )
        np.testing.assert_array_equal(
            first_fit.resolved_last_codes[plan.overflow_rows], [3, 1]
        )
        self.assertEqual(iterative.stats.relocated_count, 2)
        self.assertEqual(iterative.stats.unresolved_count, 0)
        self.assertEqual(first_fit.stats.relocated_count, 1)
        self.assertEqual(first_fit.stats.unresolved_count, 1)

    def test_origin_cap_limits_same_origin_requests(self) -> None:
        plan = create_collision_plan((4,), 1, [0, 10, 20, 30], [[1], [1], [1], [1]])
        np.testing.assert_array_equal(plan.overflow_rows, [0, 1, 2])
        candidates = np.asarray([[1, 0], [3, 1], [0, 2]], dtype=np.uint64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        # Row 2 cannot request position 0 behind row 1, so code 0 stays
        # free for row 0, whose only other candidate is its origin.
        np.testing.assert_array_equal(result.resolved_last_codes, [0, 3, 2, 1])
        self.assertEqual(result.stats.unresolved_count, 0)

    def test_uncommitted_acceptance_reopens_next_round(self) -> None:
        plan = create_collision_plan((3,), 1, [0, 10, 20], [[0], [0], [0]])
        np.testing.assert_array_equal(plan.overflow_rows, [1, 2])
        candidates = np.asarray([[2, 1], [0, 1]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 2, 1])
        self.assertEqual(result.stats.unresolved_count, 0)

    def test_slots_follow_round_then_candidate_position(self) -> None:
        plan = create_collision_plan(
            (3,), 2, [0, 10, 20, 30, 40, 50], [[0], [1], [0], [0], [0], [0]]
        )
        np.testing.assert_array_equal(plan.overflow_rows, [0, 4, 2])
        candidates = np.asarray([[1, 1], [1, 2], [2, 1]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(result.resolved_last_codes, [1, 1, 2, 0, 2, 0])
        np.testing.assert_array_equal(result.slot_indices, [2, 1, 2, 2, 1, 1])

    def test_candidate_width_is_consumed_without_truncation(self) -> None:
        plan = create_collision_plan(
            (256,),
            1,
            [10],
            [[0]],
            existing_sid_occupancy=create_existing_sid_occupancy(
                np.arange(200), np.ones(200)
            ),
        )
        candidates = np.arange(1, 201, dtype=np.int64)[None, :]

        result = IterativeCollisionResolver().resolve(plan, candidates)

        self.assertEqual(int(result.resolved_last_codes[0]), 200)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)

    def test_exhausted_candidates_keep_origin_and_dense_slots(self) -> None:
        plan = create_collision_plan((3,), 1, [10, 11, 12, 13], [[0], [0], [0], [0]])
        candidates = np.asarray([[1], [2], [0]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(
            result.resolved_last_codes[plan.overflow_rows], [1, 2, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices[plan.overflow_rows], [1, 1, 2]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [plan.overflow_rows[2]])
        self.assertEqual(result.stats.relocated_count, 2)
        np.testing.assert_array_equal(result.final_item_counts, [2, 1, 1])
        sid_item_row_index = build_resolved_sid_item_row_index(plan, result)
        np.testing.assert_array_equal(
            np.sort(sid_item_row_index.item_row_indices), np.arange(4)
        )

    def test_assignments_are_deterministic_under_input_reordering(self) -> None:
        item_ids = np.asarray([f"item-{index}" for index in range(12)])
        codes = (np.arange(item_ids.size, dtype=np.int64) % 3)[:, None]

        def assignments(order):
            ordered_ids = item_ids[order]
            plan = create_collision_plan((4,), 2, ordered_ids, codes[order])
            candidates = np.tile(
                np.asarray([1, 2, 3, 1], dtype=np.int64),
                (plan.overflow_rows.size, 1),
            )
            result = IterativeCollisionResolver().resolve(plan, candidates)
            unresolved = set(result.unresolved_rows.tolist())
            return {
                item_id: (
                    int(result.resolved_last_codes[row]),
                    int(result.slot_indices[row]),
                    row in unresolved,
                )
                for row, item_id in enumerate(ordered_ids.tolist())
            }

        expected = assignments(np.arange(item_ids.size))
        actual = assignments(np.random.default_rng(7).permutation(item_ids.size))

        self.assertEqual(actual, expected)

    def test_append_preserves_prior_over_capacity_and_moves_only_new_rows(self) -> None:
        plan = create_collision_plan(
            (4,),
            2,
            [10, 11],
            [[0], [0]],
            existing_sid_occupancy=create_existing_sid_occupancy([0, 1], [3, 1]),
        )
        candidates = np.asarray([[1, 2], [1, 2]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        self.assertEqual(plan.overflow_rows.size, 2)
        self.assertEqual(result.stats.relocated_count, 2)
        self.assertEqual(result.stats.unresolved_count, 0)
        np.testing.assert_array_equal(result.final_sid_keys, [0, 1, 2])
        np.testing.assert_array_equal(result.final_item_counts, [3, 2, 1])
        self.assertEqual(result.stats.max_final_items_per_sid, 3)
        np.testing.assert_array_equal(
            np.sort(result.slot_indices[plan.overflow_rows]), [1, 2]
        )

    def test_append_multi_prefix_preserves_prior_only_destinations(self) -> None:
        plan = create_collision_plan(
            (2, 4),
            1,
            [10, 11, 20, 21],
            [[0, 0], [0, 0], [1, 0], [1, 0]],
            existing_sid_occupancy=create_existing_sid_occupancy([1, 6], [1, 1]),
        )
        candidates = np.asarray([[1, 2], [2, 1]], dtype=np.int32)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(candidates, [[1, 2], [2, 1]])
        np.testing.assert_array_equal(
            result.resolved_last_codes[plan.overflow_rows], [2, 1]
        )
        np.testing.assert_array_equal(result.final_sid_keys, [0, 1, 2, 4, 5, 6])
        np.testing.assert_array_equal(result.final_item_counts, [1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result.slot_indices[plan.overflow_rows], [1, 1])

    def test_random_plans_keep_placement_invariants(self) -> None:
        rng = np.random.default_rng(7)
        for trial in range(100):
            last_size = rng.integers(2, 9)
            codebook = (rng.integers(1, 4), last_size)
            capacity = rng.integers(1, 3)
            item_count = rng.integers(1, 60)
            codes = np.stack(
                [rng.integers(0, size, item_count) for size in codebook], axis=1
            )
            plan = create_collision_plan(
                codebook, capacity, np.arange(item_count), codes
            )
            candidates = rng.integers(
                0, last_size, (plan.overflow_rows.size, rng.integers(1, 5))
            )

            result = IterativeCollisionResolver().resolve(plan, candidates)

            sids = codes.copy()
            sids[:, -1] = result.resolved_last_codes
            sids, sid_index_of_row, counts = np.unique(
                sids, axis=0, return_inverse=True, return_counts=True
            )
            sid_index_of_row = sid_index_of_row.reshape(-1)
            final_count = dict(zip(map(tuple, sids.tolist()), counts.tolist()))
            is_unresolved = np.zeros(item_count, dtype=bool)
            is_unresolved[result.unresolved_rows] = True
            with self.subTest(trial=trial):
                for row, candidate_row in zip(
                    plan.overflow_rows.tolist(), candidates.tolist()
                ):
                    origin = int(codes[row, -1])
                    code = int(result.resolved_last_codes[row])
                    if is_unresolved[row]:
                        self.assertEqual(code, origin)
                        for candidate in candidate_row:
                            key = (*codes[row, :-1].tolist(), candidate)
                            self.assertGreaterEqual(final_count.get(key, 0), capacity)
                    else:
                        self.assertNotEqual(code, origin)
                        self.assertIn(code, candidate_row)
                for sid in range(sids.shape[0]):
                    slots = np.sort(result.slot_indices[sid_index_of_row == sid])
                    np.testing.assert_array_equal(slots, np.arange(1, slots.size + 1))
                    unresolved_here = np.count_nonzero(
                        is_unresolved[sid_index_of_row == sid]
                    )
                    self.assertLessEqual(counts[sid] - unresolved_here, capacity)


if __name__ == "__main__":
    unittest.main()
