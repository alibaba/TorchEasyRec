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
    CollisionResolutionConfig,
    build_origin_sid_item_row_index,
    build_resolved_sid_item_row_index,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.sid.knn_collision import KnnCollisionResolver
from tzrec.utils.sid.test_util import (
    create_collision_plan,
    create_existing_sid_state,
)
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver
from tzrec.utils.test_util import parameterized_name_func


def _assert_same_assignments(test_case, actual, expected):
    np.testing.assert_array_equal(
        actual.resolved_last_codes, expected.resolved_last_codes
    )
    np.testing.assert_array_equal(actual.slot_indices, expected.slot_indices)
    np.testing.assert_array_equal(actual.unresolved_rows, expected.unresolved_rows)
    test_case.assertEqual(actual.stats, expected.stats)


class CollisionResolutionConfigTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("empty_layers", (), 1, "at least one layer"),
            ("non_positive_size", (4, 0), 1, "must be positive"),
            ("zero_capacity", (4,), 0, "capacity must be >= 1"),
            ("int64_overflow", (2**32, 2**32), 1, "int64"),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_config(
        self, _case_name, codebook, capacity, message
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            CollisionResolutionConfig(codebook, capacity)


class SidLayoutTest(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "three_layers",
                (2, 3, 4),
                [[1, 2, 3], [0, 1, 0]],
                [5, 1],
                [23, 4],
                [[1, 4, 8], [0, 3, 5]],
            ),
            ("single_layer", (5,), [[3], [0]], [0, 0], [3, 0], [[3], [0]]),
        ],
        name_func=parameterized_name_func,
    )
    def test_key_layout(
        self, _case_name, codebook, codes, prefix_ids, sid_keys, offset_codes
    ) -> None:
        config = CollisionResolutionConfig(codebook, 1)
        codes = np.asarray(codes, dtype=np.int64)

        np.testing.assert_array_equal(config.prefix_ids(codes), prefix_ids)
        np.testing.assert_array_equal(config.convert_codes_to_sid_keys(codes), sid_keys)
        np.testing.assert_array_equal(config.offset_codes(codes), offset_codes)

    def test_sid_key_near_int64_limit(self) -> None:
        codebook = (2**21 - 1, 2**21, 2**21)
        plan = create_collision_plan(
            codebook,
            1,
            [0],
            [[size - 1 for size in codebook]],
        )

        np.testing.assert_array_equal(plan.sid_keys, [(2**21 - 1) * 2**42 - 1])


class ExistingSidStateTest(unittest.TestCase):
    def test_counts_for_returns_zero_for_absent_sids(self) -> None:
        existing_sid_state = create_existing_sid_state([2, 5, 9], [1, 4, 2])
        np.testing.assert_array_equal(
            existing_sid_state.counts_for(np.asarray([9, 0, 5, 100, 2])),
            [2, 0, 4, 0, 1],
        )

    def test_restrict_to_prefixes_keeps_only_requested_prefixes(self) -> None:
        # last_size 4 -> prefix 0 spans keys 0..3, prefix 2 spans keys 8..11.
        existing_sid_state = create_existing_sid_state(
            [0, 3, 5, 9, 11, 12], [1, 2, 3, 4, 5, 6]
        )
        restricted = existing_sid_state.restrict_to_prefixes(np.asarray([0, 2]), 4)
        np.testing.assert_array_equal(restricted.sid_keys, [0, 3, 9, 11])
        np.testing.assert_array_equal(restricted.item_counts, [1, 2, 4, 5])
        self.assertTrue(
            existing_sid_state.restrict_to_prefixes(np.asarray([7]), 4).is_empty
        )


class CollisionPlanTest(unittest.TestCase):
    def test_overflow_rows_follow_sid_and_item_order(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in SID (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        plan = create_collision_plan((2, 4), 2, item_ids, codes)

        np.testing.assert_array_equal(plan.overflow_rows, [1, 3, 8])
        np.testing.assert_array_equal(plan.overflow_item_ids, [1, 3, 8])
        np.testing.assert_array_equal(
            plan.origin_sid_key_indices, [0, 0, 0, 0, 1, 2, 2, 3, 3, 3]
        )
        np.testing.assert_array_equal(plan.sid_keys, [0, 1, 2, 4])

    @parameterized.expand(
        [
            ("last_layer_too_large", [[0, 0], [0, 4]]),
            ("prefix_layer_too_large", [[0, 0], [2, 0]]),
            ("negative_code", [[0, 0], [-1, 0]]),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_out_of_range_codes(self, _case_name, codes) -> None:
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            create_collision_plan((2, 4), 2, [0, 1], codes)


class AppendPlanTest(unittest.TestCase):
    def test_existing_state_consumes_capacity_and_continues_slots(self) -> None:
        # SID (0, 0) is key 0 and already holds one published item, so with
        # capacity 2 only one of the two new rows can be retained there.
        plan = create_collision_plan(
            (2, 4),
            2,
            [10, 11],
            [[0, 0], [0, 0]],
            existing_sid_state=create_existing_sid_state([0], [1]),
        )
        np.testing.assert_array_equal(plan.existing_item_counts, [1])
        self.assertEqual(plan.overflow_rows.size, 1)
        retained = int(np.setdiff1d(np.arange(2), plan.overflow_rows)[0])
        # The retained row continues the published slot sequence at 2, not 1.
        self.assertEqual(int(plan.initial_slot_indices[retained]), 2)

    def test_rejects_existing_state_outside_the_key_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the key space"):
            create_collision_plan(
                (2, 4),
                2,
                [10],
                [[0, 0]],
                existing_sid_state=create_existing_sid_state([8], [1]),
            )


class CollisionResolverTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("knn", KnnCollisionResolver),
            ("iterative", IterativeCollisionResolver),
        ],
        name_func=parameterized_name_func,
    )
    def test_resolvers_require_candidates_for_overflow(
        self, _name, resolver_cls
    ) -> None:
        plan = create_collision_plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_last_codes are required"):
            resolver_cls().resolve(plan)

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
        plan = create_collision_plan((4,), 1, [0, 1, 2], [[0], [0], [1]])

        with self.assertRaisesRegex(exception_type, message):
            KnnCollisionResolver().resolve(plan, candidates)

    def test_no_overflow(self) -> None:
        plan = create_collision_plan((2, 4), 2, [0, 1, 2], [[0, 0], [0, 0], [1, 2]])
        resolver = KnnCollisionResolver()
        result = resolver.resolve(plan)
        rate_only = resolver.resolve(plan, collect_sid_metadata=False)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 6])
        np.testing.assert_array_equal(result.final_item_counts, [2, 1])
        self.assertEqual(dataclasses.astuple(result.stats), (3, 0, 0, 0, 0, 2))
        _assert_same_assignments(self, rate_only, result)
        self.assertTrue(result.sid_metadata_collected)
        self.assertFalse(rate_only.sid_metadata_collected)
        np.testing.assert_array_equal(rate_only.final_sid_keys, [])
        np.testing.assert_array_equal(rate_only.final_item_counts, [])

    @parameterized.expand(
        [
            ("candidate", KnnCollisionResolver, np.asarray([[1], [1]])),
            ("iterative", IterativeCollisionResolver, np.asarray([[1], [1]])),
            ("uniform", UniformCollisionResolver, None),
        ],
        name_func=parameterized_name_func,
    )
    def test_collect_sid_metadata_false_changes_only_sid_metadata(
        self, _name, resolver_cls, candidates
    ) -> None:
        plan = create_collision_plan((2,), 1, [0, 1, 2], [[0], [0], [0]])
        resolver = resolver_cls()
        result = resolver.resolve(plan, candidates)
        rate_only = resolver.resolve(plan, candidates, collect_sid_metadata=False)

        self.assertEqual(dataclasses.astuple(result.stats), (3, 1, 1, 1, 1, 2))
        _assert_same_assignments(self, rate_only, result)
        self.assertTrue(result.sid_metadata_collected)
        self.assertFalse(rate_only.sid_metadata_collected)
        np.testing.assert_array_equal(rate_only.final_sid_keys, [])
        np.testing.assert_array_equal(rate_only.final_item_counts, [])

    def test_final_counts_are_corpus_totals(self) -> None:
        plan = create_collision_plan(
            (2, 4),
            3,
            [10],
            [[0, 0]],
            existing_sid_state=create_existing_sid_state([0], [2]),
        )
        result = KnnCollisionResolver().resolve(plan, None)
        self.assertEqual(plan.overflow_rows.size, 0)
        np.testing.assert_array_equal(result.final_sid_keys, [0])
        np.testing.assert_array_equal(result.final_item_counts, [3])
        self.assertEqual(result.stats.max_final_items_per_sid, 3)


class SidItemRowIndexTest(unittest.TestCase):
    def test_empty_sid_item_row_indices(self) -> None:
        plan = create_collision_plan((2, 4), 2, [], np.empty((0, 2)))
        result = KnnCollisionResolver().resolve(plan)

        for name, sid_item_row_index in (
            ("origin", build_origin_sid_item_row_index(plan)),
            ("resolved", build_resolved_sid_item_row_index(plan, result)),
        ):
            with self.subTest(sid_item_row_index=name):
                np.testing.assert_array_equal(sid_item_row_index.sid_keys, [])
                np.testing.assert_array_equal(sid_item_row_index.item_counts, [])
                np.testing.assert_array_equal(sid_item_row_index.item_row_indices, [])
                np.testing.assert_array_equal(sid_item_row_index.offsets, [0])

    def test_origin_and_resolved_sid_item_row_indices(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in SID (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        plan = create_collision_plan((2, 4), 2, item_ids, codes)
        candidates = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 2]], dtype=np.int64)
        result = KnnCollisionResolver().resolve(plan, candidates)
        origin_sid_item_row_index = build_origin_sid_item_row_index(plan)
        resolved_sid_item_row_index = build_resolved_sid_item_row_index(plan, result)

        np.testing.assert_array_equal(origin_sid_item_row_index.sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(
            origin_sid_item_row_index.item_counts, [4, 1, 2, 3]
        )
        np.testing.assert_array_equal(
            origin_sid_item_row_index.offsets, [0, 4, 5, 7, 10]
        )
        np.testing.assert_array_equal(
            origin_sid_item_row_index.item_row_indices, [2, 0, 1, 3, 4, 6, 5, 9, 7, 8]
        )

        np.testing.assert_array_equal(
            resolved_sid_item_row_index.sid_keys, [0, 1, 2, 3, 4, 5]
        )
        np.testing.assert_array_equal(
            resolved_sid_item_row_index.item_counts, [2, 2, 2, 1, 2, 1]
        )
        np.testing.assert_array_equal(
            resolved_sid_item_row_index.item_row_indices, [2, 0, 4, 1, 6, 5, 3, 9, 7, 8]
        )

    def test_sid_item_row_indices_cover_only_the_appended_rows(self) -> None:
        existing_sid_state = create_existing_sid_state([0], [1])
        plan = create_collision_plan(
            (2, 4), 3, [10, 11], [[0, 0], [0, 1]], existing_sid_state=existing_sid_state
        )
        result = KnnCollisionResolver().resolve(plan, None)

        resolved = build_resolved_sid_item_row_index(plan, result)
        np.testing.assert_array_equal(np.sort(resolved.item_row_indices), [0, 1])
        self.assertEqual(int(resolved.item_counts.sum()), 2)
        # Key 0 already held one item, so the delta grouping keeps one row there
        # and the local slot restarts at 1 even though the global slot is 2.
        keys = CollisionResolutionConfig((2, 4), 1).convert_codes_to_sid_keys(
            np.asarray([[0, 0], [0, 1]])
        )
        np.testing.assert_array_equal(resolved.sid_keys, np.sort(keys))
        self.assertEqual(int(result.slot_indices[0]), 2)

        origin = build_origin_sid_item_row_index(plan)
        np.testing.assert_array_equal(np.sort(origin.item_row_indices), [0, 1])
        self.assertEqual(int(origin.item_counts.sum()), 2)


if __name__ == "__main__":
    unittest.main()
