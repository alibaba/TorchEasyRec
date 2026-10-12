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
    KnnCollisionResolver,
    PriorOccupancy,
    build_origin_item_grouping,
    build_resolved_item_grouping,
    concat_ranges,
    prepare_collision_plan,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver
from tzrec.utils.test_util import parameterized_name_func


def _plan(layer_sizes, capacity, item_ids, codes, prior=None):
    return prepare_collision_plan(
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
        prior=prior,
    )


def _prior(keys, counts):
    return PriorOccupancy(
        np.asarray(keys, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    )


def _assert_same_assignments(test_case, actual, expected):
    np.testing.assert_array_equal(
        actual.resolved_last_codes, expected.resolved_last_codes
    )
    np.testing.assert_array_equal(actual.slot_indices, expected.slot_indices)
    np.testing.assert_array_equal(actual.unresolved_rows, expected.unresolved_rows)
    test_case.assertEqual(actual.stats, expected.stats)


class CollisionTest(unittest.TestCase):
    def test_resolvers_reject_invalid_progress_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress_interval must be >= 1"):
            KnnCollisionResolver(progress_interval=0)

    def test_golden_candidate_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in bucket (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
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
            plan.origin_bucket_indices, [0, 0, 0, 0, 1, 2, 2, 3, 3, 3]
        )
        np.testing.assert_array_equal(plan.bucket_keys, [0, 1, 2, 4])

        np.testing.assert_array_equal(
            result.resolved_last_codes, [0, 1, 0, 3, 1, 2, 2, 0, 1, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices, [2, 2, 1, 1, 1, 2, 1, 2, 1, 1]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 2, 2, 1, 2, 1])
        self.assertTrue(result.grouping_collected)
        self.assertEqual(result.stats.raw_collision_buckets, 2)
        self.assertEqual(result.stats.final_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 3)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_bucket_size, 2)

        np.testing.assert_array_equal(origin_grouping.sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(origin_grouping.counts, [4, 1, 2, 3])
        np.testing.assert_array_equal(origin_grouping.offsets, [0, 4, 5, 7, 10])
        np.testing.assert_array_equal(
            origin_grouping.row_order, [2, 0, 1, 3, 4, 6, 5, 9, 7, 8]
        )

        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(resolved_grouping.counts, [2, 2, 2, 1, 2, 1])
        np.testing.assert_array_equal(
            resolved_grouping.row_order, [2, 0, 4, 1, 6, 5, 3, 9, 7, 8]
        )

    def test_resolution_is_deterministic_and_order_independent(self) -> None:
        rng = np.random.default_rng(0)
        item_ids = np.arange(200, dtype=np.int64)
        codes = np.column_stack(
            (rng.integers(0, 3, item_ids.size), rng.integers(0, 2, item_ids.size))
        )
        layer_sizes = (8, 8)

        def decisions(order):
            ordered_item_ids = item_ids[order]
            ordered_codes = codes[order]
            plan = _plan(layer_sizes, 2, ordered_item_ids, ordered_codes)
            candidates = np.tile(
                np.arange(layer_sizes[-1], dtype=np.int64),
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
        self.assertEqual(result.stats.final_collision_buckets, 1)
        self.assertEqual(result.stats.max_final_bucket_size, 2)
        self.assertEqual(result.stats.unresolved_count, 1)
        resolved_grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(resolved_grouping.sid_keys, [0])
        np.testing.assert_array_equal(resolved_grouping.counts, [2])
        np.testing.assert_array_equal(resolved_grouping.row_order, [0, 1])

    def test_no_overflow(self) -> None:
        plan = _plan((2, 4), 2, [0, 1, 2], [[0, 0], [0, 0], [1, 2]])
        result = KnnCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 6])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1])
        self.assertEqual(result.stats.raw_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 0)
        self.assertEqual(result.stats.max_final_bucket_size, 2)

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
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_last_codes are required"):
            resolver_cls().resolve(plan)

    def test_empty_groupings(self) -> None:
        plan = _plan((2, 4), 2, [], np.empty((0, 2)))
        result = KnnCollisionResolver().resolve(plan)

        for name, grouping in (
            ("origin", build_origin_item_grouping(plan)),
            ("resolved", build_resolved_item_grouping(plan, result)),
        ):
            with self.subTest(grouping=name):
                np.testing.assert_array_equal(grouping.sid_keys, [])
                np.testing.assert_array_equal(grouping.counts, [])
                np.testing.assert_array_equal(grouping.row_order, [])
                np.testing.assert_array_equal(grouping.offsets, [0])

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

    @parameterized.expand(
        [
            (
                f"{resolver_name}_{case_name}",
                resolver_cls,
                codes,
                candidates if takes_candidates else None,
                stats,
            )
            for resolver_name, resolver_cls, takes_candidates in (
                ("candidate", KnnCollisionResolver, True),
                ("iterative", IterativeCollisionResolver, True),
                ("uniform", UniformCollisionResolver, False),
            )
            for case_name, codes, candidates, stats in (
                ("no_overflow", [[0], [1]], None, (2, 0, 0, 0, 0, 1)),
                (
                    "full_prefix",
                    [[0], [0], [0]],
                    np.asarray([[1], [1]]),
                    (3, 1, 1, 1, 1, 2),
                ),
            )
        ],
        name_func=parameterized_name_func,
    )
    def test_collect_grouping_false_changes_only_bucket_metadata(
        self, _case_name, resolver_cls, codes, candidates, stats
    ) -> None:
        plan = _plan((2,), 1, range(len(codes)), codes)
        resolver = resolver_cls()

        grouped = resolver.resolve(plan, candidates)
        rate_only = resolver.resolve(plan, candidates, collect_grouping=False)

        self.assertEqual(dataclasses.astuple(grouped.stats), stats)
        _assert_same_assignments(self, rate_only, grouped)
        self.assertTrue(grouped.grouping_collected)
        self.assertFalse(rate_only.grouping_collected)
        np.testing.assert_array_equal(rate_only.final_bucket_keys, [])
        np.testing.assert_array_equal(rate_only.final_bucket_counts, [])

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
            _plan((2, 4), 2, [0, 1], codes)

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
        self, _case_name, layer_sizes, capacity, message
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            CollisionResolutionConfig(layer_sizes, capacity)

    def test_prefix_key_near_int64_limit(self) -> None:
        layer_size = 3_037_000_499
        plan = _plan(
            (layer_size, layer_size),
            1,
            [0],
            [[layer_size - 1, layer_size - 1]],
        )

        np.testing.assert_array_equal(plan.bucket_keys, [layer_size**2 - 1])

    def test_three_layer_sparse_prefix_relocation(self) -> None:
        plan = _plan(
            (2, 3, 4),
            2,
            [0, 1, 2, 3, 4],
            [[0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0]],
        )
        np.testing.assert_array_equal(plan.bucket_keys, [0, 8, 12])
        np.testing.assert_array_equal(plan.bucket_counts, [1, 3, 1])
        np.testing.assert_array_equal(plan.overflow_prefix_ids, [2])

        result = KnnCollisionResolver().resolve(
            plan, np.asarray([[1, 2, 3]], dtype=np.int64)
        )

        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(int(result.resolved_last_codes.sum()), 1)
        self.assertEqual(result.resolved_last_codes[3], 0)
        self.assertEqual(result.resolved_last_codes[4], 0)
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 8, 9, 12])
        np.testing.assert_array_equal(result.final_bucket_counts, [1, 2, 1, 1])
        grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(grouping.sid_keys, result.final_bucket_keys)
        np.testing.assert_array_equal(grouping.counts, result.final_bucket_counts)


class ConcatRangesTest(unittest.TestCase):
    def test_concatenates_ranges_and_skips_empty_ones(self) -> None:
        np.testing.assert_array_equal(
            concat_ranges(np.asarray([5, 10, 20]), np.asarray([2, 0, 3])),
            [5, 6, 20, 21, 22],
        )


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
        self, _case_name, layer_sizes, codes, prefix_ids, bucket_keys, offset_codes
    ) -> None:
        config = CollisionResolutionConfig(layer_sizes, 1)
        codes = np.asarray(codes, dtype=np.int64)

        np.testing.assert_array_equal(config.prefix_ids(codes), prefix_ids)
        np.testing.assert_array_equal(config.bucket_keys(codes), bucket_keys)
        np.testing.assert_array_equal(config.offset_codes(codes), offset_codes)


class PriorOccupancyTest(unittest.TestCase):
    def test_counts_for_returns_zero_for_absent_buckets(self) -> None:
        prior = _prior([2, 5, 9], [1, 4, 2])
        np.testing.assert_array_equal(
            prior.counts_for(np.asarray([9, 0, 5, 100, 2])), [2, 0, 4, 0, 1]
        )

    def test_restrict_to_prefixes_keeps_only_requested_prefixes(self) -> None:
        # last_size 4 -> prefix 0 spans keys 0..3, prefix 2 spans keys 8..11.
        prior = _prior([0, 3, 5, 9, 11, 12], [1, 2, 3, 4, 5, 6])
        restricted = prior.restrict_to_prefixes(np.asarray([0, 2]), 4)
        np.testing.assert_array_equal(restricted.bucket_keys, [0, 3, 9, 11])
        np.testing.assert_array_equal(restricted.bucket_counts, [1, 2, 4, 5])
        self.assertTrue(prior.restrict_to_prefixes(np.asarray([7]), 4).is_empty)


class AppendPlanTest(unittest.TestCase):
    def test_prior_consumes_capacity_and_continues_slots(self) -> None:
        # Bucket (0, 0) is key 0 and already holds one published item, so with
        # capacity 2 only one of the two new rows can be retained there.
        plan = _plan((2, 4), 2, [10, 11], [[0, 0], [0, 0]], prior=_prior([0], [1]))
        np.testing.assert_array_equal(plan.prior_bucket_counts, [1])
        self.assertEqual(plan.overflow_rows.size, 1)
        retained = int(np.setdiff1d(np.arange(2), plan.overflow_rows)[0])
        # The retained row continues the published slot sequence at 2, not 1.
        self.assertEqual(int(plan.initial_slot_indices[retained]), 2)

    def test_prior_over_capacity_never_reissues_a_published_slot(self) -> None:
        # A previous run left three items on key 0 at capacity 1 (unresolved
        # overflow). The new row cannot relocate, so it must land at slot 4.
        plan = _plan((2, 4), 1, [10], [[0, 0]], prior=_prior([0], [3]))
        result = KnnCollisionResolver().resolve(plan, np.asarray([[0]], dtype=np.int64))
        self.assertEqual(result.stats.unresolved_count, 1)
        self.assertEqual(int(result.slot_indices[0]), 4)

    def test_relocation_sees_prior_only_buckets_in_the_prefix(self) -> None:
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

    def test_final_counts_are_corpus_totals(self) -> None:
        plan = _plan((2, 4), 3, [10], [[0, 0]], prior=_prior([0], [2]))
        result = KnnCollisionResolver().resolve(plan, None)
        self.assertEqual(plan.overflow_rows.size, 0)
        np.testing.assert_array_equal(result.final_bucket_keys, [0])
        np.testing.assert_array_equal(result.final_bucket_counts, [3])
        self.assertEqual(result.stats.max_final_bucket_size, 3)

    def test_groupings_cover_only_the_appended_rows(self) -> None:
        prior = _prior([0], [1])
        plan = _plan((2, 4), 3, [10, 11], [[0, 0], [0, 1]], prior=prior)
        result = KnnCollisionResolver().resolve(plan, None)

        resolved = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(np.sort(resolved.row_order), [0, 1])
        self.assertEqual(int(resolved.counts.sum()), 2)
        # Key 0 already held one item, so the delta grouping keeps one row there
        # and the local slot restarts at 1 even though the global slot is 2.
        keys = CollisionResolutionConfig((2, 4), 1).bucket_keys(
            np.asarray([[0, 0], [0, 1]])
        )
        np.testing.assert_array_equal(resolved.sid_keys, np.sort(keys))
        self.assertEqual(int(result.slot_indices[0]), 2)

        origin = build_origin_item_grouping(plan)
        np.testing.assert_array_equal(np.sort(origin.row_order), [0, 1])
        self.assertEqual(int(origin.counts.sum()), 2)

    def test_rejects_prior_outside_the_key_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the key space"):
            _plan((2, 4), 2, [10], [[0, 0]], prior=_prior([8], [1]))


if __name__ == "__main__":
    unittest.main()
