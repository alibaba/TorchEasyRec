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
from parameterized import parameterized

from tzrec.utils.sid import collision
from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    KnnCollisionResolver,
    RandomCollisionResolver,
    build_original_item_grouping,
    build_resolved_item_grouping,
    prepare_collision_plan,
)
from tzrec.utils.test_util import parameterized_name_func


def _plan(layer_sizes, capacity, item_ids, codes):
    return prepare_collision_plan(
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
    )


def _assert_same_assignments(test_case, actual, expected):
    np.testing.assert_array_equal(
        actual.resolved_last_codes, expected.resolved_last_codes
    )
    np.testing.assert_array_equal(actual.slot_indices, expected.slot_indices)
    np.testing.assert_array_equal(actual.unresolved_rows, expected.unresolved_rows)
    test_case.assertEqual(actual.stats, expected.stats)


class CollisionTest(unittest.TestCase):
    def test_resolution_chunking_preserves_greedy_results(self) -> None:
        plan = _plan((4,), 1, range(8), [[0]] * 8)
        candidates = np.tile(np.asarray([1, 2, 3]), (7, 1))

        with mock.patch.object(collision, "_ROW_CHUNK_SIZE", 100):
            expected = KnnCollisionResolver().resolve(plan, candidates)
        with mock.patch.object(collision, "_ROW_CHUNK_SIZE", 2):
            actual = KnnCollisionResolver().resolve(plan, candidates)

        self.assertEqual(plan.overflow_rows.size, 7)
        self.assertEqual(actual.stats.relocated_count, 3)
        self.assertEqual(actual.stats.unresolved_count, 4)
        _assert_same_assignments(self, actual, expected)
        np.testing.assert_array_equal(
            actual.final_bucket_keys, expected.final_bucket_keys
        )
        np.testing.assert_array_equal(
            actual.final_bucket_counts, expected.final_bucket_counts
        )
        self.assertEqual(actual.grouping_collected, expected.grouping_collected)
        actual_grouping = build_resolved_item_grouping(plan, actual)
        expected_grouping = build_resolved_item_grouping(plan, expected)
        np.testing.assert_array_equal(
            actual_grouping.sid_keys, expected_grouping.sid_keys
        )
        np.testing.assert_array_equal(actual_grouping.counts, expected_grouping.counts)
        np.testing.assert_array_equal(
            actual_grouping.row_order, expected_grouping.row_order
        )

    def test_resolution_loop_reports_sample_progress(self) -> None:
        plan = _plan(
            (5,),
            1,
            [0, 1, 2, 3, 4],
            [[0], [0], [0], [0], [0]],
        )
        candidates = np.asarray([[1], [2], [3], [4]], dtype=np.int64)
        with mock.patch.object(collision, "ProgressLogger") as progress_cls:
            KnnCollisionResolver(progress_interval=3).resolve(plan, candidates)

        progress_cls.assert_called_once_with("Resolving collision overflow", start_n=0)
        self.assertEqual(
            progress_cls.return_value.log.call_args_list,
            [mock.call(3, suffix="3 samples processed")],
        )

    def test_resolvers_reject_invalid_progress_interval(self) -> None:
        with self.assertRaisesRegex(ValueError, "progress_interval must be >= 1"):
            KnnCollisionResolver(progress_interval=0)
        with self.assertRaisesRegex(ValueError, "progress_interval must be >= 1"):
            RandomCollisionResolver(num_candidates=1, progress_interval=0)

    def test_golden_candidate_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in bucket (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        with mock.patch.object(collision, "_ROW_CHUNK_SIZE", 2):
            plan = _plan((2, 4), 2, item_ids, codes)
            candidates = np.asarray([[0, 1, 2], [1, 2, 3], [0, 1, 2]], dtype=np.int64)
            result = KnnCollisionResolver().resolve(plan, candidates)
            original_grouping = build_original_item_grouping(plan)
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
        self.assertEqual(result.stats.total_items, 10)
        self.assertEqual(result.stats.raw_collision_buckets, 2)
        self.assertEqual(result.stats.final_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 3)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_bucket_size, 2)

        np.testing.assert_array_equal(original_grouping.sid_keys, [0, 1, 2, 4])
        np.testing.assert_array_equal(original_grouping.counts, [4, 1, 2, 3])
        np.testing.assert_array_equal(original_grouping.offsets, [0, 4, 5, 7, 10])
        np.testing.assert_array_equal(
            original_grouping.row_order, [2, 0, 1, 3, 4, 6, 5, 9, 7, 8]
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
                grouped_item_ids(build_original_item_grouping(plan)),
                grouped_item_ids(build_resolved_item_grouping(plan, result)),
                result.stats,
            )

        base_order = np.arange(item_ids.size)
        expected = decisions(base_order)
        self.assertEqual(decisions(base_order), expected)
        self.assertEqual(decisions(rng.permutation(item_ids.size)), expected)

    def test_keep_original_over_capacity(self) -> None:
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
        resolver = KnnCollisionResolver()
        result = resolver.resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 6])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1])
        self.assertEqual(result.stats.raw_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 0)

        without_grouping = resolver.resolve(plan, collect_grouping=False)
        np.testing.assert_array_equal(without_grouping.final_bucket_keys, [])
        np.testing.assert_array_equal(without_grouping.final_bucket_counts, [])
        self.assertFalse(without_grouping.grouping_collected)
        self.assertEqual(without_grouping.stats, result.stats)

    def test_knn_requires_candidates_for_overflow(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_codes are required"):
            KnnCollisionResolver().resolve(plan)

    def test_empty_groupings(self) -> None:
        plan = _plan((2, 4), 2, [], np.empty((0, 2)))
        result = KnnCollisionResolver().resolve(plan)

        for name, grouping in (
            ("original", build_original_item_grouping(plan)),
            ("resolved", build_resolved_item_grouping(plan, result)),
        ):
            with self.subTest(grouping=name):
                np.testing.assert_array_equal(grouping.sid_keys, [])
                np.testing.assert_array_equal(grouping.counts, [])
                np.testing.assert_array_equal(grouping.row_order, [])
                np.testing.assert_array_equal(grouping.offsets, [0])

    def test_random_candidate_golden_draws(self) -> None:
        item_ids = np.asarray([0, 1], dtype=np.int64)
        actual = RandomCollisionResolver(
            num_candidates=3
        )._generate_candidate_last_codes(item_ids, last_size=4)
        capped = RandomCollisionResolver(
            num_candidates=10
        )._generate_candidate_last_codes(item_ids, last_size=4)

        expected = [[1, 2, 0], [2, 0, 2]]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(capped, expected)

    def test_random_resolution_golden(self) -> None:
        plan = _plan((4,), 1, [0, 1, 2], [[0], [0], [3]])

        result = RandomCollisionResolver(num_candidates=3).resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 2, 3])
        np.testing.assert_array_equal(result.slot_indices, [1, 1, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 2, 3])
        np.testing.assert_array_equal(result.final_bucket_counts, [1, 1, 1])
        self.assertEqual(result.stats.raw_collision_buckets, 1)
        self.assertEqual(result.stats.final_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 1)

    def test_random_rejects_external_candidates(self) -> None:
        plan = _plan((2,), 1, [0], [[0]])

        with self.assertRaisesRegex(ValueError, "does not accept candidate_codes"):
            RandomCollisionResolver(num_candidates=1).resolve(
                plan, np.empty((0, 1), dtype=np.int64)
            )

    def test_random_no_overflow_supports_single_code_space(self) -> None:
        plan = _plan((1,), 1, [0], [[0]])

        result = RandomCollisionResolver(num_candidates=1).resolve(
            plan, collect_grouping=False
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [0])
        np.testing.assert_array_equal(result.final_bucket_keys, [])
        self.assertFalse(result.grouping_collected)

    def test_random_candidate_exhaustion_keeps_original(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        # The overflow item's single deterministic draw is its origin code.
        resolver = RandomCollisionResolver(num_candidates=1)
        grouped = resolver.resolve(plan)
        rate_only = resolver.resolve(plan, collect_grouping=False)

        np.testing.assert_array_equal(grouped.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(grouped.unresolved_rows, [1])
        self.assertEqual(grouped.stats.final_collision_buckets, 1)
        self.assertEqual(grouped.stats.max_final_bucket_size, 2)
        _assert_same_assignments(self, rate_only, grouped)
        self.assertFalse(rate_only.grouping_collected)

    @parameterized.expand(
        [("zero", 0), ("negative", -1)],
        name_func=parameterized_name_func,
    )
    def test_random_rejects_invalid_num_candidates(
        self, _case_name, num_candidates
    ) -> None:
        with self.assertRaisesRegex(ValueError, "num_candidates must be >= 1"):
            RandomCollisionResolver(num_candidates=num_candidates)

    def test_random_rejects_single_code_space(self) -> None:
        one_code_plan = _plan((1,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "last_size >= 2"):
            RandomCollisionResolver(num_candidates=1).resolve(one_code_plan)

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

    def test_fixed_width_candidates_can_contend_for_free_slots(self) -> None:
        plan = _plan((6,), 1, [0, 1, 2, 3, 4, 5], [[0], [0], [1], [3], [4], [4]])
        np.testing.assert_array_equal(plan.overflow_origin_last_codes, [0, 4])
        candidates = np.asarray([[0, 1, 2, 5], [4, 2, 1, 3]], dtype=np.int64)

        result = KnnCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(result.unresolved_rows, [5])
        self.assertEqual(result.resolved_last_codes[1], 2)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 1)

    def test_resolution_can_skip_grouping_metadata(self) -> None:
        plan = _plan((4,), 1, [0, 1, 2], [[0], [0], [1]])
        candidates = np.asarray([[2]], dtype=np.int64)
        resolver = KnnCollisionResolver()
        expected = resolver.resolve(plan, candidates)

        result = resolver.resolve(plan, candidates, collect_grouping=False)

        _assert_same_assignments(self, result, expected)
        np.testing.assert_array_equal(result.final_bucket_keys, [])
        np.testing.assert_array_equal(result.final_bucket_counts, [])
        self.assertFalse(result.grouping_collected)

        with self.assertRaisesRegex(RuntimeError, "metadata was not collected"):
            build_resolved_item_grouping(plan, result)

        regrouped = resolver.resolve(plan, candidates)
        self.assertTrue(regrouped.grouping_collected)
        _assert_same_assignments(self, regrouped, expected)

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

    def test_rejects_codebook_exceeding_int64(self) -> None:
        with self.assertRaisesRegex(ValueError, "int64"):
            CollisionResolutionConfig((2**32, 2**32), 1)

    def test_band_key_near_int64_limit(self) -> None:
        layer_size = 3_037_000_499
        plan = _plan(
            (layer_size, layer_size),
            1,
            [0],
            [[layer_size - 1, layer_size - 1]],
        )

        np.testing.assert_array_equal(plan.bucket_keys, [layer_size**2 - 1])

    def test_grouping_includes_untouched_bands(self) -> None:
        plan = _plan(
            (3, 4), 2, [0, 1, 2, 3, 4], [[0, 0], [0, 0], [0, 0], [1, 0], [2, 1]]
        )
        candidates = np.asarray([[1, 2, 3]], dtype=np.int64)

        result = KnnCollisionResolver().resolve(plan, candidates)

        # Bands 1 (key 4) and 2 (key 9) hold no overflow row; they must still
        # appear in the grouping output with their original counts, and the
        # relocation must add band-0 key 1.
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 4, 9])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1, 1, 1])
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)

        rate_only = KnnCollisionResolver().resolve(
            plan, candidates, collect_grouping=False
        )
        self.assertEqual(rate_only.stats, result.stats)

    def test_three_layer_sparse_band_relocation(self) -> None:
        plan = _plan(
            (2, 3, 4),
            2,
            [0, 1, 2, 3, 4],
            [[0, 2, 0], [0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0]],
        )
        np.testing.assert_array_equal(plan.bucket_keys, [0, 8, 12])
        np.testing.assert_array_equal(plan.bucket_counts, [1, 3, 1])
        np.testing.assert_array_equal(plan.overflow_bucket_key_prefixes, [8])

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

    def test_rate_only_matches_grouping_with_unplaceable(self) -> None:
        # One unplaceable overflow (its only candidate equals the origin) keeps
        # bucket (0, 0) over capacity, alongside an untouched band (1, 0). The
        # rate-only branch must report the same collision stats as the grouping
        # branch -- those numbers are exactly what --rate_only exists to compute.
        plan = _plan((2, 2), 1, [0, 1, 2], [[0, 0], [0, 0], [1, 0]])
        candidates = np.asarray([[0]], dtype=np.int64)  # only candidate == origin

        grouped = KnnCollisionResolver().resolve(plan, candidates)
        rate_only = KnnCollisionResolver().resolve(
            plan, candidates, collect_grouping=False
        )

        self.assertEqual(grouped.stats.unresolved_count, 1)
        self.assertEqual(grouped.stats.final_collision_buckets, 1)
        self.assertEqual(grouped.stats.max_final_bucket_size, 2)
        _assert_same_assignments(self, rate_only, grouped)
        self.assertFalse(rate_only.grouping_collected)


if __name__ == "__main__":
    unittest.main()
