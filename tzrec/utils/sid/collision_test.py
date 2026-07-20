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

from tzrec.utils.sid import collision
from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    CollisionResolver,
    KnnCollisionResolver,
    RandomCollisionResolver,
    build_original_item_grouping,
    build_resolved_item_grouping,
    prepare_collision_plan,
)


def _plan(layer_sizes, capacity, item_ids, codes):
    return prepare_collision_plan(
        np.asarray(item_ids, dtype=np.int64),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
    )


class CollisionTest(unittest.TestCase):
    def test_resolver_is_abstract(self) -> None:
        with self.assertRaises(TypeError):
            CollisionResolver()

    def test_golden_candidate_resolution(self) -> None:
        item_ids = np.arange(10, dtype=np.int64)
        # 4 items in bucket (0,0), 1 in (0,1), 2 in (0,2), 3 in (1,0).
        codes = np.asarray(
            [[0, 0]] * 4 + [[0, 1]] + [[0, 2]] * 2 + [[1, 0]] * 3, dtype=np.int64
        )
        with mock.patch.object(collision, "_GROUPING_ROW_CHUNK", 2):
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
        result = KnnCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 2, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 6])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1])
        self.assertEqual(result.stats.raw_collision_buckets, 0)
        self.assertEqual(result.stats.relocated_count, 0)

    def test_knn_requires_candidates_for_overflow(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_codes are required"):
            KnnCollisionResolver().resolve(plan)

    def test_empty_groupings(self) -> None:
        plan = _plan((2, 4), 2, [], np.empty((0, 2)))
        result = KnnCollisionResolver().resolve(plan)

        for grouping in (
            build_original_item_grouping(plan),
            build_resolved_item_grouping(plan, result),
        ):
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

    def test_random_candidate_generator_rejects_single_code_space(self) -> None:
        resolver = RandomCollisionResolver(num_candidates=1)

        with self.assertRaisesRegex(ValueError, "last_size >= 2"):
            resolver._generate_candidate_last_codes(
                np.asarray([0], dtype=np.int64), last_size=1
            )

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

    def test_random_no_overflow_skips_candidate_generation(self) -> None:
        plan = _plan((2,), 1, [0], [[0]])
        resolver = RandomCollisionResolver(num_candidates=1)
        with mock.patch.object(resolver, "_generate_candidate_last_codes") as generate:
            result = resolver.resolve(plan, collect_grouping=False)

        generate.assert_not_called()
        np.testing.assert_array_equal(result.resolved_last_codes, [0])
        np.testing.assert_array_equal(result.final_bucket_keys, [])
        self.assertFalse(result.grouping_collected)

    def test_random_candidate_exhaustion_keeps_original(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        # The overflow item's single deterministic draw is its origin code.
        resolver = RandomCollisionResolver(num_candidates=1)
        grouped = resolver.resolve(plan)
        rate_only = resolver.resolve(plan, collect_grouping=False)
        regrouped = resolver.resolve(plan)

        np.testing.assert_array_equal(grouped.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(grouped.unresolved_rows, [1])
        self.assertEqual(grouped.stats.final_collision_buckets, 1)
        self.assertEqual(grouped.stats.max_final_bucket_size, 2)
        self.assertEqual(rate_only.stats, grouped.stats)
        self.assertFalse(rate_only.grouping_collected)
        self.assertTrue(regrouped.grouping_collected)
        self.assertEqual(regrouped.stats, grouped.stats)
        for result in (rate_only, regrouped):
            np.testing.assert_array_equal(
                result.resolved_last_codes, grouped.resolved_last_codes
            )
            np.testing.assert_array_equal(result.slot_indices, grouped.slot_indices)
            np.testing.assert_array_equal(
                result.unresolved_rows, grouped.unresolved_rows
            )

    def test_random_candidate_validation(self) -> None:
        one_code_plan = _plan((1,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "num_candidates must be >= 1"):
            RandomCollisionResolver(num_candidates=0)
        with self.assertRaisesRegex(ValueError, "num_candidates must be >= 1"):
            RandomCollisionResolver(num_candidates=-1)
        with self.assertRaisesRegex(ValueError, "last_size >= 2"):
            RandomCollisionResolver(num_candidates=1).resolve(one_code_plan)

    def test_candidate_matrix_must_be_two_dimensional(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "must be 2-D"):
            KnnCollisionResolver().resolve(plan, np.asarray([1], dtype=np.int64))

    def test_candidate_matrix_must_align_with_overflow_rows(self) -> None:
        plan = _plan((2,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "row-aligned"):
            KnnCollisionResolver().resolve(plan, np.empty((0, 1), dtype=np.int64))

    def test_fixed_width_candidates_can_contend_for_free_slots(self) -> None:
        plan = _plan((6,), 1, [0, 1, 2, 3, 4, 5], [[0], [0], [1], [3], [4], [4]])
        np.testing.assert_array_equal(plan.overflow_origin_last_codes, [0, 4])
        candidates = np.asarray([[0, 1, 2, 5], [4, 2, 1, 3]], dtype=np.int64)

        result = KnnCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(result.unresolved_rows, [5])
        self.assertEqual(result.resolved_last_codes[1], 2)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 1)

    def test_rejects_out_of_range_candidate(self) -> None:
        plan = _plan((4,), 1, [0, 1, 2], [[0], [0], [1]])

        with self.assertRaisesRegex(ValueError, r"must be in \[0, 4\)"):
            KnnCollisionResolver().resolve(plan, np.asarray([[5]], dtype=np.int64))

    def test_resolution_can_skip_grouping_metadata(self) -> None:
        plan = _plan((4,), 1, [0, 1, 2], [[0], [0], [1]])
        candidates = np.asarray([[2]], dtype=np.int64)
        resolver = KnnCollisionResolver()
        expected = resolver.resolve(plan, candidates)

        result = resolver.resolve(plan, candidates, collect_grouping=False)

        np.testing.assert_array_equal(
            result.resolved_last_codes, expected.resolved_last_codes
        )
        np.testing.assert_array_equal(result.slot_indices, expected.slot_indices)
        np.testing.assert_array_equal(result.unresolved_rows, expected.unresolved_rows)
        np.testing.assert_array_equal(result.final_bucket_keys, [])
        np.testing.assert_array_equal(result.final_bucket_counts, [])
        self.assertFalse(result.grouping_collected)
        self.assertEqual(result.stats, expected.stats)

        with self.assertRaisesRegex(RuntimeError, "metadata was not collected"):
            build_resolved_item_grouping(plan, result)

        regrouped = resolver.resolve(plan, candidates)
        self.assertTrue(regrouped.grouping_collected)
        self.assertEqual(regrouped.stats, expected.stats)
        np.testing.assert_array_equal(
            regrouped.resolved_last_codes, expected.resolved_last_codes
        )
        np.testing.assert_array_equal(regrouped.slot_indices, expected.slot_indices)
        np.testing.assert_array_equal(
            regrouped.unresolved_rows, expected.unresolved_rows
        )

    def test_rejects_out_of_range_codes(self) -> None:
        for bad in ([[0, 0], [0, 4]], [[0, 0], [2, 0]], [[0, 0], [-1, 0]]):
            with self.assertRaisesRegex(ValueError, "out-of-range"):
                _plan((2, 4), 2, [0, 1], bad)

    def test_rejects_codebook_exceeding_int64(self) -> None:
        with self.assertRaisesRegex(ValueError, "int64"):
            CollisionResolutionConfig((2**32, 2**32), 1)

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

    def test_three_layer_band_fold_and_relocation(self) -> None:
        # Band = (code_0, code_1) via a mixed-radix fold with middle radix 3.
        # Items 0-2 share band (0, 0); item 3 sits at (0, 2) and item 4 at
        # (1, 0) -- distinct bands (raw 2 vs 3). A too-small middle radix would
        # fold both to 2 and merge them; dropping the middle layer would merge
        # item 3 into band (0, 0). Either regression changes the partition below.
        plan = _plan(
            (2, 3, 4),
            2,
            [0, 1, 2, 3, 4],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 2, 0], [1, 0, 0]],
        )
        # Band (0, 0) holds exactly 3 items (not 4): the middle code kept item 3
        # in its own bucket (key 4) distinct from items 0-2 (key 0).
        np.testing.assert_array_equal(plan.bucket_keys, [0, 4, 8])
        np.testing.assert_array_equal(plan.bucket_counts, [3, 1, 1])

        result = KnnCollisionResolver().resolve(
            plan, np.asarray([[1, 2, 3]], dtype=np.int64)
        )

        # The single overflow item relocates within band (0, 0) -- only its last
        # code changes -- and the two non-overflow bands stay put.
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(int(result.resolved_last_codes.sum()), 1)
        self.assertEqual(result.resolved_last_codes[3], 0)
        self.assertEqual(result.resolved_last_codes[4], 0)
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 4, 8])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1, 1, 1])

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
        self.assertEqual(rate_only.stats, grouped.stats)
        self.assertFalse(rate_only.grouping_collected)


if __name__ == "__main__":
    unittest.main()
