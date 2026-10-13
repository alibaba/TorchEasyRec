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
    PriorOccupancy,
    build_origin_sid_item_row_index,
    build_resolved_sid_item_row_index,
    prepare_collision_plan,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.sid.knn_collision import KnnCollisionResolver
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver
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


def _assert_same_assignments(test_case, actual, expected):
    np.testing.assert_array_equal(
        actual.resolved_last_codes, expected.resolved_last_codes
    )
    np.testing.assert_array_equal(actual.slot_indices, expected.slot_indices)
    np.testing.assert_array_equal(actual.unresolved_rows, expected.unresolved_rows)
    test_case.assertEqual(actual.stats, expected.stats)


class CollisionTest(unittest.TestCase):
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

    def test_empty_sid_item_row_indices(self) -> None:
        plan = _plan((2, 4), 2, [], np.empty((0, 2)))
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
    def test_collect_sid_metadata_false_changes_only_sid_metadata(
        self, _case_name, resolver_cls, codes, candidates, stats
    ) -> None:
        plan = _plan((2,), 1, range(len(codes)), codes)
        resolver = resolver_cls()

        grouped = resolver.resolve(plan, candidates)
        rate_only = resolver.resolve(plan, candidates, collect_sid_metadata=False)

        self.assertEqual(dataclasses.astuple(grouped.stats), stats)
        _assert_same_assignments(self, rate_only, grouped)
        self.assertTrue(grouped.sid_metadata_collected)
        self.assertFalse(rate_only.sid_metadata_collected)
        np.testing.assert_array_equal(rate_only.final_sid_keys, [])
        np.testing.assert_array_equal(rate_only.final_item_counts, [])

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
        self, _case_name, codebook, capacity, message
    ) -> None:
        with self.assertRaisesRegex(ValueError, message):
            CollisionResolutionConfig(codebook, capacity)

    def test_prefix_key_near_int64_limit(self) -> None:
        codebook_size = 3_037_000_499
        plan = _plan(
            (codebook_size, codebook_size),
            1,
            [0],
            [[codebook_size - 1, codebook_size - 1]],
        )

        np.testing.assert_array_equal(plan.sid_keys, [codebook_size**2 - 1])


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


class PriorOccupancyTest(unittest.TestCase):
    def test_counts_for_returns_zero_for_absent_sids(self) -> None:
        prior = _prior([2, 5, 9], [1, 4, 2])
        np.testing.assert_array_equal(
            prior.counts_for(np.asarray([9, 0, 5, 100, 2])), [2, 0, 4, 0, 1]
        )

    def test_restrict_to_prefixes_keeps_only_requested_prefixes(self) -> None:
        # last_size 4 -> prefix 0 spans keys 0..3, prefix 2 spans keys 8..11.
        prior = _prior([0, 3, 5, 9, 11, 12], [1, 2, 3, 4, 5, 6])
        restricted = prior.restrict_to_prefixes(np.asarray([0, 2]), 4)
        np.testing.assert_array_equal(restricted.sid_keys, [0, 3, 9, 11])
        np.testing.assert_array_equal(restricted.item_counts, [1, 2, 4, 5])
        self.assertTrue(prior.restrict_to_prefixes(np.asarray([7]), 4).is_empty)


class AppendPlanTest(unittest.TestCase):
    def test_prior_consumes_capacity_and_continues_slots(self) -> None:
        # SID (0, 0) is key 0 and already holds one published item, so with
        # capacity 2 only one of the two new rows can be retained there.
        plan = _plan((2, 4), 2, [10, 11], [[0, 0], [0, 0]], prior=_prior([0], [1]))
        np.testing.assert_array_equal(plan.existing_item_counts, [1])
        self.assertEqual(plan.overflow_rows.size, 1)
        retained = int(np.setdiff1d(np.arange(2), plan.overflow_rows)[0])
        # The retained row continues the published slot sequence at 2, not 1.
        self.assertEqual(int(plan.initial_slot_indices[retained]), 2)

    def test_final_counts_are_corpus_totals(self) -> None:
        plan = _plan((2, 4), 3, [10], [[0, 0]], prior=_prior([0], [2]))
        result = KnnCollisionResolver().resolve(plan, None)
        self.assertEqual(plan.overflow_rows.size, 0)
        np.testing.assert_array_equal(result.final_sid_keys, [0])
        np.testing.assert_array_equal(result.final_item_counts, [3])
        self.assertEqual(result.stats.max_final_items_per_sid, 3)

    def test_sid_item_row_indices_cover_only_the_appended_rows(self) -> None:
        prior = _prior([0], [1])
        plan = _plan((2, 4), 3, [10, 11], [[0, 0], [0, 1]], prior=prior)
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

    def test_rejects_prior_outside_the_key_space(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the key space"):
            _plan((2, 4), 2, [10], [[0, 0]], prior=_prior([8], [1]))


if __name__ == "__main__":
    unittest.main()
