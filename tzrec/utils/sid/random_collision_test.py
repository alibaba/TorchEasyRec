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

from tzrec.utils.sid.random_collision import RandomCollisionResolver
from tzrec.utils.sid.test_util import create_collision_plan
from tzrec.utils.test_util import parameterized_name_func


class RandomCollisionResolverTest(unittest.TestCase):
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
        plan = create_collision_plan((4,), 1, [0, 1, 2], [[0], [0], [3]])

        result = RandomCollisionResolver(num_candidates=3).resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 2, 3])
        np.testing.assert_array_equal(result.slot_indices, [1, 1, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_sid_keys, [0, 2, 3])
        np.testing.assert_array_equal(result.final_item_counts, [1, 1, 1])
        self.assertEqual(result.stats.raw_colliding_sid_count, 1)
        self.assertEqual(result.stats.final_colliding_sid_count, 0)
        self.assertEqual(result.stats.relocated_count, 1)

    def test_random_rejects_external_candidates(self) -> None:
        plan = create_collision_plan((2,), 1, [0], [[0]])

        with self.assertRaisesRegex(ValueError, "does not accept candidate_last_codes"):
            RandomCollisionResolver(num_candidates=1).resolve(
                plan, np.empty((0, 1), dtype=np.int64)
            )

    def test_random_no_overflow_supports_single_code_space(self) -> None:
        plan = create_collision_plan((1,), 1, [0], [[0]])

        result = RandomCollisionResolver(num_candidates=1).resolve(
            plan, collect_sid_metadata=False
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [0])
        np.testing.assert_array_equal(result.final_sid_keys, [])
        self.assertFalse(result.sid_metadata_collected)

    def test_random_candidate_exhaustion_keeps_original(self) -> None:
        plan = create_collision_plan((2,), 1, [0, 1], [[0], [0]])

        # The overflow item's single deterministic draw is its origin code.
        resolver = RandomCollisionResolver(num_candidates=1)
        grouped = resolver.resolve(plan)
        rate_only = resolver.resolve(plan, collect_sid_metadata=False)

        np.testing.assert_array_equal(grouped.resolved_last_codes, [0, 0])
        np.testing.assert_array_equal(grouped.unresolved_rows, [1])
        self.assertEqual(grouped.stats.final_colliding_sid_count, 1)
        self.assertEqual(grouped.stats.max_final_items_per_sid, 2)
        np.testing.assert_array_equal(
            rate_only.resolved_last_codes, grouped.resolved_last_codes
        )
        np.testing.assert_array_equal(rate_only.slot_indices, grouped.slot_indices)
        self.assertEqual(rate_only.stats, grouped.stats)
        self.assertFalse(rate_only.sid_metadata_collected)

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
        one_code_plan = create_collision_plan((1,), 1, [0, 1], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "last_size >= 2"):
            RandomCollisionResolver(num_candidates=1).resolve(one_code_plan)


if __name__ == "__main__":
    unittest.main()
