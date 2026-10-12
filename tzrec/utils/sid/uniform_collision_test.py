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

import collections
import dataclasses
import unittest

import numpy as np

from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    PriorOccupancy,
    build_resolved_item_grouping,
    prepare_collision_plan,
    stable_order_hash,
)
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver


def _plan(layer_sizes, capacity, item_ids, codes, prior=None):
    return prepare_collision_plan(
        np.asarray(item_ids),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
        prior=prior,
    )


def _prior(keys, counts):
    return PriorOccupancy(
        np.asarray(keys, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    )


def _rotation(prefix_id, last_size):
    return int(stable_order_hash(np.asarray([prefix_id], dtype=np.int64))[0]) % int(
        last_size
    )


def _reference(plan, codes, item_ids):
    """Place overflow rows one at a time on the least-loaded rotated code.

    Returns the code and slot of every overflow row and the final load of every
    occupied bucket, published-only buckets included.
    """
    last_size = plan.config.layer_sizes[-1]
    prefixes = plan.config.prefix_ids(np.asarray(codes))
    moving = set(plan.overflow_rows.tolist())
    loads = {}
    for row, prefix in enumerate(prefixes.tolist()):
        if row not in moving:
            key = prefix * last_size + int(codes[row][-1])
            loads[key] = loads.get(key, 0) + 1
    for key, count in zip(
        plan.prior.bucket_keys.tolist(), plan.prior.bucket_counts.tolist()
    ):
        loads[key] = loads.get(key, 0) + count
    hashes = stable_order_hash(np.asarray(item_ids)).tolist()
    codes_out, slots_out = {}, {}
    for row in sorted(moving, key=lambda r: (int(prefixes[r]), hashes[r], r)):
        prefix = int(prefixes[row])
        rotation = _rotation(prefix, last_size)
        code = min(
            range(last_size),
            key=lambda c: (
                loads.get(prefix * last_size + c, 0),
                (c - rotation) % last_size,
            ),
        )
        key = prefix * last_size + code
        loads[key] = loads.get(key, 0) + 1
        codes_out[row] = code
        slots_out[row] = loads[key]
    return codes_out, slots_out, loads


class UniformCollisionResolverTest(unittest.TestCase):
    def test_rejects_candidate_last_codes(self) -> None:
        plan = _plan((4,), 1, [10, 11], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "does not accept candidate_last_codes"):
            UniformCollisionResolver().resolve(plan, np.asarray([[1]]))

    def test_random_plans_match_least_loaded_reference(self) -> None:
        rng = np.random.default_rng(7)
        for trial in range(150):
            last_size = rng.integers(1, 9)
            layer_sizes = (rng.integers(1, 4), last_size)
            capacity = rng.integers(1, 3)
            item_count = rng.integers(1, 40)
            codes = np.stack(
                [
                    np.minimum(rng.geometric(0.4, item_count) - 1, size - 1)
                    for size in layer_sizes
                ],
                axis=1,
            )
            prior = None
            if trial % 3 == 0:
                keys = np.unique(rng.integers(0, layer_sizes[0] * last_size, 4))
                prior = _prior(keys, rng.integers(1, capacity + 3, keys.shape[0]))
            item_ids = np.arange(item_count) * 7 + 1000
            plan = _plan(layer_sizes, capacity, item_ids, codes, prior=prior)

            result = UniformCollisionResolver().resolve(plan)
            expected_codes, expected_slots, loads = _reference(plan, codes, item_ids)
            origin_keys = plan.config.bucket_keys(codes).tolist()
            published = dict(
                zip(plan.prior.bucket_keys.tolist(), plan.prior.bucket_counts.tolist())
            )
            overflow_prefixes = {
                origin_keys[row] // last_size for row in plan.overflow_rows.tolist()
            }
            final = {
                key: count
                for key, count in loads.items()
                if key in origin_keys or key // last_size in overflow_prefixes
            }
            unresolved = sum(slot > capacity for slot in expected_slots.values())

            with self.subTest(trial=trial):
                for row in plan.overflow_rows.tolist():
                    self.assertEqual(
                        int(result.resolved_last_codes[row]), expected_codes[row]
                    )
                    self.assertEqual(int(result.slot_indices[row]), expected_slots[row])
                self.assertEqual(
                    dict(
                        zip(
                            result.final_bucket_keys.tolist(),
                            result.final_bucket_counts.tolist(),
                        )
                    ),
                    final,
                )
                self.assertEqual(
                    dataclasses.astuple(result.stats),
                    (
                        item_count,
                        sum(
                            published.get(key, 0) + count > capacity
                            for key, count in collections.Counter(origin_keys).items()
                        ),
                        sum(count > capacity for count in final.values()),
                        plan.overflow_rows.shape[0] - unresolved,
                        unresolved,
                        max(final.values()),
                    ),
                )
                build_resolved_item_grouping(plan, result)


if __name__ == "__main__":
    unittest.main()
