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

import os
import unittest
from unittest import mock

from parameterized import parameterized

from tzrec.utils import dynamicemb_util


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
class StorageFormulaTest(unittest.TestCase):
    """Mode-aware ``_calculate_dynamicemb_table_storage_specific_size``."""

    ROWS = 1024
    DIM = 64
    ELEMENT_SIZE = 4
    BUCKET_CAPACITY = 128

    def _calc(self, *, cache_ratio, is_hbm, caching, only_values=False):
        return dynamicemb_util._calculate_dynamicemb_table_storage_specific_size(
            size=[self.ROWS, self.DIM],
            element_size=self.ELEMENT_SIZE,
            cache_ratio=cache_ratio,
            is_hbm=is_hbm,
            only_values=only_values,
            bucket_capacity=self.BUCKET_CAPACITY,
            caching=caching,
        )

    @parameterized.expand(
        [
            ("ratio_0_0", 0.0),
            ("ratio_0_25", 0.25),
            ("ratio_0_5", 0.5),
            ("ratio_0_75", 0.75),
            ("ratio_1_0", 1.0),
        ]
    )
    def test_hbm_identical_between_modes(self, _name, cache_ratio):
        # HBM accounting is the same in HYBRID and CACHING: HBM holds a
        # cache_ratio fraction of values plus full-row-count metadata.
        hybrid_hbm = self._calc(cache_ratio=cache_ratio, is_hbm=True, caching=False)
        caching_hbm = self._calc(cache_ratio=cache_ratio, is_hbm=True, caching=True)
        self.assertEqual(hybrid_hbm, caching_hbm)

    @parameterized.expand(
        [
            ("ratio_0_0", 0.0),
            ("ratio_0_25", 0.25),
            ("ratio_0_5", 0.5),
            ("ratio_0_75", 0.75),
            ("ratio_1_0", 1.0),
        ]
    )
    def test_ddr_hybrid_complements_cache(self, _name, cache_ratio):
        # HYBRID DDR = (1 - cache_ratio) * full-table DDR.
        full_ddr = self._calc(cache_ratio=0.0, is_hbm=False, caching=False)
        hybrid_ddr = self._calc(cache_ratio=cache_ratio, is_hbm=False, caching=False)
        self.assertEqual(hybrid_ddr, round((1.0 - cache_ratio) * full_ddr))

    @parameterized.expand(
        [
            ("ratio_0_0", 0.0),
            ("ratio_0_25", 0.25),
            ("ratio_0_5", 0.5),
            ("ratio_0_75", 0.75),
            ("ratio_1_0", 1.0),
        ]
    )
    def test_ddr_caching_holds_full_table(self, _name, cache_ratio):
        # CACHING DDR is the full backing store, independent of cache_ratio.
        full_ddr = self._calc(cache_ratio=0.0, is_hbm=False, caching=False)
        caching_ddr = self._calc(cache_ratio=cache_ratio, is_hbm=False, caching=True)
        self.assertEqual(caching_ddr, full_ddr)

    def test_caching_ddr_strictly_greater_than_hybrid_when_cached(self):
        for cache_ratio in (0.1, 0.5, 0.9):
            hybrid_ddr = self._calc(
                cache_ratio=cache_ratio, is_hbm=False, caching=False
            )
            caching_ddr = self._calc(
                cache_ratio=cache_ratio, is_hbm=False, caching=True
            )
            self.assertGreater(caching_ddr, hybrid_ddr)

    def test_only_values_drops_metadata(self):
        # only_values=True strips HBM metadata regardless of mode.
        for caching in (False, True):
            with_meta = self._calc(
                cache_ratio=0.5, is_hbm=True, caching=caching, only_values=False
            )
            without_meta = self._calc(
                cache_ratio=0.5, is_hbm=True, caching=caching, only_values=True
            )
            self.assertGreater(with_meta, without_meta)


class EffectiveCacheRatioTest(unittest.TestCase):
    """``_dynamicemb_effective_cache_ratio`` — HYBRID vs CACHING perf ratio."""

    def test_hybrid_passes_through(self):
        for x in (0.0, 0.1, 0.5, 1.0):
            self.assertEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=False), x
            )

    def test_caching_default_multiplier_saturates(self):
        # Default multiplier is 10.0 — any x >= 0.1 saturates to 1.0.
        env = {
            k: v
            for k, v in os.environ.items()
            if k != dynamicemb_util.DYNAMICEMB_CACHING_HIT_RATE_MULTIPLIER_ENV
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(0.1, caching=True),
                1.0,
            )
            self.assertEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, caching=True),
                1.0,
            )
            self.assertEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(0.01, caching=True),
                0.1,
            )

    def test_caching_env_override(self):
        with mock.patch.dict(
            os.environ,
            {dynamicemb_util.DYNAMICEMB_CACHING_HIT_RATE_MULTIPLIER_ENV: "2.0"},
        ):
            # multiplier=2: x=0.3 -> 0.6, x=0.5 -> 1.0 (clamped)
            self.assertAlmostEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(0.3, caching=True),
                0.6,
            )
            self.assertEqual(
                dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, caching=True),
                1.0,
            )

    def test_caching_invariant_monotonic_ge_hybrid(self):
        # CACHING ratio >= HYBRID ratio at every cache_load_factor.
        for x in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            hybrid = dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=False)
            caching = dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=True)
            self.assertGreaterEqual(caching, hybrid)

    def test_stats_override_uses_expected_miss_rate(self):
        class _Stats:
            expected_lookups = 1000.0

            def expected_miss_rate(self, ratio):
                return 0.05  # 95% hit rate regardless of ratio

        x_eff = dynamicemb_util._dynamicemb_effective_cache_ratio(
            0.2, caching=True, stats=_Stats()
        )
        self.assertAlmostEqual(x_eff, 0.95)

    def test_stats_override_never_drops_below_hybrid(self):
        class _Stats:
            expected_lookups = 1000.0

            def expected_miss_rate(self, ratio):
                return 0.9  # would give x_eff=0.1, below cache_load_factor=0.5

        x_eff = dynamicemb_util._dynamicemb_effective_cache_ratio(
            0.5, caching=True, stats=_Stats()
        )
        # Invariant: CACHING_bw never falls below HYBRID_bw at same ratio.
        self.assertGreaterEqual(x_eff, 0.5)


if __name__ == "__main__":
    unittest.main()
