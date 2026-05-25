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


if __name__ == "__main__":
    unittest.main()
