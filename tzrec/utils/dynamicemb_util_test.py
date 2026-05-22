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
from types import SimpleNamespace
from unittest import mock

import torch
from parameterized import parameterized
from torchrec.distributed.types import CacheParams

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
    """``_dynamicemb_effective_cache_ratio`` -- empirical fit from on-device sweep.

    The formula is fitted to the medians of an A10 benchmark sweep recorded
    in experiments/sweep_20260513-161030/full_a10gpu1.json. Reproducing the
    three-regime empirical pattern:
      * x == 1.0:            HBM-only fast path, x_eff = 1.0
      * caching=True, x<1.0: x_eff base 0.28
      * caching=False, x<1.0: x_eff base 0.11
    """

    def test_x_eq_1_returns_1_in_both_modes(self):
        # HBM-only fast path: total fits, runtime drops the host tier.
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(1.0, caching=False), 1.0
        )
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(1.0, caching=True), 1.0
        )

    def test_caching_above_hybrid_for_x_less_than_1(self):
        # Empirically CACHING is ~2x faster than HYBRID at the same ratio.
        for x in (0.1, 0.3, 0.5, 0.7, 0.9):
            hybrid = dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=False)
            caching = dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=True)
            self.assertGreater(caching, hybrid)

    def test_monotonic_within_block(self):
        # Within each mode, higher cache_load_factor -> higher x_eff (the
        # +0.01*x tiebreaker term).
        for caching in (False, True):
            prev = None
            for i in range(1, 10):  # x = 0.1 .. 0.9
                x = i / 10
                cur = dynamicemb_util._dynamicemb_effective_cache_ratio(
                    x, caching=caching
                )
                if prev is not None:
                    self.assertGreater(cur, prev)
                prev = cur

    def test_strict_block_ranking_matches_empirical(self):
        # HYBRID@1.0 > CACHING@anything < 1.0 > HYBRID@anything < 1.0
        ratios = [i / 10 for i in range(1, 10)]  # 0.1 .. 0.9
        hybrid_at_1 = dynamicemb_util._dynamicemb_effective_cache_ratio(
            1.0, caching=False
        )
        caching_block = {
            x: dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=True)
            for x in ratios
        }
        hybrid_block = {
            x: dynamicemb_util._dynamicemb_effective_cache_ratio(x, caching=False)
            for x in ratios
        }
        # Every CACHING@x<1.0 sits strictly below HYBRID@1.0.
        for x_eff in caching_block.values():
            self.assertGreater(hybrid_at_1, x_eff)
        # Every CACHING@x<1.0 sits strictly above every HYBRID@x<1.0.
        for c in caching_block.values():
            for h in hybrid_block.values():
                self.assertGreater(c, h)

    def test_stats_override_uses_expected_miss_rate(self):
        class _Stats:
            expected_lookups = 1000.0

            def expected_miss_rate(self, ratio):
                return 0.05  # 95% hit rate regardless of ratio

        x_eff = dynamicemb_util._dynamicemb_effective_cache_ratio(
            0.2, caching=True, stats=_Stats()
        )
        self.assertAlmostEqual(x_eff, 0.95)

    def test_stats_override_honored_verbatim(self):
        # Stats reflect measured behavior; trust them even if they override
        # the empirical heuristic's preferred ordering.
        class _Stats:
            expected_lookups = 1000.0

            def expected_miss_rate(self, ratio):
                return 0.9  # x_eff = 0.1 even though CACHING base = 0.28

        x_eff = dynamicemb_util._dynamicemb_effective_cache_ratio(
            0.5, caching=True, stats=_Stats()
        )
        self.assertAlmostEqual(x_eff, 0.1)


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
class BuildShardPerfContextsWrapperTest(unittest.TestCase):
    """``_dynamicemb_aware_build_shard_perf_contexts`` swap + restore.

    Mocks ``_orig_build_shard_perf_contexts`` so we can drive the wrapper
    without the heavy upstream ShardPerfContext machinery, and verify the
    boost is applied, the cache_params is restored after the call, and the
    restore still runs when the wrapped call raises.
    """

    def _call(self, sharding_option):
        # ShardPerfContext.build_shard_perf_contexts is the patched
        # classmethod after dynamicemb_util import. The classmethod
        # descriptor auto-injects ``cls``, so we pass 6 positional args
        # (config, shard_sizes, sharding_option, topology, constraints,
        # sharder).
        from torchrec.distributed.planner.estimator.types import ShardPerfContext

        return ShardPerfContext.build_shard_perf_contexts(
            None, None, sharding_option, None, None, None
        )

    def _spy_recording_cache_params(self, recorder):
        def _spy(cls, config, shard_sizes, sharding_option, *args, **kwargs):
            recorder.append(sharding_option.cache_params.load_factor)
            return []

        return _spy

    def test_boost_applied_for_caching(self):
        seen = []
        with mock.patch.object(
            dynamicemb_util,
            "_orig_build_shard_perf_contexts",
            self._spy_recording_cache_params(seen),
        ):
            so = SimpleNamespace(
                dynamicemb_options=SimpleNamespace(caching=True),
                cache_params=CacheParams(load_factor=0.5),
                cache_load_factor=0.5,
            )
            self._call(so)
        expected = dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, caching=True)
        self.assertEqual(len(seen), 1)
        self.assertAlmostEqual(seen[0], expected)

    def test_boost_applied_for_hybrid(self):
        seen = []
        with mock.patch.object(
            dynamicemb_util,
            "_orig_build_shard_perf_contexts",
            self._spy_recording_cache_params(seen),
        ):
            so = SimpleNamespace(
                dynamicemb_options=SimpleNamespace(caching=False),
                cache_params=CacheParams(load_factor=0.5),
                cache_load_factor=0.5,
            )
            self._call(so)
        expected = dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, caching=False)
        self.assertAlmostEqual(seen[0], expected)

    def test_no_boost_when_no_dynamicemb_options(self):
        # Non-dynamicemb ShardingOption has no `dynamicemb_options`
        # attribute; the wrapper must leave cache_params untouched.
        seen = []
        with mock.patch.object(
            dynamicemb_util,
            "_orig_build_shard_perf_contexts",
            self._spy_recording_cache_params(seen),
        ):
            so = SimpleNamespace(
                cache_params=CacheParams(load_factor=0.7),
                cache_load_factor=0.7,
            )
            self._call(so)
        self.assertEqual(seen[0], 0.7)

    def test_cache_params_restored_on_success(self):
        original = CacheParams(load_factor=0.3)
        with mock.patch.object(
            dynamicemb_util,
            "_orig_build_shard_perf_contexts",
            lambda cls, c, s, so, *a, **kw: [],
        ):
            so = SimpleNamespace(
                dynamicemb_options=SimpleNamespace(caching=True),
                cache_params=original,
                cache_load_factor=0.3,
            )
            self._call(so)
        # Same identity, not just same load_factor.
        self.assertIs(so.cache_params, original)

    def test_cache_params_restored_on_exception(self):
        """Restore must run even when the wrapped call raises (R1).

        Without try/finally this test fails -- the boosted cache_params
        leaks out and corrupts downstream consumers.
        """
        original = CacheParams(load_factor=0.3)

        class _Boom(RuntimeError):
            pass

        def raiser(cls, c, s, so, *a, **kw):
            raise _Boom("simulated estimator failure")

        with mock.patch.object(
            dynamicemb_util, "_orig_build_shard_perf_contexts", raiser
        ):
            so = SimpleNamespace(
                dynamicemb_options=SimpleNamespace(caching=True),
                cache_params=original,
                cache_load_factor=0.3,
            )
            with self.assertRaises(_Boom):
                self._call(so)
        self.assertIs(so.cache_params, original)


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
class DynamicEmbCalcShardStoragesTest(unittest.TestCase):
    """Direct test of ``dynamicemb_calculate_shard_storages``."""

    def _build_options(self, *, with_admission_counter=False):
        import dynamicemb

        kwargs = dict(
            max_capacity=1024,
            initializer_args=dynamicemb.DynamicEmbInitializerArgs(
                mode=dynamicemb.DynamicEmbInitializerMode.UNIFORM,
                lower=-0.01,
                upper=0.01,
            ),
            eval_initializer_args=dynamicemb.DynamicEmbInitializerArgs(
                mode=dynamicemb.DynamicEmbInitializerMode.CONSTANT, value=0.0
            ),
            score_strategy=dynamicemb.DynamicEmbScoreStrategy.STEP,
        )
        if with_admission_counter:
            kwargs["admission_counter"] = dynamicemb.KVCounter(
                capacity=1024, bucket_capacity=128
            )
            kwargs["admit_strategy"] = dynamicemb.FrequencyAdmissionStrategy(
                threshold=1,
                initializer_args=dynamicemb.DynamicEmbInitializerArgs(
                    mode=dynamicemb.DynamicEmbInitializerMode.CONSTANT, value=0.0
                ),
            )
        return dynamicemb.DynamicEmbTableOptions(**kwargs)

    def _base_kwargs(
        self, dynamicemb_options, *, compute_device="cuda", is_inference=False
    ):
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel
        from torchrec.distributed.types import ShardingType

        return dict(
            sharder=None,
            sharding_type=ShardingType.ROW_WISE.value,
            tensor=torch.empty(1024, 64, dtype=torch.float32),
            compute_device=compute_device,
            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
            shard_sizes=[[512, 64], [512, 64]],
            batch_sizes=[32, 32],
            world_size=2,
            local_world_size=2,
            input_lengths=[1.0],
            num_poolings=[1.0],
            caching_ratio=0.5,
            is_pooled=True,
            input_data_type_size=8.0,
            output_data_type_size=4.0,
            is_inference=is_inference,
            dynamicemb_options=dynamicemb_options,
        )

    def test_admission_counter_increases_hbm(self):
        baseline = dynamicemb_util.dynamicemb_calculate_shard_storages(
            **self._base_kwargs(self._build_options(with_admission_counter=False))
        )
        with_counter = dynamicemb_util.dynamicemb_calculate_shard_storages(
            **self._base_kwargs(self._build_options(with_admission_counter=True))
        )
        # Counter is HBM-side only — DDR matches, HBM grows.
        for base, w in zip(baseline, with_counter):
            self.assertGreater(w.hbm, base.hbm)
            self.assertEqual(w.ddr, base.ddr)


import os  # noqa: E402  (placed after the parameterized import block above)


class PlanLogLineTest(unittest.TestCase):
    """``_log_dynamicemb_table_plan`` -- mode label + GiB unit + rank gating."""

    def _capture(self, **kwargs):
        with mock.patch.object(dynamicemb_util.logger, "info") as m:
            dynamicemb_util._log_dynamicemb_table_plan(**kwargs)
        return [c.args[0] for c in m.call_args_list]

    def test_hybrid_partial_factor_logs_hybrid(self):
        msgs = self._capture(
            fqn="m.t",
            cache_load_factor=0.5,
            caching=False,
            hbm_bytes=1 << 30,
            ddr_bytes=2 << 30,
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("mode=HYBRID", msgs[0])
        self.assertIn("cache_load_factor=0.50", msgs[0])
        self.assertIn("local_hbm=1.000GiB", msgs[0])
        self.assertIn("local_dram=2.000GiB", msgs[0])

    def test_caching_partial_factor_logs_caching(self):
        msgs = self._capture(
            fqn="m.t",
            cache_load_factor=0.3,
            caching=True,
            hbm_bytes=1 << 30,
            ddr_bytes=5 << 30,
        )
        self.assertEqual(len(msgs), 1)
        self.assertIn("mode=CACHING", msgs[0])
        self.assertIn("local_dram=5.000GiB", msgs[0])

    def test_hybrid_at_1_0_logs_hbm_only(self):
        msgs = self._capture(
            fqn="m.t",
            cache_load_factor=1.0,
            caching=False,
            hbm_bytes=4 << 30,
            ddr_bytes=0,
        )
        self.assertIn("mode=HBM_ONLY", msgs[0])
        self.assertIn("local_dram=0.000GiB", msgs[0])

    def test_caching_at_1_0_logs_hbm_only(self):
        # Override path: even though caching=True, x=1.0 collapses to
        # HBM_ONLY at runtime and ddr is reported as 0 (would be 5GiB
        # without the override).
        msgs = self._capture(
            fqn="m.t",
            cache_load_factor=1.0,
            caching=True,
            hbm_bytes=4 << 30,
            ddr_bytes=5 << 30,
        )
        self.assertIn("mode=HBM_ONLY", msgs[0])
        self.assertIn("local_dram=0.000GiB", msgs[0])

    def test_non_rank_zero_is_silent(self):
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            msgs = self._capture(
                fqn="m.t",
                cache_load_factor=0.5,
                caching=False,
                hbm_bytes=1 << 30,
                ddr_bytes=2 << 30,
            )
        self.assertEqual(msgs, [])

    def test_just_above_1_0_does_not_relabel(self):
        # Exact-equality guard: a hypothetical leak of cache_load_factor
        # >1.0 from upstream must NOT be silently relabelled HBM_ONLY.
        msgs = self._capture(
            fqn="m.t",
            cache_load_factor=1.0001,
            caching=True,
            hbm_bytes=1 << 30,
            ddr_bytes=2 << 30,
        )
        self.assertIn("mode=CACHING", msgs[0])  # not HBM_ONLY


if __name__ == "__main__":
    unittest.main()
