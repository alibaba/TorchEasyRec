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

import math
import os
import unittest
import warnings
from functools import partial
from unittest import mock

import torch
from parameterized import param, parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim import optimizers, rowwise_adagrad

from tzrec.optim.optimizer import FTRL, has_fbgemm_ftrl
from tzrec.protos import feature_pb2
from tzrec.utils import dynamicemb_util
from tzrec.utils.plan_util import EmbeddingEnumerator, get_default_sharders
from tzrec.utils.test_util import mark_ci_scope, parameterized_name_func


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class ShardPerfContextPatchTest(unittest.TestCase):
    """Compatibility tests for the dynamicemb perf-context patch."""

    def test_forwards_version_specific_sharder_argument(self):
        sharding_option = mock.Mock()
        sharding_option.dynamicemb_options = None
        sharding_option.cache_params = object()
        expected = object()
        original = mock.Mock(return_value=expected)
        common_kwargs = {
            "config": object(),
            "shard_sizes": object(),
            "sharding_option": sharding_option,
            "topology": object(),
            "constraints": object(),
        }

        with mock.patch.object(
            dynamicemb_util, "_orig_build_shard_perf_contexts", original
        ):
            for argument_name in ("sharder", "sharder_data"):
                with self.subTest(argument_name=argument_name):
                    sharder = object()
                    result = dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
                        **common_kwargs, **{argument_name: sharder}
                    )

                    self.assertIs(result, expected)
                    self.assertEqual(original.call_count, 1)
                    self.assertIs(original.call_args.kwargs[argument_name], sharder)
                    original.reset_mock()


class DynamicEmbTableMemoryTest(unittest.TestCase):
    """Capacity-mirror invariants of the candidate kinds.

    With native fusion the planner proposes budgets and caching choices; the
    zero and full endpoints cost the per-axis maximum of their standalone and
    mixed-group allocations, so those envelopes cover -- not equal -- what
    the runtime allocates for a particular fused module.
    """

    ROWS = 100_003  # unaligned on purpose
    DIM = 64
    ELEMENT_SIZE = 4

    def _memory(self, caching, ratio, **kwargs):
        return dynamicemb_util.dynamicemb_table_memory(
            caching=caching,
            ratio=ratio,
            shard_rows=kwargs.pop("shard_rows", self.ROWS),
            dim=self.DIM,
            element_size=self.ELEMENT_SIZE,
            **kwargs,
        )

    def _aligned(self):
        return dynamicemb_util.dynamicemb_aligned_capacity(self.ROWS, None)

    def test_aligned_capacity_rounds_up_to_buckets(self):
        # ceil(100003 / 128) * 128 = 782 * 128
        self.assertEqual(
            dynamicemb_util.dynamicemb_aligned_capacity(self.ROWS, None), 782 * 128
        )
        self.assertEqual(
            dynamicemb_util.dynamicemb_aligned_capacity(self.ROWS, 1024),
            98 * 1024,
        )
        # bucket widths round up to the 16-row digest-load multiple
        self.assertEqual(dynamicemb_util.dynamicemb_aligned_capacity(128, 17), 128)
        self.assertEqual(dynamicemb_util.dynamicemb_aligned_capacity(100, 17), 128)
        self.assertEqual(dynamicemb_util.dynamicemb_aligned_capacity(100, 1024), 1024)

    def test_zero_endpoint_covers_host_only_and_mixed_minimum_hbm(self):
        # an all-zero group allocates HOST_ONLY: no HBM values, full host
        # values, and the key map on HBM. Mixed into HYBRID the member
        # acquires a 1024-row minimum HBM tier; the envelope keeps the
        # larger of the two placements per field.
        aligned = self._aligned()
        row_bytes = self.DIM * self.ELEMENT_SIZE
        zero = dynamicemb_util.dynamicemb_table_memory(
            caching=False,
            ratio=0.0,
            shard_rows=self.ROWS,
            dim=self.DIM,
            element_size=self.ELEMENT_SIZE,
        )
        self.assertEqual(zero.host_values, aligned * row_bytes)
        self.assertEqual(zero.host_rows, aligned)
        self.assertEqual(zero.hbm_values, 1024 * row_bytes)
        self.assertEqual(zero.hbm_rows, 1024)
        # one score word: 21 bytes per slot in both tiers' key maps,
        # plus each map's bucket-size counters
        self.assertEqual(
            zero.hbm_index,
            (21 * 1024 + 4) + (21 * aligned + 4 * 782),
        )
        self.assertEqual(zero.local_hbm_for_values, 0)

    def test_full_endpoint_covers_hbm_only_and_mixed_minimum_host(self):
        # an all-full group allocates HBM_ONLY at the aligned capacity; a
        # mixed group instead gives the member a minimum host bucket while
        # its HBM tier rounds to whole 1024-row buckets, and the envelope
        # keeps the larger of the two placements per field
        aligned = self._aligned()
        row_bytes = self.DIM * self.ELEMENT_SIZE
        full = self._memory(False, 1.0)
        hbm_tier_rows = 98 * 1024  # the raw shard rows rounded to 1024s
        self.assertEqual(full.hbm_values, hbm_tier_rows * row_bytes)
        self.assertEqual(full.hbm_rows, hbm_tier_rows)
        # the minimum host tier of the mixed outcome is charged, but the
        # endpoint never joins a CACHING group, so no full backing table
        self.assertEqual(full.host_values, 128 * row_bytes)
        self.assertEqual(full.host_rows, 128)
        self.assertEqual(
            full.hbm_index,
            (21 * hbm_tier_rows + 4 * 98) + (21 * 128 + 4),
        )
        self.assertEqual(full.local_hbm_for_values, aligned * row_bytes)

    def test_hybrid_splits_values_across_both_tiers(self):
        memory = self._memory(False, 0.25)
        # both tiers round up to whole buckets, so together they cover the
        # full aligned table (and may exceed it by their roundings)
        self.assertGreaterEqual(memory.hbm_rows + memory.host_rows, self._aligned())
        self.assertGreater(memory.local_hbm_for_values, 0)
        # the budget comes from the aligned full shard, not the raw rows
        self.assertEqual(
            memory.local_hbm_for_values,
            int(0.25 * self._aligned() * self.DIM * self.ELEMENT_SIZE),
        )

    def test_hybrid_near_full_split_keeps_a_minimum_host_bucket(self):
        # a near-full split still allocates at least one row on host, which
        # the runtime rounds up to a whole bucket
        memory = self._memory(False, 0.9999)
        self.assertEqual(memory.host_rows, 128)

    def test_caching_keeps_a_full_backing_table_and_bills_overflow(self):
        memory = self._memory(True, 0.1)
        aligned = self._aligned()
        row_bytes = self.DIM * self.ELEMENT_SIZE
        # host tier is the full backing table
        self.assertEqual(memory.host_rows, aligned)
        self.assertEqual(memory.host_values, aligned * row_bytes)
        # the cache realizes whole 1024-row buckets above the request
        self.assertEqual(memory.resident_rows % 1024, 0)
        self.assertGreaterEqual(memory.resident_rows, 0.1 * aligned)
        # overflow is billed to HBM but not credited to residency
        self.assertEqual(memory.hbm_rows - memory.resident_rows, 3 * 1024)

    def test_caching_request_allocates_a_larger_share_than_it_budgets(self):
        # A nominal 0.1 request on a 4096-row shard buys one whole 1024-row
        # cache bucket: the allocated capacity share is 0.25 while the
        # runtime budget stays the requested share of the aligned shard.
        memory = dynamicemb_util.dynamicemb_table_memory(
            caching=True,
            ratio=0.1,
            shard_rows=4096,
            dim=16,
            element_size=4,
        )
        self.assertEqual(memory.resident_rows, 1024)
        self.assertEqual(memory.host_rows, 4096)
        self.assertEqual(memory.local_hbm_for_values, int(0.1 * 4096 * 16 * 4))

    def test_caching_value_budget_uses_the_aligned_denominator(self):
        row_bytes = self.DIM * self.ELEMENT_SIZE
        aligned = self._aligned()
        for ratio in (0.02, 0.1, 0.5, 0.9):
            memory = self._memory(True, ratio)
            self.assertEqual(
                memory.local_hbm_for_values, int(ratio * aligned * row_bytes)
            )

    def test_partial_fit_threshold_clamp_keeps_the_candidate_partial(self):
        # an unaligned shard whose aligned capacity exceeds the runtime's
        # un-rounded fit threshold must clamp the partial budget below it,
        # so the runtime cannot turn the candidate into HBM_ONLY
        row_bytes = 1 * 4  # dim 1 fp32: tiny rows maximize the misalignment
        memory = dynamicemb_util.dynamicemb_table_memory(
            caching=False,
            ratio=0.9,
            shard_rows=17,
            dim=1,
            element_size=4,
        )
        self.assertLess(memory.local_hbm_for_values, 17 * row_bytes)

    def test_optimizer_state_widens_every_value_row(self):
        base = self._memory(False, 1.0)
        with_state = self._memory(False, 1.0, optimizer_state_dim=2 * self.DIM)
        self.assertEqual(
            with_state.hbm_values,
            base.hbm_values + base.hbm_rows * 2 * self.DIM * 4,
        )

    def test_second_score_word_widens_the_key_map(self):
        base = self._memory(False, 0.0)
        compound = self._memory(False, 0.0, score_words=2)
        aligned = self._aligned()
        # one extra 8-byte score word per slot, in both the mixed minimum
        # HBM tier and the host tier the zero envelope charges
        self.assertEqual(compound.hbm_index - base.hbm_index, 8 * (aligned + 1024))

    def test_no_eviction_value_rows_are_init_capacity_with_double_key_map(self):
        # the fork sizes the value buffer at init_capacity -- the aligned
        # per-shard rows at the planning horizon -- and the key map at twice
        # those rows, already whole buckets
        aligned = self._aligned()
        slots = 2 * aligned
        memory = self._memory(False, 1.0, no_eviction=True)
        self.assertEqual(memory.hbm_rows, aligned)
        # per slot: key 8 + digest 1 + score 8 + reference counter 4
        self.assertEqual(memory.hbm_index, 21 * slots + 4 * (slots // 128))
        self.assertGreater(memory.local_hbm_for_values, self.ROWS * self.DIM * 4)

    def test_invalid_requests_are_rejected(self):
        for bad_ratio in (-0.1, 1.5, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self._memory(False, bad_ratio)
        with self.assertRaises(ValueError):
            # the endpoints are non-caching
            self._memory(True, 0.0)
        with self.assertRaises(ValueError):
            self._memory(True, 1.0)
        with self.assertRaises(ValueError):
            self._memory(False, 0.5, bucket_capacity=0)


class OptimizerMultiplerTest(unittest.TestCase):
    """Per-element optimizer state width used to size dynamicemb values."""

    @parameterized.expand(
        [
            param("untrained", optimizer_class=None, expected=0.0),
            param("sgd", optimizer_class=optimizers.SGD, expected=0),
            param("adam", optimizer_class=optimizers.Adam, expected=2),
            param(
                "rowwise_adagrad",
                optimizer_class=rowwise_adagrad.RowWiseAdagrad,
                expected=4 / 16,
            ),
            param(
                "rowwise_adagrad_fp16",
                optimizer_class=rowwise_adagrad.RowWiseAdagrad,
                dtype=torch.float16,
                expected=8 / 16,
            ),
            param("ftrl", optimizer_class=FTRL, expected=2.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_multipler(self, _name, optimizer_class, expected, dtype=torch.float32):
        self.assertAlmostEqual(
            dynamicemb_util._get_optimizer_multipler(
                optimizer_class, torch.Size([1024, 16]), dtype
            ),
            expected,
        )

    @unittest.skipUnless(
        has_fbgemm_ftrl(), "fbgemm_gpu / torchrec have no FTRL embedding kernel."
    )
    def test_trec_ftrl_multiplier(self):
        """Torchrec's FTRL keeps two elementwise states, like Adam.

        The multiplier comes from the TorchEasyRec torchrec build's
        ``shard_estimators``; this pins that contract -- a torchrec rebuild
        dropping it would silently halve dynamicemb HBM planning.
        """
        from torchrec import optim as trec_optim

        self.assertAlmostEqual(
            dynamicemb_util._get_optimizer_multipler(
                trec_optim.FTRL, torch.Size([1024, 16]), torch.float32
            ),
            2.0,
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb and has_fbgemm_ftrl(),
    "dynamicemb or the FBGEMM FTRL embedding kernel is not installed; skipping.",
)
@mark_ci_scope("gpu")
class DynamicEmbFTRLFusedParamsTest(unittest.TestCase):
    """torchrec's fused FTRL is rewritten into dynamicemb's on dynamicemb tables."""

    _KWARGS = {
        "lr": 0.01,
        "ftrl_learning_rate_power": -0.4,
        "ftrl_beta": 1.0,
        "ftrl_l1_reg": 0.01,
        "ftrl_l2_reg": 0.02,
    }

    def test_trec_ftrl(self):
        from dynamicemb import DynamicEmbOptimType
        from torchrec import optim as trec_optim

        params = dynamicemb_util._dynamicemb_ftrl_fused_params(
            trec_optim.FTRL, dict(self._KWARGS)
        )
        self.assertEqual(
            params,
            {
                "optimizer": DynamicEmbOptimType.FTRL,
                "learning_rate_power": -0.4,
                "l1_reg": 0.01,
                "l2_reg": 0.02,
            },
        )

    @parameterized.expand(
        [
            param("untrained", optimizer_class=None),
            param("adam", optimizer_class=optimizers.Adam),
            param("placeholder_ftrl", optimizer_class=FTRL),
        ],
        name_func=parameterized_name_func,
    )
    def test_other_optimizer(self, _name, optimizer_class):
        self.assertEqual(
            dynamicemb_util._dynamicemb_ftrl_fused_params(
                optimizer_class, dict(self._KWARGS)
            ),
            {},
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class AdmissionStrategyTest(unittest.TestCase):
    """The admission oneof -> a dynamicemb strategy, and its counter's HBM cost."""

    NUM_EMBEDDINGS = 1024
    EMBEDDING_DIM = 8

    @parameterized.expand(
        [("unset", None), ("explicit", 0.25), ("all_hbm", 1.0)],
        name_func=parameterized_name_func,
    )
    def test_cache_load_factor_sets_cache_params(self, _name, cache_load_factor):
        dynamicemb_cfg = feature_pb2.DynamicEmbedding(max_capacity=self.NUM_EMBEDDINGS)
        if cache_load_factor is not None:
            dynamicemb_cfg.cache_load_factor = cache_load_factor
        constraints = dynamicemb_util.build_dynamicemb_constraints(
            dynamicemb_cfg,
            EmbeddingBagConfig(
                name="dyn_table",
                num_embeddings=self.NUM_EMBEDDINGS,
                embedding_dim=self.EMBEDDING_DIM,
                feature_names=["user_id"],
            ),
        )
        if cache_load_factor is None:
            self.assertIsNone(constraints.cache_params)
        else:
            self.assertIsInstance(constraints.cache_params, CacheParams)
            self.assertEqual(constraints.cache_params.load_factor, cache_load_factor)

    def _options(self, **admission_strategy):
        dynamicemb_cfg = feature_pb2.DynamicEmbedding(
            max_capacity=self.NUM_EMBEDDINGS, **admission_strategy
        )
        constraints = dynamicemb_util.build_dynamicemb_constraints(
            dynamicemb_cfg,
            EmbeddingBagConfig(
                name="dyn_table",
                num_embeddings=self.NUM_EMBEDDINGS,
                embedding_dim=self.EMBEDDING_DIM,
                feature_names=["user_id"],
            ),
        )
        return constraints.dynamicemb_options

    def _shard_storages(self, dynamicemb_options):
        # the estimator reads the candidate's caching choice off the options
        dynamicemb_options.caching = False
        return dynamicemb_util.dynamicemb_calculate_shard_storages(
            sharder_data=None,
            sharding_type=ShardingType.ROW_WISE.value,
            tensor=torch.empty(self.NUM_EMBEDDINGS, self.EMBEDDING_DIM),
            compute_device="cuda",
            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
            shard_sizes=[[self.NUM_EMBEDDINGS // 2, self.EMBEDDING_DIM]] * 2,
            batch_sizes=[16],
            world_size=2,
            local_world_size=2,
            input_lengths=[1.0],
            num_poolings=[1.0],
            caching_ratio=1.0,
            is_pooled=True,
            input_data_type_size=4,
            output_data_type_size=4,
            dynamicemb_options=dynamicemb_options,
        )

    def test_frequency_strategy_owns_its_counter(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            options = self._options(
                frequency_admission_strategy=(
                    feature_pb2.DynamicEmbFrequencyAdmissionStrategy(
                        threshold=5,
                        counter_capacity=2048,
                        counter_bucket_capacity=512,
                    )
                )
            )
        admit_strategy = options.admit_strategy
        self.assertIsInstance(
            admit_strategy, dynamicemb_util.FrequencyAdmissionStrategy
        )
        self.assertEqual(admit_strategy.threshold, 5)
        self.assertEqual(
            admit_strategy.counter.capacity,
            dynamicemb_util.align_to_table_size(1024),
        )
        self.assertEqual(admit_strategy.counter.bucket_capacity, 512)

    def test_counter_capacity_defaults_to_num_embeddings(self):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            options = self._options(
                frequency_admission_strategy=(
                    feature_pb2.DynamicEmbFrequencyAdmissionStrategy(threshold=1)
                )
            )
        self.assertEqual(
            options.admit_strategy.counter.capacity,
            dynamicemb_util.align_to_table_size(self.NUM_EMBEDDINGS),
        )

    def test_deprecated_admission_counter_stays_unset(self):
        # The counter now belongs to the strategy; setting the table option
        # instead is deprecated upstream and raises DeprecationWarning.
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            options = self._options(
                frequency_admission_strategy=(
                    feature_pb2.DynamicEmbFrequencyAdmissionStrategy(threshold=1)
                )
            )
        self.assertIsNone(options.admission_counter)

    def test_probabilistic_strategy_carries_its_probability(self):
        options = self._options(
            probabilistic_admission_strategy=(
                feature_pb2.DynamicEmbProbabilisticAdmissionStrategy(probability=0.25)
            )
        )
        admit_strategy = options.admit_strategy
        self.assertIsInstance(
            admit_strategy, dynamicemb_util.ProbabilisticAdmissionStrategy
        )
        self.assertEqual(admit_strategy.probability, 0.25)

    @parameterized.expand(
        [
            param(
                "frequency",
                field="frequency_admission_strategy",
                strategy=partial(
                    feature_pb2.DynamicEmbFrequencyAdmissionStrategy, threshold=1
                ),
            ),
            param(
                "probabilistic",
                field="probabilistic_admission_strategy",
                strategy=partial(
                    feature_pb2.DynamicEmbProbabilisticAdmissionStrategy,
                    probability=0.25,
                ),
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_non_admitted_initializer(self, _name, field, strategy):
        options = self._options(**{field: strategy()})
        initializer_args = options.admit_strategy.initializer_args
        self.assertEqual(
            initializer_args.mode, dynamicemb_util.DynamicEmbInitializerMode.CONSTANT
        )
        self.assertEqual(initializer_args.value, 0.0)

        options = self._options(
            **{
                field: strategy(
                    initializer_args=feature_pb2.DynamicEmbInitializerArgs(
                        mode="CONSTANT", value=0.5
                    )
                )
            }
        )
        self.assertEqual(options.admit_strategy.initializer_args.value, 0.5)

    def test_identical_configs_share_a_fused_table(self):
        strategy = feature_pb2.DynamicEmbProbabilisticAdmissionStrategy(
            probability=0.25
        )
        first = self._options(probabilistic_admission_strategy=strategy)
        second = self._options(probabilistic_admission_strategy=strategy)
        frequency = self._options(
            frequency_admission_strategy=(
                feature_pb2.DynamicEmbFrequencyAdmissionStrategy(threshold=1)
            )
        )
        self.assertEqual(first.get_grouped_key(), second.get_grouped_key())
        self.assertNotEqual(first.get_grouped_key(), frequency.get_grouped_key())

    def test_counter_hbm_lands_only_for_frequency_admission(self):
        no_admission = self._shard_storages(self._options())
        probabilistic = self._shard_storages(
            self._options(
                probabilistic_admission_strategy=(
                    feature_pb2.DynamicEmbProbabilisticAdmissionStrategy(
                        probability=0.25
                    )
                )
            )
        )
        frequency_options = self._options(
            frequency_admission_strategy=(
                feature_pb2.DynamicEmbFrequencyAdmissionStrategy(threshold=1)
            )
        )
        frequency = self._shard_storages(frequency_options)
        counter = frequency_options.admit_strategy.counter
        counter_hbm = math.ceil(
            dynamicemb_util.align_to_table_size(counter.capacity)
            * (8 + 8 + 1 + 4 / counter.bucket_capacity)
        )
        for base, prob, freq in zip(no_admission, probabilistic, frequency):
            self.assertEqual(prob.hbm, base.hbm)
            self.assertEqual(freq.hbm - base.hbm, counter_hbm)


class EffectiveCacheRatioTest(unittest.TestCase):
    """The empirical x_eff model over allocated capacity shares."""

    def test_full_endpoint_takes_the_full_hbm_path(self):
        # regardless of cache statistics inherited from a base constraint
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                1.0, False, stats=object(), full_endpoint=True
            ),
            1.0,
        )

    def test_zero_endpoint_reads_host_memory(self):
        # explicit host-access estimate: no HYBRID intercept at share zero
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(0.0, False),
            0.0,
        )

    def test_caching_share_reaching_one_stays_on_the_cache_path(self):
        # a cache covering the whole backing table is still a cache, not
        # the standalone HBM-direct path; a share above one just clamps
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(1.0, True),
            0.28 + 0.01,
        )
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(1.5, True),
            0.28 + 0.01,
        )

    def test_partial_candidates_keep_their_empirical_bases(self):
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, True),
            0.28 + 0.01 * 0.5,
        )
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(0.5, False),
            0.11 + 0.01 * 0.5,
        )

    def test_measured_stats_apply_to_caching_candidates(self):
        class _Stats:
            def expected_miss_rate(self, x):
                return 0.25

        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                0.5, True, stats=_Stats()
            ),
            0.75,
        )
        # miss statistics describe a cache, not a value partition
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                0.5, False, stats=_Stats()
            ),
            0.11 + 0.01 * 0.5,
        )

    def test_zero_ratio_of_a_caching_candidate_keeps_the_base(self):
        # a caching candidate cannot sit at ratio zero; the estimator
        # still scores the requested value, not a realized fraction
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(0.0, True),
            0.28,
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class NativeDynamicEmbOptionsGroupingTest(unittest.TestCase):
    """Project-built options through TorchRec's own grouping decide fusion.

    The planner proposes budgets and caching choices on options built by
    ``build_dynamicemb_constraints``; which tables fuse is the upstream
    grouping key's decision. Capacity and budget fields must stay out of
    that key so mixed-budget tables can share one fused module, while the
    caching choice separates the groups the runtime builds differently.
    """

    def _options(self, **overrides):
        cfg = feature_pb2.DynamicEmbedding(max_capacity=1024, score_strategy="STEP")
        constraints = dynamicemb_util.build_dynamicemb_constraints(
            cfg,
            EmbeddingBagConfig(
                num_embeddings=1024,
                embedding_dim=16,
                name="t",
                feature_names=["f"],
            ),
        )
        options = constraints.dynamicemb_options
        options.dim = 16
        options.max_capacity = 1024
        options.embedding_dtype = torch.float32
        options.index_type = torch.int64
        for key, value in overrides.items():
            setattr(options, key, value)
        return options

    def test_project_options_group_in_real_torchrec_grouping(self):
        # the real TorchRec grouping path: fused params (which carry the
        # options built by the project's own constraint builder) decide
        # which tables share one fused module. Different per-table budgets
        # fuse; different caching choices do not.
        from torchrec.distributed.embedding_sharding import (
            ShardedEmbeddingTable,
            group_tables,
        )
        from torchrec.modules.embedding_configs import PoolingType

        def table(name, **kwargs):
            return ShardedEmbeddingTable(
                num_embeddings=1024,
                embedding_dim=16,
                name=name,
                feature_names=[name],
                pooling=PoolingType.SUM,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL,
                local_rows=1024,
                local_cols=16,
                fused_params={"dynamicemb_options": self._options(**kwargs)},
            )

        tables_per_rank = [
            [
                table("t_zero", max_capacity=1024, local_hbm_for_values=0),
                table("t_partial", max_capacity=1024, local_hbm_for_values=4096),
                table("t_full", max_capacity=1024, local_hbm_for_values=65536),
                table("t_cache", max_capacity=1024, caching=True),
            ]
        ]
        grouped = group_tables(tables_per_rank)[0]
        names = sorted(
            tuple(sorted(t.name for t in g.embedding_tables)) for g in grouped
        )
        self.assertEqual(
            names,
            [("t_cache",), ("t_full", "t_partial", "t_zero")],
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class DynamicEmbCapacityNormalizationTest(unittest.TestCase):
    @parameterized.expand(
        [
            ("single_rank", 250, 1, 128, 256, 128),
            ("four_ranks", 1000, 4, 128, 256, 128),
            ("three_ranks", 100003, 3, 256, 33536, 256),
            ("one_bucket", 1000, 4, 2**63 - 1, 256, 256),
        ],
        name_func=parameterized_name_func,
    )
    def test_tensor_shards_and_budgets_match_capacity(
        self, _name, rows, world_size, bucket, local_rows, effective_bucket
    ):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}):
            table = EmbeddingBagConfig(
                name="aligned",
                num_embeddings=rows,
                embedding_dim=16,
                feature_names=["aligned"],
            )
            cfg = feature_pb2.DynamicEmbedding(
                max_capacity=rows, bucket_capacity=bucket
            )
            constraint = dynamicemb_util.build_dynamicemb_constraints(cfg, table)
            self.assertEqual(table.num_embeddings, local_rows * world_size)
            self.assertEqual(constraint.dynamicemb_options.max_capacity, local_rows)
            self.assertEqual(
                constraint.dynamicemb_options.bucket_capacity, effective_bucket
            )
            model = TestSparseNN(tables=[table], sparse_device=torch.device("meta"))
            enumerator = EmbeddingEnumerator(
                topology=Topology(world_size=world_size, compute_device="cuda"),
                batch_size=128,
                fqn_constraints={"sparse.ebc.aligned": constraint},
            )
            options = enumerator.enumerate(model, get_default_sharders())
            for option in options:
                self.assertEqual(option.tensor.shape[0], local_rows * world_size)
                for shard in option.shards:
                    self.assertEqual(shard.size[0], local_rows)
                    memory = dynamicemb_util.dynamicemb_table_memory(
                        caching=option.dynamicemb_options.caching,
                        ratio=option.cache_load_factor,
                        shard_rows=shard.size[0],
                        dim=16,
                        element_size=4,
                        bucket_capacity=effective_bucket,
                    )
                    full = local_rows * 16 * 4
                    if option.cache_load_factor == 1.0:
                        self.assertEqual(memory.local_hbm_for_values, full)
                    else:
                        self.assertLess(memory.local_hbm_for_values, full)

    @parameterized.expand(
        [("small", 1, 256), ("aligned", 512, 512), ("clamped", 1500, 1024)],
        name_func=parameterized_name_func,
    )
    def test_initial_capacity_uses_the_effective_bucket(self, _name, initial, expected):
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            table = EmbeddingBagConfig(
                name="aligned",
                num_embeddings=4000,
                embedding_dim=16,
                feature_names=["aligned"],
            )
            cfg = feature_pb2.DynamicEmbedding(
                max_capacity=4000, bucket_capacity=256, init_capacity_per_rank=initial
            )
            options = dynamicemb_util.build_dynamicemb_constraints(
                cfg, table
            ).dynamicemb_options
            self.assertEqual(options.max_capacity, 1024)
            self.assertEqual(options.init_capacity, expected)

    def test_normalized_mixed_group_has_no_alignment_surplus(self):
        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from fbgemm_gpu.split_embedding_configs import EmbOptimType

        options = []
        for rows, dim, ratio in ((4000, 64, 0.0), (897, 2048, 1.0)):
            table = EmbeddingBagConfig(
                name=f"table_{dim}",
                num_embeddings=rows,
                embedding_dim=dim,
                feature_names=[f"feature_{dim}"],
            )
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                option = dynamicemb_util.build_dynamicemb_constraints(
                    feature_pb2.DynamicEmbedding(max_capacity=rows), table
                ).dynamicemb_options
            option.dim = dim
            option.embedding_dtype = torch.float32
            option.index_type = torch.int64
            option.training = True
            option.caching = False
            option.local_hbm_for_values = int(ratio * option.max_capacity * dim * 4)
            options.append(option)
        self.assertLess(
            sum(o.local_hbm_for_values for o in options),
            sum(o.max_capacity * o.dim * 4 for o in options),
        )
        module = BatchedDynamicEmbeddingTablesV2(
            table_options=options,
            table_names=["zero", "full"],
            feature_table_map=[0, 1],
            optimizer=EmbOptimType.EXACT_SGD,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
        self.assertEqual(module._storage_layout.name, "HYBRID")


class DynamicEmbPlanConversionTest(unittest.TestCase):
    """Plan conversion turns a candidate into native runtime options."""

    def _convert(
        self,
        caching,
        ratio,
        *,
        optimizer_class=None,
        rows=4096,
        world_size=2,
        dim=32,
        bucket_capacity=None,
        score="STEP",
    ):
        from torchrec.distributed.planner import planners
        from torchrec.distributed.types import CacheParams

        cfg = feature_pb2.DynamicEmbedding()
        cfg.max_capacity = rows
        cfg.score_strategy = score
        if bucket_capacity is not None:
            cfg.bucket_capacity = bucket_capacity
        constraint = dynamicemb_util.build_dynamicemb_constraints(
            cfg,
            EmbeddingBagConfig(
                num_embeddings=rows,
                embedding_dim=dim,
                name="table_de",
                feature_names=["feat_de"],
            ),
        )
        # pin the ratio explicitly: endpoint pins select their single
        # candidate, partial pins keep both families at the ratio
        constraint.cache_params = CacheParams(load_factor=ratio)
        table = EmbeddingBagConfig(
            num_embeddings=rows,
            embedding_dim=dim,
            name="table_de",
            feature_names=["feat_de"],
        )
        model = TestSparseNN(tables=[table], sparse_device=torch.device("meta"))
        topology = Topology(world_size=world_size, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=get_default_sharders()
        )
        sharding_option = next(
            so
            for so in search_space
            if getattr(so, "use_dynamicemb", False)
            and so.dynamicemb_options.caching == caching
            and so.cache_load_factor == ratio
        )
        for rank, shard in enumerate(sharding_option.shards):
            shard.rank = rank
        if optimizer_class is not None:
            sharding_option.tensor._optimizer_classes = [optimizer_class]
        plan = planners.to_sharding_plan([sharding_option], topology)
        return (
            plan.plan[sharding_option.path][sharding_option.name],
            sharding_option,
        )

    def test_zero_pin(self):
        param_sharding, _ = self._convert(False, 0.0)
        options = param_sharding.dynamicemb_options
        self.assertFalse(options.caching)
        self.assertEqual(options.local_hbm_for_values, 0)
        self.assertTrue(hasattr(param_sharding, "initial_accumulator_value"))

    def test_full_pin(self):
        from tzrec.optim.optimizer import FTRL

        param_sharding, sharding_option = self._convert(
            False, 1.0, optimizer_class=FTRL
        )
        options = param_sharding.dynamicemb_options
        self.assertFalse(options.caching)
        self.assertTrue(options.training)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        # FTRL doubles the value row; the full aligned table is funded
        self.assertEqual(options.local_hbm_for_values, aligned * (dim + 2 * dim) * 4)

    def test_partial_non_caching_pin_budget_matches_mirror(self):
        param_sharding, sharding_option = self._convert(False, 0.5)
        options = param_sharding.dynamicemb_options
        self.assertFalse(options.caching)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        self.assertEqual(options.local_hbm_for_values, int(0.5 * aligned * dim * 4))

    def test_caching_pin_budget_matches_mirror(self):
        param_sharding, sharding_option = self._convert(True, 0.25)
        options = param_sharding.dynamicemb_options
        self.assertTrue(options.caching)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        self.assertEqual(options.local_hbm_for_values, int(0.25 * aligned * dim * 4))

    def test_zero_survives_estimation_and_conversion(self):
        # the whole enumerate->estimate->convert path must keep the zero
        # budget and the non-caching choice of the zero endpoint

        param_sharding, _ = self._convert(False, 0.0)
        self.assertEqual(param_sharding.dynamicemb_options.local_hbm_for_values, 0)


class CapacityMirrorRuntimeParityTest(unittest.TestCase):
    """The capacity mirror against what the pinned runtime really allocates.

    Single-table cases compare the mirror's branch arithmetic exactly with
    the constructed storage. The zero and full endpoints are conservative
    envelopes over the mixed-group outcomes, so single-table modules check
    the realized branch exactly while fused mixed groups check that the
    additive per-table envelope covers the whole module's allocation.
    """

    def _make_options(
        self,
        *,
        rows,
        dim,
        ratio,
        caching,
        bucket_capacity=None,
        score="STEP",
        optimizer_state_dim=0,
    ):
        cfg = feature_pb2.DynamicEmbedding()
        cfg.max_capacity = rows
        cfg.score_strategy = score
        if bucket_capacity is not None:
            cfg.bucket_capacity = bucket_capacity
        constraint = dynamicemb_util.build_dynamicemb_constraints(
            cfg,
            EmbeddingBagConfig(
                num_embeddings=rows,
                embedding_dim=dim,
                name="table_parity",
                feature_names=["feat_parity"],
            ),
        )
        options = constraint.dynamicemb_options
        options.dim = dim
        options.max_capacity = rows
        # the planner aligns init_capacity to whole buckets; the fork sizes
        # a NO_EVICTION value buffer at exactly this capacity
        options.init_capacity = dynamicemb_util.dynamicemb_aligned_capacity(
            rows, options.bucket_capacity
        )
        options.embedding_dtype = torch.float32
        options.index_type = torch.int64
        memory = dynamicemb_util.dynamicemb_table_memory(
            caching=caching,
            ratio=ratio,
            shard_rows=rows,
            dim=dim,
            element_size=4,
            optimizer_state_dim=optimizer_state_dim,
            bucket_capacity=options.bucket_capacity,
            score_words=dynamicemb_util._dynamicemb_score_words(options),
            no_eviction=dynamicemb_util._dynamicemb_no_eviction(options),
        )
        options.local_hbm_for_values = memory.local_hbm_for_values
        options.caching = caching
        return options, memory

    def _keymap_bytes(self, state):
        # every fork key map allocates the same fields: keys/digests/scores
        # in the table storage, one bucket-size counter per bucket, and a
        # per-slot reference counter (which spans the overflow area too)
        km = state.key_index_map
        total = km.table_storage_.numel()
        if km.overflow_table_storage_ is not None:
            total += km.overflow_table_storage_.numel()
        total += km.bucket_sizes.numel() * 4
        if km.overflow_bucket_sizes is not None:
            total += km.overflow_bucket_sizes.numel() * 4
        total += km._ref_counter.numel() * 4
        return int(total)

    def _tier_bytes(self, state):
        return int(sum(t.num_bytes() for t in state.tables))

    def _module_bytes(self, module):
        """(hbm values, host values, hbm keymap) of a constructed module."""
        storage = module._storage
        cache = module._cache
        layout = module._storage_layout
        from dynamicemb.batched_dynamicemb_tables import StorageLayout

        if layout == StorageLayout.CACHING:
            return (
                self._tier_bytes(cache._state),
                self._tier_bytes(storage._state),
                self._keymap_bytes(cache._state) + self._keymap_bytes(storage._state),
            )
        if layout == StorageLayout.HYBRID:
            return (
                self._tier_bytes(storage._hbm),
                self._tier_bytes(storage._host),
                self._keymap_bytes(storage._hbm) + self._keymap_bytes(storage._host),
            )
        if layout == StorageLayout.HOST_ONLY:
            return (
                0,
                self._tier_bytes(storage._state),
                self._keymap_bytes(storage._state),
            )
        assert layout == StorageLayout.HBM_ONLY
        return (
            self._tier_bytes(storage._state),
            0,
            self._keymap_bytes(storage._state),
        )

    def _build_module(self, options_list, optim="SGD"):
        from dynamicemb.batched_dynamicemb_tables import (
            BatchedDynamicEmbeddingTablesV2,
        )
        from fbgemm_gpu.split_embedding_configs import EmbOptimType

        optim_type = {
            "SGD": EmbOptimType.EXACT_SGD,
            "ROWWISE": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        }[optim]
        device = torch.device("cuda", torch.cuda.current_device())
        return BatchedDynamicEmbeddingTablesV2(
            table_options=list(options_list),
            table_names=[f"table_parity_{i}" for i in range(len(options_list))],
            feature_table_map=list(range(len(options_list))),
            device=device,
            optimizer=optim_type,
        )

    @parameterized.expand(
        [
            param("hybrid_small", False, 0.02, 100_003, 64, None, "STEP", "SGD", 0),
            param("hybrid_half", False, 0.5, 100_003, 64, None, "STEP", "SGD", 0),
            param(
                "hybrid_near_full", False, 0.9999, 100_003, 64, None, "STEP", "SGD", 0
            ),
            param("hybrid_wide_bucket", False, 0.3, 100_003, 64, 256, "STEP", "SGD", 0),
            param("hybrid_ftrl", False, 0.25, 100_003, 64, None, "STEP", "FTRL", 128),
            param(
                "hybrid_rowwise", False, 0.25, 100_003, 63, None, "STEP", "ROWWISE", 4
            ),
            param("caching_small", True, 0.02, 100_003, 64, None, "STEP", "SGD", 0),
            param("caching_tenth", True, 0.1, 100_003, 64, None, "STEP", "SGD", 0),
            param("caching_most", True, 0.9, 100_003, 64, None, "STEP", "SGD", 0),
            param("caching_small_table", True, 0.1, 1000, 64, None, "STEP", "SGD", 0),
            param(
                "caching_no_eviction",
                True,
                0.25,
                100_003,
                64,
                None,
                "NO_EVICTION",
                "SGD",
                0,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_partial_branch_matches_constructed_storage(
        self,
        name,
        caching,
        ratio,
        rows,
        dim,
        bucket_capacity,
        score,
        optim,
        optimizer_state_dim,
    ):
        options, memory = self._make_options(
            rows=rows,
            dim=dim,
            ratio=ratio,
            caching=caching,
            bucket_capacity=bucket_capacity,
            score=score,
            optimizer_state_dim=optimizer_state_dim,
        )
        module = self._build_module(
            [options], optim="SGD" if optim != "ROWWISE" else "ROWWISE"
        )
        if optim == "FTRL":
            # rebuild with the dynamicemb FTRL optimizer type
            from dynamicemb.batched_dynamicemb_tables import (
                BatchedDynamicEmbeddingTablesV2,
            )
            from dynamicemb.optimizer import DynamicEmbOptimType

            device = torch.device("cuda", torch.cuda.current_device())
            module = BatchedDynamicEmbeddingTablesV2(
                table_options=[options],
                table_names=["table_parity_0"],
                feature_table_map=[0],
                device=device,
                optimizer=DynamicEmbOptimType.FTRL,
            )
        hbm_values, host_values, keymap = self._module_bytes(module)
        self.assertEqual(hbm_values, memory.hbm_values)
        self.assertEqual(host_values, memory.host_values)
        self.assertEqual(keymap, memory.hbm_index)

    @parameterized.expand(
        [
            param("full_sgd", "SGD", 0, "STEP"),
            param("full_ftrl", "FTRL", 128, "STEP"),
            param("full_rowwise", "ROWWISE", 4, "STEP"),
            # a policy the runtime cannot split reaches only the HBM_ONLY
            # branch, so the envelope equals it exactly
            param("full_no_eviction", "SGD", 0, "NO_EVICTION"),
        ],
        name_func=parameterized_name_func,
    )
    def test_full_endpoint_covers_hbm_only_module(self, name, optim, state_dim, score):
        # a lone full-budget table forms a fitting group: the runtime
        # allocates HBM_ONLY, and the envelope (which also reserves the
        # mixed-group minimum host bucket and HYBRID rounding) covers it
        options, memory = self._make_options(
            rows=100_003,
            dim=64,
            ratio=1.0,
            caching=False,
            score=score,
            optimizer_state_dim=state_dim,
        )
        module = self._build_module(
            [options], optim="ROWWISE" if optim == "ROWWISE" else "SGD"
        )
        if optim == "FTRL":
            from dynamicemb.batched_dynamicemb_tables import (
                BatchedDynamicEmbeddingTablesV2,
            )
            from dynamicemb.optimizer import DynamicEmbOptimType

            module = BatchedDynamicEmbeddingTablesV2(
                table_options=[options],
                table_names=["table_parity_0"],
                feature_table_map=[0],
                device=torch.device("cuda", torch.cuda.current_device()),
                optimizer=DynamicEmbOptimType.FTRL,
            )
        hbm_values, host_values, keymap = self._module_bytes(module)
        self.assertEqual(module._storage_layout.name, "HBM_ONLY")
        self.assertEqual(host_values, 0)
        if score == "NO_EVICTION":
            # the exact branch: HYBRID is unreachable for this policy
            self.assertEqual(hbm_values, memory.hbm_values)
            self.assertEqual(keymap, memory.hbm_index)
        else:
            self.assertLessEqual(hbm_values, memory.hbm_values)
            self.assertLessEqual(keymap, memory.hbm_index)
        self.assertEqual(memory.local_hbm_for_values, options.local_hbm_for_values)

    def test_zero_endpoint_covers_host_only_module(self):
        # a lone zero-budget table forms an all-zero group: the runtime
        # allocates HOST_ONLY and the envelope's host branch matches
        options, memory = self._make_options(
            rows=100_003, dim=64, ratio=0.0, caching=False
        )
        module = self._build_module([options])
        hbm_values, host_values, keymap = self._module_bytes(module)
        self.assertEqual(hbm_values, 0)
        self.assertEqual(host_values, memory.host_values)
        self.assertLessEqual(keymap, memory.hbm_index)

    def test_mixed_zero_and_partial_group_covered_by_envelope(self):
        # a zero-budget member fused with a partial member goes HYBRID: the
        # member acquires a minimum HBM tier, and the additive envelope
        # must cover the whole module's allocation
        zero, zero_mem = self._make_options(
            rows=100_003, dim=64, ratio=0.0, caching=False
        )
        partial, partial_mem = self._make_options(
            rows=100_003, dim=64, ratio=0.25, caching=False
        )
        module = self._build_module([zero, partial])
        hbm_values, host_values, keymap = self._module_bytes(module)
        self.assertEqual(module._storage_layout.name, "HYBRID")
        self.assertLessEqual(hbm_values, zero_mem.hbm_values + partial_mem.hbm_values)
        self.assertLessEqual(
            host_values, zero_mem.host_values + partial_mem.host_values
        )
        self.assertLessEqual(keymap, zero_mem.hbm_index + partial_mem.hbm_index)
        # budgets stay per-table: the zero member keeps its zero budget
        self.assertEqual(zero.local_hbm_for_values, 0)
        self.assertEqual(partial.local_hbm_for_values, partial_mem.local_hbm_for_values)

    def test_mixed_full_and_partial_group_covered_by_envelope(self):
        # a fully funded member fused with a partial member goes HYBRID: it
        # acquires a minimum host tier instead of the HBM_DIRECT path
        full, full_mem = self._make_options(
            rows=100_003, dim=64, ratio=1.0, caching=False
        )
        partial, partial_mem = self._make_options(
            rows=100_003, dim=64, ratio=0.125, caching=False
        )
        module = self._build_module([full, partial])
        hbm_values, host_values, keymap = self._module_bytes(module)
        self.assertEqual(module._storage_layout.name, "HYBRID")
        self.assertGreater(host_values, 0)
        self.assertLessEqual(hbm_values, full_mem.hbm_values + partial_mem.hbm_values)
        self.assertLessEqual(
            host_values, full_mem.host_values + partial_mem.host_values
        )
        self.assertLessEqual(keymap, full_mem.hbm_index + partial_mem.hbm_index)

    def test_all_zero_group_is_host_only(self):
        zero, _ = self._make_options(rows=100_003, dim=64, ratio=0.0, caching=False)
        zero2, _ = self._make_options(rows=1000, dim=32, ratio=0.0, caching=False)
        module = self._build_module([zero, zero2])
        _, host_values, _ = self._module_bytes(module)
        self.assertEqual(module._storage_layout.name, "HOST_ONLY")
        self.assertGreater(host_values, 0)

    def test_all_full_group_is_hbm_only(self):
        full, _ = self._make_options(rows=100_003, dim=64, ratio=1.0, caching=False)
        full2, _ = self._make_options(rows=1000, dim=32, ratio=1.0, caching=False)
        module = self._build_module([full, full2])
        hbm_values, host_values, _ = self._module_bytes(module)
        self.assertEqual(module._storage_layout.name, "HBM_ONLY")
        self.assertGreater(hbm_values, 0)
        self.assertEqual(host_values, 0)

    def test_reordering_members_keeps_allocations(self):
        specs = [
            self._make_options(rows=100_003, dim=64, ratio=0.0, caching=False),
            self._make_options(rows=50_000, dim=64, ratio=0.25, caching=False),
            self._make_options(rows=1000, dim=32, ratio=1.0, caching=False),
        ]
        forward = self._build_module([o for o, _ in specs])
        reverse = self._build_module([o for o, _ in reversed(specs)])
        self.assertEqual(forward._storage_layout.name, reverse._storage_layout.name)
        self.assertEqual(self._module_bytes(forward), self._module_bytes(reverse))


if __name__ == "__main__":
    unittest.main()
