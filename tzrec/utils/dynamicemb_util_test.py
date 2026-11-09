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
from torchrec.distributed.types import ShardingType
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
    """Pure capacity-mirror invariants of the four layouts."""

    ROWS = 100_003  # unaligned on purpose
    DIM = 64
    ELEMENT_SIZE = 4

    def _aligned(self):
        return dynamicemb_util.dynamicemb_aligned_capacity(self.ROWS, None)

    def _memory(self, layout, ratio, **kwargs):
        return dynamicemb_util.dynamicemb_table_memory(
            layout=layout,
            ratio=ratio,
            shard_rows=kwargs.pop("shard_rows", self.ROWS),
            dim=self.DIM,
            element_size=self.ELEMENT_SIZE,
            **kwargs,
        )

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

    def test_host_only_keeps_hbm_metadata_but_no_hbm_values(self):
        memory = self._memory(dynamicemb_util.DYNAMICEMB_HOST_ONLY, 0.0)
        aligned = 782 * 128
        row_bytes = self.DIM * self.ELEMENT_SIZE
        self.assertEqual(memory.hbm_values, 0)
        self.assertEqual(memory.hbm_rows, 0)
        self.assertEqual(memory.resident_rows, 0)
        self.assertEqual(memory.host_rows, aligned)
        self.assertEqual(memory.host_values, aligned * row_bytes)
        self.assertEqual(memory.local_hbm_for_values, 0)
        self.assertEqual(memory.value_fraction, 0.0)
        # per-slot: key 8 + digest 1 + score 8 + row index 8 = 25 bytes
        self.assertEqual(memory.hbm_index, 25 * aligned + 4 * 782)

    def test_hbm_only_is_fully_funded_with_no_host_tier(self):
        memory = self._memory(dynamicemb_util.DYNAMICEMB_HBM_ONLY, 1.0)
        aligned = 782 * 128
        row_bytes = self.DIM * self.ELEMENT_SIZE
        self.assertEqual(memory.host_values, 0)
        self.assertEqual(memory.host_rows, 0)
        self.assertEqual(memory.hbm_rows, aligned)
        self.assertEqual(memory.hbm_values, aligned * row_bytes)
        # the budget funds the complete runtime value size
        self.assertEqual(memory.local_hbm_for_values, aligned * row_bytes)
        self.assertEqual(memory.value_fraction, 1.0)
        # per-slot: key + digest + score + ref counter 4 + row index 8 = 29
        self.assertEqual(memory.hbm_index, 29 * aligned + 4 * 782)

    def test_hybrid_splits_values_across_both_tiers(self):
        memory = self._memory(dynamicemb_util.DYNAMICEMB_HYBRID, 0.25)
        row_bytes = self.DIM * self.ELEMENT_SIZE
        self.assertEqual(memory.hbm_values, memory.hbm_rows * row_bytes)
        self.assertEqual(memory.host_values, memory.host_rows * row_bytes)
        self.assertGreater(memory.local_hbm_for_values, 0)
        self.assertLess(memory.value_fraction, 1.0)
        # both tiers round up to whole buckets, so together they cover the
        # full aligned table (and may exceed it by their roundings)
        self.assertGreaterEqual(memory.hbm_rows + memory.host_rows, self._aligned())
        # the budget comes from the aligned full shard, not the raw rows
        self.assertEqual(
            memory.local_hbm_for_values,
            int(0.25 * self._aligned() * row_bytes),
        )

    def test_hybrid_near_full_split_keeps_a_minimum_host_bucket(self):
        # a near-full split still allocates at least one row on host, which
        # the runtime rounds up to a whole bucket
        memory = self._memory(dynamicemb_util.DYNAMICEMB_HYBRID, 0.9999)
        self.assertEqual(memory.host_rows, 128)
        self.assertLess(memory.value_fraction, 1.0)

    def test_caching_keeps_a_full_backing_table_and_bills_overflow(self):
        memory = self._memory(dynamicemb_util.DYNAMICEMB_CACHING, 0.1)
        row_bytes = self.DIM * self.ELEMENT_SIZE
        aligned = 782 * 128
        # host tier is the full backing table
        self.assertEqual(memory.host_rows, aligned)
        self.assertEqual(memory.host_values, aligned * row_bytes)
        # the cache realizes whole 1024-row buckets above the request
        self.assertEqual(memory.resident_rows % 1024, 0)
        self.assertGreaterEqual(memory.resident_rows, 0.1 * aligned)
        # overflow is billed to HBM but not credited to residency
        self.assertEqual(memory.hbm_rows - memory.resident_rows, 3 * 1024)
        self.assertEqual(memory.hbm_values, memory.hbm_rows * row_bytes)
        self.assertAlmostEqual(memory.value_fraction, memory.resident_rows / aligned)

    def test_caching_value_budget_uses_the_aligned_denominator(self):
        row_bytes = self.DIM * self.ELEMENT_SIZE
        aligned = 782 * 128
        for ratio in (0.02, 0.1, 0.5, 0.9):
            memory = self._memory(dynamicemb_util.DYNAMICEMB_CACHING, ratio)
            self.assertEqual(
                memory.local_hbm_for_values, int(ratio * aligned * row_bytes)
            )

    def test_optimizer_state_widens_every_value_row(self):
        base = self._memory(dynamicemb_util.DYNAMICEMB_HBM_ONLY, 1.0)
        with_ftrl = self._memory(
            dynamicemb_util.DYNAMICEMB_HBM_ONLY, 1.0, optimizer_state_dim=2 * self.DIM
        )
        self.assertEqual(
            with_ftrl.hbm_values, base.hbm_values + base.hbm_rows * 2 * self.DIM * 4
        )

    def test_second_score_word_widens_the_key_map(self):
        base = self._memory(dynamicemb_util.DYNAMICEMB_HOST_ONLY, 0.0)
        compound = self._memory(
            dynamicemb_util.DYNAMICEMB_HOST_ONLY, 0.0, score_words=2
        )
        aligned = 782 * 128
        self.assertEqual(compound.hbm_index - base.hbm_index, 8 * aligned)

    def test_no_eviction_key_map_is_twice_the_value_buffer(self):
        memory = self._memory(
            dynamicemb_util.DYNAMICEMB_HBM_ONLY, 1.0, no_eviction=True
        )
        # the key map holds 1/0.5 = 2x the rows, rounded to whole buckets
        slots = -(-2 * self.ROWS // 128) * 128
        rows = slots // 2
        self.assertEqual(memory.hbm_rows, rows)
        # per slot: key + digest + score + ref + row index
        self.assertEqual(memory.hbm_index, 29 * slots + 4 * (slots // 128))
        self.assertGreater(memory.local_hbm_for_values, self.ROWS * 64 * 4)

    def test_invalid_requests_are_rejected(self):
        for bad_ratio in (-0.1, 1.5, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self._memory(dynamicemb_util.DYNAMICEMB_HYBRID, bad_ratio)
        with self.assertRaises(ValueError):
            self._memory(dynamicemb_util.DYNAMICEMB_HYBRID, 0.5, no_eviction=True)
        with self.assertRaises(ValueError):
            self._memory(dynamicemb_util.DYNAMICEMB_HOST_ONLY, 0.5)
        with self.assertRaises(ValueError):
            self._memory(dynamicemb_util.DYNAMICEMB_HBM_ONLY, 0.5)
        with self.assertRaises(ValueError):
            self._memory("SOME_OTHER_LAYOUT", 0.5)
        with self.assertRaises(ValueError):
            self._memory(dynamicemb_util.DYNAMICEMB_HYBRID, 0.5, bucket_capacity=0)


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
        # the estimator reads the placement off the planner layout
        dynamicemb_options.tzrec_layout = dynamicemb_util.DYNAMICEMB_HBM_ONLY
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
    """Layout-dispatched effective HBM-hit ratio for the perf model."""

    def test_hbm_only_takes_the_full_hbm_path(self):
        # regardless of cache statistics inherited from a base constraint
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_HBM_ONLY, 0.0, stats=object()
            ),
            1.0,
        )

    def test_host_only_has_zero_hbm_hit_contribution(self):
        # no HYBRID intercept at ratio zero: every read reaches the host
        self.assertEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_HOST_ONLY, 0.0
            ),
            0.0,
        )

    def test_partial_layouts_keep_their_empirical_bases(self):
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_CACHING, 0.5
            ),
            0.28 + 0.01 * 0.5,
        )
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_HYBRID, 0.5
            ),
            0.11 + 0.01 * 0.5,
        )

    def test_measured_stats_apply_to_caching_only(self):
        class _Stats:
            def expected_miss_rate(self, x):
                return 0.25

        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_CACHING, 0.5, stats=_Stats()
            ),
            0.75,
        )
        # miss statistics describe a cache, not a value partition
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_HYBRID, 0.5, stats=_Stats()
            ),
            0.11 + 0.01 * 0.5,
        )

    def test_zero_fraction_of_partial_layouts_keeps_the_base(self):
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_CACHING, 0.0
            ),
            0.28,
        )
        self.assertAlmostEqual(
            dynamicemb_util._dynamicemb_effective_cache_ratio(
                dynamicemb_util.DYNAMICEMB_HYBRID, 0.0
            ),
            0.11,
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class TZRecDynamicEmbTableOptionsTest(unittest.TestCase):
    """The planner layout rides the upstream grouping key through TorchRec."""

    def _options(self, layout=None, **kwargs):
        return dynamicemb_util.TZRecDynamicEmbTableOptions(
            tzrec_layout=layout, **kwargs
        )

    def test_layout_joins_the_grouped_key(self):
        base = self._options().get_grouped_key()
        self.assertIn("tzrec_layout", base)
        self.assertEqual(base["tzrec_layout"], None)
        self.assertEqual(
            self._options(layout="HOST_ONLY").get_grouped_key()["tzrec_layout"],
            "HOST_ONLY",
        )

    def test_different_layouts_never_group(self):
        host = self._options(layout="HOST_ONLY")
        hybrid = self._options(layout="HYBRID")
        self.assertNotEqual(host.get_grouped_key(), hybrid.get_grouped_key())
        self.assertNotEqual(hash(host), hash(hybrid))

    def test_same_layout_different_ratio_groups(self):
        # a table's HBM share is not a grouping field: same-layout tables
        # keep fusing and keep their individual budgets
        low = self._options(layout="HYBRID")
        high = self._options(layout="HYBRID")
        self.assertEqual(low, high)
        self.assertEqual(hash(low), hash(high))
        self.assertNotEqual(
            low.get_grouped_key(),
            self._options(layout="CACHING").get_grouped_key(),
        )

    def test_layout_inequality_is_not_field_wise(self):
        # the underlying dataclass fields may differ without breaking fusion
        a = self._options(layout="HYBRID", max_capacity=10, dim=8)
        b = self._options(layout="HYBRID", max_capacity=20, dim=16)
        self.assertEqual(a, b)

    def test_deepcopy_and_pickle_preserve_the_subtype(self):
        import copy
        import pickle

        options = self._options(layout="HOST_ONLY")
        self.assertIs(options.__class__, dynamicemb_util.TZRecDynamicEmbTableOptions)
        self.assertEqual(
            copy.deepcopy(options).get_grouped_key(), options.get_grouped_key()
        )
        self.assertIs(type(copy.deepcopy(options)), type(options))
        roundtrip = pickle.loads(pickle.dumps(options))
        self.assertIs(type(roundtrip), type(options))
        self.assertEqual(roundtrip.get_grouped_key(), options.get_grouped_key())

    def test_torchrec_group_tables_separates_layouts(self):
        # the real TorchRec grouping path: fused params (which carry the
        # options object) decide which tables share one fused module
        from torchrec.distributed.embedding_sharding import (
            ShardedEmbeddingTable,
            group_tables,
        )
        from torchrec.distributed.embedding_types import EmbeddingComputeKernel
        from torchrec.modules.embedding_configs import PoolingType

        def table(name, layout):
            options = self._options(layout=layout)
            return ShardedEmbeddingTable(
                num_embeddings=1024,
                embedding_dim=16,
                name=name,
                feature_names=[name],
                pooling=PoolingType.SUM,
                compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL,
                local_rows=1024,
                local_cols=16,
                fused_params={"dynamicemb_options": options},
            )

        tables_per_rank = [
            [
                table("t_host", "HOST_ONLY"),
                table("t_host2", "HOST_ONLY"),
                table("t_hyb", "HYBRID"),
                table("t_cache", "CACHING"),
                table("t_hbm", "HBM_ONLY"),
            ]
        ]
        grouped = group_tables(tables_per_rank)[0]
        layouts = []
        for group in grouped:
            names = [t.name for t in group.embedding_tables]
            layouts.append(tuple(sorted(names)))
        self.assertEqual(
            sorted(layouts),
            [("t_cache",), ("t_hbm",), ("t_host", "t_host2"), ("t_hyb",)],
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@mark_ci_scope("gpu")
class DynamicEmbPlanConversionTest(unittest.TestCase):
    """Plan conversion turns a normalized candidate into runtime options."""

    def _convert(
        self,
        layout,
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
        # layout, partial pins keep both families at the ratio
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
            and so.dynamicemb_options.tzrec_layout == layout
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

    def test_host_only_pin(self):
        param_sharding, _ = self._convert("HOST_ONLY", 0.0)
        options = param_sharding.dynamicemb_options
        self.assertEqual(options.tzrec_layout, "HOST_ONLY")
        self.assertFalse(options.caching)
        self.assertEqual(options.local_hbm_for_values, 0)
        self.assertTrue(hasattr(param_sharding, "initial_accumulator_value"))

    def test_hbm_only_pin(self):
        from tzrec.optim.optimizer import FTRL

        param_sharding, sharding_option = self._convert(
            "HBM_ONLY", 1.0, optimizer_class=FTRL
        )
        options = param_sharding.dynamicemb_options
        self.assertEqual(options.tzrec_layout, "HBM_ONLY")
        self.assertFalse(options.caching)
        self.assertTrue(options.training)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        # FTRL doubles the value row; the full aligned table is funded
        self.assertEqual(options.local_hbm_for_values, aligned * (dim + 2 * dim) * 4)

    def test_hybrid_pin_budget_matches_mirror(self):
        param_sharding, sharding_option = self._convert("HYBRID", 0.5)
        options = param_sharding.dynamicemb_options
        self.assertEqual(options.tzrec_layout, "HYBRID")
        self.assertFalse(options.caching)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        self.assertEqual(options.local_hbm_for_values, int(0.5 * aligned * dim * 4))

    def test_caching_pin_budget_matches_mirror(self):
        param_sharding, sharding_option = self._convert("CACHING", 0.25)
        options = param_sharding.dynamicemb_options
        self.assertEqual(options.tzrec_layout, "CACHING")
        self.assertTrue(options.caching)
        rows, dim = sharding_option.shards[0].size
        aligned = -(-rows // 128) * 128
        self.assertEqual(options.local_hbm_for_values, int(0.25 * aligned * dim * 4))

    def test_missing_layout_is_rejected(self):
        # an unassigned base option never reaches an emitted plan
        from torchrec.distributed.planner import planners

        cfg = feature_pb2.DynamicEmbedding()
        cfg.max_capacity = 4096
        cfg.score_strategy = "STEP"
        constraint = dynamicemb_util.build_dynamicemb_constraints(
            cfg,
            EmbeddingBagConfig(
                num_embeddings=4096,
                embedding_dim=32,
                name="table_de",
                feature_names=["feat_de"],
            ),
        )
        table = EmbeddingBagConfig(
            num_embeddings=4096,
            embedding_dim=32,
            name="table_de",
            feature_names=["feat_de"],
        )
        model = TestSparseNN(tables=[table], sparse_device=torch.device("meta"))
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=get_default_sharders()
        )
        sharding_option = next(
            so for so in search_space if getattr(so, "use_dynamicemb", False)
        )
        # a caller that strips the planner layout must fail loudly
        sharding_option.dynamicemb_options.tzrec_layout = None
        for rank, shard in enumerate(sharding_option.shards):
            shard.rank = rank
        with self.assertRaisesRegex(ValueError, "no planner layout"):
            planners.to_sharding_plan([sharding_option], topology)


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb, "dynamicemb is not installed; skipping."
)
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required.")
@mark_ci_scope("gpu")
class CapacityMirrorRuntimeParityTest(unittest.TestCase):
    """The capacity mirror matches what the pinned runtime really allocates.

    Constructs the planner's options for one table per case, builds the
    actual storage through ``BatchedDynamicEmbeddingTablesV2`` (the same
    constructor the sharded module uses), and compares every tier's value
    bytes and HBM key-map metadata against the mirror.
    """

    def _build_options(
        self,
        layout,
        ratio,
        *,
        rows=100_003,
        dim=64,
        bucket_capacity=None,
        score="STEP",
        optim="SGD",
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
        options.init_capacity = rows
        options.embedding_dtype = torch.float32
        options.index_type = torch.int64
        options.tzrec_layout = layout
        memory = dynamicemb_util.dynamicemb_table_memory(
            layout=layout,
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
        options.caching = layout == "CACHING"
        return options, memory

    def _keymap_bytes(self, state):
        km = state.key_index_map
        total = km.table_storage_.numel()
        if km.overflow_table_storage_ is not None:
            total += km.overflow_table_storage_.numel()
        total += km.bucket_sizes.numel() * 4
        if km.overflow_bucket_sizes is not None:
            total += km.overflow_bucket_sizes.numel() * 4
        if km.rows_ is not None:
            total += km.rows_.numel() * 8
        if km._ref_counter is not None:
            total += km._ref_counter.numel() * 4
        return int(total)

    def _tier_bytes(self, state):
        return int(sum(t.num_bytes() for t in state.tables))

    @parameterized.expand(
        [
            param(
                "host_only_sgd", "HOST_ONLY", 0.0, 100_003, 64, None, "STEP", "SGD", 0
            ),
            param("hbm_only_sgd", "HBM_ONLY", 1.0, 100_003, 64, None, "STEP", "SGD", 0),
            param(
                "hbm_only_ftrl", "HBM_ONLY", 1.0, 100_003, 64, None, "STEP", "FTRL", 128
            ),
            param(
                "hbm_only_rowwise",
                "HBM_ONLY",
                1.0,
                100_003,
                63,
                None,
                "STEP",
                "ROWWISE",
                4,
            ),
            param(
                "hbm_only_no_eviction",
                "HBM_ONLY",
                1.0,
                100_003,
                64,
                None,
                "NO_EVICTION",
                "SGD",
                0,
            ),
            param("hybrid_small", "HYBRID", 0.02, 100_003, 64, None, "STEP", "SGD", 0),
            param("hybrid_half", "HYBRID", 0.5, 100_003, 64, None, "STEP", "SGD", 0),
            param(
                "hybrid_near_full",
                "HYBRID",
                0.9999,
                100_003,
                64,
                None,
                "STEP",
                "SGD",
                0,
            ),
            param(
                "hybrid_wide_bucket", "HYBRID", 0.3, 100_003, 64, 256, "STEP", "SGD", 0
            ),
            param(
                "caching_small", "CACHING", 0.02, 100_003, 64, None, "STEP", "SGD", 0
            ),
            param("caching_tenth", "CACHING", 0.1, 100_003, 64, None, "STEP", "SGD", 0),
            param("caching_most", "CACHING", 0.9, 100_003, 64, None, "STEP", "SGD", 0),
            param(
                "host_only_small_table",
                "HOST_ONLY",
                0.0,
                1000,
                64,
                None,
                "STEP",
                "SGD",
                0,
            ),
            param(
                "caching_small_table", "CACHING", 0.1, 1000, 64, None, "STEP", "SGD", 0
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_mirror_matches_constructed_storage(
        self,
        name,
        layout,
        ratio,
        rows,
        dim,
        bucket_capacity,
        score,
        optim,
        optimizer_state_dim,
    ):
        from dynamicemb.batched_dynamicemb_tables import (
            BatchedDynamicEmbeddingTablesV2,
        )
        from fbgemm_gpu.split_embedding_configs import EmbOptimType

        options, memory = self._build_options(
            layout,
            ratio,
            rows=rows,
            dim=dim,
            bucket_capacity=bucket_capacity,
            score=score,
            optimizer_state_dim=optimizer_state_dim,
        )
        optim_type = {
            "SGD": EmbOptimType.EXACT_SGD,
            "FTRL": EmbOptimType.NONE,  # replaced below by dynamicemb FTRL
            "ROWWISE": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
        }[optim]
        if optim == "FTRL":
            from dynamicemb.optimizer import DynamicEmbOptimType

            optim_type = DynamicEmbOptimType.FTRL
        device = torch.device("cuda", torch.cuda.current_device())
        module = BatchedDynamicEmbeddingTablesV2(
            table_options=[options],
            table_names=["table_parity"],
            feature_table_map=[0],
            device=device,
            optimizer=optim_type,
        )
        storage = module._storage
        cache = module._cache
        self.assertEqual(module._storage_layout.name, layout)
        if layout == "CACHING":
            runtime_hbm_values = self._tier_bytes(cache._state)
            runtime_host_values = self._tier_bytes(storage._state)
            runtime_keymap = self._keymap_bytes(cache._state) + self._keymap_bytes(
                storage._state
            )
        elif layout == "HYBRID":
            runtime_hbm_values = self._tier_bytes(storage._hbm)
            runtime_host_values = self._tier_bytes(storage._host)
            runtime_keymap = self._keymap_bytes(storage._hbm) + self._keymap_bytes(
                storage._host
            )
        elif layout == "HOST_ONLY":
            runtime_hbm_values = 0
            runtime_host_values = self._tier_bytes(storage._state)
            runtime_keymap = self._keymap_bytes(storage._state)
        else:
            runtime_hbm_values = self._tier_bytes(storage._state)
            runtime_host_values = 0
            runtime_keymap = self._keymap_bytes(storage._state)

        self.assertEqual(runtime_hbm_values, memory.hbm_values)
        self.assertEqual(runtime_host_values, memory.host_values)
        self.assertEqual(runtime_keymap, memory.hbm_index)


if __name__ == "__main__":
    unittest.main()
