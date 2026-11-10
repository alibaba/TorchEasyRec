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
import warnings
from functools import partial
from unittest import mock

import torch
from parameterized import param, parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.types import CacheParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.optim import optimizers, rowwise_adagrad

from tzrec.optim.optimizer import FTRL, has_fbgemm_ftrl
from tzrec.protos import feature_pb2
from tzrec.utils import dynamicemb_util
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
    def test_normal_cache_residency_excludes_overflow(self):
        size = partial(
            dynamicemb_util.dynamicemb_table_memory,
            caching=True,
            ratio=0.1,
            shard_rows=4096,
            dim=32,
            element_size=4,
        )
        m = size()
        self.assertEqual((m.resident_rows, m.hbm_rows, m.host_rows), (1024, 4096, 4096))
        self.assertEqual(m.local_hbm_for_values, int(0.1 * 4096 * 32 * 4))
        self.assertEqual(size(score_words=2).hbm_index - m.hbm_index, 8192 * 8)

    def test_invalid_placements(self):
        for caching, ratio in [
            (False, -1),
            (False, float("nan")),
            (True, 0),
            (True, 1),
        ]:
            with (
                self.subTest(caching=caching, ratio=ratio),
                self.assertRaises(ValueError),
            ):
                dynamicemb_util.dynamicemb_table_memory(
                    caching=caching, ratio=ratio, shard_rows=128, dim=16, element_size=4
                )


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
        rows = self.NUM_EMBEDDINGS // 2
        return dynamicemb_util.dynamicemb_calculate_shard_storages(
            sharder_data=None,
            sharding_type=ShardingType.ROW_WISE.value,
            tensor=torch.empty(self.NUM_EMBEDDINGS, self.EMBEDDING_DIM),
            compute_device="cuda",
            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
            shard_sizes=[[rows, self.EMBEDDING_DIM]] * 2,
            batch_sizes=[16],
            world_size=2,
            local_world_size=2,
            input_lengths=[1.0],
            num_poolings=[1.0],
            dynamicemb_memory=dynamicemb_util.dynamicemb_table_memory(
                shard_rows=rows,
                dim=self.EMBEDDING_DIM,
                element_size=4,
                ratio=1.0,
                caching=False,
            ),
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

    @parameterized.expand(
        [("aligned", 2048, 1024), ("unaligned", 1026, 65)],
        name_func=parameterized_name_func,
    )
    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA counter")
    def test_counter_hbm_lands_only_for_frequency_admission(self, _, capacity, width):
        from dynamicemb.embedding_admission import MultiTableKVCounter

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
        strategy = feature_pb2.DynamicEmbFrequencyAdmissionStrategy(
            threshold=1, counter_capacity=capacity, counter_bucket_capacity=width
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            frequency_options = self._options(frequency_admission_strategy=strategy)
        frequency = self._shard_storages(frequency_options)
        table = MultiTableKVCounter(
            [frequency_options.admit_strategy.counter], torch.device("cuda")
        ).table_
        # The pinned runtime's memory_usage() omits its reference counters.
        counter_hbm = (
            table.table_storage_.nbytes
            + table.bucket_sizes.nbytes
            + table._ref_counter.nbytes
        )
        for base, prob, freq in zip(no_admission, probabilistic, frequency):
            self.assertEqual(prob.hbm, base.hbm)
            self.assertEqual(freq.hbm - base.hbm, counter_hbm)


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
            ("single", 250, 1, 128, 1, 256, 128, 128),
            ("four_ranks", 1000, 4, 128, 512, 256, 128, 256),
            ("three_ranks", 100003, 3, 256, 512, 33536, 256, 512),
            ("one_bucket", 1000, 4, 2**63 - 1, 1, 256, 256, 256),
        ],
        name_func=parameterized_name_func,
    )
    def test_capacity(self, _, rows, ranks, bucket, initial, local, width, init):
        table = EmbeddingBagConfig(num_embeddings=rows, embedding_dim=16)
        cfg = feature_pb2.DynamicEmbedding(
            max_capacity=rows, bucket_capacity=bucket, init_capacity_per_rank=initial
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": str(ranks)}):
            opt = dynamicemb_util.build_dynamicemb_constraints(
                cfg, table
            ).dynamicemb_options
        self.assertEqual(table.num_embeddings, local * ranks)
        self.assertEqual(
            (opt.max_capacity, opt.bucket_capacity, opt.init_capacity),
            (local, width, init),
        )


@unittest.skipUnless(
    dynamicemb_util.has_dynamicemb and torch.cuda.is_available(),
    "requires dynamicemb on CUDA",
)
@mark_ci_scope("gpu")
class CapacityMirrorRuntimeParityTest(unittest.TestCase):
    @staticmethod
    def _allocated(module):
        storage = module._storage
        layout = module._storage_layout.name
        if layout == "CACHING":
            hbm, host = module._cache._state, storage._state
        elif layout == "HYBRID":
            hbm, host = storage._hbm, storage._host
        elif layout == "HOST_ONLY":
            hbm, host = None, storage._state
        else:
            assert layout == "HBM_ONLY"
            hbm, host = storage._state, None
        values = [
            sum(t.num_bytes() for t in state.tables) if state else 0
            for state in (hbm, host)
        ]
        index = 0
        for state in (hbm, host):
            if state is not None:
                km = state.key_index_map
                index += sum(
                    t.nbytes
                    for t in (
                        km.table_storage_,
                        km.bucket_sizes,
                        km._ref_counter,
                        km.overflow_table_storage_,
                        km.overflow_bucket_sizes,
                    )
                    if t is not None
                )
        return (*values, index)

    @parameterized.expand(
        [
            ("cache", "CACHING", (0.1,)),
            ("small_cache", "CACHING", (0.1,), "SGD", "STEP", 1000),
            ("cache_no_eviction", "CACHING", (0.25,), "SGD", "NO_EVICTION"),
            ("hybrid_small", "HYBRID", (0.02,)),
            ("hybrid_near_full", "HYBRID", (0.9999,)),
            ("hybrid_wide_bucket", "HYBRID", (0.3,), "SGD", "STEP", 100003, 256),
            ("ftrl", "HYBRID", (0.25,), "FTRL"),
            ("rowwise", "HYBRID", (0.25,), "ROWWISE"),
            ("zero_group", "HOST_ONLY", (0, 0)),
            ("full_group", "HBM_ONLY", (1, 1)),
            ("full_many_buckets", "HBM_ONLY", (1,), "SGD", "STEP", 262144, 16),
            ("full_ftrl", "HBM_ONLY", (1,), "FTRL"),
            ("full_rowwise", "HBM_ONLY", (1,), "ROWWISE"),
            ("no_eviction", "HBM_ONLY", (1,), "SGD", "NO_EVICTION"),
            ("host_no_eviction", "HOST_ONLY", (0,), "SGD", "NO_EVICTION"),
            ("mixed_zero", "HYBRID", (0, 0.25)),
            ("mixed_full", "HYBRID", (1, 0.125)),
            ("no_alignment_surplus", "HYBRID", (0, 1), "SGD", "STEP", 4000),
        ],
        name_func=parameterized_name_func,
    )
    def test_native_allocations(
        self, _, layout, ratios, optim="SGD", score="STEP", rows=100003, width=128
    ):
        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from dynamicemb.optimizer import DynamicEmbOptimType
        from fbgemm_gpu.split_embedding_configs import EmbOptimType

        optimizer = {
            "SGD": EmbOptimType.EXACT_SGD,
            "ROWWISE": EmbOptimType.EXACT_ROWWISE_ADAGRAD,
            "FTRL": DynamicEmbOptimType.FTRL,
        }[optim]
        options, estimates = [], []
        for i, ratio in enumerate(ratios):
            dim = 2048 if i else (63 if optim == "ROWWISE" else 64)
            capacity = 897 if i else rows
            state_dim = {"SGD": 0, "ROWWISE": 4, "FTRL": 2 * dim}[optim]
            table = EmbeddingBagConfig(num_embeddings=capacity, embedding_dim=dim)
            cfg = feature_pb2.DynamicEmbedding(
                max_capacity=capacity, bucket_capacity=width, score_strategy=score
            )
            with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
                opt = dynamicemb_util.build_dynamicemb_constraints(
                    cfg, table
                ).dynamicemb_options
            opt.dim, opt.embedding_dtype, opt.index_type = (
                dim,
                torch.float32,
                torch.int64,
            )
            opt.init_capacity = opt.max_capacity
            opt.caching = layout == "CACHING"
            m = dynamicemb_util.dynamicemb_table_memory(
                caching=opt.caching,
                ratio=ratio,
                shard_rows=opt.max_capacity,
                dim=dim,
                element_size=4,
                optimizer_state_dim=state_dim,
                bucket_capacity=width,
                no_eviction=score == "NO_EVICTION",
            )
            opt.local_hbm_for_values = m.local_hbm_for_values
            self.assertEqual(
                opt.local_hbm_for_values,
                int(ratio * opt.max_capacity * (dim + state_dim) * 4),
            )
            options.append(opt)
            estimates.append((m.hbm_values, m.host_values, m.hbm_index))

        def build(opts):
            return BatchedDynamicEmbeddingTablesV2(
                table_options=opts,
                table_names=[f"t{i}" for i in range(len(opts))],
                feature_table_map=list(range(len(opts))),
                device=torch.device("cuda", torch.cuda.current_device()),
                optimizer=optimizer,
            )

        module = build(options)
        self.assertEqual(module._storage_layout.name, layout)
        actual = self._allocated(module)
        expected = tuple(map(sum, zip(*estimates)))
        if layout in ("CACHING", "HYBRID") or score == "NO_EVICTION":
            self.assertEqual(actual, expected)
        else:
            for used, reserved in zip(actual, expected):
                self.assertLessEqual(used, reserved)
        if ratios == (0, 1):
            self.assertEqual(self._allocated(build(options[::-1])), actual)


if __name__ == "__main__":
    unittest.main()
