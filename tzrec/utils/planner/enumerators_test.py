# Copyright (c) 2024, Alibaba Group;
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
import torch
from parameterized import parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.estimator.types import HardwarePerfConfig
from torchrec.distributed.planner.types import SharderData, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ShardingType, StorageUsageType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.utils import dynamicemb_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.planner.enumerators import (
    EmbeddingEnumerator,
    _emit_dynamicemb_variants,
)
from tzrec.utils.test_util import (
    mark_ci_scope,
    parameterized_name_func,
)


class EmbeddingEnumeratorTest(unittest.TestCase):
    def test_global_constraints_apply_behind_statistics_only_entries(self) -> None:
        # a parameter entry that only carries measured factors keeps the
        # global sharding and kernel lists applied to its table
        from torchrec.distributed.planner.types import ParameterConstraints

        topology = Topology(world_size=1, compute_device="cpu")
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=8,
                name="table_a",
                feature_names=["feature_a"],
            )
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        global_constraints = ParameterConstraints(
            sharding_types=["table_wise", "row_wise"]
        )
        statistics_only = ParameterConstraints(
            pooling_factors=[2.0], feature_names=["feature_a"]
        )
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=16,
            fqn_constraints={"sparse.ebc.table_a": statistics_only},
            global_constraints=global_constraints,
        )
        search_space = enumerator.enumerate(
            module=model, sharders=get_default_sharders()
        )
        self.assertTrue(search_space)
        # the global list restricts the table; data_parallel is gone
        self.assertNotIn(
            "data_parallel", {option.sharding_type for option in search_space}
        )
        # an entry that restricts sharding itself still outranks the global
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=16,
            fqn_constraints={
                "sparse.ebc.table_a": ParameterConstraints(
                    sharding_types=["table_wise"]
                )
            },
            global_constraints=global_constraints,
        )
        search_space = enumerator.enumerate(
            module=model, sharders=get_default_sharders()
        )
        self.assertEqual(
            {option.sharding_type for option in search_space}, {"table_wise"}
        )


@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@mark_ci_scope("gpu")
class DynamicEmbEnumeratorTest(unittest.TestCase):
    """Test candidate generation and scoring with meta tensors."""

    def _options(self, factor=None, ratio=None, *, rows=4096, pooling=1):
        from dynamicemb import DynamicEmbScoreStrategy, DynamicEmbTableOptions
        from torchrec.distributed.planner.types import Shard, ShardingOption

        base = ShardingOption(
            name="t",
            module=("", torch.nn.Module()),
            partition_by="uniform",
            compute_kernel=EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
            sharding_type=ShardingType.ROW_WISE.value,
            batch_size=128,
            input_lengths=[pooling],
            tensor=torch.empty((rows, 1), dtype=torch.float32, device="meta"),
            shards=[Shard(size=[rows, 1], offset=[0, 0])],
            cache_params=None if ratio is None else CacheParams(load_factor=ratio),
        )
        base.dynamicemb_options = DynamicEmbTableOptions(
            max_capacity=rows,
            bucket_capacity=16,
            score_strategy=DynamicEmbScoreStrategy.STEP,
        )
        return _emit_dynamicemb_variants(base, factor)

    @parameterized.expand(
        [
            ("unmeasured", None, 0.1, 0.1, 3),
            ("measured_zero", 0, 1 / 4096, 0.1, 3),
            ("below_one_percent", 0.001, 1 / 4096, 0.1, 3),
            ("small_working_set", 0.1, 64 / 4096, 0.1, 3),
            ("above_default", 1.92, 0.1, 1229 / 4096, 2),
            ("half", 3.2, 0.1, 0.5, 2),
            ("above_half", 4.8, 0.1, 0.75, 1),
            ("full", 6.4, 0.1, None, 0),
            ("overfull", 12.8, 0.1, None, 0),
        ],
        name_func=parameterized_name_func,
    )
    def test_auto_grid(self, _, factor, hybrid_start, cache_start, ncache):
        partial_ratios = [*np.geomspace(hybrid_start, 0.5, 5), 0.6, 0.7, 0.8, 0.9]
        if hybrid_start == 1 / 4096:
            del partial_ratios[1]  # Same physical storage as the first point.
        opts = self._options(factor)
        noncache = [o for o in opts if not o.dynamicemb_options.caching]
        cache = [o for o in opts if o.dynamicemb_options.caching]
        full_bytes = 4096 * 4
        self.assertEqual(
            [o.dynamicemb_storage.local_hbm_for_values for o in noncache],
            [int(r * full_bytes) for r in [0, *partial_ratios, 1]],
        )
        self.assertEqual(len(cache), ncache)
        if cache:
            self.assertEqual(
                cache[0].dynamicemb_storage.local_hbm_for_values,
                int(cache_start * full_bytes),
            )

    @parameterized.expand(
        [
            ("host", 0, 1, [False]),
            ("full", 1, 1, [False]),
            ("insufficient_cache", 0.1, 9, [False]),
            ("one_byte", 1 / 16384, 1, [False, True]),
        ],
        name_func=parameterized_name_func,
    )
    def test_ratio_pin(self, _, ratio, pooling, families):
        opts = self._options(ratio=ratio, pooling=pooling)
        self.assertEqual(
            [
                (
                    o.dynamicemb_storage.local_hbm_for_values,
                    o.dynamicemb_options.caching,
                )
                for o in opts
            ],
            [(int(ratio * 16384), family) for family in families],
        )

    def test_subbyte_budget_becomes_zero_endpoint(self):
        opts = self._options(ratio=0.5 / 4096, rows=1024)
        self.assertEqual(
            [(o.cache_load_factor, o.dynamicemb_options.caching) for o in opts],
            [(0.0, False)],
        )

    @parameterized.expand(
        [("exact", 1024), ("oversized_bucket", 512)], name_func=parameterized_name_func
    )
    def test_full_cache_becomes_one_non_caching_endpoint(self, _, rows):
        options = self._options(rows=rows)
        self.assertFalse(any(o.dynamicemb_options.caching for o in options))
        self.assertEqual(sum(o.cache_load_factor == 1 for o in options), 1)

        pinned = self._options(ratio=0.1, rows=rows)
        self.assertEqual(
            [
                (
                    o.dynamicemb_storage.local_hbm_for_values,
                    o.dynamicemb_options.caching,
                )
                for o in pinned
            ],
            [(int(0.1 * rows * 4), False), (rows * 4, False)],
        )

    def test_physical_dedup_keeps_first_budget(self):
        pinned = [
            next(o for o in self._options(ratio=ratio) if o.dynamicemb_options.caching)
            for ratio in (0.6, 0.7)
        ]
        first, second = [o.dynamicemb_storage for o in pinned]
        for candidate in (first, second):
            self.assertEqual(
                (candidate.resident_rows, candidate.backing_rows), (3072, 4096)
            )
        self.assertEqual(first.storage, second.storage)
        self.assertNotEqual(first.local_hbm_for_values, second.local_hbm_for_values)
        opts = self._options()
        caches = [o for o in opts if o.dynamicemb_options.caching]
        self.assertEqual(
            [
                (o.cache_load_factor, o.dynamicemb_storage.local_hbm_for_values)
                for o in caches
            ],
            [
                (0.25, int(0.1 * 16384)),
                (0.5, int(0.1 * 5**0.75 * 16384)),
                (0.75, int(0.6 * 16384)),
            ],
        )
        self.assertEqual(caches[-1].dynamicemb_storage, first)
        hybrid, cache = self._options(ratio=0.15)
        self.assertFalse(hybrid.dynamicemb_options.caching)
        self.assertTrue(cache.dynamicemb_options.caching)
        self.assertEqual(cache.cache_load_factor, 0.25)
        for option in (hybrid, cache):
            self.assertEqual(
                option.dynamicemb_storage.local_hbm_for_values, int(0.15 * 16384)
            )

    def test_equal_tier_rows_keep_zero_partial_and_full(self):
        opts = self._options(rows=16)
        self.assertEqual(
            [
                (o.cache_load_factor, o.dynamicemb_storage.local_hbm_for_values)
                for o in opts
            ],
            [(0, 0), (1024 / 1040, 6), (1, 64)],
        )
        self.assertEqual(
            {
                (o.dynamicemb_storage.resident_rows, o.dynamicemb_storage.backing_rows)
                for o in opts
            },
            {(1024, 16)},
        )

    def test_invalid_factor(self):
        for factor in (float("nan"), -1):
            with self.assertRaisesRegex(ValueError, "unique_id_factor"):
                self._options(factor)

    @parameterized.expand(
        [
            ("zero", 0, False, None, 0),
            ("full", 1, False, None, 1),
            ("hybrid", 0.1, False, None, 0.11 + 0.01 * 1024 / (1024 + 3696)),
            ("cache", 0.1, True, None, 0.28 + 0.01 * 0.25),
            ("measured_cache", 0.1, True, 2.4, 0.6),
            ("miss_rate_above_one", 0.1, True, -1.0, 0),
            ("miss_rate_below_zero", 0.1, True, 8.0, 1),
        ],
        name_func=parameterized_name_func,
    )
    def test_performance_bandwidth_preserves_allocated_share(
        self, _, ratio, caching, stats_scale, expected_weight
    ):
        rows = 4096
        opt = next(
            o
            for o in self._options(ratio=ratio, rows=rows)
            if o.dynamicemb_options.caching == caching
        )
        original = opt.cache_params
        allocated = opt.cache_load_factor
        if stats_scale is not None:
            original.stats = mock.Mock()
            original.stats.expected_miss_rate.side_effect = lambda share: (
                1 - stats_scale * share
            )
        topology = Topology(
            world_size=1, compute_device="cuda", hbm_mem_bw=1000, hbm_to_ddr_mem_bw=20
        )
        config = HardwarePerfConfig()
        fused_key = ("cuda", EmbeddingComputeKernel.FUSED.value)
        config.kernel_device_bandwidths = {fused_key: 37.0}
        sharder_data = SharderData(
            fused_params={},
            qcomm_dtype_sizes={},
            storage_usage_type=StorageUsageType.DEFAULT,
        )
        contexts = dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
            config, [[rows, 1]], opt, topology, {}, sharder_data
        )
        self.assertEqual(len(contexts), 1)
        self.assertAlmostEqual(contexts[0].device_bw, 2 + 98 * expected_weight)
        self.assertEqual(contexts[0].caching_ratio, allocated)
        self.assertEqual(contexts[0].hbm_to_ddr_mem_bw, 20)
        self.assertEqual(
            dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
                config, [], opt, topology, {}, sharder_data
            ),
            [],
        )
        self.assertIs(opt.cache_params, original)
        self.assertEqual(opt.cache_load_factor, allocated)
        self.assertEqual(config.kernel_device_bandwidths, {fused_key: 37.0})
        opt.dynamicemb_options = None
        opt.compute_kernel = EmbeddingComputeKernel.FUSED.value
        contexts = dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
            config, [[rows, 1]], opt, topology, {}, sharder_data
        )
        self.assertEqual(contexts[0].device_bw, 37.0)


if __name__ == "__main__":
    unittest.main()
