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
from types import SimpleNamespace

import torch
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.plan_util import DynamicProgrammingProposer, _emit_dynamicemb_variants


class PlanUtilTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    def test_dp_proposer(self) -> None:
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = EmbeddingEnumerator(topology=topology, batch_size=8196)
        partitioner = GreedyPerfPartitioner()

        tables = [
            EmbeddingBagConfig(
                num_embeddings=1000**i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = enumerator.enumerate(
            module=model,
            sharders=get_default_sharders(),
        )

        dp_proposer = DynamicProgrammingProposer()
        dp_proposer.load(search_space)
        best_dp_perf = float("inf")
        best_dp_proposal = None
        num_proposals = 0
        proposal = dp_proposer.propose()
        while proposal:
            num_proposals += 1
            try:
                partitioner.partition(proposal, topology)
                cur_perf = sum([x.total_perf for x in proposal])
                if cur_perf < best_dp_perf:
                    best_dp_proposal = {x.fqn: x for x in proposal}
                    best_dp_perf = cur_perf
            except PlannerError:
                pass
            dp_proposer.feedback(partitionable=True, storage_constraint=topology)
            proposal = dp_proposer.propose()
        self.assertEqual(num_proposals, 3)

        grid_proposer = GridSearchProposer()
        grid_proposer.load(search_space)
        best_grid_perf = float("inf")
        best_grid_proposal = None
        proposal = grid_proposer.propose()
        while proposal:
            try:
                partitioner.partition(proposal, topology)
                cur_perf = sum([x.total_perf for x in proposal])
                if cur_perf < best_grid_perf:
                    best_grid_proposal = {x.fqn: x for x in proposal}
                    best_grid_perf = cur_perf
            except PlannerError:
                pass
            grid_proposer.feedback(partitionable=True)
            proposal = grid_proposer.propose()

        self.assertAlmostEqual(best_dp_perf, best_grid_perf)
        for k, v in best_grid_proposal.items():
            self.assertEqual(str(v), str(best_dp_proposal[k]))

    def test_dp_proposer_with_prune(self) -> None:
        topology = Topology(
            world_size=2,
            hbm_cap=(1000**3) * 10 * 2 * 4,
            compute_device="cuda" if torch.cuda.is_available() else "cpu",
        )
        enumerator = EmbeddingEnumerator(topology=topology, batch_size=8196)
        partitioner = GreedyPerfPartitioner()

        tables = [
            EmbeddingBagConfig(
                num_embeddings=1000**i,
                embedding_dim=10 * i,
                name="table_" + str(i),
                feature_names=["feature_" + str(i)],
            )
            for i in range(1, 4)
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        search_space = enumerator.enumerate(
            module=model,
            sharders=get_default_sharders(),
        )

        dp_proposer = DynamicProgrammingProposer()
        dp_proposer.load(search_space)
        best_dp_perf = float("inf")
        best_dp_proposal = None
        num_proposals = 0
        proposal = dp_proposer.propose()
        while proposal:
            num_proposals += 1
            try:
                partitioner.partition(proposal, topology)
                cur_perf = sum([x.total_perf for x in proposal])
                if cur_perf < best_dp_perf:
                    best_dp_proposal = {x.fqn: x for x in proposal}
                    best_dp_perf = cur_perf
            except PlannerError:
                pass
            dp_proposer.feedback(partitionable=True, storage_constraint=topology)
            proposal = dp_proposer.propose()
        self.assertEqual(
            best_dp_proposal["sparse.ebc.table_3"].sharding_type,
            "row_wise" if torch.cuda.is_available() else "table_wise",
        )


class EmitDynamicEmbVariantsTest(unittest.TestCase):
    """``_emit_dynamicemb_variants`` produces { HYBRID, CACHING } × factors."""

    def _make_base(self, cache_params=None):
        # _emit_dynamicemb_variants only touches cache_params and
        # dynamicemb_options on the option, so a SimpleNamespace stand-in is
        # sufficient and avoids the heavy ShardingOption constructor surface.
        return SimpleNamespace(
            cache_params=cache_params,
            dynamicemb_options=SimpleNamespace(caching=False, bucket_capacity=128),
        )

    def test_unfixed_factor_emits_twenty_variants(self):
        variants = _emit_dynamicemb_variants(self._make_base(cache_params=None))
        self.assertEqual(len(variants), 20)
        cache_modes = sorted({v.dynamicemb_options.caching for v in variants})
        self.assertEqual(cache_modes, [False, True])
        for caching_mode in (False, True):
            factors = sorted(
                v.cache_params.load_factor
                for v in variants
                if v.dynamicemb_options.caching is caching_mode
            )
            self.assertEqual(factors, [round((i + 1) / 10, 4) for i in range(10)])

    def test_fixed_factor_emits_two_variants(self):
        variants = _emit_dynamicemb_variants(
            self._make_base(cache_params=CacheParams(load_factor=0.3))
        )
        self.assertEqual(len(variants), 2)
        cache_modes = sorted(v.dynamicemb_options.caching for v in variants)
        self.assertEqual(cache_modes, [False, True])
        for v in variants:
            self.assertEqual(v.cache_params.load_factor, 0.3)

    def test_variants_own_dynamicemb_options(self):
        # Per-variant mutation of caching must not bleed across variants.
        base = self._make_base()
        original_opts = base.dynamicemb_options
        variants = _emit_dynamicemb_variants(base)
        for v in variants:
            self.assertIsNot(v.dynamicemb_options, original_opts)

    def test_stats_preserved_on_clone(self):
        class _Stats:
            expected_lookups = 100.0

            def expected_miss_rate(self, ratio):
                return 0.1

        stats = _Stats()
        variants = _emit_dynamicemb_variants(
            self._make_base(cache_params=CacheParams(load_factor=0.2, stats=stats))
        )
        for v in variants:
            self.assertIs(v.cache_params.stats, stats)


class _FakeStorage:
    def __init__(self, hbm, ddr):
        self.hbm = hbm
        self.ddr = ddr


class _FakeShard:
    def __init__(self, hbm, ddr):
        self.storage = _FakeStorage(hbm, ddr)


class _FakeShardingOption:
    """Minimal ShardingOption stand-in: only the fields the DP proposer reads."""

    def __init__(self, fqn, hbm, ddr, perf):
        self.fqn = fqn
        # Total = single shard for simplicity (single-rank assignment).
        self.shards = [_FakeShard(hbm, ddr)]
        self.total_storage = _FakeStorage(hbm, ddr)
        self.total_perf = perf


def _make_topology(num_devices, hbm_per_device, ddr_per_device, local_world_size=None):
    return SimpleNamespace(
        devices=[
            SimpleNamespace(
                storage=_FakeStorage(hbm=hbm_per_device, ddr=ddr_per_device)
            )
            for _ in range(num_devices)
        ],
        local_world_size=local_world_size or num_devices,
    )


class DynamicProgrammingProposerTest(unittest.TestCase):
    """2D DP across HBM × DDR picks per-table mode under joint budgets."""

    def _run(self, search_space, topology):
        proposer = DynamicProgrammingProposer(
            hbm_bins_per_device=20, ddr_bins_per_device=20
        )
        proposer.load(search_space)
        # First propose returns the lowest-mem-per-table seed.
        proposer.propose()
        proposer.feedback(partitionable=True, storage_constraint=topology)
        proposals = []
        proposal = proposer.propose()
        while proposal:
            proposals.append(proposal)
            proposer.feedback(partitionable=True, storage_constraint=topology)
            proposal = proposer.propose()
        return proposals

    def test_caching_preferred_when_ddr_is_generous(self):
        # Three options for one table:
        #   HYBRID @ x=1.0: hbm = T,   ddr = 0,   perf = high  (HBM-only)
        #   HYBRID @ x=0.1: hbm = .1T, ddr = .9T, perf = high  (slow)
        #   CACHING @ x=0.1: hbm = .1T, ddr = T,  perf = low   (fast — modeled hits)
        opts = [
            _FakeShardingOption("table_a", hbm=1000, ddr=0, perf=50.0),
            _FakeShardingOption("table_a", hbm=100, ddr=900, perf=80.0),
            _FakeShardingOption("table_a", hbm=100, ddr=1000, perf=10.0),
        ]
        topology = _make_topology(
            num_devices=2, hbm_per_device=2000, ddr_per_device=2000
        )
        proposals = self._run(opts, topology)
        # Best plan must be the CACHING option (perf=10).
        best = min(proposals, key=lambda p: sum(o.total_perf for o in p))
        self.assertEqual(best[0].total_perf, 10.0)

    def test_caching_rejected_when_ddr_is_tight(self):
        # Host budget is only 950 — CACHING (ddr=1000) cannot fit; HYBRID can.
        opts = [
            _FakeShardingOption("table_a", hbm=100, ddr=900, perf=80.0),
            _FakeShardingOption("table_a", hbm=100, ddr=1000, perf=10.0),
        ]
        topology = _make_topology(
            num_devices=1, hbm_per_device=2000, ddr_per_device=950
        )
        proposals = self._run(opts, topology)
        # Every proposed plan must pick the HYBRID option (perf=80).
        for p in proposals:
            self.assertEqual(p[0].total_perf, 80.0)

    def test_high_factor_collapses_modes(self):
        # At x=1.0 HYBRID == CACHING in HBM and CACHING.ddr = T = HYBRID.hbm.
        # If we offer just the high-factor options, DP picks one of them.
        opts = [
            _FakeShardingOption("table_a", hbm=1000, ddr=0, perf=50.0),  # HYBRID x=1.0
            _FakeShardingOption(
                "table_a", hbm=1000, ddr=1000, perf=50.0
            ),  # CACHING x=1.0
        ]
        topology = _make_topology(
            num_devices=1, hbm_per_device=1100, ddr_per_device=2000
        )
        proposals = self._run(opts, topology)
        # Either option is fine — they're tied. Just verify the proposer
        # returned something feasible.
        self.assertGreater(len(proposals), 0)
        for p in proposals:
            self.assertEqual(p[0].total_perf, 50.0)

    def test_two_tables_pick_mixed_modes_under_joint_budget(self):
        # Two tables, each with HYBRID@1.0 (all-HBM, no DDR) and CACHING@0.1
        # (small HBM, full-T DDR). Topology HBM=2000 admits exactly one
        # full HYBRID + one small CACHING shard (1500+100), and host DDR
        # admits exactly one full-T CACHING backing (1500). Both-HYBRID is
        # HBM-infeasible (3000>2000), both-CACHING is DDR-infeasible
        # (3000>2000). Only the mixed plan fits. Exercises the
        # cross-table DP transition at plan_util.py table_i==1.
        opts = [
            _FakeShardingOption("table_a", hbm=1500, ddr=0, perf=50.0),  # HYBRID@1.0
            _FakeShardingOption("table_a", hbm=100, ddr=1500, perf=40.0),  # CACHING@0.1
            _FakeShardingOption("table_b", hbm=1500, ddr=0, perf=50.0),  # HYBRID@1.0
            _FakeShardingOption("table_b", hbm=100, ddr=1500, perf=40.0),  # CACHING@0.1
        ]
        topology = _make_topology(
            num_devices=1, hbm_per_device=2000, ddr_per_device=2000
        )
        proposals = self._run(opts, topology)
        self.assertGreater(len(proposals), 0)
        best = min(proposals, key=lambda p: sum(o.total_perf for o in p))
        styles = sorted(
            "hybrid" if o.shards[0].storage.ddr == 0 else "caching" for o in best
        )
        self.assertEqual(styles, ["caching", "hybrid"])

    def test_empty_search_space_returns_empty_proposal(self):
        proposer = DynamicProgrammingProposer()
        proposer.load([])
        # Seed proposal is the per-table first option; with no tables the
        # list is empty.
        self.assertEqual(proposer.propose(), [])
        # Feedback must not raise on an empty proposer (it short-circuits
        # via `table_count == 0`).
        topology = _make_topology(
            num_devices=1, hbm_per_device=1000, ddr_per_device=1000
        )
        proposer.feedback(partitionable=True, storage_constraint=topology)
        # After feedback, no proposals are available.
        self.assertIsNone(proposer.propose())


@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for dynamicemb.")
class PlanUtilDynamicEmbE2ETest(unittest.TestCase):
    """End-to-end exercise of the dynamicemb planner integration."""

    def _build_constraint(self, max_capacity=4096):
        import dynamicemb
        from dynamicemb.planner import DynamicEmbParameterConstraints

        opts = dynamicemb.DynamicEmbTableOptions(
            max_capacity=max_capacity,
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
        return DynamicEmbParameterConstraints(
            use_dynamicemb=True,
            sharding_types=[ShardingType.ROW_WISE.value],
            compute_kernels=[EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value],
            dynamicemb_options=opts,
        )

    def _build_model(self):
        table = EmbeddingBagConfig(
            num_embeddings=4096,
            embedding_dim=32,
            name="table_de",
            feature_names=["feat_de"],
        )
        return TestSparseNN(tables=[table], sparse_device=torch.device("meta"))

    def test_enumerate_yields_both_modes_and_all_factors(self):
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        model = self._build_model()
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = _TzrecEmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": self._build_constraint()},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )
        self.assertEqual(len(search_space), 20)
        caching_modes = sorted(
            {
                so.dynamicemb_options.caching
                for so in search_space
                if getattr(so, "use_dynamicemb", False)
            }
        )
        self.assertEqual(caching_modes, [False, True])
        load_factors = sorted(
            {
                round(so.cache_load_factor, 4)
                for so in search_space
                if getattr(so, "use_dynamicemb", False)
            }
        )
        self.assertEqual(load_factors, [round((i + 1) / 10, 4) for i in range(10)])
        # Each option must carry a non-zero perf and storage estimate.
        for so in search_space:
            self.assertGreater(so.total_perf, 0)
            self.assertGreaterEqual(so.total_storage.hbm, 0)
            self.assertGreaterEqual(so.total_storage.ddr, 0)

    def test_dp_proposer_picks_feasible_dynamicemb_plan(self):
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        model = self._build_model()
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = _TzrecEmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": self._build_constraint()},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )

        proposer = DynamicProgrammingProposer()
        proposer.load(search_space)
        proposal = proposer.propose()
        self.assertIsNotNone(proposal)
        proposer.feedback(partitionable=True, storage_constraint=topology)

        # At least one further proposal should be generated by the 2D DP.
        count = 0
        proposal = proposer.propose()
        while proposal is not None and count < 5:
            count += 1
            for so in proposal:
                if getattr(so, "use_dynamicemb", False):
                    self.assertIn(so.dynamicemb_options.caching, (False, True))
            proposer.feedback(partitionable=True, storage_constraint=topology)
            proposal = proposer.propose()
        self.assertGreater(count, 0)


if __name__ == "__main__":
    unittest.main()
