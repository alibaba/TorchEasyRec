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

import random
import unittest
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch
from parameterized import parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.enumerators import EmbeddingEnumerator
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.plan_util import DynamicProgrammingProposer, _sparse_dp_proposor_numpy


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


def _dense_dp_proposor_python_reference(
    table_opts: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    hbm_bins: int,
    ddr_bins: int,
) -> List[List[int]]:
    """Dense T x H x D Python DP -- the oracle for sparse-K NumPy equivalence.

    Same input shape as plan_util._sparse_dp_proposor_numpy: per table, a tuple of
    (opt_h, opt_d, opt_perf, opt_global_idx) in bin-units. Algorithmically
    identical to the in-tree DP that the NumPy version replaced (dense state +
    full backtrack), tightened with a 2-row dp ring buffer so it stays
    tractable on small property-test problem sizes.
    """
    T = len(table_opts)
    if T == 0:
        return []
    INF = float("inf")
    empty = (INF, INF, INF)
    prev_dp = [[empty] * ddr_bins for _ in range(hbm_bins)]
    backtrack = [[[(-1, -1, -1)] * ddr_bins for _ in range(hbm_bins)] for _ in range(T)]

    # Seed table 0.
    h0, d0, p0, g0 = table_opts[0]
    for j in range(len(p0)):
        h, d, p = float(h0[j]), float(d0[j]), float(p0[j])
        if h >= hbm_bins or d >= ddr_bins:
            continue
        hi, di = int(h), int(d)
        if prev_dp[hi][di][0] > p:
            prev_dp[hi][di] = (p, h, d)
            backtrack[0][hi][di] = (int(g0[j]), -1, -1)

    # Transitions.
    for t in range(1, T):
        h_t, d_t, p_t, g_t = table_opts[t]
        cur_dp = [[empty] * ddr_bins for _ in range(hbm_bins)]
        for h in range(hbm_bins):
            for d in range(ddr_bins):
                pp, ph, pd = prev_dp[h][d]
                if pp == INF:
                    continue
                for j in range(len(p_t)):
                    new_h = ph + float(h_t[j])
                    new_d = pd + float(d_t[j])
                    if new_h >= hbm_bins or new_d >= ddr_bins:
                        continue
                    nhi, ndi = int(new_h), int(new_d)
                    new_p = pp + float(p_t[j])
                    if cur_dp[nhi][ndi][0] > new_p:
                        cur_dp[nhi][ndi] = (new_p, new_h, new_d)
                        backtrack[t][nhi][ndi] = (int(g_t[j]), h, d)
        prev_dp = cur_dp

    # Per-HBM-bin best, decreasing HBM order.
    proposals: List[List[int]] = []
    last_back = backtrack[T - 1]
    for h in range(hbm_bins - 1, -1, -1):
        best_p, best_d = INF, -1
        for d in range(ddr_bins):
            if last_back[h][d][0] >= 0 and prev_dp[h][d][0] < best_p:
                best_p, best_d = prev_dp[h][d][0], d
        if best_d < 0:
            continue
        opt, ph, pd = last_back[h][best_d]
        ids = [-1] * T
        ids[T - 1] = opt
        for t in range(T - 2, -1, -1):
            ids[t], ph, pd = backtrack[t][ph][pd]
        proposals.append(ids)
    return proposals


def _make_random_table_opts(
    seed: int, T: int = 5, N: int = 4, H: int = 8, D: int = 8
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Seeded random per-table option arrays for property testing.

    Per-option bin contributions stay in [0, axis/2) so combined plans can fit
    several tables without saturating either axis.
    """
    rng = random.Random(seed)
    table_opts = []
    for _ in range(T):
        h = np.asarray([rng.uniform(0, H / 2) for _ in range(N)], dtype=np.float32)
        d = np.asarray([rng.uniform(0, D / 2) for _ in range(N)], dtype=np.float32)
        p = np.asarray([rng.uniform(1, 100) for _ in range(N)], dtype=np.float32)
        g = np.arange(N, dtype=np.int32)
        table_opts.append((h, d, p, g))
    return table_opts


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

    def test_per_machine_ddr_prune_on_multi_host_topology(self):
        # 4 GPUs across 2 machines (local_world_size=2). Each machine has
        # 1000 DDR; total = 2000. An option whose per-shard ddr is 1500
        # exceeds the 1000 per-machine cap and must be pruned, even
        # though 1500 < ddr_total. The 900-ddr option fits.
        topology = _make_topology(
            num_devices=4,
            local_world_size=2,
            hbm_per_device=2000,
            ddr_per_device=500,
        )
        # 1500 > per-machine cap (1000) -> pruned, no proposal.
        proposals_pruned = self._run(
            [_FakeShardingOption("t", hbm=100, ddr=1500, perf=10.0)], topology
        )
        self.assertEqual(proposals_pruned, [])
        # 900 <= per-machine cap (1000) -> survives, proposal emitted.
        proposals_fit = self._run(
            [_FakeShardingOption("t", hbm=100, ddr=900, perf=10.0)], topology
        )
        self.assertGreater(len(proposals_fit), 0)

    @parameterized.expand([(seed,) for seed in [0, 1, 7, 42, 1337]])
    def test_sparse_numpy_matches_dense_reference(self, seed):
        # Property test: sparse-K NumPy DP must produce the same proposal set
        # as the dense T x H x D Python reference oracle for any random input.
        table_opts = _make_random_table_opts(seed, T=5, N=4, H=8, D=8)
        np_proposals = _sparse_dp_proposor_numpy(table_opts, hbm_bins=8, ddr_bins=8)
        ref_proposals = _dense_dp_proposor_python_reference(
            table_opts, hbm_bins=8, ddr_bins=8
        )
        self.assertEqual(
            {tuple(p) for p in np_proposals},
            {tuple(p) for p in ref_proposals},
        )

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
