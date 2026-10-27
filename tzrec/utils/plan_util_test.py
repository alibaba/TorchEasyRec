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
from unittest import mock

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

from tzrec.protos import feature_pb2, train_pb2
from tzrec.utils import plan_util, rocksdb_capacity, rocksdb_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.plan_util import DynamicProgrammingProposer, _sparse_dp_proposor_numpy
from tzrec.utils.test_util import mark_ci_scope


class HostMemoryTest(unittest.TestCase):
    """The planner's host memory respects a container memory limit."""

    def _run(self, limit_text):
        files = {"/sys/fs/cgroup/memory.max": limit_text}

        def fake_open(path, *args, **kwargs):
            if path in files:
                return mock.mock_open(read_data=files[path])()
            raise OSError(path)

        with (
            mock.patch.object(
                plan_util.psutil,
                "virtual_memory",
                return_value=SimpleNamespace(total=1 << 40),
            ),
            mock.patch("builtins.open", side_effect=fake_open),
        ):
            return plan_util.host_memory_bytes()

    def test_cgroup_limit_caps_host_memory(self):
        self.assertEqual(self._run(str(170 << 30)), 170 << 30)

    def test_unlimited_cgroup_keeps_ram(self):
        self.assertEqual(self._run("max"), 1 << 40)


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

    Same input shape as plan_util._sparse_dp_proposor_numpy: per table, a tuple
    of ``(opt_hbm, opt_ddr, opt_perf, opt_global_id)`` in bin-units.
    Algorithmically identical to the in-tree DP that the NumPy version
    replaced (dense state + full backtrack), tightened with a 2-row dp ring
    buffer so it stays tractable on small property-test problem sizes.
    """
    table_count = len(table_opts)
    if table_count == 0:
        return []
    INF = float("inf")
    empty_state = (INF, INF, INF)
    prev_dp = [[empty_state] * ddr_bins for _ in range(hbm_bins)]
    backtrack = [
        [[(-1, -1, -1)] * ddr_bins for _ in range(hbm_bins)] for _ in range(table_count)
    ]

    # Seed table 0.
    seed_hbm, seed_ddr, seed_perf, seed_opt_id = table_opts[0]
    for opt_j in range(len(seed_perf)):
        hbm = float(seed_hbm[opt_j])
        ddr = float(seed_ddr[opt_j])
        perf = float(seed_perf[opt_j])
        if hbm >= hbm_bins or ddr >= ddr_bins:
            continue
        hbm_i, ddr_i = int(hbm), int(ddr)
        if prev_dp[hbm_i][ddr_i][0] > perf:
            prev_dp[hbm_i][ddr_i] = (perf, hbm, ddr)
            backtrack[0][hbm_i][ddr_i] = (int(seed_opt_id[opt_j]), -1, -1)

    # Transitions.
    for table_i in range(1, table_count):
        cur_table_hbm, cur_table_ddr, cur_table_perf, cur_table_opt_id = table_opts[
            table_i
        ]
        cur_dp = [[empty_state] * ddr_bins for _ in range(hbm_bins)]
        for hbm_i in range(hbm_bins):
            for ddr_i in range(ddr_bins):
                prev_perf, prev_hbm, prev_ddr = prev_dp[hbm_i][ddr_i]
                if prev_perf == INF:
                    continue
                for opt_j in range(len(cur_table_perf)):
                    new_hbm = prev_hbm + float(cur_table_hbm[opt_j])
                    new_ddr = prev_ddr + float(cur_table_ddr[opt_j])
                    if new_hbm >= hbm_bins or new_ddr >= ddr_bins:
                        continue
                    new_hbm_i, new_ddr_i = int(new_hbm), int(new_ddr)
                    new_perf = prev_perf + float(cur_table_perf[opt_j])
                    if cur_dp[new_hbm_i][new_ddr_i][0] > new_perf:
                        cur_dp[new_hbm_i][new_ddr_i] = (new_perf, new_hbm, new_ddr)
                        backtrack[table_i][new_hbm_i][new_ddr_i] = (
                            int(cur_table_opt_id[opt_j]),
                            hbm_i,
                            ddr_i,
                        )
        prev_dp = cur_dp

    # Per-HBM-bin best, decreasing HBM order.
    proposals: List[List[int]] = []
    last_back = backtrack[table_count - 1]
    for hbm_i in range(hbm_bins - 1, -1, -1):
        best_perf, best_ddr_i = INF, -1
        for ddr_i in range(ddr_bins):
            if last_back[hbm_i][ddr_i][0] >= 0 and prev_dp[hbm_i][ddr_i][0] < best_perf:
                best_perf, best_ddr_i = prev_dp[hbm_i][ddr_i][0], ddr_i
        if best_ddr_i < 0:
            continue
        cur_opt_j, prev_hbm_i, prev_ddr_i = last_back[hbm_i][best_ddr_i]
        proposal_indices = [-1] * table_count
        proposal_indices[table_count - 1] = cur_opt_j
        for table_i in range(table_count - 2, -1, -1):
            proposal_indices[table_i], prev_hbm_i, prev_ddr_i = backtrack[table_i][
                prev_hbm_i
            ][prev_ddr_i]
        proposals.append(proposal_indices)
    return proposals


def _make_random_table_opts(
    seed: int,
    table_count: int = 5,
    option_count: int = 4,
    hbm_bins: int = 8,
    ddr_bins: int = 8,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Seeded random per-table option arrays for property testing.

    Per-option bin contributions stay in [0, axis/2) so combined plans can fit
    several tables without saturating either axis.
    """
    rng = random.Random(seed)
    table_opts = []
    for _ in range(table_count):
        hbm = np.asarray(
            [rng.uniform(0, hbm_bins / 2) for _ in range(option_count)],
            dtype=np.float32,
        )
        ddr = np.asarray(
            [rng.uniform(0, ddr_bins / 2) for _ in range(option_count)],
            dtype=np.float32,
        )
        perf = np.asarray(
            [rng.uniform(1, 100) for _ in range(option_count)], dtype=np.float32
        )
        opt_id = np.arange(option_count, dtype=np.int32)
        table_opts.append((hbm, ddr, perf, opt_id))
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
        table_opts = _make_random_table_opts(
            seed, table_count=5, option_count=4, hbm_bins=8, ddr_bins=8
        )
        actual_proposals = _sparse_dp_proposor_numpy(table_opts, hbm_bins=8, ddr_bins=8)
        expected_proposals = _dense_dp_proposor_python_reference(
            table_opts, hbm_bins=8, ddr_bins=8
        )
        self.assertEqual(
            {tuple(p) for p in actual_proposals},
            {tuple(p) for p in expected_proposals},
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
@mark_ci_scope("gpu")
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

    def _build_model(self, with_plain_table=False):
        tables = [
            EmbeddingBagConfig(
                num_embeddings=4096,
                embedding_dim=32,
                name="table_de",
                feature_names=["feat_de"],
            )
        ]
        if with_plain_table:
            tables.append(
                EmbeddingBagConfig(
                    num_embeddings=4096,
                    embedding_dim=32,
                    name="table_plain",
                    feature_names=["feat_plain"],
                )
            )
        return TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

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
                round(so.cache_load_factor or 0.0, 4)
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

    def test_cache_load_factor_pins_the_ratio(self):
        from tzrec.protos import feature_pb2 as feature_pb2_
        from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        config = EmbeddingBagConfig(
            num_embeddings=4096,
            embedding_dim=32,
            name="table_de",
            feature_names=["feat_de"],
        )
        constraint = build_dynamicemb_constraints(
            feature_pb2_.DynamicEmbedding(max_capacity=4096, cache_load_factor=0.5),
            config,
        )
        enumerator = _TzrecEmbeddingEnumerator(
            topology=Topology(world_size=2, compute_device="cuda"),
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        search_space = enumerator.enumerate(
            module=self._build_model(), sharders=_tzrec_get_default_sharders()
        )
        # One HYBRID and one CACHING variant, both at the configured factor.
        self.assertEqual(len(search_space), 2)
        self.assertEqual({so.cache_load_factor for so in search_space}, {0.5})

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

    def test_sharding_plan_carries_initial_accumulator_value(self):
        import inspect

        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from dynamicemb.planner import DynamicEmbParameterSharding
        from torchrec.distributed.planner import planners

        from tzrec.optim.optimizer import set_sparse_init_accumulator_value
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        # The value travels as a plain kwarg of the dynamicemb table module.
        self.assertIn(
            "initial_accumulator_value",
            inspect.signature(BatchedDynamicEmbeddingTablesV2.__init__).parameters,
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
        sharding_option = next(
            so for so in search_space if getattr(so, "use_dynamicemb", False)
        )
        for rank, shard in enumerate(sharding_option.shards):
            shard.rank = rank

        set_sparse_init_accumulator_value(0.1)
        try:
            plan = planners.to_sharding_plan([sharding_option], topology)
        finally:
            set_sparse_init_accumulator_value(0.0)

        param_sharding = plan.plan[sharding_option.path][sharding_option.name]
        fused_params = param_sharding.get_additional_fused_params()
        self.assertAlmostEqual(fused_params["initial_accumulator_value"], 0.1)
        # dynamicemb strips only its planner-only keys before the table module.
        DynamicEmbParameterSharding.pop_additional_fused_params(fused_params)
        self.assertIn("initial_accumulator_value", fused_params)

    def _enumerate(self, with_plain_table=False):
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        model = self._build_model(with_plain_table=with_plain_table)
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = _TzrecEmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": self._build_constraint()},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )
        return search_space, topology

    def test_sharding_plan_sizes_ftrl_optimizer_state(self):
        import inspect

        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from torchrec.distributed.planner import planners

        from tzrec.optim.optimizer import FTRL
        from tzrec.utils.dynamicemb_util import (
            _calculate_dynamicemb_table_storage_specific_size,
        )

        # FTRL's knobs travel as plain kwargs of the dynamicemb table module.
        signature = inspect.signature(BatchedDynamicEmbeddingTablesV2.__init__)
        for name in ("learning_rate_power", "ftrl_beta", "l1_reg", "l2_reg"):
            self.assertIn(name, signature.parameters)

        search_space, topology = self._enumerate()
        sharding_option = next(
            so for so in search_space if getattr(so, "use_dynamicemb", False)
        )
        for rank, shard in enumerate(sharding_option.shards):
            shard.rank = rank
        sharding_option.tensor._optimizer_classes = [FTRL]

        plan = planners.to_sharding_plan([sharding_option], topology)

        param_sharding = plan.plan[sharding_option.path][sharding_option.name]
        options = param_sharding.dynamicemb_options
        self.assertTrue(options.training)
        # FTRL keeps `linear` and `accum` per element, so the values are sized
        # as if the row were three times the embedding width.
        self.assertEqual(
            options.local_hbm_for_values,
            _calculate_dynamicemb_table_storage_specific_size(
                sharding_option.shards[0].size,
                sharding_option.tensor.element_size(),
                2.0,
                sharding_option.cache_load_factor,
                is_hbm=True,
                only_values=True,
                bucket_capacity=options.bucket_capacity,
            ),
        )

    def test_sharding_plan_rejects_ftrl_on_non_dynamicemb_table(self):
        from torchrec.distributed.planner import planners

        from tzrec.optim.optimizer import FTRL

        search_space, topology = self._enumerate(with_plain_table=True)
        for sharding_type, raises in (
            (ShardingType.ROW_WISE.value, True),
            (ShardingType.DATA_PARALLEL.value, False),
        ):
            sharding_option = next(
                so
                for so in search_space
                if not getattr(so, "use_dynamicemb", False)
                and so.sharding_type == sharding_type
            )
            for rank, shard in enumerate(sharding_option.shards):
                shard.rank = rank
            sharding_option.tensor._optimizer_classes = [FTRL]
            if raises:
                with self.assertRaisesRegex(ValueError, "ftrl_optimizer"):
                    planners.to_sharding_plan([sharding_option], topology)
            else:
                planners.to_sharding_plan([sharding_option], topology)


@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@mark_ci_scope("gpu")
class RocksDBCacheRatioTest(unittest.TestCase):
    """RocksDB tables share one rank-wide GPU cache ratio."""

    def tearDown(self):
        rocksdb_util.set_rocksdb_config(None)

    def _option(self, rows, dim, rocksdb=True):
        return SimpleNamespace(
            use_dynamicemb=True,
            dynamicemb_options=SimpleNamespace(caching=None, rocksdb=rocksdb),
            shards=[SimpleNamespace(size=[rows, dim])],
            tensor=torch.empty(0, dim),
            cache_params=None,
        )

    def _pin(self, options, budget):
        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(
                local_path="x", gpu_cache_value_bytes_per_rank=budget
            )
        )
        with mock.patch.object(
            rocksdb_util, "is_rocksdb_options", side_effect=lambda o: o.rocksdb
        ):
            plan_util._pin_rocksdb_cache_ratio(options)

    def test_uniform_ratio_and_mode(self):
        a, b, other = (
            self._option(1 << 16, 16),
            self._option(1 << 16, 48),
            self._option(1 << 16, 16, rocksdb=False),
        )
        total = (1 << 16) * (16 + 48) * 4
        self._pin([a, b, other], total // 4)
        for o in (a, b):
            self.assertAlmostEqual(o.cache_params.load_factor, 0.25)
            self.assertTrue(o.dynamicemb_options.caching)
        self.assertIsNone(other.cache_params)
        self.assertIsNone(other.dynamicemb_options.caching)

    def test_zero_budget_rejected(self):
        # RocksDB tables always run behind the GPU cache.
        with self.assertRaisesRegex(ValueError, "gpu_cache_value_bytes_per_rank"):
            self._pin([self._option(1 << 16, 16)], 0)

    def test_full_budget_rejected(self):
        with self.assertRaises(ValueError):
            self._pin([self._option(1 << 10, 16)], 1 << 30)


GIB = 1 << 30


def _obs(rank=0, free=10 << 40, job=0, source="nas.example:/share", fstype="nfs"):
    return rocksdb_capacity.DomainObservation(
        rank=rank,
        host=f"h{rank}",
        path="/mnt/xnas/db",
        fstype=fstype,
        source=source,
        mount_point="/mnt/xnas",
        st_dev=42,
        free_bytes=free,
        total_bytes=20 << 40,
        job_bytes=job,
    )


@unittest.skipUnless(has_dynamicemb, "dynamicemb is required")
@mark_ci_scope("gpu")
class RocksDBPlacementTest(unittest.TestCase):
    """Backend selection of dynamicemb tables under memory and disk budgets."""

    def setUp(self):
        try:
            import tzrec_rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_rocksdb is not installed")

    def tearDown(self):
        rocksdb_util.set_rocksdb_config(None)

    def _place(
        self,
        tables,
        hbm_gib,
        ddr_gib,
        placement="AUTO",
        budget=GIB << 10,
        observations=None,
        cache=GIB // 16,
    ):
        """tables: name -> (rows, dim, dict of DynamicEmbedding fields)."""
        from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints

        rocksdb_util.set_rocksdb_config(
            train_pb2.RocksDBConfig(
                local_path="/unused",
                gpu_cache_value_bytes_per_rank=cache,
                placement=train_pb2.DynamicEmbPlacement.Value(placement),
                disk_budget_bytes=budget,
                block_cache_bytes_per_rank=GIB,
                write_buffer_bytes_per_rank=GIB // 4,
                staging_buffer_bytes_per_rank=GIB // 16,
            )
        )
        configs, constraints = [], {}
        for name, (rows, dim, fields) in tables.items():
            fields = dict(fields)
            explicit = fields.pop("rocksdb", False)
            cfg = feature_pb2.DynamicEmbedding(max_capacity=rows, **fields)
            if explicit:
                cfg.rocksdb.SetInParent()
            config = EmbeddingBagConfig(
                num_embeddings=rows,
                embedding_dim=dim,
                name=name,
                feature_names=[f"f_{name}"],
            )
            configs.append(config)
            constraints[f"sparse.ebc.{name}"] = build_dynamicemb_constraints(
                cfg, config
            )
        model = TestSparseNN(tables=configs, sparse_device=torch.device("meta"))
        topology = Topology(
            world_size=1,
            compute_device="cuda",
            hbm_cap=int(hbm_gib * GIB),
            ddr_cap=int(ddr_gib * GIB),
        )
        placement_ = plan_util.RocksDBPlacement(observations or [_obs()], 0.15)
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
        )
        sharders = plan_util.get_default_sharders()
        placement_.decide(enumerator.enumerate(model, sharders), topology)
        enumerator.placement = placement_
        options = enumerator.enumerate(model, sharders)
        rocksdb = {
            o.name
            for o in options
            if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        }
        return placement_, rocksdb, topology

    # 2^24 x 64 fp32 with SGD: 4 GiB of values per table.
    BIG = (1 << 24, 64, {})

    def test_all_in_memory(self):
        p, rocksdb, topo = self._place({"a": self.BIG, "b": self.BIG}, 80, 64)
        self.assertEqual(rocksdb, set())
        self.assertEqual(p.domains, [])
        # Nothing in RocksDB: no host reserve is taken.
        self.assertEqual(topo.devices[0].storage.ddr, 64 * GIB)

    def test_mixed(self):
        # 8.25 GiB of tables against 3.4 GiB usable HBM + 4.75 GiB DRAM: moving
        # one big table covers the deficit plus the RocksDB host reserve.
        p, rocksdb, topo = self._place(
            {"a": self.BIG, "b": self.BIG, "c": (1 << 20, 64, {})}, 4, 4.75
        )
        self.assertEqual(len(rocksdb), 1)
        self.assertNotIn("c", rocksdb)
        self.assertEqual(
            topo.devices[0].storage.ddr,
            int(4.75 * GIB) - rocksdb_util.host_reserve_bytes(),
        )
        self.assertEqual(len(p.domains), 1)
        self.assertTrue(p.domains[0].feasible)

    def test_mostly_rocksdb(self):
        tables = {n: self.BIG for n in "abcd"}
        _, rocksdb, _ = self._place(tables, 4, 4.75)
        self.assertGreaterEqual(len(rocksdb), 3)

    def test_overrides(self):
        tables = {
            "a": (1 << 20, 64, {"rocksdb": True}),
            "b": (1 << 24, 64, {"cache_load_factor": 0.5}),
            "c": (1 << 24, 64, {"score_strategy": "CUSTOMIZED"}),
            "d": (1 << 24, 64, {}),
        }
        _, rocksdb, _ = self._place(tables, 4, 12)
        # rocksdb {} always moves; pinned built-in and CUSTOMIZED never do.
        self.assertIn("a", rocksdb)
        self.assertNotIn("b", rocksdb)
        self.assertNotIn("c", rocksdb)
        self.assertIn("d", rocksdb)

    def test_explicit_mode_needs_no_memory_pressure(self):
        _, rocksdb, _ = self._place(
            {"a": (1 << 20, 64, {"rocksdb": True}), "b": self.BIG},
            80,
            64,
            placement="EXPLICIT",
        )
        self.assertEqual(rocksdb, {"a"})

    def test_disk_pressure_picks_another_table(self):
        # "a" (8 GiB) would be moved first, but only "b" (4 GiB) fits the disk
        # budget, and moving "b" alone covers the memory deficit.
        tables = {
            "a": (1 << 25, 64, {}),
            "b": (1 << 24, 64, {}),
            "c": (1 << 20, 64, {}),
        }
        p, rocksdb, _ = self._place(tables, 4, 9.0, budget=49 * GIB)
        self.assertEqual(rocksdb, {"b"})
        self.assertTrue(p.domains[0].feasible)

    def test_no_feasible_plan(self):
        tables = {"a": (1 << 26, 64, {}), "b": (1 << 26, 64, {})}
        with self.assertRaisesRegex(rocksdb_capacity.DiskCapacityError, "short"):
            self._place(tables, 4, 4.75, budget=40 * GIB)

    def test_explicit_table_over_disk_budget_fails(self):
        with self.assertRaisesRegex(rocksdb_capacity.DiskCapacityError, "budget"):
            self._place(
                {"a": (1 << 26, 64, {"rocksdb": True})},
                80,
                64,
                placement="EXPLICIT",
                budget=40 * GIB,
            )

    def test_shared_domain_counts_every_rank(self):
        obs = [_obs(0), _obs(1)]
        p, _, _ = self._place(
            {"a": (1 << 20, 64, {"rocksdb": True})},
            80,
            64,
            placement="EXPLICIT",
            observations=obs,
        )
        self.assertEqual(len(p.domains), 1)
        self.assertEqual(p.domains[0].ranks, [0, 1])
        solo, _, _ = self._place(
            {"a": (1 << 20, 64, {"rocksdb": True})}, 80, 64, placement="EXPLICIT"
        )
        self.assertEqual(p.domains[0].sst_bytes, 2 * solo.domains[0].sst_bytes)

    def test_retry_spills_one_more(self):
        tables = {n: self.BIG for n in "abc"}
        p, rocksdb, _ = self._place(tables, 80, 64)
        self.assertEqual(rocksdb, set())
        self.assertTrue(p.spill_more())
        _, rocksdb_after, _ = self._place_again(p, tables)
        self.assertEqual(len(rocksdb_after), 1)

    def _place_again(self, placement, tables):
        from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints

        configs, constraints = [], {}
        for name, (rows, dim, fields) in tables.items():
            cfg = feature_pb2.DynamicEmbedding(max_capacity=rows, **fields)
            config = EmbeddingBagConfig(
                num_embeddings=rows,
                embedding_dim=dim,
                name=name,
                feature_names=[f"f_{name}"],
            )
            configs.append(config)
            constraints[f"sparse.ebc.{name}"] = build_dynamicemb_constraints(
                cfg, config
            )
        model = TestSparseNN(tables=configs, sparse_device=torch.device("meta"))
        topology = Topology(
            world_size=1, compute_device="cuda", hbm_cap=80 * GIB, ddr_cap=64 * GIB
        )
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology, batch_size=128, fqn_constraints=constraints
        )
        sharders = plan_util.get_default_sharders()
        placement.decide(enumerator.enumerate(model, sharders), topology)
        enumerator.placement = placement
        options = enumerator.enumerate(model, sharders)
        return (
            placement,
            {
                o.name
                for o in options
                if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            },
            topology,
        )


if __name__ == "__main__":
    unittest.main()
