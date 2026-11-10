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

import os
import random
import unittest
from types import SimpleNamespace
from typing import List, Tuple
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2
from tzrec.utils import dynamicemb_util
from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints, has_dynamicemb
from tzrec.utils.plan_util import (
    DynamicProgrammingProposer,
    EmbeddingEnumerator,
    _emit_dynamicemb_variants,
    _sparse_dp_proposor_numpy,
    create_planner,
)
from tzrec.utils.test_util import (
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)


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

    def test_global_pooling_factor_is_rejected(self) -> None:
        # ids per sample differ per feature, so a global value means nothing;
        # the check runs before any distributed setup
        global_cfg = feature_pb2.ParameterConstraints()
        global_cfg.pooling_factor = 2.0
        with self.assertRaisesRegex(
            ValueError, "global_embedding_constraints cannot set pooling_factor"
        ):
            create_planner(
                device=torch.device("cpu"),
                batch_size=32,
                global_constraints_cfg=global_cfg,
            )

    @parameterized.expand(
        [
            param(
                "single_input",
                table_name="table_a",
                feature_names=["feature_a"],
                pooling_factors=[3.0],
            ),
            param(
                "shared_inputs",
                table_name="table_shared",
                feature_names=["feature_a", "feature_b"],
                pooling_factors=[1.0, 20.0],
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_pooling_factors_affect_shared_table_storage_estimates(
        self, name, table_name, feature_names, pooling_factors
    ) -> None:
        # one entry per input feature survives enumeration; the estimation
        # path then prices a shared pooled bag one output row per input,
        # while an unpooled collection keeps every position
        from torchrec.distributed.planner.shard_estimators import (
            calculate_shard_storages,
            get_num_poolings,
        )
        from torchrec.distributed.planner.types import ParameterConstraints

        topology = Topology(world_size=1, compute_device="cpu")
        tables = [
            EmbeddingBagConfig(
                num_embeddings=100,
                embedding_dim=8,
                name=table_name,
                feature_names=feature_names,
            )
        ]
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        constraint = ParameterConstraints(
            pooling_factors=pooling_factors, feature_names=feature_names
        )
        fqn = f"sparse.ebc.{table_name}"
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=16,
            fqn_constraints={fqn: constraint},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=get_default_sharders()
        )
        self.assertTrue(search_space)
        for option in search_space:
            self.assertEqual(option.input_lengths, pooling_factors)
            self.assertEqual(option.num_inputs, len(feature_names))
        table_wise = [
            option
            for option in search_space
            if option.sharding_type == ShardingType.TABLE_WISE.value
        ][0]
        self.assertEqual(
            get_num_poolings({fqn: constraint}, table_wise),
            [1.0] * len(feature_names),
        )
        # the estimation the planner runs: batch sizes and poolings straight
        # off the enumerated option
        from torchrec.distributed.planner.types import (
            SharderData,
            StorageUsageType,
        )

        sharder_data = SharderData(
            fused_params={},
            qcomm_dtype_sizes={},
            storage_usage_type=StorageUsageType.DEFAULT,
        )
        batch_sizes = [table_wise.batch_size] * table_wise.num_inputs
        kwargs = dict(
            sharder_data=sharder_data,
            sharding_type=ShardingType.TABLE_WISE.value,
            tensor=torch.zeros((100, 8)),
            compute_device="cpu",
            compute_kernel=EmbeddingComputeKernel.FUSED.value,
            shard_sizes=[[100, 8]],
            batch_sizes=batch_sizes,
            world_size=1,
            local_world_size=1,
            num_poolings=[1.0] * table_wise.num_inputs,
            caching_ratio=1.0,
            input_data_type_size=4,
            output_data_type_size=4,
        )

        def _total(storages):
            return storages[0].hbm + storages[0].ddr + storages[0].ssd

        shared = calculate_shard_storages(
            input_lengths=pooling_factors, is_pooled=True, **kwargs
        )
        collapsed = calculate_shard_storages(
            input_lengths=[sum(pooling_factors)], is_pooled=True, **kwargs
        )
        # a pooled bag outputs one row per input: len(features) rows, where a
        # single summed factor would have collapsed them to one
        self.assertEqual(
            _total(shared) - _total(collapsed),
            (len(pooling_factors) - 1) * 16 * 8 * 4,
        )
        # an unpooled collection outputs every position
        unpooled = calculate_shard_storages(
            input_lengths=pooling_factors, is_pooled=False, **kwargs
        )
        self.assertEqual(
            _total(unpooled) - _total(shared),
            (sum(pooling_factors) - len(pooling_factors)) * 16 * 8 * 4,
        )

    def _planner_with_saved_dp_plan(self, test_dir, constraint):
        """A planner over one table whose saved plan pinned it data_parallel.

        Sets up the single-process environment create_planner needs; the
        caller restores it in a finally block.
        """
        import json as _json

        import torch.distributed as dist

        os.environ.update(
            RANK="0",
            WORLD_SIZE="1",
            MASTER_ADDR="127.0.0.1",
            MASTER_PORT="29713",
        )
        if not dist.is_initialized():
            dist.init_process_group(backend="gloo")

        class _SparseNN(TestSparseNN):
            def parameter_constraints(self, prefix: str = ""):
                return {f"{prefix}sparse.ebc.table_a": constraint}

        model = _SparseNN(
            tables=[
                EmbeddingBagConfig(
                    num_embeddings=100,
                    embedding_dim=8,
                    name="table_a",
                    feature_names=["feature_a"],
                )
            ],
            sparse_device=torch.device("meta"),
        )
        plan_path = os.path.join(test_dir, "plan.json")
        with open(plan_path, "w") as f:
            _json.dump(
                {"sparse": {"ebc.table_a": {"sharding_type": "data_parallel"}}}, f
            )
        planner = create_planner(
            device=torch.device("cuda"),
            batch_size=16,
            ckpt_plan_path=plan_path,
            model=model,
        )
        return planner, model

    def _run_with_saved_dp_plan(self, constraint):
        """Plan one table whose saved plan pinned it data_parallel."""
        import torch.distributed as dist

        env = dict(os.environ)
        test_dir = make_test_dir()
        try:
            planner, model = self._planner_with_saved_dp_plan(test_dir, constraint)
            return planner.plan(model, get_default_sharders())
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()
            os.environ.clear()
            os.environ.update(env)

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

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for create_planner."
    )
    @mark_ci_scope("gpu")
    def test_checkpoint_sharding_applies_with_statistics_only_constraints(self):
        # every sparse table now carries a constraint entry (measured pooling
        # factors); an entry that does not restrict sharding must not disable
        # the data_parallel pin that resume depends on
        from torchrec.distributed.planner.types import ParameterConstraints

        constraint = ParameterConstraints(
            pooling_factors=[2.0], feature_names=["feature_a"]
        )
        plan = self._run_with_saved_dp_plan(constraint)
        self.assertEqual(
            plan.plan["sparse.ebc"]["table_a"].sharding_type, "data_parallel"
        )
        # planning must not mutate the model's own constraint objects
        self.assertIsNone(constraint.sharding_types)

    @unittest.skipUnless(
        torch.cuda.is_available(), "CUDA is required for create_planner."
    )
    @mark_ci_scope("gpu")
    def test_explicit_sharding_overrides_checkpoint_plan(self):
        from torchrec.distributed.planner.types import ParameterConstraints

        constraint = ParameterConstraints(
            sharding_types=["row_wise"], feature_names=["feature_a"]
        )
        plan = self._run_with_saved_dp_plan(constraint)
        self.assertEqual(plan.plan["sparse.ebc"]["table_a"].sharding_type, "row_wise")

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


def _dp_proposals(search_space, topology):
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


class DynamicProgrammingProposerTest(unittest.TestCase):
    """2D DP across HBM × DDR picks per-table mode under joint budgets."""

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
        proposals = _dp_proposals(opts, topology)
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
        proposals = _dp_proposals(opts, topology)
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
        proposals = _dp_proposals(opts, topology)
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
        proposals = _dp_proposals(opts, topology)
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
        proposals_pruned = _dp_proposals(
            [_FakeShardingOption("t", hbm=100, ddr=1500, perf=10.0)], topology
        )
        self.assertEqual(proposals_pruned, [])
        # 900 <= per-machine cap (1000) -> survives, proposal emitted.
        proposals_fit = _dp_proposals(
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

    def _enumerate(self, with_plain_table=False, rows=4096, factor=None):
        from tzrec.utils.plan_util import get_default_sharders as dynamicemb_sharders

        table = EmbeddingBagConfig(
            num_embeddings=rows,
            embedding_dim=32,
            name="table_de",
            feature_names=["feat_de"],
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            constraint = build_dynamicemb_constraints(
                feature_pb2.DynamicEmbedding(max_capacity=rows), table
            )
        constraint.unique_id_factor = factor
        tables = [table]
        if with_plain_table:
            tables.append(
                EmbeddingBagConfig(
                    num_embeddings=rows,
                    embedding_dim=32,
                    name="table_plain",
                    feature_names=["feat_plain"],
                )
            )
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        topology = Topology(world_size=2, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        return enumerator.enumerate(model, dynamicemb_sharders()), topology

    def test_normalized_shards_and_converted_budgets(self):
        from torchrec.distributed.planner import planners

        options, topology = self._enumerate(rows=8001, factor=3.2)
        # 2048 distinct rows per shard: the first cache must cover half of 4096.
        self.assertEqual(
            min(
                o.cache_params.load_factor
                for o in options
                if o.dynamicemb_options.caching
            ),
            0.5,
        )
        for opt in options:
            ratio, caching = opt.cache_load_factor, opt.dynamicemb_options.caching
            self.assertEqual(opt.tensor.shape[0], 8192)
            self.assertGreater(opt.total_perf, 0)
            for rank, shard in enumerate(opt.shards):
                self.assertEqual(shard.size[0], 4096)
                if ratio == 1:
                    self.assertEqual(shard.storage.ddr, 128 * 32 * 4)
                shard.rank = rank
            target = (
                planners.to_sharding_plan([opt], topology)
                .plan[opt.path][opt.name]
                .dynamicemb_options
            )
            self.assertEqual(
                (target.caching, target.max_capacity, target.local_hbm_for_values),
                (caching, 4096, int(ratio * 4096 * 32 * 4)),
            )

    def test_large_table_under_tight_hbm(self):
        options, _ = self._enumerate(rows=40_000_000)
        budget = _make_topology(2, 520 << 20, 10 << 30)
        self.assertFalse(
            _dp_proposals([o for o in options if o.cache_load_factor >= 0.1], budget)
        )
        plan = _dp_proposals(options, budget)[0]
        self.assertEqual(
            [(o.dynamicemb_options.caching, o.cache_load_factor) for o in plan],
            [(False, 0.0)],
        )
        self.assertFalse(_dp_proposals(options, _make_topology(2, 300 << 20, 10 << 30)))

    def test_dp_proposer_picks_feasible_dynamicemb_plan(self):
        search_space, topology = self._enumerate()

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

        # The value travels as a plain kwarg of the dynamicemb table module.
        self.assertIn(
            "initial_accumulator_value",
            inspect.signature(BatchedDynamicEmbeddingTablesV2.__init__).parameters,
        )

        search_space, topology = self._enumerate()
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

    def test_sharding_plan_sizes_ftrl_optimizer_state(self):
        import inspect

        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from torchrec.distributed.planner import planners

        from tzrec.optim.optimizer import FTRL

        # FTRL's knobs travel as plain kwargs of the dynamicemb table module.
        signature = inspect.signature(BatchedDynamicEmbeddingTablesV2.__init__)
        for name in ("learning_rate_power", "ftrl_beta", "l1_reg", "l2_reg"):
            self.assertIn(name, signature.parameters)

        search_space, topology = self._enumerate()
        sharding_option = next(
            so
            for so in search_space
            if getattr(so, "use_dynamicemb", False)
            and so.dynamicemb_options.caching
            and so.cache_load_factor == 0.1
        )
        for rank, shard in enumerate(sharding_option.shards):
            shard.rank = rank
        sharding_option.tensor._optimizer_classes = [FTRL]

        plan = planners.to_sharding_plan([sharding_option], topology)

        param_sharding = plan.plan[sharding_option.path][sharding_option.name]
        options = param_sharding.dynamicemb_options
        self.assertTrue(options.training)
        self.assertTrue(options.caching)
        # FTRL keeps `linear` and `accum` per element, so a value row is three
        # times the embedding width, and the budget is the requested share of
        # the bucket-aligned full shard.
        rows, dim = sharding_option.shards[0].size
        aligned_rows = -(-rows // 128) * 128
        row_bytes = (dim + 2 * dim) * sharding_option.tensor.element_size()
        self.assertEqual(
            options.local_hbm_for_values,
            int(0.1 * aligned_rows * row_bytes),
        )
        self.assertGreater(options.local_hbm_for_values, 0)

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
class DynamicEmbCandidateGridTest(unittest.TestCase):
    def _options(
        self, factor=None, ratio=None, *, rows=4096, dtype=torch.float32, pooling=1
    ):
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
            tensor=torch.empty((rows, 1), dtype=dtype, device="meta"),
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
            ("unmeasured", None, 0.1, 0.1, 7),
            ("measured_zero", 0, 0.1, 0.1, 7),
            ("small_working_set", 0.1, 64 / 4096, 0.1, 7),
            ("half", 3.2, 0.1, 0.5, 3),
            ("above_half", 4.8, 0.1, 0.75, 1),
            ("full", 6.4, 0.1, None, 0),
        ],
        name_func=parameterized_name_func,
    )
    def test_auto_grid(self, _, factor, hybrid_floor, cache_floor, ncache):
        opts = self._options(factor)
        noncache = [
            o.cache_params.load_factor for o in opts if not o.dynamicemb_options.caching
        ]
        cache = [
            o.cache_params.load_factor for o in opts if o.dynamicemb_options.caching
        ]
        self.assertEqual(noncache[0:2], [0, hybrid_floor])
        self.assertEqual(noncache[5:], [0.5, 0.6, 0.7, 0.8, 0.9, 1])
        np.testing.assert_allclose(
            np.diff(np.log(noncache[1:6])), np.log(0.5 / hybrid_floor) / 4
        )
        self.assertEqual(len(cache), ncache)
        self.assertEqual(cache[:1], [] if cache_floor is None else [cache_floor])

    @parameterized.expand(
        [
            ("host", 0, 1, [False]),
            ("full", 1, 1, [False]),
            ("both", 0.37, 1, [False, True]),
            ("insufficient_cache", 0.1, 9, [False]),
            ("one_byte", 1 / 16384, 1, [False, True]),
        ],
        name_func=parameterized_name_func,
    )
    def test_ratio_pin(self, _, ratio, pooling, families):
        opts = self._options(ratio=ratio, pooling=pooling)
        self.assertEqual(
            [(o.cache_params.load_factor, o.dynamicemb_options.caching) for o in opts],
            [(ratio, family) for family in families],
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
            [(o.cache_load_factor, o.dynamicemb_options.caching) for o in pinned],
            [(0.1, False), (1.0, False)],
        )

    def test_physical_dedup_keeps_distinct_budgets(self):
        # The five low ratios buy three byte budgets in the same partial cache.
        opts = self._options(factor=2047 / 640, rows=4096, dtype=torch.float16)
        ratios = [
            o.cache_params.load_factor for o in opts if o.dynamicemb_options.caching
        ]
        self.assertEqual(ratios[0], 2047 / 4096)
        self.assertEqual(
            [int(r * 8192) for r in ratios], [4094, 4095, 4096, 4915, 5734]
        )

    def test_invalid_factor(self):
        for factor in (float("nan"), -1):
            with self.assertRaisesRegex(ValueError, "unique_id_factor"):
                self._options(factor)

    @parameterized.expand(
        [
            ("zero", 0, False, 4096, False, 0),
            ("full", 1, False, 4096, False, 1),
            ("hybrid", 0.5, False, 4096, False, 0.11 + 0.01 * 0.5),
            ("cache", 0.1, True, 4096, False, 0.28 + 0.01 * 0.25),
            ("measured_cache", 0.1, True, 4096, True, 0.6),
        ],
        name_func=parameterized_name_func,
    )
    def test_performance_uses_allocated_share_only(
        self, _, ratio, caching, rows, measured, expected
    ):
        opt = next(
            o
            for o in self._options(ratio=ratio, rows=rows)
            if o.dynamicemb_options.caching == caching
        )
        original = opt.cache_params
        if measured:
            original.stats = mock.Mock()
            original.stats.expected_miss_rate.return_value = 0.4
        with mock.patch.object(
            dynamicemb_util,
            "_orig_build_shard_perf_contexts",
            side_effect=lambda *args: opt.cache_params.load_factor,
        ):
            hit_rate = dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
                None, [[rows, 1]], opt, None, None, None
            )
        if measured:
            original.stats.expected_miss_rate.assert_called_once_with(0.25)
        self.assertAlmostEqual(hit_rate, expected)
        self.assertIs(opt.cache_params, original)
        self.assertEqual(opt.cache_params.load_factor, ratio)


if __name__ == "__main__":
    unittest.main()
