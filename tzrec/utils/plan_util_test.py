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

import math
import os
import random
import unittest
from types import SimpleNamespace
from typing import List, Tuple, cast

import numpy as np
import torch
from parameterized import param, parameterized
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.plan_util import (
    DynamicProgrammingProposer,
    EmbeddingEnumerator,
    _dynamicemb_low_segment,
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

    def _build_model(self, with_plain_table=False, num_embeddings=4096):
        tables = [
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=32,
                name="table_de",
                feature_names=["feat_de"],
            )
        ]
        if with_plain_table:
            tables.append(
                EmbeddingBagConfig(
                    num_embeddings=num_embeddings,
                    embedding_dim=32,
                    name="table_plain",
                    feature_names=["feat_plain"],
                )
            )
        return TestSparseNN(tables=tables, sparse_device=torch.device("meta"))

    def _enumerate_with(
        self, constraint, batch_size=128, world_size=2, num_embeddings=4096
    ):
        from tzrec.utils.plan_util import (
            EmbeddingEnumerator as _TzrecEmbeddingEnumerator,
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        model = self._build_model(num_embeddings=num_embeddings)
        topology = Topology(world_size=world_size, compute_device="cuda")
        enumerator = _TzrecEmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )
        return search_space, topology

    def test_enumerate_yields_all_candidates_and_default_grids(self):
        # a table large enough that every grid ratio keeps a distinct
        # budget and tier rows; smaller tables collapse physically equal
        # candidates (see the candidate-grid tests)
        num_embeddings = 200_000
        search_space, _ = self._enumerate_with(
            self._build_constraint(max_capacity=num_embeddings),
            num_embeddings=num_embeddings,
        )

        dynamicemb_opts = [
            so for so in search_space if getattr(so, "use_dynamicemb", False)
        ]
        # One sharding type/kernel combination, expanded to the default 20
        # candidates before physical deduplication: 9 caching + 9 partial
        # non-caching + the zero endpoint + the full endpoint.
        self.assertEqual(len(dynamicemb_opts), 20)
        by_caching = {True: [], False: []}
        for so in dynamicemb_opts:
            by_caching[bool(so.dynamicemb_options.caching)].append(so)
        self.assertEqual(len(by_caching[True]), 9)
        self.assertEqual(len(by_caching[False]), 11)
        ratios_nc = sorted(
            cast(float, so.cache_load_factor) for so in by_caching[False]
        )
        self.assertEqual(ratios_nc[0], 0.0)
        self.assertEqual(ratios_nc[-1], 1.0)
        self.assertEqual(
            [round(r, 9) for r in ratios_nc[1:-1]],
            [
                0.1,
                0.149534878,
                0.223606798,
                0.334370152,
                0.5,
                0.6,
                0.7,
                0.8,
                0.9,
            ],
        )

        # Unmeasured floors at 0.1: five geometric points to 0.5 plus the
        # fixed high grid, with the shared 0.5 point merged -- nine ratios.
        expected_low = [
            0.1,
            0.14953487812212204,
            0.22360679774997896,
            0.3343701524878291,
            0.5,
        ]
        for caching in (True, False):
            ratios = sorted(
                cast(float, so.cache_load_factor)
                for so in by_caching[caching]
                if 0.0 < so.cache_load_factor < 1.0
            )
            self.assertEqual(len(ratios), 9)
            for want, got in zip(expected_low, ratios[:5]):
                self.assertAlmostEqual(want, got, places=9)
            self.assertEqual(ratios[5:], [0.6, 0.7, 0.8, 0.9])
            # monotonic, distinct ratios
            self.assertEqual(ratios, sorted(set(ratios)))

        # Each option must carry a non-zero perf and a real storage
        # estimate; the zero endpoint's HBM estimate covers its mixed-group
        # minimum tier and metadata, and the full endpoint's DDR estimate
        # covers its mixed-group minimum host bucket.
        for so in dynamicemb_opts:
            self.assertGreater(so.total_perf, 0)
            self.assertGreater(so.total_storage.hbm, 0)
            self.assertGreaterEqual(so.total_storage.ddr, 0)
        zero = by_caching[False][0]
        self.assertEqual(zero.cache_load_factor, 0.0)
        self.assertGreater(zero.total_storage.hbm, 0)
        self.assertGreater(zero.total_storage.ddr, 0)
        full = by_caching[False][-1]
        self.assertEqual(full.cache_load_factor, 1.0)
        self.assertGreater(full.total_storage.ddr, 0)

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
    """Layout candidate generation: grids, endpoints, eligibility, residency."""

    BATCH = 128
    # large enough that every grid ratio realizes a distinct placement:
    # per-rank rows 100_000, aligned 100_096
    BIG = 200_000

    def _constraint(
        self,
        *,
        unique_id_factor=None,
        explicit_ratio=None,
        score_strategy="STEP",
        num_embeddings=None,
        pooling=None,
    ):
        from torchrec.distributed.types import CacheParams

        from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints

        num_embeddings = num_embeddings or self.BIG
        cfg = feature_pb2.DynamicEmbedding()
        cfg.max_capacity = num_embeddings
        cfg.score_strategy = score_strategy
        constraint = build_dynamicemb_constraints(
            cfg,
            EmbeddingBagConfig(
                num_embeddings=num_embeddings,
                embedding_dim=32,
                name="table_de",
                feature_names=["feat_de"],
            ),
        )
        if pooling is not None:
            constraint.pooling_factors = [pooling]
        if unique_id_factor is not None:
            constraint.unique_id_factor = float(unique_id_factor)
        if explicit_ratio is not None:
            constraint.cache_params = CacheParams(load_factor=explicit_ratio)
        return constraint

    def _candidates(
        self, constraint, batch_size=None, world_size=2, num_embeddings=None
    ):
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        num_embeddings = num_embeddings or self.BIG
        batch_size = batch_size or self.BATCH
        model = TestSparseNN(
            tables=[
                EmbeddingBagConfig(
                    num_embeddings=num_embeddings,
                    embedding_dim=32,
                    name="table_de",
                    feature_names=["feat_de"],
                )
            ],
            sparse_device=torch.device("meta"),
        )
        topology = Topology(world_size=world_size, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )
        families = {True: [], False: []}
        for so in search_space:
            if getattr(so, "use_dynamicemb", False):
                families[bool(so.dynamicemb_options.caching)].append(so)
        for candidates in families.values():
            candidates.sort(key=lambda so: so.cache_load_factor)
        return families

    def _ratios(self, families, caching):
        return [so.cache_load_factor for so in families.get(caching, [])]

    def _partial_ratios(self, families, caching):
        """The strictly-interior ratios of one family (endpoints excluded)."""
        return [
            so.cache_load_factor
            for so in families.get(caching, [])
            if 0.0 < so.cache_load_factor < 1.0
        ]

    @staticmethod
    def _expected_floors(factor, batch_size, aligned_rows):
        """The RFC's lower endpoints from a measured factor."""
        if factor is None:
            return 0.1, 0.1
        working_set = math.ceil(5 * factor * batch_size)
        q = working_set / aligned_rows
        return max(q, 0.1), min(q, 0.1)

    @parameterized.expand(
        [
            param("unmeasured", factor=None),
            param("q_small", factor=0.032),
            param("q_just_below_0_1", factor=15.0),
            param("q_just_above_0_1", factor=16.0),
            param("q_at_0_5", factor=78.2),
            param("q_in_0_5_1", factor=85.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_measured_factor_moves_the_floors(self, name, factor):
        aligned_rows = 100_096
        caching_floor, hybrid_floor = self._expected_floors(
            factor, self.BATCH, aligned_rows
        )
        families = self._candidates(self._constraint(unique_id_factor=factor))
        self.assertEqual(
            [round(r, 9) for r in self._partial_ratios(families, True)],
            [round(r, 9) for r in _expected_caching_ratios(caching_floor)],
        )
        self.assertEqual(
            [round(r, 9) for r in self._partial_ratios(families, False)],
            [round(r, 9) for r in _expected_hybrid_ratios(hybrid_floor)],
        )
        self.assertEqual(self._ratios(families, False)[:1], [0.0])
        self.assertEqual(self._ratios(families, False)[-1:], [1.0])

    def test_high_lower_endpoint_inserts_itself(self):
        # q in (0.7, 0.8): caching keeps the useful high ratios plus the
        # endpoint itself instead of discarding caching entirely.
        factor = 114.0
        caching_floor, _ = self._expected_floors(factor, self.BATCH, 100_096)
        self.assertGreater(caching_floor, 0.7)
        self.assertLess(caching_floor, 0.8)
        families = self._candidates(self._constraint(unique_id_factor=factor))
        self.assertEqual(
            [round(r, 9) for r in self._partial_ratios(families, True)],
            [round(caching_floor, 9), 0.8, 0.9],
        )
        # the non-caching segment is exempt from the cache-residency floor
        self.assertEqual(len(self._partial_ratios(families, False)), 9)

    def test_working_set_reaching_the_table_drops_partial_caching(self):
        # q >= 1: no partial cache is offered; HBM_ONLY stays valid.
        factor = 160.0
        caching_floor, _ = self._expected_floors(factor, self.BATCH, 100_096)
        self.assertGreaterEqual(caching_floor, 1.0)
        families = self._candidates(self._constraint(unique_id_factor=factor))
        self.assertEqual(self._ratios(families, True), [])
        self.assertEqual(len(self._partial_ratios(families, False)), 9)
        self.assertEqual(len(self._ratios(families, False)[:1]), 1)
        self.assertEqual(len(self._ratios(families, False)[-1:]), 1)

    def test_measured_zero_keeps_the_unmeasured_floors(self):
        # The zero endpoint already represents the empty working set, so a
        # measured zero keeps both positive segments at the 0.1 unmeasured
        # floor instead of introducing a minimum-allocation search.
        families = self._candidates(self._constraint(unique_id_factor=0.0))
        self.assertEqual(self._ratios(families, False)[:1], [0.0])
        self.assertEqual(
            [round(r, 9) for r in self._partial_ratios(families, False)],
            [round(r, 9) for r in _expected_hybrid_ratios(0.1)],
        )
        self.assertEqual(
            [round(r, 9) for r in self._partial_ratios(families, True)],
            [round(r, 9) for r in _expected_caching_ratios(0.1)],
        )

    def test_low_segment_is_geometric_with_five_points(self):
        points = _dynamicemb_low_segment(0.02)
        self.assertEqual(len(points), 5)
        self.assertEqual(points[0], 0.02)
        self.assertEqual(points[-1], 0.5)
        # geometric spacing: constant ratio between consecutive points
        for i in range(len(points) - 2):
            self.assertAlmostEqual(
                points[i + 1] / points[i], points[i + 2] / points[i + 1], places=9
            )
        self.assertEqual(points, sorted(set(points)))
        self.assertEqual(_dynamicemb_low_segment(0.5), [0.5])
        self.assertEqual(_dynamicemb_low_segment(0.73), [])

    def test_world_size_is_not_applied_twice(self):
        # f = 78.2, B = 128 gives U5 = 50_048 = N/2, so the CACHING floor
        # sits at 0.5. A second world_size division would lower it to 0.25
        # and admit grid points that cannot hold the working set.
        families = self._candidates(self._constraint(unique_id_factor=78.2))
        self.assertEqual(round(min(self._ratios(families, True)), 9), 0.5)

    def test_unmeasured_large_pooling_drops_small_caches_only(self):
        # Without a factor the fallback estimate is the batch lookup count;
        # caches that cannot hold it are dropped, while HYBRID is exempt
        # from the cache check and keeps its whole grid.
        families = self._candidates(self._constraint(pooling=500.0), batch_size=100)
        self.assertEqual(
            self._partial_ratios(families, True), [0.5, 0.6, 0.7, 0.8, 0.9]
        )
        self.assertEqual(len(self._partial_ratios(families, False)), 9)

    def test_explicit_zero_is_one_host_only(self):
        families = self._candidates(self._constraint(explicit_ratio=0.0))
        self.assertEqual(sorted(k for k in families if families[k]), [False])
        self.assertEqual(self._ratios(families, False)[:1], [0.0])

    def test_explicit_one_is_one_hbm_only(self):
        families = self._candidates(self._constraint(explicit_ratio=1.0))
        self.assertEqual(sorted(k for k in families if families[k]), [False])
        self.assertEqual(self._ratios(families, False)[-1:], [1.0])

    def test_explicit_partial_pins_both_families(self):
        families = self._candidates(self._constraint(explicit_ratio=0.37))
        self.assertEqual([round(r, 9) for r in self._ratios(families, False)], [0.37])
        self.assertEqual([round(r, 9) for r in self._ratios(families, True)], [0.37])

    def test_explicit_caching_failure_keeps_hybrid(self):
        # The pinned cache cannot hold the batch, but the same-ratio HYBRID
        # needs no cache and survives.
        families = self._candidates(
            self._constraint(explicit_ratio=0.1, pooling=500.0), batch_size=100
        )
        self.assertEqual([round(r, 9) for r in self._ratios(families, False)], [0.1])
        self.assertEqual(self._ratios(families, True), [])

    def test_explicit_no_eligible_variant_raises(self):
        # NO_EVICTION refuses HYBRID and the pinned cache cannot hold the
        # batch: the table has no eligible variant at the pin.
        constraint = self._constraint(
            explicit_ratio=0.1, pooling=500.0, score_strategy="NO_EVICTION"
        )
        with self.assertRaisesRegex(ValueError, "no eligible variant"):
            self._candidates(constraint, batch_size=100)

    def test_ineligible_factor_is_rejected(self):
        constraint = self._constraint(unique_id_factor=float("nan"))
        with self.assertRaisesRegex(ValueError, "unique_id_factor"):
            self._candidates(constraint)
        constraint = self._constraint(unique_id_factor=-1.0)
        with self.assertRaisesRegex(ValueError, "unique_id_factor"):
            self._candidates(constraint)

    def test_unsplittable_policy_keeps_non_caching_full_budget_only(self):
        # HybridStorage refuses NO_EVICTION tables. Automatic non-caching
        # choices stay full-budget only -- a zero or partial member could
        # fuse with budgeted members into an unsupported mixed HYBRID --
        # while the supported caching choices keep their grid.
        families = self._candidates(self._constraint(score_strategy="NO_EVICTION"))
        self.assertEqual(self._ratios(families, False), [1.0])
        self.assertEqual(len(self._partial_ratios(families, True)), 9)

    def test_clamp_collapses_equal_budgets_keeping_best_score(self):
        # A 17-row shard aligned to 128 rows: every partial ratio whose
        # aligned budget reaches the un-rounded fit threshold clamps to the
        # same budget and realizes the same tier rows, so the physically
        # equal candidates collapse. The retained estimator scores nominal
        # ratios differently, so the best-scored (highest-ratio) candidate
        # wins rather than the smallest.
        families = self._candidates(
            self._constraint(num_embeddings=34), num_embeddings=34
        )
        non_caching = [r for r in self._ratios(families, False) if 0.0 < r < 1.0]
        # 0.1's budget stays below the threshold; every ratio from the
        # second grid point up clamps to it
        self.assertEqual(non_caching, [0.1, 0.9])
        # the caching family collapses the same way, also keeping 0.9
        caching = self._ratios(families, True)
        self.assertEqual(caching, [0.1, 0.9])

    def test_equal_rows_with_different_budgets_do_not_collapse(self):
        # On a table whose tier rows are identical across ratios (a cache
        # bucket already covers the shard) but whose budgets differ, the
        # candidates are NOT equivalent: native fusion decides the group
        # from summed budgets, and a different budget can flip the fit.
        families = self._candidates(
            self._constraint(num_embeddings=4096), num_embeddings=4096
        )
        caching = self._ratios(families, True)
        self.assertEqual(len(caching), 9)
        # every surviving candidate realizes the same 1024-row resident
        # tier here, so what keeps them distinct is the budget the mirror
        # computes for the dedup key -- verify by recomputing it
        from tzrec.utils.dynamicemb_util import dynamicemb_table_memory

        budgets = {
            dynamicemb_table_memory(
                caching=True,
                ratio=float(so.cache_load_factor),
                shard_rows=so.shards[0].size[0],
                dim=so.tensor.shape[1],
                element_size=so.tensor.element_size(),
            ).local_hbm_for_values
            for so in families[True]
        }
        self.assertEqual(len(budgets), len(caching))


@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@mark_ci_scope("gpu")
class DynamicEmbPlannerRegressionTest(unittest.TestCase):
    """Feasibility regressions for the four-layout candidate space."""

    def _enumerate(self, num_embeddings, dim, world_size=2, batch_size=32):
        from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints

        table = EmbeddingBagConfig(
            num_embeddings=num_embeddings,
            embedding_dim=dim,
            name="table_big",
            feature_names=["feat_big"],
        )
        cfg = feature_pb2.DynamicEmbedding()
        cfg.max_capacity = num_embeddings
        cfg.score_strategy = "STEP"
        constraint = build_dynamicemb_constraints(cfg, table)
        model = TestSparseNN(tables=[table], sparse_device=torch.device("meta"))
        topology = Topology(world_size=world_size, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=batch_size,
            fqn_constraints={"sparse.ebc.table_big": constraint},
        )
        from tzrec.utils.plan_util import (
            get_default_sharders as _tzrec_get_default_sharders,
        )

        search_space = enumerator.enumerate(
            module=model, sharders=_tzrec_get_default_sharders()
        )
        return [so for so in search_space if getattr(so, "use_dynamicemb", False)]

    def _dp_proposals(self, search_space, world_size, hbm_cap, ddr_cap):
        constraint = SimpleNamespace(
            devices=[
                SimpleNamespace(storage=SimpleNamespace(hbm=hbm_cap, ddr=ddr_cap))
                for _ in range(world_size)
            ],
            local_world_size=world_size,
        )
        proposer = DynamicProgrammingProposer(
            hbm_bins_per_device=100, ddr_bins_per_device=25
        )
        proposer.load(search_space)
        if not proposer.propose():
            return []
        proposer.feedback(partitionable=True, storage_constraint=constraint)
        proposals = []
        proposal = proposer.propose()
        while proposal:
            proposals.append(proposal)
            proposer.feedback(partitionable=True, storage_constraint=constraint)
            proposal = proposer.propose()
        return proposals

    def test_large_table_fits_only_with_new_candidates(self):
        # Per-rank rows = 20M, dim 32 fp32: values = 2.56 GB/rank. With the
        # old candidate space the smallest HBM value share is 10%, which
        # together with the always-resident hash-map metadata exceeds a
        # 520 MB budget; HOST_ONLY fits with metadata alone.
        search_space = self._enumerate(40_000_000, 32)
        old_space = [so for so in search_space if so.cache_load_factor >= 0.1]
        hbm_cap = 520 * (1 << 20)
        for so in old_space:
            self.assertGreater(max(shard.storage.hbm for shard in so.shards), hbm_cap)
        zero = next(
            so
            for so in search_space
            if not so.dynamicemb_options.caching and so.cache_load_factor == 0.0
        )
        self.assertLessEqual(max(shard.storage.hbm for shard in zero.shards), hbm_cap)

        # the old candidate space has no feasible DP proposal...
        self.assertEqual(self._dp_proposals(old_space, 2, hbm_cap, 10 << 30), [])
        # ...while the full space yields one, placing the table at the zero
        # endpoint
        proposals = self._dp_proposals(search_space, 2, hbm_cap, 10 << 30)
        self.assertTrue(proposals)
        chosen = {so.fqn: so for so in proposals[0]}
        placed = chosen["sparse.ebc.table_big"]
        self.assertEqual(placed.cache_load_factor, 0.0)
        self.assertFalse(placed.dynamicemb_options.caching)
        self.assertEqual(placed.dynamicemb_options.local_hbm_for_values, 0)

    def test_metadata_alone_exceeding_hbm_still_fails(self):
        # HOST_ONLY is not free GPU memory: its hash-map metadata alone
        # (>= 21 bytes per row over 20M rows) exceeds a 300 MB budget, so
        # even the new candidate space has no feasible placement.
        search_space = self._enumerate(40_000_000, 32)
        self.assertEqual(
            self._dp_proposals(search_space, 2, 300 * (1 << 20), 10 << 30),
            [],
        )

    def test_zero_ddr_topology_envelope_is_conservative(self):
        # A topology without host memory cannot hold zero, partial or
        # caching values. The full endpoint's DDR estimate is the
        # mixed-group minimum host bucket of the conservative envelope, so
        # it too is pruned here even though an all-full group would really
        # allocate no host tier -- the envelope approximation the design
        # accepts rather than promising exact zero-DDR feasibility.
        search_space = self._enumerate(4096, 32)
        hbm_cap = 64 * (1 << 20)
        survivors = []
        for so in search_space:
            max_hbm = max(shard.storage.hbm for shard in so.shards)
            max_ddr = max(shard.storage.ddr for shard in so.shards)
            if max_hbm <= hbm_cap and max_ddr <= 0:
                survivors.append((so.cache_load_factor, so.dynamicemb_options.caching))
        self.assertEqual(survivors, [])

    def test_tight_ddr_keeps_high_hybrid_candidates(self):
        # DDR pressure must not remove the useful high-HBM HYBRID points:
        # the candidate set still spans up to 0.9, so a nearly-full HBM
        # split remains plannable.
        search_space = self._enumerate(40_000_000, 32)
        non_caching_partials = sorted(
            (
                so
                for so in search_space
                if not so.dynamicemb_options.caching
                and 0.0 < so.cache_load_factor < 1.0
            ),
            key=lambda so: so.cache_load_factor,
        )
        self.assertEqual(
            [
                round(cast(float, so.cache_load_factor), 9)
                for so in non_caching_partials
            ][-4:],
            [0.6, 0.7, 0.8, 0.9],
        )

    def test_full_hbm_candidate_is_not_billed_twice(self):
        # The HBM_ONLY candidate charges values once and no host tier.
        search_space = self._enumerate(4096, 32)
        full = next(
            so
            for so in search_space
            if not so.dynamicemb_options.caching and so.cache_load_factor == 1.0
        )
        # values are charged once; the only DDR in the estimate is the
        # mixed-group minimum host bucket, never a full backing table
        for shard in full.shards:
            self.assertLessEqual(shard.storage.ddr, 128 * 32 * 4)
            self.assertGreater(shard.storage.hbm, 0)


def _expected_caching_ratios(floor):
    ratios = set(_dynamicemb_low_segment(floor))
    ratios.update(r for r in (0.5, 0.6, 0.7, 0.8, 0.9) if r >= floor)
    if 0.5 < floor < 1.0:
        ratios.add(floor)
    return sorted(r for r in ratios if 0.0 < r < 1.0)


def _expected_hybrid_ratios(floor):
    ratios = set(_dynamicemb_low_segment(floor))
    ratios.update((0.5, 0.6, 0.7, 0.8, 0.9))
    return sorted(r for r in ratios if 0.0 < r < 1.0)


if __name__ == "__main__":
    unittest.main()
