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
from typing import List, Tuple
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer, UniformProposer
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import PlannerError, Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2, train_pb2
from tzrec.utils import plan_util, rocksdb_capacity, rocksdb_util
from tzrec.utils.dynamicemb_util import has_dynamicemb
from tzrec.utils.plan_util import (
    DynamicProgrammingProposer,
    EmbeddingEnumerator,
    _sparse_dp_proposor_numpy,
    create_planner,
)
from tzrec.utils.test_util import (
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)


class PlannerHostMemoryTest(unittest.TestCase):
    """The planner holds back HOST_MEMORY_RESERVE_PERCENT of host memory."""

    @parameterized.expand(
        [param("default", env=None, share=0.85), param("set", env="0.5", share=0.5)],
        name_func=parameterized_name_func,
    )
    def test_reserve(self, _, env, share):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        environ = dict(os.environ)
        environ.pop("HOST_MEMORY_RESERVE_PERCENT", None)
        if env is not None:
            environ["HOST_MEMORY_RESERVE_PERCENT"] = env
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(plan_util, "host_memory_bytes", return_value=100 << 30),
            mock.patch.object(plan_util, "get_local_size", return_value=1),
            mock.patch.object(plan_util.dist, "get_world_size", return_value=1),
        ):
            planner = plan_util.create_planner(device, batch_size=8, model=nn.Module())
        self.assertEqual(
            planner._topology.devices[0].storage.ddr, int((100 << 30) * share)
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

    def test_reload_runs_the_dp_again(self):
        topology = _make_topology(
            num_devices=1, hbm_per_device=2000, ddr_per_device=2000
        )
        proposer = DynamicProgrammingProposer()
        proposer.load([_FakeShardingOption("table_a", hbm=100, ddr=0, perf=1.0)])
        proposer.propose()
        proposer.feedback(partitionable=False, storage_constraint=topology)

        fits = _FakeShardingOption("table_a", hbm=200, ddr=0, perf=2.0)
        oversized = _FakeShardingOption("table_a", hbm=100, ddr=9000, perf=3.0)
        proposer.load([oversized, fits])
        proposer.propose()
        proposer.feedback(partitionable=False, storage_constraint=topology)
        self.assertEqual(proposer.propose(), [fits])

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


GIB = 1 << 30


def _view(rank=0, free=10 << 40, job=0, pool="net:nas.example"):
    return rocksdb_capacity.DiskView(
        rank=rank, pool=pool, free_bytes=free, job_bytes=job
    )


def _model_and_constraints(tables):
    """tables: name -> (rows, dim, DynamicEmbedding fields[, pooling factor])."""
    from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints

    configs, constraints = [], {}
    for name, (rows, dim, fields, *pooling) in tables.items():
        config = EmbeddingBagConfig(
            num_embeddings=rows,
            embedding_dim=dim,
            name=name,
            feature_names=[f"f_{name}"],
        )
        configs.append(config)
        constraints[f"sparse.ebc.{name}"] = build_dynamicemb_constraints(
            feature_pb2.DynamicEmbedding(max_capacity=rows, **fields),
            config,
            *pooling,
        )
    return TestSparseNN(tables=configs, sparse_device=torch.device("meta")), constraints


@unittest.skipUnless(has_dynamicemb, "dynamicemb is required")
@mark_ci_scope("gpu")
class RocksDBOffloadPlannerTest(unittest.TestCase):
    """The DP proposer chooses offloaded tables, cache ratios, host and disk."""

    # 2^24 x 64 fp32 with SGD: 4 GiB of values per table.
    BIG = (1 << 24, 64, {})

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")

    def _planner(self, tables, hbm_gib, ddr_gib, budget=None, views=None):
        model, constraints = _model_and_constraints(tables)
        topology = Topology(
            world_size=1,
            compute_device="cuda",
            hbm_cap=int(hbm_gib * GIB),
            ddr_cap=int(ddr_gib * GIB),
        )
        cfg = train_pb2.RocksDBConfig(root_dir="/unused")
        if budget is not None:
            cfg.disk_budget_bytes = budget
        planner = plan_util.RocksDBOffloadPlanner(
            rocksdb=SimpleNamespace(config=cfg),
            views=views or [_view()],
            topology=topology,
            batch_size=128,
            enumerator=plan_util.EmbeddingEnumerator(
                topology=topology,
                batch_size=128,
                fqn_constraints=constraints,
                rocksdb=True,
            ),
            storage_reservation=HeuristicalStorageReservation(percentage=0.15),
            proposer=[DynamicProgrammingProposer(), UniformProposer()],
        )
        return planner, model

    def _plan(self, tables, hbm_gib, ddr_gib, **kwargs):
        planner, model = self._planner(tables, hbm_gib, ddr_gib, **kwargs)
        plan = planner.plan(model, plan_util.get_default_sharders())
        offloaded = {
            name: p.dynamicemb_options
            for m in plan.plan.values()
            for name, p in m.items()
            if rocksdb_util.is_rocksdb_options(p.dynamicemb_options)
        }
        return plan, offloaded, planner

    @parameterized.expand(
        [
            param("rocksdb", rocksdb=True, modes={True}),
            param("builtin", rocksdb=False, modes={False, True}),
        ],
        name_func=parameterized_name_func,
    )
    def test_memory_table_layouts(self, _, rocksdb, modes):
        # With RocksDB configured, a table stays in memory only in a layout
        # every restore preserves.
        base = SimpleNamespace(
            dynamicemb_options=SimpleNamespace(caching=False, external_storage=None),
            cache_params=None,
        )
        variants = plan_util._emit_dynamicemb_variants(base, rocksdb)
        self.assertEqual({v.dynamicemb_options.caching for v in variants}, modes)

    def test_all_in_memory(self):
        _, offloaded, planner = self._plan({"a": self.BIG, "b": self.BIG}, 80, 64)
        self.assertEqual(offloaded, {})
        # RocksDB's fixed host memory is charged only to an offloading plan.
        self.assertEqual(planner._topology.devices[0].storage.ddr, 64 * GIB)

    def test_offload_when_memory_is_short(self):
        tables = {"a": self.BIG, "b": self.BIG, "c": (1 << 20, 64, {})}
        _, offloaded, planner = self._plan(tables, 4, 4.75)
        self.assertTrue(offloaded)
        self.assertNotIn("c", offloaded)
        configs = {o.external_storage.backend_config for o in offloaded.values()}
        self.assertEqual(len(configs), 1)
        (config,) = configs
        self.assertGreater(
            config.block_cache_bytes, rocksdb_util.BLOCK_CACHE_DATA_BYTES
        )
        self.assertGreater(config.import_budget_bytes, 0)
        for name, o in offloaded.items():
            factory = o.external_storage
            self.assertEqual(factory.table_namespace, f"sparse.ebc.{name}")
            # DynamicEmb sizes a fused group's cache with its budget over its
            # value bytes: the budget must be exactly a planned ratio's bytes.
            per_row = o.max_capacity * o.dim * 4
            budgets = {math.ceil(r * per_row) for r in rocksdb_util.CACHE_RATIOS}
            self.assertIn(o.local_hbm_for_values, budgets)
        self.assertEqual(planner._topology.devices[0].storage.ddr, int(4.75 * GIB))

    def test_overrides_stay_in_memory(self):
        tables = {
            "b": (1 << 24, 64, {"cache_load_factor": 0.5}),
            "c": (1 << 24, 64, {"score_strategy": "CUSTOMIZED"}),
            "d": (1 << 24, 64, {}),
        }
        _, offloaded, _ = self._plan(tables, 4, 16)
        self.assertEqual(set(offloaded), {"d"})

    def test_cache_holds_the_batch_estimate(self):
        # 128 samples x 1000 ids: ratios whose cache is below 128k rows of a
        # 16M-row table (0.0001, 0.001) are not offered.
        planner, model = self._planner({"a": (1 << 24, 64, {}, 1000.0)}, 4, 4)
        enumerator = planner._enumerator
        enumerator.offload = True
        options = enumerator.enumerate(model, plan_util.get_default_sharders())
        ratios = sorted(
            o.cache_params.load_factor
            for o in options
            if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        )
        self.assertEqual(ratios, list(rocksdb_util.CACHE_RATIOS[2:]))

    def test_rocksdb_misses_cost_disk_bandwidth(self):
        planner, model = self._planner({"a": self.BIG}, 4, 4)
        planner._enumerator.offload = True
        options = planner._enumerator.enumerate(model, plan_util.get_default_sharders())
        perf = {}
        for o in options:
            rocksdb = rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            perf[(rocksdb, o.cache_load_factor)] = o.total_perf
        for ratio in (0.1, 0.5, 0.9):
            self.assertGreater(perf[(True, ratio)], perf[(False, ratio)])

    def test_disk_axis_picks_the_table_that_fits(self):
        # Offloading "a" or "b" (4 GiB of values each) covers the GPU deficit;
        # "a" is cheaper (100x fewer lookups) but only "b" fits the disk.
        tables = {
            "a": (1 << 25, 32, {}, 1.0),
            "b": (1 << 24, 64, {}, 100.0),
            "c": (1 << 20, 64, {}),
        }
        _, offloaded, planner = self._plan(tables, 1.76, 12)
        self.assertEqual(set(offloaded), {"a"})
        backend = rocksdb_util._import_backend()
        overheads = backend.disk_overheads(
            planner._backend_config(block_cache_bytes=0, import_budget_bytes=0)
        )

        def growth(rows, dim, ratio, lookups):
            return rocksdb_capacity.TableDisk(
                "t",
                rows,
                backend.logical_row_bytes(dim, 0, 4),
                backend.index_entry_bytes(),
                "",
                rocksdb_util.cache_rows(rows, ratio),
                lookups,
            ).growth_bytes

        # Room for "b" at its smallest cache ratio, and for "a" at none.
        room = (
            growth(1 << 24, 64, 0.001, 12800) + growth(1 << 25, 32, 0.0001, 128)
        ) // 2
        budget = (
            room
            + rocksdb_capacity.DISK_RESERVE_BYTES
            + rocksdb_capacity.pool_fixed_bytes(
                overheads, (1 << 25) * backend.logical_row_bytes(32, 0, 4)
            )
        )
        _, offloaded, _ = self._plan(tables, 1.76, 12, budget=budget)
        self.assertEqual(set(offloaded), {"b"})

    def test_no_plan_fits(self):
        tables = {"a": (1 << 26, 64, {}), "b": (1 << 26, 64, {})}
        with self.assertRaises(PlannerError):
            self._plan(tables, 4, 4.75, budget=40 * GIB)


if __name__ == "__main__":
    unittest.main()
