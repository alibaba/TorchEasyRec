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
from functools import partial
from types import SimpleNamespace
from typing import Any, List, Tuple
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner import EmbeddingShardingPlanner
from torchrec.distributed.planner.partitioners import GreedyPerfPartitioner
from torchrec.distributed.planner.proposers import GridSearchProposer
from torchrec.distributed.planner.storage_reservations import (
    HeuristicalStorageReservation,
)
from torchrec.distributed.planner.types import (
    PlannerError,
    PlannerErrorType,
    Storage,
    Topology,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import CacheParams, ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2
from tzrec.utils import dynamicemb_util, plan_util, resource_util, rocksdb_util
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


class PlannerHostMemoryTest(unittest.TestCase):
    """Every configuration sizes host memory by the container-aware budget.

    ``HOST_MEMORY_RESERVE_PERCENT`` (default 0) holds back host RAM for
    non-embedding use in ordinary CPU/HYBRID/CACHING plans as well as in
    RocksDB plans; zero reserve disables only the margin, never the
    container limit.
    """

    @parameterized.expand(
        [
            param("default_reserve", env=None, reserve=0.0),
            param("explicit_reserve", env="0.5", reserve=0.5),
            param("zero_reserve", env="0.0", reserve=0.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_reserve(self, _, env, reserve):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        environ = dict(os.environ)
        environ.pop("HOST_MEMORY_RESERVE_PERCENT", None)
        if env is not None:
            environ["HOST_MEMORY_RESERVE_PERCENT"] = env
        container = 100 << 30
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(
                resource_util,
                "_container_memory_limit_bytes",
                return_value=container,
            ),
            mock.patch.object(plan_util, "get_local_size", return_value=1),
            mock.patch.object(plan_util.dist, "get_world_size", return_value=1),
        ):
            # The ordinary, no-RocksDB planner uses the same container-aware
            # budget as a RocksDB one.
            planner = plan_util.create_planner(device, batch_size=8, model=nn.Module())
            self.assertEqual(
                planner._topology.devices[0].storage.ddr,
                int(container * (1 - reserve)),
            )
            rocksdb = mock.Mock(
                rank_dir="/tmp/db", config=mock.Mock(root_dir="/tmp/db")
            )
            with mock.patch.object(
                resource_util, "ssd_capacity_per_rank", return_value=0
            ):
                planner = plan_util.create_planner(
                    device, batch_size=8, model=nn.Module(), rocksdb=rocksdb
                )
            self.assertEqual(
                planner._topology.devices[0].storage.ddr,
                int(container * (1 - reserve)),
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

    @parameterized.expand(
        [
            param("default_reserve", env=None, reserve=0.0),
            param("explicit_reserve", env="0.5", reserve=0.5),
            param("zero_reserve", env="0.0", reserve=0.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_host_memory_reserve(self, _, env, reserve):
        """The planner sizes its DDR budget by container RAM less a reserve.

        ``HOST_MEMORY_RESERVE_PERCENT`` (default 0.0) can hold back host RAM
        for everything but embeddings; zero reserve disables only the
        margin, never the container limit.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        environ = dict(os.environ)
        environ.pop("HOST_MEMORY_RESERVE_PERCENT", None)
        if env is not None:
            environ["HOST_MEMORY_RESERVE_PERCENT"] = env
        container = 100 << 30
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(
                resource_util,
                "_container_memory_limit_bytes",
                return_value=container,
            ),
            mock.patch.object(plan_util, "get_local_size", return_value=1),
            mock.patch.object(plan_util.dist, "get_world_size", return_value=1),
        ):
            planner = plan_util.create_planner(device, batch_size=8, model=nn.Module())
            self.assertEqual(
                planner._topology.devices[0].storage.ddr,
                int(container * (1 - reserve)),
            )

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
    def __init__(self, hbm, ddr, ssd=0):
        self.hbm = hbm
        self.ddr = ddr
        self.ssd = ssd

    def __sub__(self, other):
        return _FakeStorage(
            self.hbm - other.hbm, self.ddr - other.ddr, self.ssd - other.ssd
        )


class _FakeShard:
    def __init__(self, hbm, ddr, ssd=0):
        self.storage = _FakeStorage(hbm, ddr, ssd)


class _FakePartitioner:
    """Places a proposal when every shard fits the first device's storage."""

    def __init__(self):
        self.seen = []

    def partition(self, proposal, storage_constraint):
        self.seen.append(storage_constraint)
        device = storage_constraint.devices[0]
        for option in proposal:
            for shard in option.shards:
                if not (
                    shard.storage.hbm <= device.storage.hbm
                    and shard.storage.ddr <= device.storage.ddr
                    and shard.storage.ssd <= device.storage.ssd
                ):
                    raise PlannerError(
                        "shard does not fit", PlannerErrorType.INSUFFICIENT_STORAGE
                    )
        return proposal


class _FakeShardingOption:
    """Minimal ShardingOption stand-in: only the fields the DP proposer reads."""

    def __init__(self, fqn, hbm, ddr, perf, ssd=0):
        self.fqn = fqn
        # Total = single shard for simplicity (single-rank assignment).
        self.shards = [_FakeShard(hbm, ddr, ssd)]
        self.total_storage = _FakeStorage(hbm, ddr, ssd)
        self.total_perf = perf


def _make_topology(
    num_devices,
    hbm_per_device,
    ddr_per_device,
    local_world_size=None,
    ssd_per_device=0,
):
    return SimpleNamespace(
        devices=[
            SimpleNamespace(
                storage=_FakeStorage(
                    hbm=hbm_per_device, ddr=ddr_per_device, ssd=ssd_per_device
                )
            )
            for _ in range(num_devices)
        ],
        local_world_size=local_world_size or num_devices,
    )


def _make_random_table_opts(
    seed: int,
    table_count: int = 5,
    option_count: int = 4,
    hbm_bins: int = 8,
    ddr_bins: int = 8,
    ssd_bins: int = 8,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Seeded random per-table option arrays for property testing.

    Per-option bin contributions stay in [0, axis/2) so combined plans can fit
    several tables without saturating any axis. Some options carry SSD cost
    so the activation state is exercised.
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
        ssd = np.asarray(
            [
                rng.uniform(0, ssd_bins / 2) if rng.random() < 0.4 else 0.0
                for _ in range(option_count)
            ],
            dtype=np.float32,
        )
        perf = np.asarray(
            [rng.uniform(1, 100) for _ in range(option_count)], dtype=np.float32
        )
        opt_id = np.arange(option_count, dtype=np.int32)
        table_opts.append((hbm, ddr, ssd, perf, opt_id))
    return table_opts


def _exhaustive_best_proposals(
    table_opts,
    hbm_bins,
    ddr_bins,
    ssd_bins,
    activation_ddr_units=0.0,
    activation_ssd_units=0.0,
):
    """Exhaustive-combination oracle for the binned DP on a tiny input.

    Enumerates every per-table option combination directly -- no second
    implementation of the sparse DP -- keeps the feasible ones (exact
    equality to a capacity lands in the last bin), charges the shared
    activation on the first SSD-billed option, and keeps the best proposal
    per (HBM bin, activation state). Mirrors the semantics specified for
    the in-tree DP: same feasibility rule, same grouping, same tie order
    (inactive first within a bin).
    """
    import itertools

    best = {}
    for combo in itertools.product(*(range(len(t[3])) for t in table_opts)):
        hbm = ddr = ssd = perf = 0.0
        active = False
        for table_i, opt_j in enumerate(combo):
            _, _, opt_ssd, opt_perf, _ = table_opts[table_i]
            if opt_ssd[opt_j] > 0 and not active:
                active = True
                ddr += activation_ddr_units
                ssd += activation_ssd_units
            hbm += float(table_opts[table_i][0][opt_j])
            ddr += float(table_opts[table_i][1][opt_j])
            ssd += float(opt_ssd[opt_j])
            perf += float(opt_perf[opt_j])
        if hbm > hbm_bins or ddr > ddr_bins or ssd > ssd_bins:
            continue
        hbm_i = min(int(hbm), hbm_bins - 1)
        key = (hbm_i, active)
        if key not in best or perf < best[key][0]:
            best[key] = (perf, combo)
    ordered = sorted(best.items(), key=lambda kv: (-kv[0][0], kv[0][1]))
    return [list(kv[1][1]) for kv in ordered]


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
    """Binned DP across HBM × DDR (× SSD for RocksDB) picks per-table modes."""

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

    @parameterized.expand(
        [
            param("no_ssd", ssd_cap=0),
            param("one_ssd_tight", ssd_cap=10_000),
            param("nonbinding_axis", ssd_cap=1_000_000),
        ],
        name_func=parameterized_name_func,
    )
    def test_shared_activation_cost_combinations(self, _, ssd_cap):
        # One table offers an SSD option (cheap) beside a memory option; the
        # other is memory-only. The activation is charged once to any
        # combination that selects the SSD option: its 1000 ddr plus the 500
        # activation fits the 1500 device exactly, so a missing charge admits
        # an infeasible plan and a second charge breaks the active one. The
        # all-memory alternative never pays it, and survives even when the
        # SSD axis cannot bind (the fixed DDR charge and the inactive
        # alternative both survive the collapse).
        tables = [
            _FakeShardingOption("t.a", hbm=100, ddr=1000, perf=10.0, ssd=2000),
            _FakeShardingOption("t.a", hbm=100, ddr=0, perf=90.0, ssd=0),
            _FakeShardingOption("t.b", hbm=100, ddr=0, perf=50.0, ssd=0),
        ]
        topology = _make_topology(
            num_devices=1,
            hbm_per_device=1000,
            ddr_per_device=1500,
            ssd_per_device=ssd_cap,
        )
        activation = Storage(hbm=0, ddr=500, ssd=3000)
        proposer = DynamicProgrammingProposer(
            hbm_bins_per_device=20,
            ddr_bins_per_device=20,
            ssd_bins_per_device=20,
            ssd_activation=activation,
        )
        # load() groups by fqn; each fqn's options keep insertion order.
        proposer.load(tables)
        proposer.propose()
        proposer.feedback(partitionable=True, storage_constraint=topology)
        proposals = []
        proposal = proposer.propose()
        while proposal:
            proposals.append(proposal)
            proposer.feedback(partitionable=True, storage_constraint=topology)
            proposal = proposer.propose()

        active = [p for p in proposals if any(o.total_storage.ssd > 0 for o in p)]
        inactive = [p for p in proposals if not any(o.total_storage.ssd > 0 for o in p)]
        self.assertTrue(inactive, "the all-memory alternative must survive")
        if ssd_cap == 0:
            # Positive SSD cost with zero capacity is infeasible.
            self.assertFalse(active)
            return
        self.assertTrue(active, "the one-charge active plan must fit exactly")
        # The active plan pays the activation exactly once: its shards carry
        # 1000 ddr and the budget is 1500, so a second 500 charge would
        # leave it infeasible.
        for plan in active:
            self.assertEqual(sorted(o.fqn for o in plan), ["t.a", "t.b"])

    def test_shortcut_pays_activation_before_acceptance(self):
        # The initial shortcut (each table's smallest-memory option) fits by
        # additive cost but exceeds capacity once the shared activation is
        # charged; the adapter must reject it, and the shared topology must
        # stay untouched while a pure-memory proposal delegates untouched.
        ssd_table = _FakeShardingOption("t.ssd", hbm=100, ddr=1000, perf=10.0, ssd=100)
        mem_table = _FakeShardingOption("t.b", hbm=100, ddr=0, perf=50.0, ssd=0)
        topology = _make_topology(
            num_devices=1, hbm_per_device=1000, ddr_per_device=1200, ssd_per_device=5000
        )
        activation = Storage(hbm=0, ddr=500, ssd=0)
        before = (
            topology.devices[0].storage.hbm,
            topology.devices[0].storage.ddr,
            topology.devices[0].storage.ssd,
        )
        partitioner = plan_util._ActivationAwarePartitioner(
            _FakePartitioner(), activation
        )
        # Additive cost fits (1000 + 0 <= 1200); after the activation the
        # device keeps 700, below the SSD option's 1000 ddr.
        with self.assertRaises(PlannerError):
            partitioner.partition([ssd_table, mem_table], topology)
        self.assertEqual(
            (
                topology.devices[0].storage.hbm,
                topology.devices[0].storage.ddr,
                topology.devices[0].storage.ssd,
            ),
            before,
            "the shared topology must not be mutated",
        )
        # A pure-memory proposal delegates untouched and passes.
        self.assertEqual(partitioner.partition([mem_table], topology), [mem_table])
        # The charged copy is fresh each time: a retry starts from full
        # capacity, never from the previous deduction.
        with self.assertRaises(PlannerError):
            partitioner.partition([ssd_table, mem_table], topology)

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

    @parameterized.expand(
        [
            (seed, act_ddr, act_ssd)
            for seed in [0, 1, 7, 42, 1337]
            for act_ddr, act_ssd in [(0.0, 0.0), (1.5, 2.0), (0.0, 3.0)]
        ]
    )
    def test_sparse_numpy_matches_exhaustive_reference(
        self, seed, activation_ddr, activation_ssd
    ):
        # Property test: the sparse-K NumPy DP must produce the same proposal
        # set as exhaustively enumerating every tiny input's combinations,
        # including SSD costs and the shared activation charge.
        table_opts = _make_random_table_opts(
            seed, table_count=4, option_count=3, hbm_bins=8, ddr_bins=8, ssd_bins=8
        )
        actual = _sparse_dp_proposor_numpy(
            table_opts,
            hbm_bins=8,
            ddr_bins=8,
            ssd_bins=8,
            activation_ddr_units=activation_ddr,
            activation_ssd_units=activation_ssd,
        )
        expected = _exhaustive_best_proposals(
            table_opts,
            hbm_bins=8,
            ddr_bins=8,
            ssd_bins=8,
            activation_ddr_units=activation_ddr,
            activation_ssd_units=activation_ssd,
        )
        self.assertEqual(
            {tuple(p) for p in actual},
            {tuple(p) for p in expected},
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

    def _enumerate(
        self, with_plain_table=False, rows=4096, factor=None, optimizer=None
    ):
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
        if optimizer is not None:
            model.sparse.ebc.embedding_bags["table_de"].weight._optimizer_classes = [
                optimizer
            ]
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
        noncache = [o for o in options if not o.dynamicemb_options.caching]
        for opt, requested in ((noncache[0], 0), (noncache[1], 0.1), (noncache[-1], 1)):
            self.assertEqual(opt.tensor.shape[0], 8192)
            for rank, shard in enumerate(opt.shards):
                self.assertEqual(shard.size[0], 4096)
                if requested == 1:
                    self.assertEqual(shard.storage.ddr, 128 * 32 * 4)
                shard.rank = rank
            target = (
                planners.to_sharding_plan([opt], topology)
                .plan[opt.path][opt.name]
                .dynamicemb_options
            )
            self.assertEqual(
                (target.caching, target.max_capacity, target.local_hbm_for_values),
                (False, 4096, int(requested * 4096 * 32 * 4)),
            )

    def test_large_table_under_tight_hbm(self):
        options, _ = self._enumerate(rows=40_000_000)
        budget = Topology(
            world_size=2, compute_device="cuda", hbm_cap=520 << 20, ddr_cap=10 << 30
        )
        self.assertFalse(
            _dp_proposals([o for o in options if o.cache_load_factor >= 0.1], budget)
        )
        plan = _dp_proposals(options, budget)[0]
        self.assertEqual(
            [(o.dynamicemb_options.caching, o.cache_load_factor) for o in plan],
            [(False, 0.0)],
        )
        metadata_only = Topology(
            world_size=2, compute_device="cuda", hbm_cap=300 << 20, ddr_cap=10 << 30
        )
        self.assertFalse(_dp_proposals(options, metadata_only))

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

    def test_sharding_plan_sizes_training_and_inference_values(self):
        import inspect

        from dynamicemb.batched_dynamicemb_tables import BatchedDynamicEmbeddingTablesV2
        from torchrec.distributed.planner import planners

        from tzrec.optim.optimizer import FTRL

        # FTRL's knobs travel as plain kwargs of the dynamicemb table module.
        signature = inspect.signature(BatchedDynamicEmbeddingTablesV2.__init__)
        for name in ("learning_rate_power", "ftrl_beta", "l1_reg", "l2_reg"):
            self.assertIn(name, signature.parameters)

        # FTRL stores two state elements per embedding element; inference stores none.
        for optimizer, row_bytes in ((FTRL, 3 * 32 * 4), (None, 32 * 4)):
            with self.subTest(optimizer=optimizer):
                options, topology = self._enumerate(optimizer=optimizer)
                cache = next(
                    o
                    for o in options
                    if o.dynamicemb_options.caching and o.cache_load_factor == 0.5
                )
                hybrid = next(
                    o
                    for o in options
                    if not o.dynamicemb_options.caching and 0 < o.cache_load_factor < 1
                )
                self.assertEqual(
                    [s.storage.ddr for s in hybrid.shards], [1920 * row_bytes] * 2
                )
                for rank, shard in enumerate(cache.shards):
                    shard.rank = rank
                target = (
                    planners.to_sharding_plan([cache], topology)
                    .plan[cache.path][cache.name]
                    .dynamicemb_options
                )
                self.assertEqual(target.training, optimizer is not None)
                self.assertTrue(target.caching)
                # The cache covers half the shard on a 10% value budget.
                self.assertEqual(
                    target.local_hbm_for_values, int(0.1 * 2048 * row_bytes)
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
class DynamicEmbPlannerTest(unittest.TestCase):
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
            ("measured_zero", 0, 0.1, 0.1, 3),
            ("small_working_set", 0.1, 64 / 4096, 0.1, 3),
            ("half", 3.2, 0.1, 0.5, 2),
            ("above_half", 4.8, 0.1, 0.75, 1),
            ("full", 6.4, 0.1, None, 0),
        ],
        name_func=parameterized_name_func,
    )
    def test_auto_grid(self, _, factor, hybrid_floor, cache_floor, ncache):
        opts = self._options(factor)
        noncache = [o for o in opts if not o.dynamicemb_options.caching]
        cache = [o for o in opts if o.dynamicemb_options.caching]
        full_bytes = 4096 * 4
        grid = [0, *np.geomspace(hybrid_floor, 0.5, 5), 0.6, 0.7, 0.8, 0.9, 1]
        self.assertEqual(
            [o.dynamicemb_storage.local_hbm_for_values for o in noncache],
            [int(r * full_bytes) for r in grid],
        )
        self.assertEqual(len(cache), ncache)
        if cache:
            self.assertEqual(
                cache[0].dynamicemb_storage.local_hbm_for_values,
                int(cache_floor * full_bytes),
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
            ("zero", 0, False, 4096, False, 0),
            ("full", 1, False, 4096, False, 1),
            ("hybrid", 0.1, False, 4096, False, 0.11 + 0.01 * 1024 / (1024 + 3696)),
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
        allocated = opt.cache_load_factor
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
        self.assertEqual(opt.cache_load_factor, allocated)


GIB = 1 << 30


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
        constraint = build_dynamicemb_constraints(
            feature_pb2.DynamicEmbedding(max_capacity=rows, **fields), config
        )
        if pooling:
            constraint.pooling_factors = list(pooling)
        constraints[f"sparse.ebc.{name}"] = constraint
    return TestSparseNN(tables=configs, sparse_device=torch.device("meta")), constraints


@unittest.skipUnless(has_dynamicemb, "dynamicemb is required")
@mark_ci_scope("gpu")
class RocksDBPlannerTest(unittest.TestCase):
    """One combined search places dynamicemb tables in memory or RocksDB."""

    # 2^24 x 64 fp32 with SGD: 4 GiB of values per table.
    BIG = (1 << 24, 64, {})

    def setUp(self):
        try:
            import tzrec_ext.rocksdb  # noqa: F401
        except ImportError:
            self.skipTest("tzrec_ext.rocksdb is not installed")

    def _planner(self, tables, hbm_gib, ddr_gib, ssd_gib=None, ssd_mem_bw=None):
        """Build the unified planner over a synthetic topology.

        Mirrors ``create_planner``'s RocksDB path -- one memoized
        enumeration, the activation estimate, the SSD-aware DP as the only
        proposer, and the activation-aware partitioner -- but takes the
        HBM/DDR/SSD capacities directly so tests are deterministic.
        """
        model, constraints = _model_and_constraints(tables)
        topo_kwargs = {}
        if ssd_mem_bw is not None:
            topo_kwargs["ssd_mem_bw"] = ssd_mem_bw
        topology = Topology(
            world_size=1,
            compute_device="cuda",
            hbm_cap=int(hbm_gib * GIB),
            ddr_cap=int(ddr_gib * GIB),
            ssd_cap=0 if ssd_gib is None else int(ssd_gib * GIB),
            **topo_kwargs,
        )
        enumerator: Any = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
            allow_rocksdb=True,
        )
        sharders = plan_util.get_default_sharders()
        search_space = enumerator.enumerate(model, sharders)
        enumerator = plan_util._MemoizedSearchSpace(enumerator, search_space)
        activation = plan_util._ssd_activation_storage(search_space, 1)
        planner = EmbeddingShardingPlanner(
            topology=topology,
            batch_size=128,
            enumerator=enumerator,
            storage_reservation=HeuristicalStorageReservation(percentage=0.15),
            proposer=[
                DynamicProgrammingProposer(
                    hbm_bins_per_device=20,
                    ddr_bins_per_device=20,
                    ssd_bins_per_device=20,
                    ssd_activation=activation,
                )
            ],
            partitioner=plan_util._ActivationAwarePartitioner(
                GreedyPerfPartitioner(), activation
            ),
            callbacks=[
                partial(
                    plan_util._attach_backend_config,
                    ddr_caps=tuple(d.storage.ddr or 0 for d in topology.devices),
                    local_world_size=1,
                )
            ],
            debug=True,
        )
        return planner, model, search_space

    def _plan(self, tables, hbm_gib, ddr_gib, **kwargs):
        planner, model, _ = self._planner(tables, hbm_gib, ddr_gib, **kwargs)
        plan = planner.plan(model, plan_util.get_default_sharders())
        offloaded = {
            name: p.dynamicemb_options
            for m in plan.plan.values()
            for name, p in m.items()
            if rocksdb_util.is_rocksdb_options(p.dynamicemb_options)
        }
        return plan, offloaded, planner

    def _plan_with_ssd_room(self, tables, hbm_gib, ddr_gib, **kwargs):
        """Plan with an SSD budget that fits offloading every table fully.

        The backend's fixed activation grows with the machine's visible
        CPUs, so a hardcoded budget makes offload infeasible on large CI
        runners; derive the budget from the real activation and the worst
        per-table candidate cost instead.
        """
        _, _, space = self._planner(tables, hbm_gib, ddr_gib, **kwargs)
        activation = plan_util._ssd_activation_storage(space, 1)
        worst = {}
        for o in space:
            ssd = o.total_storage.ssd or 0
            if ssd:
                worst[o.fqn] = max(ssd, worst.get(o.fqn, 0))
        budget_gib = (activation.ssd + sum(worst.values())) / GIB
        return self._plan(tables, hbm_gib, ddr_gib, ssd_gib=budget_gib, **kwargs)

    @parameterized.expand(
        [
            param("rocksdb", input_lengths=[1.0]),
            param("rocksdb_lookup_floor", input_lengths=[5000.0]),
        ],
        name_func=parameterized_name_func,
    )
    def test_memory_table_layouts(self, _, input_lengths):
        # With RocksDB planning the built-in family keeps only the layouts
        # every restore preserves -- partial CACHING plus the full endpoint
        # with its empty host tier; no zero or partial HYBRID. The RocksDB
        # CACHING_PS candidates span one geometric grid whose lower
        # endpoint the working-set floor sets.
        rows = 1 << 24
        model, constraints = _model_and_constraints(
            {"a": (rows, 64, {}, input_lengths[0])}
        )
        topology = Topology(world_size=1, compute_device="cuda")
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
            allow_rocksdb=True,
        )
        options = enumerator.enumerate(model, plan_util.get_default_sharders())
        for o in options:
            if not rocksdb_util.is_rocksdb_options(o.dynamicemb_options):
                self.assertTrue(
                    o.dynamicemb_options.caching or o.cache_params.load_factor == 1.0,
                    "only CACHING and the full endpoint stay in memory",
                )
        ps = [
            o for o in options if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        ]
        required = math.ceil(128 * input_lengths[0])
        self.assertTrue(ps)
        for o in ps:
            # The working-set floor holds in the rows the runtime would
            # actually allocate, and each factory carries its table's own
            # namespace from construction.
            self.assertGreaterEqual(o.dynamicemb_storage.resident_rows, required)
            self.assertLess(o.dynamicemb_storage.resident_rows, rows)
            self.assertEqual(
                o.dynamicemb_options.external_storage.table_namespace,
                "sparse.ebc.a",
            )

    def test_all_in_memory(self):
        _, offloaded, planner = self._plan_with_ssd_room(
            {"a": self.BIG, "b": self.BIG}, 80, 64
        )
        self.assertEqual(offloaded, {})
        # The topology carries pre-activation capacity; nothing is deducted
        # from it for a memory-only plan.
        self.assertEqual(planner._topology.devices[0].storage.ddr, 64 * GIB)

    def test_faster_mixed_choice_beats_the_fitting_memory_plan(self):
        # The old absolute memory preference is gone: an all-memory plan
        # (small-ratio CACHING, host-backed misses) fits, but with SSD reads
        # priced faster than host transfers the mixed RocksDB plan has the
        # better estimated objective and must win.
        _, offloaded, _ = self._plan_with_ssd_room(
            {"a": self.BIG, "b": self.BIG},
            4.75,
            16,
            ssd_mem_bw=100 * 1024**3 / 1000,
        )
        self.assertTrue(offloaded, "the faster mixed plan must win")

    def test_offload_when_memory_is_short(self):
        tables = {"a": self.BIG, "b": self.BIG, "c": (1 << 20, 64, {})}
        _, offloaded, planner = self._plan_with_ssd_room(tables, 4, 4.75)
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
            # The fused group keeps each table's own budget: the runtime
            # budget must be exactly one planned grid ratio's requested
            # bytes (SGD carries no optimizer state).
            rows, required = o.max_capacity, 128
            budgets = {
                int(r * rows * o.dim * 4)
                for r in np.geomspace(required / rows, 0.5, num=10)
            }
            self.assertIn(o.local_hbm_for_values, budgets)
        # No temporary DDR deduction: the topology keeps its original caps.
        self.assertEqual(planner._topology.devices[0].storage.ddr, int(4.75 * GIB))

    def test_overrides_stay_in_memory(self):
        tables = {
            "b": (1 << 24, 64, {"cache_load_factor": 0.5}),
            "c": (1 << 24, 64, {"score_strategy": "CUSTOMIZED"}),
            "d": (1 << 24, 64, {}),
        }
        _, offloaded, _ = self._plan_with_ssd_room(tables, 4, 16)
        self.assertEqual(set(offloaded), {"d"})

    @parameterized.expand(
        [
            param("grid", factor=1024.0),
            param("equal_endpoints", factor=13107.2),
            param("floor_above_cap", factor=20000.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_cache_floor_with_configured_working_set(self, _, factor):
        # A configured unique_id_factor anchors the RocksDB grid's lower
        # endpoint at the working set over local rows. With 128 samples
        # and a five-times headroom: factor 1024 needs 655360 rows
        # (endpoint 5/128 of the table), factor 13107.2 needs exactly
        # half the table (equal endpoints, one candidate), and factor
        # 20000 needs more than half (no RocksDB candidates).
        rows = 1 << 24
        required = math.ceil(5 * factor * 128)
        model, constraints = _model_and_constraints({"a": (rows, 64, {})})
        constraints["sparse.ebc.a"].unique_id_factor = factor
        topology = Topology(world_size=1, compute_device="cuda")
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
            allow_rocksdb=True,
        )
        options = enumerator.enumerate(model, plan_util.get_default_sharders())
        ps = [
            o for o in options if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        ]
        if required > rows // 2:
            self.assertEqual(ps, [], "a floor above half the table offloads nothing")
            return
        # The grid maps to the requested byte budgets (SGD carries no
        # optimizer state, so one value row is dim * 4 bytes).
        expected = sorted(
            {int(r * rows * 64 * 4) for r in np.geomspace(required / rows, 0.5, num=10)}
        )
        self.assertEqual(
            sorted(o.dynamicemb_storage.local_hbm_for_values for o in ps), expected
        )
        if required * 2 == rows:
            self.assertEqual(
                expected, [int(0.5 * rows * 256)], "equal endpoints leave one candidate"
            )
        if factor == 1024.0:
            # The lower endpoint holds exactly the required capacity.
            self.assertEqual(ps[0].dynamicemb_storage.resident_rows, required)
        for o in ps:
            self.assertGreaterEqual(o.dynamicemb_storage.resident_rows, required)
            self.assertLess(o.dynamicemb_storage.resident_rows, rows)

    def test_shared_table_factor_takes_the_maximum(self):
        # Two features share table "a"; the working-set floor uses the
        # maximum configured factor, not a sum, and deduplication is the
        # user's measurement to configure.
        model, constraints = _model_and_constraints({"a": (1 << 24, 64, {})})
        constraints["sparse.ebc.a"].unique_id_factor = 3000.0
        topology = Topology(world_size=1, compute_device="cuda")
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
            allow_rocksdb=True,
        )
        options = enumerator.enumerate(model, plan_util.get_default_sharders())
        for o in options:
            if rocksdb_util.is_rocksdb_options(o.dynamicemb_options):
                self.assertGreaterEqual(
                    o.dynamicemb_storage.resident_rows, 5 * 3000 * 128
                )

    def test_rocksdb_misses_cost_disk_bandwidth(self):
        _, model, _ = self._planner({"a": self.BIG}, 4, 4, ssd_gib=100)
        topology = Topology(world_size=1, compute_device="cuda", ssd_mem_bw=104857.6)
        enumerator = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={
                "sparse.ebc.a": build_dynamicemb_constraints(
                    feature_pb2.DynamicEmbedding(max_capacity=1 << 24),
                    EmbeddingBagConfig(
                        num_embeddings=1 << 24,
                        embedding_dim=64,
                        name="a",
                        feature_names=["f_a"],
                    ),
                )
            },
            allow_rocksdb=True,
        )
        options = enumerator.enumerate(model, plan_util.get_default_sharders())
        perf = {}
        for o in options:
            rocksdb = rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            perf[(rocksdb, float(o.cache_load_factor))] = o.total_perf
        ps_ratios = sorted(r for rocksdb, r in perf if rocksdb)
        linear = sorted(r for rocksdb, r in perf if not rocksdb)
        self.assertTrue(ps_ratios)
        for ratio in ps_ratios:
            # The cheapest built-in CACHING choice with at least as large a
            # cache is still faster: its misses are fewer and served from
            # host memory, not the disk.
            faster_memory = min(b for b in linear if b >= ratio)
            self.assertGreater(perf[(True, ratio)], perf[(False, faster_memory)])

    def test_ssd_axis_picks_the_table_that_fits(self):
        # Offloading "a" or "b" covers the GPU deficit; "a" is cheaper in
        # the objective but only "b" fits the SSD budget, so the search
        # must select "b".
        tables = {
            "a": (1 << 25, 32, {}, 1.0),
            "b": (1 << 24, 64, {}, 100.0),
            "c": (1 << 20, 64, {}),
        }
        _, model, space = self._planner(tables, 1.76, 12, ssd_gib=100)
        ssd_cost = {}
        for fqn in ("sparse.ebc.a", "sparse.ebc.b"):
            costs = [
                o.total_storage.ssd
                for o in space
                if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
                and o.fqn == fqn
            ]
            ssd_cost[fqn] = min(costs, default=0)
        planner, _, _ = self._planner(tables, 1.76, 12, ssd_gib=100)
        activation = planner._partitioner._activation
        self.assertGreater(ssd_cost["sparse.ebc.a"], ssd_cost["sparse.ebc.b"])
        # A budget that admits "b" plus the shared activation but not "a"
        # plus it: only the "b" plan can pass the exact SSD check.
        fits_b = ssd_cost["sparse.ebc.b"] + activation.ssd
        fits_a = ssd_cost["sparse.ebc.a"] + activation.ssd
        ssd_gib = (fits_b + (fits_a - fits_b) / 2) / GIB
        _, offloaded, _ = self._plan(tables, 1.76, 12, ssd_gib=ssd_gib)
        self.assertEqual(set(offloaded), {"b"})

    def test_no_plan_fits(self):
        tables = {"a": (1 << 26, 64, {}), "b": (1 << 26, 64, {})}
        with self.assertRaises(PlannerError):
            self._plan(tables, 4, 4.75, ssd_gib=40)

    def test_infeasible_ssd_leaves_feasible_memory_plan(self):
        # Every RocksDB candidate needs more SSD than a 1 GiB budget
        # admits, so the search must fall back to the feasible all-memory
        # plan, and the failed attempts never mutate the shared topology.
        tables = {"a": self.BIG, "b": self.BIG}
        planner, model, _ = self._planner(
            tables, 80, 64, ssd_gib=1, ssd_mem_bw=100 * 1024**3 / 1000
        )
        before = (
            planner._topology.devices[0].storage.hbm,
            planner._topology.devices[0].storage.ddr,
            planner._topology.devices[0].storage.ssd,
        )
        plan = planner.plan(model, plan_util.get_default_sharders())
        offloaded = {
            name: p.dynamicemb_options
            for m in plan.plan.values()
            for name, p in m.items()
            if rocksdb_util.is_rocksdb_options(p.dynamicemb_options)
        }
        self.assertEqual(offloaded, {}, "the SSD plan must be infeasible")
        self.assertEqual(
            (
                planner._topology.devices[0].storage.hbm,
                planner._topology.devices[0].storage.ddr,
                planner._topology.devices[0].storage.ssd,
            ),
            before,
        )


if __name__ == "__main__":
    unittest.main()
