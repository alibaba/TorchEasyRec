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
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from parameterized import param, parameterized
from torch import nn
from torchrec.distributed.embedding_types import EmbeddingComputeKernel
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.estimator.types import HardwarePerfConfig
from torchrec.distributed.planner.types import (
    PlannerError,
    SharderData,
    Topology,
)
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType, StorageUsageType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2, train_pb2
from tzrec.utils import dynamicemb_util, plan_util, resource_util, rocksdb_util
from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints, has_dynamicemb
from tzrec.utils.plan_util import create_planner
from tzrec.utils.planner.enumerators import (
    EmbeddingEnumerator,
)
from tzrec.utils.planner.proposers_test import _dp_proposals
from tzrec.utils.test_util import (
    ci_resources,
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)


@ci_resources(slots=None)
class PlanUtilTest(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None

    @parameterized.expand(
        [("single_host", 2), ("two_hosts", 4)],
        name_func=parameterized_name_func,
    )
    @unittest.skipUnless(
        has_dynamicemb and torch.cuda.is_available(), "CUDA and dynamicemb are required"
    )
    @mark_ci_scope("gpu")
    def test_milp_plans_mixed_dynamic_and_static_tables(self, name, world_size):
        from torchrec.distributed.planner.types import ParameterConstraints

        tables = [
            EmbeddingBagConfig(
                num_embeddings=1024,
                embedding_dim=256,
                name=table,
                feature_names=[f"f_{table}"],
            )
            for table in ("dynamic", "static_tw", "static_cw")
        ]
        with mock.patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}):
            dynamic_constraint = build_dynamicemb_constraints(
                feature_pb2.DynamicEmbedding(max_capacity=1024), tables[0]
            )
        constraints = {
            "sparse.ebc.dynamic": dynamic_constraint,
            "sparse.ebc.static_tw": ParameterConstraints(
                sharding_types=["table_wise"], compute_kernels=["fused"]
            ),
            "sparse.ebc.static_cw": ParameterConstraints(
                sharding_types=["column_wise"],
                compute_kernels=["fused"],
                min_partition=128,
            ),
        }
        model = TestSparseNN(tables=tables, sparse_device=torch.device("meta"))
        with (
            mock.patch.dict(
                os.environ,
                {"TZREC_PLANNER": "milp", "TZREC_MILP_TIME_LIMIT_SECONDS": "12.5"},
            ),
            mock.patch.object(
                model, "parameter_constraints", return_value=constraints, create=True
            ),
            mock.patch.object(plan_util, "get_local_size", return_value=2),
            mock.patch.object(
                plan_util.dist, "get_world_size", return_value=world_size
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                return_value=SimpleNamespace(total_memory=1 << 30),
            ),
            mock.patch.object(
                resource_util, "host_memory_capacity_per_rank", return_value=1 << 30
            ),
        ):
            planner = create_planner(torch.device("cuda"), batch_size=16, model=model)
        for _ in range(2):
            plan = planner.plan(model, plan_util.get_default_sharders())
            table_plans = plan.plan["sparse.ebc"]
            self.assertEqual(set(table_plans), {t.name for t in tables})
            self.assertEqual(table_plans["dynamic"].ranks, list(range(world_size)))
            self.assertEqual(
                table_plans["dynamic"].dynamicemb_options.max_capacity,
                tables[0].num_embeddings // world_size,
            )
            self.assertEqual(table_plans["static_tw"].sharding_type, "table_wise")
            self.assertEqual(len(table_plans["static_cw"].ranks), 2)
            self.assertTrue(
                all(
                    0 <= rank < world_size
                    for table in table_plans.values()
                    for rank in table.ranks
                )
            )
            used = [0] * world_size
            for option in planner.get_selected_options():
                if option.name == "dynamic":
                    self.assertEqual(
                        table_plans["dynamic"].dynamicemb_options.local_hbm_for_values,
                        option.dynamicemb_storage.local_hbm_for_values,
                    )
                for shard in option.shards:
                    used[shard.rank] += shard.storage.hbm
            self.assertTrue(all(hbm <= (1 << 30) * 0.85 for hbm in used))

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


@ci_resources(slots=None)
@unittest.skipUnless(has_dynamicemb, "dynamicemb is not installed; skipping.")
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for dynamicemb.")
@mark_ci_scope("gpu")
class PlanUtilDynamicEmbE2ETest(unittest.TestCase):
    """End-to-end exercise of the dynamicemb planner integration."""

    def _enumerate(
        self,
        with_plain_table=False,
        rows=4096,
        factor=None,
        optimizer=None,
        world_size=2,
    ):
        from tzrec.utils.plan_util import get_default_sharders as dynamicemb_sharders

        table = EmbeddingBagConfig(
            num_embeddings=rows,
            embedding_dim=32,
            name="table_de",
            feature_names=["feat_de"],
        )
        with mock.patch.dict(os.environ, {"WORLD_SIZE": str(world_size)}):
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
        topology = Topology(world_size=world_size, compute_device="cuda")
        enumerator = EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints={"sparse.ebc.table_de": constraint},
        )
        return enumerator.enumerate(model, dynamicemb_sharders()), topology

    @parameterized.expand([(2,), (4,)], name_func=parameterized_name_func)
    def test_normalized_shards_and_converted_budgets(self, world_size):
        from torchrec.distributed.planner import planners

        options, topology = self._enumerate(
            rows=world_size * 4000 + 1, factor=3.2, world_size=world_size
        )
        # Local batch 128 requires 2048 rows, without another rank division.
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
            self.assertEqual(opt.tensor.shape[0], world_size * 4096)
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


GIB = 1 << 30


def _model_and_constraints(tables):
    """Map names to (rows, dim, DynamicEmbedding fields[, pooling]); None is static."""
    configs, constraints = [], {}
    for name, (rows, dim, fields, *pooling) in tables.items():
        config = EmbeddingBagConfig(
            num_embeddings=rows,
            embedding_dim=dim,
            name=name,
            feature_names=[f"f_{name}"],
        )
        configs.append(config)
        if fields is None:
            continue
        constraint = build_dynamicemb_constraints(
            feature_pb2.DynamicEmbedding(max_capacity=rows, **fields), config
        )
        if pooling:
            constraint.pooling_factors = list(pooling)
        constraints[f"sparse.ebc.{name}"] = constraint
    return TestSparseNN(tables=configs, sparse_device=torch.device("meta")), constraints


@ci_resources(slots=None)
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
        """Build the production planner with deterministic resource capacities."""
        model, constraints = _model_and_constraints(tables)
        topo_kwargs = {}
        if ssd_mem_bw is not None:
            topo_kwargs["ssd_mem_bw"] = ssd_mem_bw
        topology = Topology(
            world_size=1,
            compute_device="cuda",
            hbm_cap=int(hbm_gib * GIB),
            ddr_cap=int(ddr_gib * GIB),
            ssd_cap=(1 << 50) if ssd_gib is None else int(ssd_gib * GIB),
            **topo_kwargs,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with (
            mock.patch.object(
                model, "parameter_constraints", return_value=constraints, create=True
            ),
            mock.patch.object(plan_util, "Topology", return_value=topology),
            mock.patch.object(plan_util, "get_local_size", return_value=1),
            mock.patch.object(plan_util.dist, "get_world_size", return_value=1),
            mock.patch.object(
                resource_util,
                "host_memory_capacity_per_rank",
                return_value=int(ddr_gib * GIB),
            ),
            mock.patch.object(
                resource_util,
                "ssd_capacity_per_rank",
                return_value=topology.devices[0].storage.ssd,
            ),
            mock.patch.dict(
                os.environ,
                {
                    "STORAGE_RESERVE_PERCENT": "0.15",
                    "TZREC_PLANNER": "dp",
                },
            ),
        ):
            planner = create_planner(
                device,
                batch_size=128,
                model=model,
                rocksdb=mock.Mock(config=train_pb2.RocksDBConfig(root_dir="unused")),
            )
        return planner, model

    def _plan(self, tables, hbm_gib, ddr_gib, **kwargs):
        planner, model = self._planner(tables, hbm_gib, ddr_gib, **kwargs)
        plan = planner.plan(model, plan_util.get_default_sharders())
        offloaded = {
            name: p.dynamicemb_options
            for m in plan.plan.values()
            for name, p in m.items()
            if rocksdb_util.is_rocksdb_options(getattr(p, "dynamicemb_options", None))
        }
        return plan, offloaded, planner

    def _plan_with_ssd_room(self, tables, hbm_gib, ddr_gib, **kwargs):
        """Plan with an SSD budget that fits offloading every table fully.

        The backend's shared reserve grows with the machine's visible
        CPUs, so a hardcoded budget makes offload infeasible on large CI
        runners; derive the budget from the shared reserve and the worst
        per-table candidate cost instead.
        """
        planner, model = self._planner(tables, hbm_gib, ddr_gib, **kwargs)
        space = planner._enumerator.enumerate(model, plan_util.get_default_sharders())
        shared = rocksdb_util.shared_storage(1)
        worst = {}
        for o in space:
            ssd = o.total_storage.ssd or 0
            if ssd:
                worst[o.fqn] = max(ssd, worst.get(o.fqn, 0))
        budget_gib = (shared.ssd + sum(worst.values())) / GIB
        return self._plan(tables, hbm_gib, ddr_gib, ssd_gib=budget_gib, **kwargs)

    @parameterized.expand(
        [
            param("rocksdb", input_lengths=[1.0], rows=1 << 24, first_grid_index=0),
            param(
                "rocksdb_lookup_floor",
                input_lengths=[5000.0],
                rows=1 << 24,
                first_grid_index=4,
            ),
            param("rounded_cache", input_lengths=[1.0], rows=4096, first_grid_index=0),
        ],
        name_func=parameterized_name_func,
    )
    def test_memory_table_layouts(self, _, input_lengths, rows, first_grid_index):
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
        ordinary_options = plan_util.EmbeddingEnumerator(
            topology=topology,
            batch_size=128,
            fqn_constraints=constraints,
            allow_rocksdb=False,
        ).enumerate(model, plan_util.get_default_sharders())
        self.assertEqual(
            [
                (
                    o.dynamicemb_options.caching,
                    o.cache_load_factor,
                    o.dynamicemb_storage,
                )
                for o in options
                if not rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            ],
            [
                (
                    o.dynamicemb_options.caching,
                    o.cache_load_factor,
                    o.dynamicemb_storage,
                )
                for o in ordinary_options
            ],
        )
        ps = [
            o for o in options if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        ]
        required = math.ceil(128 * input_lengths[0])
        self.assertTrue(ps)
        expected_budgets = [
            int(r * rows * 64 * 4)
            for r in np.geomspace(0.01, 0.5, num=10)[first_grid_index:]
        ]
        self.assertEqual(
            [o.dynamicemb_storage.local_hbm_for_values for o in ps], expected_budgets
        )
        memory_ratios = {
            o.cache_load_factor
            for o in options
            if o.dynamicemb_options.caching
            and not rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
        }
        self.assertTrue(memory_ratios.intersection(o.cache_load_factor for o in ps))
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
        self.assertEqual(
            planner._topology.devices[0].storage.ddr,
            64 * GIB - rocksdb_util.shared_storage(1).ddr,
        )

    @parameterized.expand(
        [("memory", 80, 64, False), ("offload", 4, 4.75, True)],
        name_func=parameterized_name_func,
    )
    def test_mixed_static_dynamic_tables(self, _, hbm, ddr, offload):
        tables = {
            "dynamic_a": self.BIG,
            "dynamic_b": self.BIG,
            "static": (1024, 64, None),
        }
        plan, offloaded, _ = self._plan_with_ssd_room(tables, hbm, ddr)
        self.assertEqual(set(plan.plan["sparse.ebc"]), set(tables))
        self.assertNotEqual(
            plan.plan["sparse.ebc"]["static"].compute_kernel,
            EmbeddingComputeKernel.CUSTOMIZED_KERNEL.value,
        )
        self.assertEqual(
            set(offloaded), {"dynamic_a", "dynamic_b"} if offload else set()
        )

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
            rows = o.max_capacity
            budgets = {
                int(r * rows * o.dim * 4) for r in np.geomspace(0.01, 0.5, num=10)
            }
            self.assertIn(o.local_hbm_for_values, budgets)
        self.assertEqual(
            planner._topology.devices[0].storage.ddr,
            int(4.75 * GIB) - rocksdb_util.shared_storage(1).ddr,
        )

    def test_rocksdb_preserves_explicit_cache_budget(self):
        rows, dim = 1 << 24, 64
        tables = {
            "pinned": (rows, dim, {"cache_load_factor": 0.8}),
            "custom": (1 << 16, dim, {"score_strategy": "CUSTOMIZED"}),
        }
        _, offloaded, planner = self._plan_with_ssd_room(tables, 8, 4.75)
        expected = int(float(np.float32(0.8)) * rows * dim * 4)
        self.assertEqual(
            [
                (o.name, o.dynamicemb_storage.local_hbm_for_values)
                for o in planner.get_search_space()
                if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            ],
            [("pinned", expected)],
        )
        self.assertEqual(set(offloaded), {"pinned"})
        self.assertEqual(offloaded["pinned"].local_hbm_for_values, expected)

    def test_explicit_rocksdb_cache_boundaries(self):
        rows = 1 << 16
        tables = {
            "small": (rows, 64, {"cache_load_factor": 0.005}),
            "below_batch": (rows, 64, {"cache_load_factor": 0.005}, 32),
            "rounds_to_full": (1024, 64, {"cache_load_factor": 0.8}),
            "host": (rows, 64, {"cache_load_factor": 0}),
            "hbm": (rows, 64, {"cache_load_factor": 1}),
        }
        planner, model = self._planner(tables, 8, 16)
        space = planner._enumerator.enumerate(model, plan_util.get_default_sharders())
        self.assertEqual(
            [
                (o.name, o.dynamicemb_storage.local_hbm_for_values)
                for o in space
                if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            ],
            [("small", int(float(np.float32(0.005)) * rows * 64 * 4))],
        )

    @parameterized.expand(
        [
            param("zero", factor=0.0),
            param("below_one_percent", factor=100.0),
            param("grid", factor=1024.0),
            param("equal_endpoints", factor=13107.2),
            param("floor_above_cap", factor=20000.0),
        ],
        name_func=parameterized_name_func,
    )
    def test_cache_floor_with_configured_working_set(self, _, factor):
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
        floor = max(required, 1) / rows
        for caching in (False, True):
            partial = [
                o
                for o in options
                if not rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
                and o.dynamicemb_options.caching == caching
                and 0 < o.dynamicemb_storage.local_hbm_for_values < rows * 64 * 4
            ]
            self.assertTrue(partial)
            builtin_floor = max(floor, 0.1) if caching else min(floor, 0.1)
            self.assertEqual(
                partial[0].dynamicemb_storage.local_hbm_for_values,
                int(builtin_floor * rows * 64 * 4),
            )
        if required > rows // 2:
            self.assertEqual(ps, [], "a floor above half the table offloads nothing")
            return
        # The grid maps to the requested byte budgets (SGD carries no
        # optimizer state, so one value row is dim * 4 bytes).
        expected = sorted(
            {int(r * rows * 64 * 4) for r in np.geomspace(floor, 0.5, num=10)}
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
        model, _ = _model_and_constraints({"a": self.BIG})
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
        config = HardwarePerfConfig()
        sharder_data = SharderData(
            fused_params={},
            qcomm_dtype_sizes={},
            storage_usage_type=StorageUsageType.DEFAULT,
        )
        for o in options:
            rocksdb = rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
            if not rocksdb and not o.dynamicemb_options.caching:
                continue
            contexts = dynamicemb_util.ShardPerfContext.build_shard_perf_contexts(
                config, [s.size for s in o.shards], o, topology, {}, sharder_data
            )
            weight = 0.28 + 0.01 * o.cache_load_factor
            backing_bw = topology.ssd_mem_bw if rocksdb else topology.hbm_to_ddr_mem_bw
            for ctx in contexts:
                self.assertEqual(ctx.caching_ratio, o.cache_load_factor)
                self.assertEqual(ctx.hbm_to_ddr_mem_bw, topology.hbm_to_ddr_mem_bw)
                self.assertAlmostEqual(
                    ctx.device_bw,
                    (weight * topology.hbm_mem_bw + (1 - weight) * backing_bw) / 10,
                )
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

    def test_ssd_capacity_picks_the_table_that_fits(self):
        # Offloading "a" or "b" covers the GPU deficit; "a" is cheaper in
        # the objective but only "b" fits the SSD budget, so the search
        # must select "b".
        tables = {
            "a": (1 << 25, 32, {}, 1.0),
            "b": (1 << 24, 64, {}, 100.0),
            "c": (1 << 20, 64, {}),
        }
        # Construction only needs the model and the candidate space; the
        # capless default keeps the machine-scaled shared reserve in budget.
        planner, model = self._planner(tables, 1.76, 12, ssd_gib=None)
        space = planner._enumerator.enumerate(model, plan_util.get_default_sharders())
        ssd_cost = {}
        for fqn in ("sparse.ebc.a", "sparse.ebc.b"):
            costs = [
                o.total_storage.ssd
                for o in space
                if rocksdb_util.is_rocksdb_options(o.dynamicemb_options)
                and o.fqn == fqn
            ]
            ssd_cost[fqn] = min(costs, default=0)
        shared = rocksdb_util.shared_storage(1)
        self.assertGreater(ssd_cost["sparse.ebc.a"], ssd_cost["sparse.ebc.b"])
        # A budget that admits "b" plus the shared reserve but not "a"
        # plus it: only the "b" plan can pass the exact SSD check.
        fits_b = ssd_cost["sparse.ebc.b"] + shared.ssd
        fits_a = ssd_cost["sparse.ebc.a"] + shared.ssd
        ssd_gib = (fits_b + (fits_a - fits_b) / 2) / GIB
        _, offloaded, _ = self._plan(tables, 1.76, 12, ssd_gib=ssd_gib)
        self.assertEqual(set(offloaded), {"b"})

    def test_no_plan_fits(self):
        tables = {"a": (1 << 26, 64, {}), "b": (1 << 26, 64, {})}
        with self.assertRaises(PlannerError):
            self._plan(tables, 4, 4.75, ssd_gib=40)

    def test_infeasible_ssd_leaves_feasible_memory_plan(self):
        # Reject all SSD candidates without changing the input topology.
        tables = {"a": self.BIG, "b": self.BIG}
        planner, model = self._planner(
            tables,
            80,
            64,
            ssd_gib=rocksdb_util.shared_storage(1).ssd / GIB + 1,
            ssd_mem_bw=100 * 1024**3 / 1000,
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
            if rocksdb_util.is_rocksdb_options(getattr(p, "dynamicemb_options", None))
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
