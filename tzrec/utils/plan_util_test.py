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
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from parameterized import param, parameterized
from torch import nn
from torchrec.distributed.model_parallel import get_default_sharders
from torchrec.distributed.planner.types import Topology
from torchrec.distributed.test_utils.test_model import TestSparseNN
from torchrec.distributed.types import ShardingType
from torchrec.modules.embedding_configs import EmbeddingBagConfig

from tzrec.protos import feature_pb2
from tzrec.utils import plan_util, resource_util
from tzrec.utils.dynamicemb_util import build_dynamicemb_constraints, has_dynamicemb
from tzrec.utils.plan_util import create_planner
from tzrec.utils.planner.enumerators import EmbeddingEnumerator
from tzrec.utils.planner.proposers_test import _dp_proposals
from tzrec.utils.test_util import (
    make_test_dir,
    mark_ci_scope,
    parameterized_name_func,
)


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


if __name__ == "__main__":
    unittest.main()
