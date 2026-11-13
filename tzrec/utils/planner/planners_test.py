# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from parameterized import parameterized
from scipy.optimize import LinearConstraint
from torch import nn
from torchrec.distributed.planner.enumerators import get_partition_by_type
from torchrec.distributed.planner.storage_reservations import (
    FixedPercentageStorageReservation,
)
from torchrec.distributed.planner.types import (
    Perf,
    PlannerError,
    PlannerErrorType,
    Shard,
    ShardingOption,
    Storage,
    Topology,
)

from tzrec.utils.planner import planners
from tzrec.utils.planner.planners import (
    MILPEmbeddingShardingPlanner,
    _solve_milp_placement,
)
from tzrec.utils.test_util import parameterized_name_func


def _option(name, kind, costs, perfs=None, dependency=None, kernel="fused"):
    return ShardingOption(
        name=name,
        tensor=torch.empty(16, 16, device="meta"),
        module=("ebc", nn.Module()),
        input_lengths=[1.0],
        batch_size=16,
        sharding_type=kind,
        partition_by=get_partition_by_type(kind),
        compute_kernel=kernel,
        shards=[
            Shard(
                size=[16, 16] if kind == "data_parallel" else [8, 16],
                offset=[0, 0] if kind == "data_parallel" else [i * 8, 0],
                storage=Storage(*cost),
                perf=Perf(perfs[i] if perfs else 1, 0, 0, 0),
            )
            for i, cost in enumerate(costs)
        ],
        is_pooled=True,
        dependency=dependency,
    )


def _topology(world=2, local=2, capacity=10):
    return Topology(
        world_size=world,
        local_world_size=local,
        compute_device="cuda",
        hbm_cap=capacity,
        ddr_cap=capacity,
    )


class MILPEmbeddingShardingPlannerTest(unittest.TestCase):
    def test_plan_respects_storage_reservation(self):
        options = [
            _option("a", "table_wise", [[9, 0]], [1]),
            _option("a", "table_wise", [[4, 0]], [3]),
        ]
        planner = MILPEmbeddingShardingPlanner(
            topology=_topology(),
            batch_size=16,
            enumerator=mock.Mock(enumerate=mock.Mock(return_value=options)),
            storage_reservation=FixedPercentageStorageReservation(0.5),
            stats=[],
        )
        plan = planner.plan(nn.Module(), [])
        self.assertEqual(len(plan.plan["ebc"]), 1)
        self.assertEqual(planner.get_selected_options()[0].total_storage.hbm, 4)

    @parameterized.expand(
        [("row_wise", "fused"), ("data_parallel", "dense")],
        name_func=parameterized_name_func,
    )
    def test_hbm_capacity_per_rank(self, kind, kernel):
        space = [_option(str(i), "table_wise", [[6, 0]]) for i in range(3)]
        with self.assertRaises(PlannerError) as cm:
            _solve_milp_placement(space, _topology(), 0)
        self.assertEqual(cm.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS)
        plan = _solve_milp_placement(
            [_option("uniform", kind, [[4, 0], [4, 0]], kernel=kernel)] + space[:2],
            _topology(),
            0,
        )
        self.assertEqual([s.rank for s in plan[0].shards], [0, 1])
        self.assertEqual({o.shards[0].rank for o in plan[1:]}, {0, 1})

    @parameterized.expand(
        [
            ("uneven", 18, True),
            ("exact_capacity", 20, True),
            ("exceeds_node", 21, False),
        ],
        name_func=parameterized_name_func,
    )
    def test_ddr_capacity_per_node(self, name, ddr, fits):
        topology = _topology(world=4, local=2)
        space = [_option("a", "table_wise", [[1, ddr]])]
        if fits:
            plan = _solve_milp_placement(space, topology, 0)
            shard = plan[0].shards[0]
            self.assertGreater(
                shard.storage.ddr, topology.devices[shard.rank].storage.ddr
            )
            node_start = shard.rank // 2 * 2
            self.assertLessEqual(
                shard.storage.ddr,
                sum(
                    d.storage.ddr for d in topology.devices[node_start : node_start + 2]
                ),
            )
        else:
            with self.assertRaises(PlannerError) as cm:
                _solve_milp_placement(space, topology, 0)
            self.assertEqual(
                cm.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS
            )

    def test_ddr_pool_does_not_span_a_pod(self):
        topology = Topology(
            world_size=4,
            local_world_size=2,
            pod_size=2,
            compute_device="cuda",
            hbm_cap=10,
            ddr_cap=10,
        )
        space = [_option("rw", "row_wise", [[1, 11], [1, 11], [1, 0], [1, 0]])]
        with self.assertRaises(PlannerError) as cm:
            _solve_milp_placement(space, topology, 0)
        self.assertEqual(cm.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS)

    @parameterized.expand(
        [
            ("independent", None, 6),
            ("dependent_fits", "tower", 4),
            ("dependent_exceeds_host", "tower", 6),
        ],
        name_func=parameterized_name_func,
    )
    def test_host_colocation(self, name, dependency, size):
        space = [
            _option("a", "table_row_wise", [[size, 0]] * 2, dependency=dependency),
            _option("b", "table_column_wise", [[size, 0]] * 2, dependency=dependency),
        ]
        topology = _topology(world=4)
        if dependency and size == 6:
            with self.assertRaises(PlannerError) as cm:
                _solve_milp_placement(space, topology, 0)
            self.assertEqual(
                cm.exception.error_type, PlannerErrorType.STRICT_CONSTRAINTS
            )
        else:
            plan = _solve_milp_placement(space, topology, 0)
            hosts = [{s.rank // 2 for s in o.shards} for o in plan]
            self.assertTrue(all(len(h) == 1 for h in hosts))
            if dependency:
                self.assertEqual(hosts[0], hosts[1])
            else:
                self.assertNotEqual(hosts[0], hosts[1])
            self.assertEqual([s.rank % 2 for s in plan[0].shards], [0, 1])

    @parameterized.expand(
        [("fused",), ("key_value",)], name_func=parameterized_name_func
    )
    def test_column_wise_rank_restrictions(self, kernel):
        topology = _topology()
        topology.devices[1].storage.hbm = 0
        space = [_option("cw", "column_wise", [[2, 0]] * 2, kernel=kernel)]
        if kernel == "key_value":
            with self.assertRaises(PlannerError):
                _solve_milp_placement(space, topology, 0)
        else:
            plan = _solve_milp_placement(space, topology, 0)
            self.assertEqual([s.rank for s in plan[0].shards], [0, 0])

    @parameterized.expand([(0,), (1,), (2,)], name_func=parameterized_name_func)
    def test_matches_exhaustive_selection_and_placement(self, seed):
        rng = np.random.default_rng(seed)
        groups = []
        for i in range(4):
            replica_cost = rng.integers(1, 5, 2).tolist()
            replica_perf = int(rng.integers(1, 6))
            groups.append(
                [
                    _option(
                        str(i),
                        "row_wise",
                        rng.integers(1, 5, (2, 2)).tolist(),
                        rng.integers(1, 6, 2).tolist(),
                    ),
                    _option(
                        str(i),
                        "table_wise",
                        rng.integers(3, 9, (1, 2)).tolist(),
                        rng.integers(2, 10, 1).tolist(),
                    ),
                    _option(
                        str(i),
                        "data_parallel",
                        [replica_cost] * 2,
                        [replica_perf] * 2,
                        kernel="dense",
                    ),
                ]
            )
        best = float("inf")
        for choices in itertools.product(*groups):
            movable = sum(o.sharding_type == "table_wise" for o in choices)
            for positions in itertools.product(range(2), repeat=movable):
                positions = iter(positions)
                used = np.zeros((2, 2), dtype=np.int64)
                perfs = [0.0, 0.0]
                for option in choices:
                    ranks = (
                        [0, 1]
                        if option.sharding_type in ("row_wise", "data_parallel")
                        else [next(positions)]
                    )
                    for rank, shard in zip(ranks, option.shards):
                        used[rank] += [
                            shard.storage.hbm,
                            shard.storage.ddr,
                        ]
                        perfs[rank] += shard.perf.total
                if np.all(used[:, 0] <= 14) and used[:, 1].sum() <= 28:
                    best = min(best, max(perfs))
        plan = _solve_milp_placement(
            [o for group in groups for o in group], _topology(capacity=14), 0
        )
        self.assertEqual(len({o.fqn for o in plan}), 4)
        used = np.zeros((2, 2), dtype=np.int64)
        perfs = [0.0, 0.0]
        for option in plan:
            for shard in option.shards:
                used[shard.rank] += [
                    shard.storage.hbm,
                    shard.storage.ddr,
                ]
                perfs[shard.rank] += shard.perf.total
        self.assertTrue(np.all(used[:, 0] <= 14))
        self.assertLessEqual(used[:, 1].sum(), 28)
        self.assertEqual(max(perfs), best)

    def test_timeout_without_solution_is_not_infeasibility(self):
        space = [_option("a", "table_wise", [[1, 0]])]
        with mock.patch.object(
            planners,
            "milp",
            return_value=SimpleNamespace(status=1, x=None),
        ):
            with self.assertRaisesRegex(
                PlannerError, "does not prove insufficient"
            ) as cm:
                _solve_milp_placement(space, _topology(), 0)
        self.assertEqual(cm.exception.error_type, PlannerErrorType.OTHER)

    def test_timeout_uses_feasible_incumbent(self):
        solve = planners.milp

        def timed_out(*args, **kwargs):
            result = solve(*args, **kwargs)
            result.status = 1
            result.mip_gap = 0.02
            return result

        space = [_option("a", "column_wise", [[6, 0]] * 2)]
        planner = MILPEmbeddingShardingPlanner(
            topology=_topology(),
            batch_size=16,
            enumerator=mock.Mock(enumerate=mock.Mock(return_value=space)),
            storage_reservation=FixedPercentageStorageReservation(0),
            time_limit=12.5,
            stats=[],
        )
        with mock.patch.object(planners, "milp", side_effect=timed_out):
            with self.assertLogs("tzrec", level="WARNING") as logs:
                plan = planner.plan(nn.Module(), [])
        self.assertEqual(set(plan.plan["ebc"]["a"].ranks), {0, 1})
        self.assertIn("12.5s solver limit", " ".join(logs.output))
        self.assertIn("gap=0.02", " ".join(logs.output))

    @parameterized.expand(
        [
            ("time_limit", 0.0),
            ("time_limit", float("nan")),
            ("relative_gap", -0.001),
            ("relative_gap", float("nan")),
        ],
        name_func=parameterized_name_func,
    )
    def test_invalid_solver_settings(self, setting, value):
        with self.assertRaisesRegex(ValueError, f"{setting} must be"):
            MILPEmbeddingShardingPlanner(
                topology=_topology(),
                batch_size=16,
                enumerator=mock.Mock(),
                storage_reservation=FixedPercentageStorageReservation(0),
                **{setting: value},
            )

    @parameterized.expand(
        [("hbm", 0, "rank"), ("ddr", 1, "node")],
        name_func=parameterized_name_func,
    )
    def test_rejects_timeout_plan_over_byte_capacity(self, name, axis, scope):
        capacity = 1 << 30
        cost = [0, 0]
        cost[axis] = capacity + 1
        space = [_option("a", "table_wise", [cost])]
        solve = planners.milp

        def rounded_timeout(*args, **kwargs):
            constraints = kwargs["constraints"]
            # Model solver feasibility tolerance without assuming variable indices.
            kwargs["constraints"] = LinearConstraint(
                constraints.A, constraints.lb, constraints.ub + 1e-6
            )
            result = solve(*args, **kwargs)
            self.assertEqual(result.status, 0)
            result.status = 1
            return result

        with mock.patch.object(planners, "milp", side_effect=rounded_timeout):
            with self.assertRaisesRegex(
                PlannerError, f"exact byte capacity on {scope} 0"
            ) as cm:
                _solve_milp_placement(
                    space, _topology(world=1, local=1, capacity=capacity), 0
                )
        self.assertEqual(cm.exception.error_type, PlannerErrorType.PARTITION)

    def test_grid_shard_is_not_silently_ignored(self):
        with self.assertRaisesRegex(NotImplementedError, "grid_shard"):
            _solve_milp_placement(
                [_option("a", "grid_shard", [[1, 0]] * 2)], _topology(), 0
            )


if __name__ == "__main__":
    unittest.main()
