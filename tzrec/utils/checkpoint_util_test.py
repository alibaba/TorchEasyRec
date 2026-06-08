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


import multiprocessing as mp
import os
import shutil
import tempfile
import unittest
from unittest import mock

import torch
import torch.distributed as dist
import torchrec
from parameterized import parameterized
from torch import nn
from torchrec import EmbeddingBagCollection
from torchrec.distributed.model_parallel import (
    DistributedModelParallel,
    get_default_sharders,
)
from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
from torchrec.optim import optimizers
from torchrec.optim.apply_optimizer_in_backward import apply_optimizer_in_backward
from torchrec.optim.keyed import CombinedOptimizer, KeyedOptimizerWrapper
from torchrec.optim.optimizers import in_backward_optimizer_filter
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from tzrec.constant import TRAIN_EVAL_RESULT_FILENAME
from tzrec.protos.export_pb2 import ExportConfig
from tzrec.utils import checkpoint_util, misc_util


def _create_test_model(large_table_cnt=2, small_table_cnt=2):
    large_tables = [
        torchrec.EmbeddingBagConfig(
            name="large_table_" + str(i),
            embedding_dim=64,
            num_embeddings=4096,
            feature_names=["large_table_feature_" + str(i)],
        )
        for i in range(large_table_cnt)
    ]
    small_tables = [
        torchrec.EmbeddingBagConfig(
            name="small_table_" + str(i),
            embedding_dim=64,
            num_embeddings=1024,
            feature_names=["small_table_feature_" + str(i)],
        )
        for i in range(small_table_cnt)
    ]

    class DebugModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ebc = EmbeddingBagCollection(
                tables=large_tables + small_tables, device="meta"
            )
            self.linear = nn.Linear(64 * (small_table_cnt + large_table_cnt), 1)

        def forward(self, kjt: KeyedJaggedTensor):
            emb = self.ebc(kjt)
            return torch.mean(self.linear(emb.values()))

    device: torch.device = torch.device("cpu")
    world_size = dist.get_world_size()

    model = DebugModel()
    apply_optimizer_in_backward(
        optimizers.Adagrad, model.ebc.parameters(), {"lr": 0.001}
    )

    topology = Topology(world_size=world_size, compute_device=device.type)
    planner = EmbeddingShardingPlanner(
        topology=topology,
    )
    plan = planner.collective_plan(
        model, get_default_sharders(), dist.GroupMember.WORLD
    )

    model = DistributedModelParallel(
        model,
        plan=plan,
        device=device,
    )
    dense_optimizer = KeyedOptimizerWrapper(
        dict(in_backward_optimizer_filter(model.named_parameters())),
        lambda params: torch.optim.Adam(params, lr=0.001),
    )
    optimizer = CombinedOptimizer([model.fused_optimizer, dense_optimizer])

    batch_size = 64
    kjt = KeyedJaggedTensor(
        keys=["large_table_feature_" + str(i) for i in range(large_table_cnt)]
        + ["small_table_feature_" + str(i) for i in range(small_table_cnt)],
        values=torch.cat(
            [
                torch.randint(0, 4095, (batch_size * large_table_cnt,), device=device),
                torch.randint(0, 1023, (batch_size * small_table_cnt,), device=device),
            ]
        ),
        lengths=torch.ones(
            batch_size * (small_table_cnt + large_table_cnt),
            dtype=torch.int32,
            device=device,
        ),
    ).to(device=device)
    losses = model.forward(kjt)
    torch.sum(losses, dim=0).backward()
    optimizer.step()
    return model, optimizer


def _save_restore_worker(test_dir, rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo")
    model, optimizer = _create_test_model()
    checkpoint_util.save_model(test_dir, model, optimizer)
    checkpoint_util.restore_model(test_dir, model, optimizer)


def _partial_restore_worker(test_dir, rank, world_size, port):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo")
    model, optimizer = _create_test_model()
    checkpoint_util.save_model(test_dir, model, optimizer)
    model, optimizer = _create_test_model(3, 3)
    checkpoint_util.restore_model(test_dir, model, optimizer)


def _remap_restore_worker(test_dir, rank, world_size, port, remap_file_path):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo")
    model1, optimizer = _create_test_model()
    checkpoint_util.save_model(test_dir, model1, optimizer)
    model2, optimizer = _create_test_model(3, 3)
    checkpoint_util.restore_model(test_dir, model2, optimizer, remap_file_path)
    shard_w_0_m1 = model1.state_dict()["ebc.embedding_bags.large_table_0.weight"]
    shard_w_0_m2 = model2.state_dict()["ebc.embedding_bags.large_table_0.weight"]
    shard_w_2_m2 = model2.state_dict()["ebc.embedding_bags.large_table_2.weight"]
    if rank == 0:
        w_0_m1 = torch.empty(shard_w_0_m1.size())
        w_0_m2 = torch.empty(shard_w_0_m2.size())
        w_2_m2 = torch.empty(shard_w_2_m2.size())
        shard_w_0_m1.gather(0, w_0_m1)
        shard_w_0_m2.gather(0, w_0_m2)
        shard_w_2_m2.gather(0, w_2_m2)
        torch.testing.assert_close(w_0_m1, w_0_m2)
        torch.testing.assert_close(w_0_m1, w_2_m2)
    else:
        shard_w_0_m1.gather(0)
        shard_w_0_m2.gather(0)
        shard_w_2_m2.gather(0)


class CheckpointUtilTest(unittest.TestCase):
    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_latest_checkpoint_with_model_dir(self):
        os.makedirs(os.path.join(self.test_dir, "model.ckpt-0"))
        os.makedirs(os.path.join(self.test_dir, "model.ckpt-10"))
        ckpt_path, step = checkpoint_util.latest_checkpoint(self.test_dir)
        self.assertEqual(ckpt_path, os.path.join(self.test_dir, "model.ckpt-10"))
        self.assertEqual(step, 10)
        ckpt_path, step = checkpoint_util.latest_checkpoint(
            os.path.join(self.test_dir, "model.ckpt-0")
        )
        self.assertEqual(ckpt_path, os.path.join(self.test_dir, "model.ckpt-0"))
        self.assertEqual(step, 0)

    def test_best_checkpoint_model_dir(self):
        os.makedirs(os.path.join(self.test_dir, "model.ckpt-0"))
        os.makedirs(os.path.join(self.test_dir, "model.ckpt-10"))
        os.makedirs(os.path.join(self.test_dir, "model.ckpt-20"))
        with open(os.path.join(self.test_dir, TRAIN_EVAL_RESULT_FILENAME), "w+") as f:
            f.write('{"global_step":0, "auc": 0.633, "grouped_auc": 0.565}\n')
            f.write('{"global_step":10, "auc": 0.632, "grouped_auc": 0.570}\n')
            f.write('{"global_step":20, "auc": 0.631, "grouped_auc": 0.567}\n')
        config = ExportConfig(exporter_type="best", best_exporter_metric="grouped_auc")
        ckpt_path, step = checkpoint_util.best_checkpoint(self.test_dir, config)
        self.assertEqual(ckpt_path, os.path.join(self.test_dir, "model.ckpt-10"))
        self.assertEqual(step, 10)
        config.metric_larger_is_better = False
        ckpt_path, step = checkpoint_util.best_checkpoint(self.test_dir, config)
        self.assertEqual(ckpt_path, os.path.join(self.test_dir, "model.ckpt-0"))
        self.assertEqual(step, 0)

    def test_latest_checkpoint_with_custom_dir(self):
        os.makedirs(os.path.join(self.test_dir, "custom/model"))
        ckpt_path, step = checkpoint_util.latest_checkpoint(
            os.path.join(self.test_dir, "custom")
        )
        self.assertEqual(ckpt_path, os.path.join(self.test_dir, "custom"))
        self.assertEqual(step, 0)

    def _remaining_ckpt_steps(self):
        return sorted(
            int(p.rsplit("-", 1)[1])
            for p in os.listdir(self.test_dir)
            if p.startswith("model.ckpt-")
        )

    @parameterized.expand(
        [
            # name, keep_checkpoint_max, protect_best, expected remaining steps
            ("keep_all", 0, False, [0, 10, 20, 30]),
            ("recent_2", 2, False, [20, 30]),
            ("protect_best", 2, True, [10, 20, 30]),
            ("keep_ge_count", 10, False, [0, 10, 20, 30]),
        ]
    )
    def test_checkpoint_manager_prune(
        self, _name, keep_checkpoint_max, protect_best, expected_steps
    ):
        for step in [0, 10, 20, 30]:
            os.makedirs(os.path.join(self.test_dir, f"model.ckpt-{step}"))
        export_config = None
        if protect_best:
            with open(
                os.path.join(self.test_dir, TRAIN_EVAL_RESULT_FILENAME), "w"
            ) as f:
                f.write('{"global_step":0, "auc":0.50}\n')
                f.write('{"global_step":10, "auc":0.99}\n')  # best, outside recent-2
                f.write('{"global_step":20, "auc":0.60}\n')
                f.write('{"global_step":30, "auc":0.70}\n')
            export_config = ExportConfig(
                exporter_type="best",
                best_exporter_metric="auc",
                metric_larger_is_better=True,
            )
        manager = checkpoint_util.CheckpointManager(
            self.test_dir,
            keep_checkpoint_max=keep_checkpoint_max,
            export_config=export_config,
        )
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            manager.prune()
            manager.close()  # drains the async worker -> deterministic
        self.assertEqual(self._remaining_ckpt_steps(), expected_steps)

    def test_checkpoint_manager_prune_non_rank_zero(self):
        for step in [0, 10, 20, 30]:
            os.makedirs(os.path.join(self.test_dir, f"model.ckpt-{step}"))
        manager = checkpoint_util.CheckpointManager(
            self.test_dir, keep_checkpoint_max=2
        )
        with mock.patch.dict(os.environ, {"RANK": "1"}):
            manager.prune()
            manager.close()
        self.assertEqual(self._remaining_ckpt_steps(), [0, 10, 20, 30])

    def test_checkpoint_manager_prune_idempotent(self):
        for step in [0, 10, 20, 30]:
            os.makedirs(os.path.join(self.test_dir, f"model.ckpt-{step}"))
        manager = checkpoint_util.CheckpointManager(
            self.test_dir, keep_checkpoint_max=2
        )
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            manager.prune()
            manager.prune()  # coalesced; must not double-delete or error
            manager.close()
        self.assertEqual(self._remaining_ckpt_steps(), [20, 30])

    def test_checkpoint_manager_finalizer_drains(self):
        # Simulate the exception path: prune() runs but close() is never reached.
        # The weakref.finalize safety net must drain the worker at interpreter exit.
        for step in [0, 10, 20, 30]:
            os.makedirs(os.path.join(self.test_dir, f"model.ckpt-{step}"))
        manager = checkpoint_util.CheckpointManager(
            self.test_dir, keep_checkpoint_max=2
        )
        with mock.patch.dict(os.environ, {"RANK": "0"}):
            manager.prune()
        self.assertTrue(manager._finalizer.alive)
        manager._finalizer()  # invoke directly to simulate at-exit
        self.assertEqual(self._remaining_ckpt_steps(), [20, 30])
        self.assertFalse(manager._prune_worker.is_alive())
        self.assertFalse(manager._finalizer.alive)

    def test_checkpoint_manager_discovery(self):
        for step in [0, 10, 20]:
            os.makedirs(os.path.join(self.test_dir, f"model.ckpt-{step}"))
        with open(os.path.join(self.test_dir, TRAIN_EVAL_RESULT_FILENAME), "w") as f:
            f.write('{"global_step":0, "auc":0.6}\n')
            f.write('{"global_step":10, "auc":0.7}\n')  # best
            f.write('{"global_step":20, "auc":0.5}\n')
        export_config = ExportConfig(
            exporter_type="best",
            best_exporter_metric="auc",
            metric_larger_is_better=True,
        )
        manager = checkpoint_util.CheckpointManager(
            self.test_dir, export_config=export_config
        )
        self.assertEqual(
            manager.latest_checkpoint(),
            checkpoint_util.latest_checkpoint(self.test_dir),
        )
        self.assertEqual(
            manager.best_checkpoint(),
            checkpoint_util.best_checkpoint(self.test_dir, export_config),
        )
        self.assertEqual(manager.best_checkpoint()[1], 10)

    def test_dist_save_restore_model(self):
        port = misc_util.get_free_port()
        procs = []
        ctx = mp.get_context("spawn")
        for i in range(2):
            p = ctx.Process(
                target=_save_restore_worker, args=(self.test_dir, i, 2, port)
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed.")
        param_names = checkpoint_util.list_distcp_param(self.test_dir)
        self.assertEqual(
            sorted(param_names),
            [
                "ebc.embedding_bags.large_table_0.weight",
                "ebc.embedding_bags.large_table_1.weight",
                "ebc.embedding_bags.small_table_0.weight",
                "ebc.embedding_bags.small_table_1.weight",
                "linear.bias",
                "linear.weight",
                "state.ebc.embedding_bags.large_table_0.weight.large_table_0.momentum1",
                "state.ebc.embedding_bags.large_table_1.weight.large_table_1.momentum1",
                "state.ebc.embedding_bags.small_table_0.weight.small_table_0.momentum1",
                "state.ebc.embedding_bags.small_table_1.weight.small_table_1.momentum1",
                "state.linear.bias.exp_avg",
                "state.linear.bias.exp_avg_sq",
                "state.linear.bias.step",
                "state.linear.weight.exp_avg",
                "state.linear.weight.exp_avg_sq",
                "state.linear.weight.step",
            ],
        )

    def test_dist_partial_restore_model(self):
        port = misc_util.get_free_port()
        procs = []
        ctx = mp.get_context("spawn")
        for i in range(2):
            p = ctx.Process(
                target=_partial_restore_worker, args=(self.test_dir, i, 2, port)
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed.")

    def test_dist_remap_restore_model(self):
        port = misc_util.get_free_port()
        procs = []
        ctx = mp.get_context("spawn")
        remap_file_path = os.path.join(self.test_dir, "mapping.txt")
        with open(remap_file_path, "w") as f:
            f.write(
                "ebc.embedding_bags.large_table_2.weight\tebc.embedding_bags.large_table_0.weight\n"
            )
            f.write(
                "ebc.embedding_bags.small_table_2.weight\tebc.embedding_bags.small_table_0.weight\n"
            )
        for i in range(2):
            p = ctx.Process(
                target=_remap_restore_worker,
                args=(self.test_dir, i, 2, port, remap_file_path),
            )
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed.")


class DataloaderCheckpointTest(unittest.TestCase):
    """Tests for dataloader checkpoint utilities."""

    def setUp(self):
        if not os.path.exists("./tmp"):
            os.makedirs("./tmp")
        self.test_dir = tempfile.mkdtemp(prefix="tzrec_", dir="./tmp")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_restore_dataloader_state(self):
        """Test saving and restoring dataloader state."""
        checkpoint_state = {
            "/data/test.parquet:0": 499,
            "/data/test.parquet:500": 999,
        }

        # Save
        os.environ["RANK"] = "0"
        checkpoint_util.save_dataloader_state(self.test_dir, checkpoint_state)

        # Verify file exists
        ckpt_path = os.path.join(
            self.test_dir, checkpoint_util.DATALOADER_CKPT_FILENAME
        )
        self.assertTrue(os.path.exists(ckpt_path))

        # Restore
        restored_state = checkpoint_util.restore_dataloader_state(self.test_dir)
        self.assertEqual(restored_state, checkpoint_state)

    def test_restore_dataloader_state_not_found(self):
        """Test restore returns None when no checkpoint exists."""
        restored_state = checkpoint_util.restore_dataloader_state(self.test_dir)
        self.assertIsNone(restored_state)

    def test_update_checkpoint_state(self):
        """Test merging checkpoint info by taking max per key."""
        checkpoint_state = {"path:0": 100, "path:500": 200}

        # Update with higher values
        checkpoint_info = {"path:0": 150, "path:500": 180}  # 150 > 100, 180 < 200
        checkpoint_util.update_dataloder_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state["path:0"], 150)  # Updated
        self.assertEqual(checkpoint_state["path:500"], 200)  # Not updated (200 > 180)

    def test_update_checkpoint_state_with_new_keys(self):
        """Test adding new keys to checkpoint state."""
        checkpoint_state = {"path:0": 100}

        checkpoint_info = {"path:500": 200}  # New key
        checkpoint_util.update_dataloder_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state["path:0"], 100)
        self.assertEqual(checkpoint_state["path:500"], 200)

    def test_update_checkpoint_state_with_none(self):
        """Test handling None checkpoint info."""
        checkpoint_state = {"path:0": 100}

        checkpoint_util.update_dataloder_state(checkpoint_state, None)

        # State should be unchanged
        self.assertEqual(checkpoint_state, {"path:0": 100})

    def test_update_checkpoint_state_empty(self):
        """Test updating empty checkpoint state."""
        checkpoint_state = {}

        checkpoint_info = {"path:0": 100, "path:500": 200}
        checkpoint_util.update_dataloder_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state, {"path:0": 100, "path:500": 200})

    def test_should_save_on_timestamp_first_batch(self):
        """No reference yet -> only initialize, never save."""
        self.assertFalse(
            checkpoint_util.should_save_on_timestamp(
                1000, None, interval_s=600, target_ts_list=[500]
            )
        )

    def test_should_save_on_timestamp_interval_crossed(self):
        """Crossing an epoch-aligned interval boundary fires."""
        # floor(3600/3600)=1 > floor(3599/3600)=0
        self.assertTrue(
            checkpoint_util.should_save_on_timestamp(
                3600, 3599, interval_s=3600, target_ts_list=[]
            )
        )

    def test_should_save_on_timestamp_interval_not_crossed(self):
        """Same interval bucket does not fire."""
        # floor(3500/3600) == floor(3000/3600) == 0
        self.assertFalse(
            checkpoint_util.should_save_on_timestamp(
                3500, 3000, interval_s=3600, target_ts_list=[]
            )
        )

    def test_should_save_on_timestamp_interval_disabled(self):
        """interval_s=0 disables the interval trigger."""
        self.assertFalse(
            checkpoint_util.should_save_on_timestamp(
                10000, 0, interval_s=0, target_ts_list=[]
            )
        )

    def test_should_save_on_timestamp_target_crossed(self):
        """Crossing an absolute target fires (boundary inclusive on the right)."""
        self.assertTrue(
            checkpoint_util.should_save_on_timestamp(
                1500, 1000, interval_s=0, target_ts_list=[1500]
            )
        )
        self.assertTrue(
            checkpoint_util.should_save_on_timestamp(
                1600, 1000, interval_s=0, target_ts_list=[1500]
            )
        )

    def test_should_save_on_timestamp_target_not_reached(self):
        """Target ahead of consumed event-time does not fire."""
        self.assertFalse(
            checkpoint_util.should_save_on_timestamp(
                1400, 1000, interval_s=0, target_ts_list=[1500]
            )
        )

    def test_should_save_on_timestamp_target_no_refire(self):
        """A target already behind the reference does not refire."""
        self.assertFalse(
            checkpoint_util.should_save_on_timestamp(
                2000, 1500, interval_s=0, target_ts_list=[1500]
            )
        )

    def test_should_save_on_timestamp_multiple_targets(self):
        """Any one crossed target in the window fires."""
        self.assertTrue(
            checkpoint_util.should_save_on_timestamp(
                1600, 1000, interval_s=0, target_ts_list=[900, 1500, 5000]
            )
        )

    def test_should_save_on_timestamp_interval_and_targets(self):
        """Targets fire even when the interval boundary is not crossed."""
        # same interval bucket (floor(1600/3600)==floor(1000/3600)==0) but a target
        # at 1500 is crossed.
        self.assertTrue(
            checkpoint_util.should_save_on_timestamp(
                1600, 1000, interval_s=3600, target_ts_list=[1500]
            )
        )

    def test_quorum_event_time_empty(self):
        self.assertIsNone(checkpoint_util.quorum_event_time([], 0.5))

    def test_quorum_event_time_single(self):
        self.assertEqual(checkpoint_util.quorum_event_time([42.0], 0.5), 42.0)

    def test_quorum_event_time_all_equal(self):
        self.assertEqual(checkpoint_util.quorum_event_time([5.0, 5.0, 5.0], 0.5), 5.0)

    def test_quorum_event_time_full_quorum_is_min(self):
        # quorum=1.0 -> all workers must be past T -> min
        self.assertEqual(
            checkpoint_util.quorum_event_time([10.0, 20.0, 30.0, 40.0], 1.0), 10.0
        )

    def test_quorum_event_time_tiny_quorum_is_max(self):
        # quorum near 0 -> any one worker -> max
        self.assertEqual(
            checkpoint_util.quorum_event_time([10.0, 20.0, 30.0, 40.0], 0.01), 40.0
        )

    def test_quorum_event_time_half_even(self):
        # m=4, k=ceil(0.5*4)=2 -> vals[4-2]=vals[2] = 2nd largest, half are >= it
        self.assertEqual(
            checkpoint_util.quorum_event_time([10.0, 20.0, 30.0, 40.0], 0.5), 30.0
        )

    def test_quorum_event_time_half_odd(self):
        # m=3, k=ceil(0.5*3)=ceil(1.5)=2 -> vals[3-2]=vals[1] = median
        self.assertEqual(
            checkpoint_util.quorum_event_time([10.0, 30.0, 20.0], 0.5), 20.0
        )

    def test_quorum_event_time_robust_to_outlier(self):
        # a single far-future garbage value does not move the median (quorum 0.5)
        self.assertEqual(
            checkpoint_util.quorum_event_time([100.0, 101.0, 1e18], 0.5), 101.0
        )

    def _policy_manager(self, **policy):
        """A CheckpointManager with save() mocked and a save policy applied."""
        tmp_dir = tempfile.mkdtemp(dir="./")
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        mgr = checkpoint_util.CheckpointManager(tmp_dir)
        defaults = dict(
            save_steps=0,
            save_epochs=0,
            ts_interval_s=0,
            ts_targets=[],
            ts_quorum=0.5,
        )
        defaults.update(policy)
        mgr.set_save_policy(**defaults)
        mgr.save = mock.MagicMock(return_value="ckpt")
        return mgr

    def test_maybe_save_step_interval(self):
        mgr = self._policy_manager(save_steps=10)
        self.assertFalse(mgr.maybe_save(5, model=None))
        self.assertTrue(mgr.maybe_save(10, model=None))
        self.assertEqual(mgr.save.call_count, 1)

    def test_maybe_save_epoch_interval(self):
        mgr = self._policy_manager(save_epochs=2)
        self.assertFalse(mgr.maybe_save(100, model=None, epoch=0))
        self.assertTrue(mgr.maybe_save(100, model=None, epoch=1))

    def test_maybe_save_dedupe_epoch_after_same_step(self):
        # regression: an epoch save right after a same-step save must not re-save
        mgr = self._policy_manager(save_steps=10, save_epochs=1)
        self.assertTrue(mgr.maybe_save(10, model=None))  # step trigger
        self.assertFalse(mgr.maybe_save(10, model=None, epoch=0))  # same step -> no-op
        self.assertEqual(mgr.save.call_count, 1)

    def test_maybe_save_final_dedupe(self):
        mgr = self._policy_manager(save_steps=10)
        self.assertTrue(mgr.maybe_save(10, model=None))
        self.assertFalse(mgr.maybe_save(10, model=None, final=True))  # already saved
        self.assertTrue(mgr.maybe_save(11, model=None, final=True))  # new step

    def test_maybe_save_timestamp_init_then_fire(self):
        mgr = self._policy_manager(ts_interval_s=3600)
        # first observed event-time only initializes the reference, no save
        self.assertFalse(mgr.maybe_save(1, model=None, worker_ts_list=[3599.0]))
        # crossing the next Unix-epoch-aligned boundary fires
        self.assertTrue(mgr.maybe_save(2, model=None, worker_ts_list=[3600.0]))

    def test_maybe_save_timestamp_quorum(self):
        # quorum=1.0: only fires once the slowest worker crosses
        mgr = self._policy_manager(ts_interval_s=3600, ts_quorum=1.0)
        self.assertFalse(mgr.maybe_save(1, model=None, worker_ts_list=[3000.0, 3000.0]))
        # one worker crossed, the other not -> min still in old bucket -> no save
        self.assertFalse(mgr.maybe_save(2, model=None, worker_ts_list=[3700.0, 3500.0]))
        # both crossed -> save
        self.assertTrue(mgr.maybe_save(3, model=None, worker_ts_list=[3700.0, 3601.0]))

    def test_maybe_save_stamps_watermark(self):
        mgr = self._policy_manager(ts_interval_s=3600)
        state = {}
        self.assertFalse(
            mgr.maybe_save(
                1, model=None, dataloader_state=state, worker_ts_list=[3599.0]
            )
        )
        self.assertTrue(
            mgr.maybe_save(
                2, model=None, dataloader_state=state, worker_ts_list=[3600.0]
            )
        )
        self.assertEqual(state[checkpoint_util.DATA_TS_WATERMARK], 3600.0)

    def test_restore_seeds_watermark(self):
        mgr = self._policy_manager(ts_interval_s=3600)
        tmp_dir = tempfile.mkdtemp(dir="./")
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        checkpoint_util.save_dataloader_state(
            tmp_dir, {"topic:0": 5, checkpoint_util.DATA_TS_WATERMARK: 7200.0}
        )
        state = mgr.restore_dataloader_state(tmp_dir)
        self.assertEqual(state[checkpoint_util.DATA_TS_WATERMARK], 7200.0)
        # seeded reference: a value in the same bucket does not re-fire
        self.assertFalse(mgr.maybe_save(1, model=None, worker_ts_list=[7201.0]))
        # crossing the next boundary fires
        self.assertTrue(mgr.maybe_save(2, model=None, worker_ts_list=[10800.0]))


if __name__ == "__main__":
    unittest.main()
