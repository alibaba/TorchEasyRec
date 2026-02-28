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

import torch
import torch.distributed as dist
import torchrec
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
        ckpt_path = os.path.join(self.test_dir, checkpoint_util.DATALOADER_CKPT_FILENAME)
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
        checkpoint_util.update_checkpoint_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state["path:0"], 150)  # Updated
        self.assertEqual(checkpoint_state["path:500"], 200)  # Not updated (200 > 180)

    def test_update_checkpoint_state_with_new_keys(self):
        """Test adding new keys to checkpoint state."""
        checkpoint_state = {"path:0": 100}

        checkpoint_info = {"path:500": 200}  # New key
        checkpoint_util.update_checkpoint_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state["path:0"], 100)
        self.assertEqual(checkpoint_state["path:500"], 200)

    def test_update_checkpoint_state_with_none(self):
        """Test handling None checkpoint info."""
        checkpoint_state = {"path:0": 100}

        checkpoint_util.update_checkpoint_state(checkpoint_state, None)

        # State should be unchanged
        self.assertEqual(checkpoint_state, {"path:0": 100})

    def test_update_checkpoint_state_empty(self):
        """Test updating empty checkpoint state."""
        checkpoint_state = {}

        checkpoint_info = {"path:0": 100, "path:500": 200}
        checkpoint_util.update_checkpoint_state(checkpoint_state, checkpoint_info)

        self.assertEqual(checkpoint_state, {"path:0": 100, "path:500": 200})


if __name__ == "__main__":
    unittest.main()
