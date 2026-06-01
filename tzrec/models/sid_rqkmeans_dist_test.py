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

"""Multi-process tests for SidRqkmeans.on_train_end's DDP code path.

This exercises the collective sequence the single-process unit test
cannot reach: the cross-rank empty-buffer all_reduce, ``gather_object``
of the per-rank embedding buffers to rank 0, the FAISS fit, and the
``broadcast`` of centroids + ``_is_initialized`` fill on every rank.

Uses NCCL on GPU when >=2 devices are available (the production backend
the reviewer flagged for ``gather_object``), else gloo/CPU.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.protos import model_pb2
from tzrec.protos.models import sid_model_pb2
from tzrec.utils import misc_util

WORLD_SIZE = 2


def _init(rank: int, world_size: int, port: int) -> torch.device:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    use_cuda = torch.cuda.is_available() and torch.cuda.device_count() >= world_size
    if use_cuda:
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        return torch.device(f"cuda:{rank}")
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    return torch.device("cpu")


def _make_batch(batch_size: int, input_dim: int, device: torch.device) -> Batch:
    dense = KeyedTensor.from_tensor_list(
        keys=["item_emb"], tensors=[torch.randn(batch_size, input_dim, device=device)]
    )
    return Batch(
        dense_features={BASE_DATA_GROUP: dense}, sparse_features={}, labels={}
    )


def _create_model(input_dim: int, n_layers: int, k: int):
    from google.protobuf.struct_pb2 import Struct

    from tzrec.models.sid_rqkmeans import SidRqkmeans

    faiss_kwargs = Struct()
    faiss_kwargs.update({"niter": 5, "verbose": False, "seed": 1234})
    cfg = sid_model_pb2.SidRqkmeans(
        input_dim=input_dim,
        codebook=[k] * n_layers,
        normalize_residuals=False,
        faiss_kmeans_kwargs=faiss_kwargs,
        embedding_feature_name="item_emb",
    )
    model_config = model_pb2.ModelConfig(sid_rqkmeans=cfg)
    return SidRqkmeans(model_config=model_config, features=[], labels=[])


def _on_train_end_worker(rank: int, world_size: int, port: int) -> None:
    device = _init(rank, world_size, port)
    input_dim, n_layers, k = 16, 2, 16
    model = _create_model(input_dim, n_layers, k).to(device)
    model.train()

    torch.manual_seed(100 + rank)
    for _ in range(6):
        model.predict(_make_batch(32, input_dim, device))
    assert model._n_seen == 6 * 32, f"rank{rank}: reservoir not filled"

    # The collective sequence under test: empty-flag all_reduce ->
    # gather_object -> rank0 FAISS fit -> broadcast centroids + fill flag.
    model.on_train_end()

    # Every rank must end initialized with non-zero centroids.
    for layer in model._quantizer.layers:
        assert bool(layer._is_initialized.item()), f"rank{rank}: layer uninit"
        assert layer.centroids.abs().sum().item() > 0.0, f"rank{rank}: zero centroids"

    # Centroids were broadcast from rank0 -> must be bit-identical across
    # ranks (min == max under all_reduce).
    for layer in model._quantizer.layers:
        cmin = layer.centroids.clone()
        cmax = layer.centroids.clone()
        dist.all_reduce(cmin, op=dist.ReduceOp.MIN)
        dist.all_reduce(cmax, op=dist.ReduceOp.MAX)
        assert torch.allclose(cmin, cmax), f"rank{rank}: centroids differ across ranks"

    # After the fit, eval predict emits valid codes.
    model.eval()
    codes = model.predict(_make_batch(8, input_dim, device))["codes"]
    assert codes.shape == (8, n_layers), f"rank{rank}: bad codes shape {codes.shape}"
    assert (codes >= 0).all() and (codes < k).all(), f"rank{rank}: codes out of range"
    dist.destroy_process_group()


def _run(target) -> None:
    port = misc_util.get_free_port()
    ctx = mp.get_context("spawn")
    procs = []
    for rank in range(WORLD_SIZE):
        p = ctx.Process(target=target, args=(rank, WORLD_SIZE, port))
        p.start()
        procs.append(p)
    for i, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"worker-{i} failed (exitcode={p.exitcode}).")


class SidRqkmeansDistTest(unittest.TestCase):
    """2-rank test for SidRqkmeans.on_train_end."""

    def test_on_train_end_ddp(self) -> None:
        _run(_on_train_end_worker)


if __name__ == "__main__":
    unittest.main()
