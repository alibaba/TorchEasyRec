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

"""Multi-process test for ResidualVectorQuantizer FAISS kmeans-init.

Validates the DDP path of ``init_embed_``: the codebook is fit on rank 0
only and broadcast, so every rank ends with a bit-identical warm start.
(The previous behavior averaged permutation-misaligned per-rank centroids,
which the review flagged as a near-random init.) Uses NCCL on GPU when
>=2 devices are available, else gloo/CPU.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tzrec.modules.sid_generation.residual_vector_quantizer import (
    ResidualVectorQuantizer,
)
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


def _init_embed_worker(rank: int, world_size: int, port: int) -> None:
    device = _init(rank, world_size, port)
    # Rank-distinct data: a per-rank average/init would diverge; only a
    # broadcast-from-rank0 init yields identical codebooks.
    torch.manual_seed(rank)
    rvq = ResidualVectorQuantizer(
        embed_dim=8, n_layers=2, n_embed=16, kmeans_init=True
    ).to(device)
    rvq.train()
    rvq(torch.randn(512, 8, device=device))  # first forward triggers init_embed_
    assert bool(rvq.initted.item()), f"rank{rank}: not initialized"

    for layer in rvq.layers:
        w = layer.embedding.weight.detach().clone()
        wmin, wmax = w.clone(), w.clone()
        dist.all_reduce(wmin, op=dist.ReduceOp.MIN)
        dist.all_reduce(wmax, op=dist.ReduceOp.MAX)
        assert torch.allclose(wmin, wmax), (
            f"rank{rank}: codebook differs across ranks (init not broadcast)"
        )
    dist.destroy_process_group()


class ResidualVectorQuantizerDistTest(unittest.TestCase):
    """2-rank test for the FAISS kmeans-init broadcast."""

    def test_init_embed_broadcast(self) -> None:
        port = misc_util.get_free_port()
        ctx = mp.get_context("spawn")
        procs = []
        for rank in range(WORLD_SIZE):
            p = ctx.Process(target=_init_embed_worker, args=(rank, WORLD_SIZE, port))
            p.start()
            procs.append(p)
        for i, p in enumerate(procs):
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"worker-{i} failed (exitcode={p.exitcode}).")


if __name__ == "__main__":
    unittest.main()
