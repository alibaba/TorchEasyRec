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

"""Multi-process tests for the CLIP distributed all-gather path.

Validates ``_all_gather_with_grad`` (built on the differentiable
``torch.distributed.nn.functional.all_gather``) and ``MaskedCLIPLoss``
across ranks. Uses NCCL on GPU when >=2 devices are available (the
production path the reviewer cared about), else falls back to gloo/CPU,
so the test is runnable on a multi-GPU box and in CPU CI alike.
"""

import os
import unittest

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tzrec.loss.clip_loss import MaskedCLIPLoss
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


def _all_gather_worker(rank: int, world_size: int, port: int) -> None:
    device = _init(rank, world_size, port)
    # Each rank holds a distinct, rank-identifying tensor.
    x = torch.full((2, 3), float(rank + 1), device=device, requires_grad=True)
    gathered = MaskedCLIPLoss._all_gather_with_grad([x])[0]

    # Forward: gathered is (world_size*2, 3); rank r contributes rows
    # [2r : 2r+2] all equal to (r+1).
    assert gathered.shape == (world_size * 2, 3), gathered.shape
    for r in range(world_size):
        block = gathered[2 * r : 2 * r + 2]
        assert torch.allclose(block, torch.full_like(block, float(r + 1))), (
            f"rank{rank}: gathered block {r} wrong: {block}"
        )

    # Backward: identical scalar loss on every rank -> grad to every gathered
    # element is 1; the differentiable all_gather sum-reduces across ranks,
    # so the local input grad is world_size * ones.
    gathered.sum().backward()
    assert x.grad is not None, f"rank{rank}: no grad"
    assert torch.isfinite(x.grad).all(), f"rank{rank}: non-finite grad"
    expected = torch.full_like(x, float(world_size))
    assert torch.allclose(x.grad, expected), f"rank{rank}: grad {x.grad} != {expected}"
    dist.destroy_process_group()


def _masked_clip_worker(rank: int, world_size: int, port: int) -> None:
    device = _init(rank, world_size, port)
    torch.manual_seed(1234 + rank)
    B, D = 4, 8
    scale = torch.tensor(np.log(1 / 0.07)).exp().to(device)
    feats = {
        "image_embed": torch.randn(B, D, device=device, requires_grad=True),
        "text_embed": torch.randn(B, D, device=device, requires_grad=True),
        "image_embed_ori": torch.randn(B, D, device=device),
        "text_embed_ori": torch.randn(B, D, device=device),
        "logit_scale_self": scale,
        "logit_scale_cl": scale,
        "logit_scale": scale,
    }
    mask = torch.ones(B, dtype=torch.bool, device=device)

    loss_fn = MaskedCLIPLoss().to(device)
    out = loss_fn(feats, mask)
    clip_loss = out["clip_loss"]
    assert torch.isfinite(clip_loss).all(), f"rank{rank}: non-finite clip_loss"
    assert clip_loss.item() > 0.0, f"rank{rank}: clip_loss not positive"

    clip_loss.backward()
    g = feats["image_embed"].grad
    assert g is not None and torch.isfinite(g).all(), f"rank{rank}: bad grad"
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


class ClipLossDistTest(unittest.TestCase):
    """2-rank tests for the CLIP distributed collectives."""

    def test_all_gather_with_grad(self) -> None:
        _run(_all_gather_worker)

    def test_masked_clip_loss(self) -> None:
        _run(_masked_clip_worker)


if __name__ == "__main__":
    unittest.main()
