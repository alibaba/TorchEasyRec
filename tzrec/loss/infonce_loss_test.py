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

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tzrec.loss.infonce_loss import MaskedInfoNCELoss
from tzrec.utils import misc_util


class AllGatherWithGradTest(unittest.TestCase):
    def test_single_process_identity(self) -> None:
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        out = MaskedInfoNCELoss._all_gather_with_grad([a, b])
        self.assertIs(out[0], a)
        self.assertIs(out[1], b)


class MaskedInfoNCELossTest(unittest.TestCase):
    """Single-process tests for the masked CLIP loss."""

    def _features(self, B: int, D: int) -> dict:
        torch.manual_seed(0)
        return {
            "embed_a": torch.randn(B, D, requires_grad=True),
            "embed_b": torch.randn(B, D, requires_grad=True),
            "embed_a_ori": torch.randn(B, D),
            "embed_b_ori": torch.randn(B, D),
        }

    def test_forward_all_clip_finite(self) -> None:
        loss_fn = MaskedInfoNCELoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        out = loss_fn(feats, mask)
        self.assertIn("loss", out)
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertGreater(out["loss"].item(), 0.0)

    def test_all_recon_mask_zero_loss(self) -> None:
        loss_fn = MaskedInfoNCELoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)  # no clip rows
        out = loss_fn(feats, mask)
        # No clip rows -> masked average is exactly zero (and finite).
        self.assertTrue(torch.isfinite(out["loss"]))
        self.assertAlmostEqual(out["loss"].item(), 0.0, places=6)

    def test_all_recon_mask_finite_gradient(self) -> None:
        # Regression: with float("-inf") column fill an all-recon batch produced
        # a NaN gradient (0 * NaN) that survived the row mask. The finite fill
        # must keep the backward finite (and zero, since no clip row contributes).
        loss_fn = MaskedInfoNCELoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)
        loss_fn(feats, mask)["loss"].backward()
        grad = feats["embed_a"].grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertAlmostEqual(grad.abs().sum().item(), 0.0, places=6)

    def test_backward_flows_to_embeddings(self) -> None:
        loss_fn = MaskedInfoNCELoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        loss_fn(feats, mask)["loss"].backward()
        self.assertIsNotNone(feats["embed_a"].grad)
        self.assertTrue(torch.isfinite(feats["embed_a"].grad).all())

    def test_recon_columns_excluded_from_negatives(self) -> None:
        """A recon row's embedding must not affect a clip row's loss.

        Recon rows are dropped as queries (row mask) AND their columns are
        masked out of the negatives (col_mask). Perturbing the recon rows of
        EVERY column operand — ``embed_b`` (the self group) and both
        ``*_ori`` operands (the ori/cl groups) — must leave the clip rows' loss
        unchanged; a dropped or inverted ``col_mask`` on any group would fail.
        Distinct ``embed_a_ori`` / ``embed_b_ori`` so the ori/cl masking
        is actually exercised (not hidden by a shared tensor).
        """
        torch.manual_seed(0)
        B, D = 4, 8
        img = torch.randn(B, D)
        mask = torch.tensor([True, True, False, False])  # rows 2,3 are recon

        def feats(txt: torch.Tensor, txt_ori: torch.Tensor, img_ori: torch.Tensor):
            return {
                "embed_a": img,
                "embed_b": txt,
                "embed_a_ori": img_ori,
                "embed_b_ori": txt_ori,
            }

        txt, txt_ori, img_ori = (torch.randn(B, D) for _ in range(3))
        loss_fn = MaskedInfoNCELoss()
        loss_fn.eval()
        base = loss_fn(feats(txt, txt_ori, img_ori), mask)["loss"]
        # Perturb ONLY the recon rows of every column operand that feeds negatives.
        txt2, txt_ori2, img_ori2 = txt.clone(), txt_ori.clone(), img_ori.clone()
        for t in (txt2, txt_ori2, img_ori2):
            t[2:] = torch.randn(2, D)
        after = loss_fn(feats(txt2, txt_ori2, img_ori2), mask)["loss"]
        torch.testing.assert_close(base, after)

    def test_mask_holds_under_large_scale(self) -> None:
        # The column fill is finfo.min (below any real logit) rather than a
        # hardcoded -1e4, so masking holds even when the temperature is large and
        # the *_ori operands are un-normalized (real logits can dwarf 1e4). The
        # loss's internal clamp caps exp() at <= 100; loss/grad must stay finite.
        loss_fn = MaskedInfoNCELoss()
        with torch.no_grad():
            for p in (
                loss_fn.logit_scale,
                loss_fn.logit_scale_self,
                loss_fn.logit_scale_cl,
            ):
                p.fill_(3000.0)
        loss_fn.eval()
        feats = self._features(6, 8)
        feats["embed_a_ori"] = feats["embed_a_ori"] * 50
        feats["embed_b_ori"] = feats["embed_b_ori"] * 50
        mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        out = loss_fn(feats, mask)
        self.assertTrue(torch.isfinite(out["loss"]))
        loss_fn.train()
        feats["embed_a"].grad = None
        loss_fn(feats, mask)["loss"].backward()
        self.assertTrue(torch.isfinite(feats["embed_a"].grad).all())


# --- Multi-process tests for the CLIP distributed all-gather path. ---
# Validates ``_all_gather_with_grad`` (built on the differentiable
# ``torch.distributed.nn.functional.all_gather``) and ``MaskedInfoNCELoss`` across
# ranks. Uses NCCL on GPU when >=2 devices are available (the production path the
# reviewer cared about), else falls back to gloo/CPU, so it runs on a multi-GPU
# box and in CPU CI alike.

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
    gathered = MaskedInfoNCELoss._all_gather_with_grad([x])[0]

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
    feats = {
        "embed_a": torch.randn(B, D, device=device, requires_grad=True),
        "embed_b": torch.randn(B, D, device=device, requires_grad=True),
        "embed_a_ori": torch.randn(B, D, device=device),
        "embed_b_ori": torch.randn(B, D, device=device),
    }
    mask = torch.ones(B, dtype=torch.bool, device=device)

    loss_fn = MaskedInfoNCELoss().to(device)
    out = loss_fn(feats, mask)
    clip_loss = out["loss"]
    assert torch.isfinite(clip_loss).all(), f"rank{rank}: non-finite clip_loss"
    assert clip_loss.item() > 0.0, f"rank{rank}: clip_loss not positive"

    clip_loss.backward()
    g = feats["embed_a"].grad
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


class InfoNCEDistTest(unittest.TestCase):
    """2-rank tests for the CLIP distributed collectives."""

    def test_all_gather_with_grad(self) -> None:
        _run(_all_gather_worker)

    def test_masked_clip_loss(self) -> None:
        _run(_masked_clip_worker)


if __name__ == "__main__":
    unittest.main()
