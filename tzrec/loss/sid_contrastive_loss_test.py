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
from functools import partial

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from parameterized import parameterized
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel

from tzrec.loss.sid_contrastive_loss import SidContrastiveLoss
from tzrec.utils import misc_util
from tzrec.utils.test_util import mark_ci_scope, parameterized_name_func


def _reference_loss(loss_fn, embed_a, embed_b, embed_a_ori, embed_b_ori, pair_mask):
    """Compute CE after explicitly selecting each query's allowed candidates."""
    a, b = F.normalize(embed_a, dim=-1), F.normalize(embed_b, dim=-1)
    originals = [embed_a_ori.tolist(), embed_b_ori.tolist()]
    groups = (
        (loss_fn.logit_scale_self, (a, b), False),
        (loss_fn.logit_scale_ori, (embed_a_ori, embed_b_ori), False),
        (loss_fn.logit_scale_cl, (embed_a_ori, embed_b_ori), True),
    )
    loss = a.new_zeros(())
    for log_scale, targets, own_view in groups:
        scale = log_scale.clamp(max=4.605170185988092).exp()
        for side, query in enumerate((a, b)):
            target_side = side if own_view else 1 - side
            for row in range(len(pair_mask)):
                if not pair_mask[row]:
                    continue
                candidates = [row]
                for col in range(len(pair_mask)):
                    if col == row or not pair_mask[col]:
                        continue
                    candidate = originals[target_side][col]
                    if candidate in (originals[0][row], originals[1][row]):
                        continue
                    if not own_view and originals[side][col] == originals[side][row]:
                        continue
                    candidates.append(col)
                logits = scale * query[row] @ targets[target_side][candidates].t()
                loss = loss + F.cross_entropy(
                    logits.unsqueeze(0),
                    torch.zeros(1, dtype=torch.long, device=a.device),
                )
    return loss / (6 * pair_mask.sum().clamp(min=1))


class AllGatherWithGradTest(unittest.TestCase):
    def test_single_process_identity(self) -> None:
        a, b = torch.randn(3, 4), torch.randn(3, 4)
        out = SidContrastiveLoss._all_gather_with_grad([a, b])
        self.assertIs(out[0], a)
        self.assertIs(out[1], b)


class SidContrastiveLossTest(unittest.TestCase):
    """Single-process tests for the masked pair-contrastive loss."""

    def _features(self, B: int, D: int) -> dict:
        torch.manual_seed(0)
        return {
            "embed_a": torch.randn(B, D, requires_grad=True),
            "embed_b": torch.randn(B, D, requires_grad=True),
            "embed_a_ori": torch.randn(B, D),
            "embed_b_ori": torch.randn(B, D),
        }

    def test_forward_all_pairs_finite(self) -> None:
        loss_fn = SidContrastiveLoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        loss = loss_fn(**feats, pair_mask=mask)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)

    def test_all_recon_mask_zero_loss(self) -> None:
        loss_fn = SidContrastiveLoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)  # no pair rows
        loss = loss_fn(**feats, pair_mask=mask)
        # No pair rows -> masked average is exactly zero (and finite).
        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_all_recon_mask_finite_gradient(self) -> None:
        # Regression: with float("-inf") column fill an all-recon batch produced
        # a NaN gradient (0 * NaN) that survived the row mask. The finite fill
        # must keep the backward finite (and zero, since no pair row contributes).
        loss_fn = SidContrastiveLoss()
        feats = self._features(6, 8)
        mask = torch.zeros(6, dtype=torch.bool)
        loss_fn(**feats, pair_mask=mask).backward()
        grad = feats["embed_a"].grad
        self.assertIsNotNone(grad)
        self.assertTrue(torch.isfinite(grad).all())
        self.assertAlmostEqual(grad.abs().sum().item(), 0.0, places=6)

    def test_backward_flows_to_embeddings(self) -> None:
        loss_fn = SidContrastiveLoss()
        feats = self._features(6, 8)
        mask = torch.ones(6, dtype=torch.bool)
        loss_fn(**feats, pair_mask=mask).backward()
        self.assertIsNotNone(feats["embed_a"].grad)
        self.assertTrue(torch.isfinite(feats["embed_a"].grad).all())

    def test_recon_columns_excluded_from_negatives(self) -> None:
        """A recon row's embedding must not affect a pair row's loss.

        Recon rows are dropped as queries (row mask) AND their columns are
        masked out of the negatives (col_mask). Perturbing the recon rows of
        EVERY column operand — ``embed_b`` (the self group) and both
        ``*_ori`` operands (the ori/cl groups) — must leave the pair rows' loss
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
        loss_fn = SidContrastiveLoss()
        loss_fn.eval()
        base = loss_fn(**feats(txt, txt_ori, img_ori), pair_mask=mask)
        # Perturb ONLY the recon rows of every column operand that feeds negatives.
        txt2, txt_ori2, img_ori2 = txt.clone(), txt_ori.clone(), img_ori.clone()
        for t in (txt2, txt_ori2, img_ori2):
            t[2:] = torch.randn(2, D)
        after = loss_fn(**feats(txt2, txt_ori2, img_ori2), pair_mask=mask)
        torch.testing.assert_close(base, after)

    def test_autocast_bf16_does_not_overflow_masked_fill(self) -> None:
        # Regression: the column fill must come from the LOGITS dtype, not the
        # (fp32) embeddings. Under autocast the matmul emits bf16, so
        # finfo(fp32).min would raise "cannot be converted to BFloat16 without
        # overflow" in masked_fill. A mixed pair/non-pair mask exercises the fill.
        loss_fn = SidContrastiveLoss()
        feats = self._features(6, 8)
        mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = loss_fn(**feats, pair_mask=mask)
        self.assertTrue(torch.isfinite(loss))

    def test_mask_holds_under_large_scale(self) -> None:
        # The column fill is finfo.min (below any real logit) rather than a
        # hardcoded -1e4, so masking holds even when the temperature is large and
        # the *_ori operands are un-normalized (real logits can dwarf 1e4). The
        # loss's internal clamp caps exp() at <= 100; loss/grad must stay finite.
        loss_fn = SidContrastiveLoss()
        with torch.no_grad():
            for p in (
                loss_fn.logit_scale_ori,
                loss_fn.logit_scale_self,
                loss_fn.logit_scale_cl,
            ):
                p.fill_(3000.0)
        loss_fn.eval()
        feats = self._features(6, 8)
        feats["embed_a_ori"] = feats["embed_a_ori"] * 50
        feats["embed_b_ori"] = feats["embed_b_ori"] * 50
        mask = torch.tensor([1, 1, 1, 0, 0, 0], dtype=torch.bool)
        loss = loss_fn(**feats, pair_mask=mask)
        self.assertTrue(torch.isfinite(loss))
        loss_fn.train()
        feats["embed_a"].grad = None
        loss_fn(**feats, pair_mask=mask).backward()
        self.assertTrue(torch.isfinite(feats["embed_a"].grad).all())

    @parameterized.expand(
        [
            ("distinct", [0, 2, 4, 6], [1, 3, 5, 7]),
            ("duplicate_target", [0, 2, 4, 6], [1, 1, 5, 7]),
            ("duplicate_anchor", [0, 0, 4, 6], [1, 3, 5, 7]),
            ("query_as_candidate", [0, 2, 4, 6], [1, 3, 0, 7]),
            ("same_item_pair", [0, 2, 4, 6], [0, 3, 5, 7]),
            ("all_identical", [0, 0, 0, 0], [0, 0, 0, 0]),
        ],
        name_func=parameterized_name_func,
    )
    def test_duplicate_candidates_match_filtered_reference(self, _name, a_ids, b_ids):
        feats = self._features(4, 8)
        originals = torch.randn(8, 8)
        feats["embed_a_ori"] = originals[a_ids]
        feats["embed_b_ori"] = originals[b_ids]
        reference_feats = {
            key: value.detach().clone().requires_grad_(value.requires_grad)
            for key, value in feats.items()
        }
        mask = torch.ones(4, dtype=torch.bool)
        loss_fn, reference_fn = SidContrastiveLoss(), SidContrastiveLoss()
        actual = loss_fn(**feats, pair_mask=mask)
        expected = _reference_loss(reference_fn, **reference_feats, pair_mask=mask)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        for name in ("embed_a", "embed_b"):
            torch.testing.assert_close(feats[name].grad, reference_feats[name].grad)
        for actual_param, expected_param in zip(
            loss_fn.parameters(), reference_fn.parameters()
        ):
            torch.testing.assert_close(actual_param.grad, expected_param.grad)

    def test_exact_equality_uses_original_precision(self):
        feats = self._features(4, 8)
        feats["embed_b_ori"][1] = feats["embed_b_ori"][0]
        feats["embed_b_ori"][1, 0] += 1e-6
        self.assertFalse(torch.equal(feats["embed_b_ori"][0], feats["embed_b_ori"][1]))
        self.assertTrue(
            torch.equal(
                feats["embed_b_ori"][0].bfloat16(), feats["embed_b_ori"][1].bfloat16()
            )
        )
        mask = torch.ones(4, dtype=torch.bool)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            actual = SidContrastiveLoss()(**feats, pair_mask=mask)
            expected = _reference_loss(SidContrastiveLoss(), **feats, pair_mask=mask)
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.01)


# --- Multi-process tests for the contrastive distributed all-gather path. ---
# Validates ``_all_gather_with_grad`` (built on the differentiable
# ``torch.distributed.nn.functional.all_gather``) and ``SidContrastiveLoss`` across
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
    gathered = SidContrastiveLoss._all_gather_with_grad([x])[0]

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


def _contrastive_worker(rank, world_size, port, use_bf16) -> None:
    device = _init(rank, world_size, port)
    torch.manual_seed(42)
    originals = torch.randn(8, 8, device=device)
    full = {
        "embed_a": torch.randn(4, 8, device=device),
        "embed_b": torch.randn(4, 8, device=device),
        "embed_a_ori": originals[[0, 2, 0, 6]],
        "embed_b_ori": originals[[1, 1, 5, 0]],
    }
    selection = slice(2 * rank, 2 * rank + 2)
    local = {key: value[selection].clone() for key, value in full.items()}
    for key in ("embed_a", "embed_b"):
        local[key].requires_grad_()
    loss_fn = SidContrastiveLoss().to(device)
    model = DistributedDataParallel(
        loss_fn,
        device_ids=[rank] if device.type == "cuda" else None,
        static_graph=True,
    )
    for flags in (
        [True, True, True, True],
        [False, True, True, False],
        [False, False, True, True],
        [False, False, False, False],
    ):
        global_mask = torch.tensor(flags, device=device)
        local_mask = global_mask[selection]
        model.zero_grad()
        for key in ("embed_a", "embed_b"):
            local[key].grad = None
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
            actual = model(**local, pair_mask=local_mask) * local_mask.float().mean()
            with torch.no_grad():
                expected = _reference_loss(loss_fn, **full, pair_mask=global_mask)
                expected = expected * global_mask.float().mean()
        averaged = actual.detach().clone()
        dist.all_reduce(averaged)
        tolerance = {"atol": 0.04, "rtol": 0.04} if use_bf16 else {}
        torch.testing.assert_close(averaged / world_size, expected, **tolerance)
        actual.backward()
        for key in ("embed_a", "embed_b"):
            assert local[key].grad is not None and torch.isfinite(local[key].grad).all()
            torch.testing.assert_close(
                local[key].grad[~local_mask],
                torch.zeros_like(local[key].grad[~local_mask]),
            )
        for param in model.parameters():
            assert param.grad is not None and torch.isfinite(param.grad).all()
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


@mark_ci_scope("gpu")
class SidContrastiveDistTest(unittest.TestCase):
    """2-rank tests for the contrastive distributed collectives."""

    def test_all_gather_with_grad(self) -> None:
        _run(_all_gather_worker)

    @parameterized.expand(
        [("fp32", False), ("bf16", True)], name_func=parameterized_name_func
    )
    def test_masked_contrastive_loss(self, _name, use_bf16):
        _run(partial(_contrastive_worker, use_bf16=use_bf16))


if __name__ == "__main__":
    unittest.main()
