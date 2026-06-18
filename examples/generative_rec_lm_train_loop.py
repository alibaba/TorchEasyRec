"""Multi-step CPU training loop for ``GenerativeRecLM``.

Exercises ~hundreds of forward+backward+optimizer steps on synthetic batches
to verify the model learns. The data is a tiny next-SID-prediction task:
each row maps a sampled ``user_sequence`` of SIDs to a deterministic
``label`` SID list derived from it via a fixed permutation, so a sufficiently
expressive LM can drive the CE loss down toward 0.

Emits one ``[step …]`` line every ``--log-every`` steps so a Monitor wrapper
can stream loss progress. Failure modes (Traceback, NaN/Inf loss) also go to
stdout so the monitor sees them.

Usage:
    cd /workspace/fangtinglin/codework/feat/support_qwen/TorchEasyRec
    /opt/conda/bin/python -m examples.generative_rec_lm_train_loop \\
        --steps 500 --bsz 2 --user-len 6 --label-len 3 \\
        --log-every 5
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List

import torch

from tzrec.datasets.utils import Batch
from tzrec.models import generative_rec_lm  # noqa: F401  registers class
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


HF_MODEL_ID_DEFAULT = "/workspace/fangtinglin/_hf_stage/Qwen2.5-0.5B"


class _DummyJagged:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor):
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


def _make_proto(codebook: List[int], hf_model_id: str) -> ModelConfig:
    cfg = ModelConfig()
    grl = cfg.generative_rec_lm
    grl.class_name = "Qwen2RecLM"
    grl.hf_model_id = hf_model_id
    for c in codebook:
        grl.codebook.append(c)
    grl.user_sequence_feature_name = "user_sequence"
    grl.label_feature_name = "label"
    grl.ignore_index = -100
    return cfg


def _sample_batch(
    rng: torch.Generator,
    sum_codebook: int,
    bsz: int,
    user_len: int,
    label_len: int,
) -> Batch:
    """Synthesise a deterministic-mapping batch: ``label = (user mod K) + 1``
    for the first ``label_len`` positions. Easy to memorise; gives the LM
    something to drive CE down."""
    user_vals = torch.randint(
        1, sum_codebook + 1, (bsz, user_len), generator=rng, dtype=torch.long,
    )
    # Deterministic label: pick the first label_len user positions, modded
    # back into [1, sum_codebook]. Small enough to fit in CE in ~hundreds of
    # steps even on a frozen-most-of-the-net base.
    label_vals = ((user_vals[:, :label_len] % sum_codebook) + 1).contiguous()
    user_flat = user_vals.reshape(-1)
    label_flat = label_vals.reshape(-1)
    user_lens = torch.full((bsz,), user_len, dtype=torch.long)
    label_lens = torch.full((bsz,), label_len, dtype=torch.long)
    return Batch(sequence_dense_features={
        "user_sequence": _DummyJagged(user_flat, user_lens),
        "label":         _DummyJagged(label_flat, label_lens),
    })


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--user-len", type=int, default=6)
    ap.add_argument("--label-len", type=int, default=3)
    ap.add_argument("--log-every", type=int, default=5)
    # 1e-5 matches algr's `qwen2.5_05b_3layer_s1tiny.json` (see
    # [[project-tzrec-qwen2-integration]]).
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--hf-model-id", default=HF_MODEL_ID_DEFAULT)
    ap.add_argument(
        "--codebook", default="64,64",
        help="comma-separated codebook sizes; 64,64 fast / 32768,32768 algr-like",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(
        f"[init] torch={torch.__version__} device={device} "
        f"steps={args.steps} bsz={args.bsz} user_len={args.user_len} "
        f"label_len={args.label_len} lr={args.lr} hf={args.hf_model_id}",
        flush=True,
    )

    codebook = [int(x) for x in args.codebook.split(",") if x]
    sum_codebook = sum(codebook)
    cfg = _make_proto(codebook, args.hf_model_id)

    model_cls = BaseModel.create_class("GenerativeRecLM")
    t0 = time.time()
    model = model_cls(cfg, features=[], labels=[], sample_weights=None)
    model.train()
    if device == "cuda":
        model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[init] construct ok ({time.time()-t0:.1f}s) "
        f"base_vocab={model._base_vocab} "
        f"final_vocab={model.lm.config.vocab_size} "
        f"sum_codebook={sum_codebook}",
        flush=True,
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8,
    )

    rng = torch.Generator(device="cpu").manual_seed(0)
    # Warmup: discount the first 3 steps from throughput accounting (CUDA
    # kernel autotune, cudnn benchmark, allocator warm-up).
    warmup_steps = 3 if device == "cuda" else 0
    start = None
    losses: List[float] = []
    step_walls: List[float] = []
    for step in range(1, args.steps + 1):
        batch = _sample_batch(
            rng, sum_codebook, args.bsz, args.user_len, args.label_len,
        )
        if device == "cuda":
            # Move tensors to GPU — small batches, ok to do per-step.
            for k, v in batch.sequence_dense_features.items():
                v._values = v._values.cuda(non_blocking=True)
                v._lengths = v._lengths.cuda(non_blocking=True)
            torch.cuda.synchronize()
        t_step = time.perf_counter()
        pred = model.predict(batch)
        loss = pred["loss"]

        loss_val = loss.detach().item()
        if not math.isfinite(loss_val):
            print(f"[step {step}] FAIL: non-finite loss {loss_val}", flush=True)
            return 1

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t_step
        losses.append(loss.detach().item())

        if step == warmup_steps:
            start = time.time()
            step_walls = []
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        elif step > warmup_steps:
            step_walls.append(dt)

        if step % args.log_every == 0 or step == 1:
            mem_str = ""
            if device == "cuda":
                peak_gb = torch.cuda.max_memory_allocated() / 1024**3
                resv_gb = torch.cuda.max_memory_reserved() / 1024**3
                mem_str = f" peak_alloc={peak_gb:.2f}GB peak_reserved={resv_gb:.2f}GB"
            if start is not None and step_walls:
                avg_dt = sum(step_walls) / len(step_walls)
                it_s = 1.0 / avg_dt
                wall_str = f" avg_step={avg_dt*1000:.1f}ms it/s={it_s:.2f}"
            else:
                wall_str = " (warmup)"
            print(
                f"[step {step}/{args.steps}] ce_loss={loss_val:.4f} "
                f"avg_last_{min(20, len(losses))}={sum(losses[-20:])/min(20, len(losses)):.4f}"
                f"{wall_str}{mem_str}",
                flush=True,
            )

    # ---- final report ----
    start_avg = sum(losses[:5]) / min(5, len(losses))
    end_avg = sum(losses[-5:]) / min(5, len(losses))
    if step_walls:
        median_dt = sorted(step_walls)[len(step_walls) // 2]
        avg_dt = sum(step_walls) / len(step_walls)
        peak_str = ""
        if device == "cuda":
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            resv_gb = torch.cuda.max_memory_reserved() / 1024**3
            peak_str = (
                f" peak_alloc={peak_gb:.2f}GB peak_reserved={resv_gb:.2f}GB"
            )
        print(
            f"[done] steps={args.steps} bsz={args.bsz} "
            f"start_avg5={start_avg:.4f} end_avg5={end_avg:.4f} "
            f"delta={start_avg-end_avg:+.4f} "
            f"avg_step={avg_dt*1000:.1f}ms median_step={median_dt*1000:.1f}ms "
            f"it/s={1.0/avg_dt:.2f}{peak_str}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
