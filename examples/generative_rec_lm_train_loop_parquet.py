"""Parquet-fed multi-step training loop for ``GenerativeRecLM``.

Same instrumentation as ``generative_rec_lm_train_loop.py`` (GPU memory +
throughput) but consumes real ``user_sequence + label`` rows from a parquet
directory instead of synthesising them. Used for apples-to-apples
comparison with algr on the same data.

Usage on remote::

    /opt/conda/bin/python -m examples.generative_rec_lm_train_loop_parquet \\
        --parquet_dir /home/admin/workspace/aop_lab/data/AL-GR-Tiny/train_data_genreclm_100k \\
        --steps 25 --log-every 1 --bsz 10 --device cuda \\
        --codebook 32768,32768 \\
        --hf-model-id /home/admin/workspace/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
from typing import List

import pyarrow.parquet as pq
import torch

from tzrec.datasets.utils import Batch
from tzrec.models import generative_rec_lm  # noqa: F401  registers class
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


class _DummyJagged:
    def __init__(self, values: torch.Tensor, lengths: torch.Tensor):
        self._values = values
        self._lengths = lengths

    def values(self) -> torch.Tensor:
        return self._values

    def lengths(self) -> torch.Tensor:
        return self._lengths


class _ParquetIter:
    """Round-robin iterator over rows in a parquet directory.

    Loads each shard into memory as Arrow Tables (small for our 100K split).
    Yields ``(user_sids: list[int], label_sids: list[int])`` rows. Loops to
    the start when exhausted so the training loop never runs out of data.
    """
    def __init__(self, parquet_dir: str, max_source_len: int):
        paths = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
        assert paths, f"no parquet files under {parquet_dir!r}"
        self._rows: List = []
        for p in paths:
            t = pq.read_table(p)
            u_col = t["user_sequence"].to_pylist()
            l_col = t["label"].to_pylist()
            for u, lab in zip(u_col, l_col):
                # algr truncates user prompts to ``max_source_length`` tokens
                # via ``prompt_ids[:max_source_length]``. We mirror that at
                # the SID-list level (each SID is one token after splice).
                if max_source_len and len(u) > max_source_len:
                    u = u[-max_source_len:]  # keep most-recent items
                self._rows.append((u, lab))
        self._n = len(self._rows)
        self._i = 0

    def __len__(self) -> int:
        return self._n

    def next_batch(self, bsz: int) -> Batch:
        rows = []
        for _ in range(bsz):
            rows.append(self._rows[self._i])
            self._i = (self._i + 1) % self._n
        user_lens = torch.tensor([len(r[0]) for r in rows], dtype=torch.long)
        label_lens = torch.tensor([len(r[1]) for r in rows], dtype=torch.long)
        # Flatten to JaggedTensor (values, lengths) shape
        user_vals: List[int] = []
        for r in rows:
            user_vals.extend(r[0])
        label_vals: List[int] = []
        for r in rows:
            label_vals.extend(r[1])
        return Batch(sequence_dense_features={
            "user_sequence": _DummyJagged(
                torch.tensor(user_vals, dtype=torch.long), user_lens,
            ),
            "label": _DummyJagged(
                torch.tensor(label_vals, dtype=torch.long), label_lens,
            ),
        })


def _make_proto(
    codebook: List[int],
    hf_model_id: str,
    system_instruction: str = "",
    user_prefix_text: str = "",
    user_suffix_text: str = "",
) -> ModelConfig:
    cfg = ModelConfig()
    grl = cfg.generative_rec_lm
    grl.class_name = "Qwen2RecLM"
    grl.hf_model_id = hf_model_id
    for c in codebook:
        grl.codebook.append(c)
    grl.user_sequence_feature_name = "user_sequence"
    grl.label_feature_name = "label"
    grl.ignore_index = -100
    if system_instruction:
        grl.system_instruction = system_instruction
    if user_prefix_text:
        grl.user_prefix_text = user_prefix_text
    if user_suffix_text:
        grl.user_suffix_text = user_suffix_text
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_dir", required=True)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--bsz", type=int, default=10)
    ap.add_argument("--log-every", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-source-len", type=int, default=1020,
                    help="cap user_sequence SID count (mirrors algr max_source_length)")
    ap.add_argument("--codebook", default="32768,32768")
    ap.add_argument("--hf-model-id", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--system-instruction", default="",
                    help="CN/EN system prompt; matches algr's `default_instruction` / row.system")
    ap.add_argument("--user-prefix-text", default="",
                    help="text wrapped before SID list in user message (e.g. algr CN prefix)")
    ap.add_argument("--user-suffix-text", default="",
                    help="text wrapped after SID list in user message (e.g. algr CN suffix)")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    print(
        f"[init] torch={torch.__version__} device={device} steps={args.steps} "
        f"bsz={args.bsz} lr={args.lr} parquet={args.parquet_dir}",
        flush=True,
    )

    t_load = time.time()
    data = _ParquetIter(args.parquet_dir, args.max_source_len)
    print(f"[init] parquet loaded ({time.time()-t_load:.1f}s) rows={len(data)}", flush=True)

    codebook = [int(x) for x in args.codebook.split(",") if x]
    cfg = _make_proto(
        codebook, args.hf_model_id,
        system_instruction=args.system_instruction,
        user_prefix_text=args.user_prefix_text,
        user_suffix_text=args.user_suffix_text,
    )
    model_cls = BaseModel.create_class("GenerativeRecLM")
    t0 = time.time()
    model = model_cls(cfg, features=[], labels=[], sample_weights=None)
    model.train()
    if device == "cuda":
        model.to("cuda")
        torch.cuda.reset_peak_memory_stats()
    print(
        f"[init] construct ok ({time.time()-t0:.1f}s) base_vocab={model._base_vocab} "
        f"final_vocab={model.lm.config.vocab_size}",
        flush=True,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    warmup = 3 if device == "cuda" else 0
    start = None
    step_walls: List[float] = []
    losses: List[float] = []
    for step in range(1, args.steps + 1):
        batch = data.next_batch(args.bsz)
        if device == "cuda":
            for v in batch.sequence_dense_features.values():
                v._values = v._values.cuda(non_blocking=True)
                v._lengths = v._lengths.cuda(non_blocking=True)
            torch.cuda.synchronize()
        t = time.perf_counter()
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
        dt = time.perf_counter() - t
        losses.append(loss_val)
        if step == warmup:
            start = time.time()
            step_walls = []
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
        elif step > warmup:
            step_walls.append(dt)
        if step % args.log_every == 0 or step == 1:
            mem_str = ""
            if device == "cuda":
                pa = torch.cuda.max_memory_allocated() / 1024**3
                pr = torch.cuda.max_memory_reserved() / 1024**3
                mem_str = f" peak_alloc={pa:.2f}GB peak_reserved={pr:.2f}GB"
            wall_str = (
                f" avg_step={sum(step_walls)/len(step_walls)*1000:.1f}ms "
                f"it/s={1.0/(sum(step_walls)/len(step_walls)):.2f}"
                if step_walls else " (warmup)"
            )
            ph_t = batch.sequence_dense_features["user_sequence"]._values.shape[0]
            print(
                f"[step {step}/{args.steps}] ce_loss={loss_val:.4f} "
                f"avg_last_{min(20, len(losses))}={sum(losses[-20:])/min(20, len(losses)):.4f}"
                f" sum_lens={ph_t}{wall_str}{mem_str}",
                flush=True,
            )
    if step_walls:
        pa = (torch.cuda.max_memory_allocated() / 1024**3) if device == "cuda" else 0
        pr = (torch.cuda.max_memory_reserved() / 1024**3) if device == "cuda" else 0
        print(
            f"[done] steps={args.steps} bsz={args.bsz} "
            f"start_avg5={sum(losses[:5])/min(5, len(losses)):.4f} "
            f"end_avg5={sum(losses[-5:])/min(5, len(losses)):.4f} "
            f"avg_step={sum(step_walls)/len(step_walls)*1000:.1f}ms "
            f"median_step={sorted(step_walls)[len(step_walls)//2]*1000:.1f}ms "
            f"it/s={1.0/(sum(step_walls)/len(step_walls)):.2f}"
            f" peak_alloc={pa:.2f}GB peak_reserved={pr:.2f}GB",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
