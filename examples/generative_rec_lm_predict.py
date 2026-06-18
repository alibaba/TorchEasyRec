"""Beam-search SID generation for an exported `GenerativeRecLM` checkpoint.

Mirrors algr's predict flow (config/qwen_predict_tiny_*.json: beam=50,
num_return=50, max_new_tokens=3, greedy) and emits `output_sids.jsonl`
lines in the exact shape `calc_hr_fast.py` consumes::

    {"_generated_text_": ["C{a}C{b}C{c}", ... x num_return], "answer": "item1;item2"}

The generated C-codes carry the dataset's layer offsets verbatim
(lv2 ∈ [8192, 16383], lv3 ∈ [16384, ...]) because the model is trained on
offset codes — decoding is just ``token_id - sid_base``.

Prompts are rebuilt with the SAME splice as training, minus the answer:
``system + user_prefix + SID atoms + user_suffix + asst_prefix``,
left-padded with eos. The SID base is read from the exported tokenizer
(``convert_tokens_to_ids("C0")``), so tokenizer-layout drift is impossible.

Run sharded across GPUs (one process per GPU, simple row interleave)::

    PYTHONPATH=. python -m examples.generative_rec_lm_predict \\
        --export_dir experiments/<run>/export_hf_40000 \\
        --test_parquet_dir /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data_genreclm \\
        --test_csv /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data/s1_tiny_test.csv \\
        --out logs/ter_predict_40000/output_sids.rank0.jsonl \\
        --rank 0 --world_size 8
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import List

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from tzrec.models.escalating_beam import escalating_beam_search

csv.field_size_limit(10 * 1024 * 1024)


def _enc(tok, text: str) -> List[int]:
    return tok.encode(text, add_special_tokens=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export_dir", required=True)
    ap.add_argument("--test_parquet_dir", required=True)
    ap.add_argument(
        "--test_csv",
        required=True,
        help="raw test CSV; supplies the ground-truth `answer` item-id strings",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world_size", type=int, default=1)
    ap.add_argument(
        "--bsz",
        type=int,
        default=4,
        help="algr per_device_eval_batch_size=4; 8 OOMs at beam 50",
    )
    ap.add_argument("--num_beams", type=int, default=50)
    ap.add_argument("--num_return", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=3)
    ap.add_argument(
        "--dynamic_beam",
        action="store_true",
        help="ALGR-style escalating beam (width doubles per level); "
        "returns num_beams * 2**max_new_tokens candidates, "
        "ignoring --num_return.",
    )
    ap.add_argument(
        "--codebook",
        type=int,
        default=8192,
        help="per-level SID codebook size (for --dynamic_beam band masking)",
    )
    ap.add_argument(
        "--max_rows", type=int, default=0, help="cap rows (0 = all); for smoke tests"
    )
    # algr CN prompt fragments (must match the training config)
    ap.add_argument(
        "--system-instruction",
        default=(
            "你是一个推荐系统，根据用户的历史行为，预测用户在电商场景的下一步行为。"
            "我会给你一串连续行为的语义编码，按照用户点击的时间顺序排列，每个行为用三个词表示。"
        ),
    )
    ap.add_argument("--user-prefix-text", default="当前用户的历史行为如下：")
    ap.add_argument(
        "--user-suffix-text", default="，请预测用户在电商推荐场景后续行为的语义编码"
    )
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.export_dir, use_fast=True)
    base = tok.convert_tokens_to_ids("C0")
    assert base is not None and base > 0, "exported tokenizer lacks C atoms"
    eos = tok.eos_token_id

    model = (
        AutoModelForCausalLM.from_pretrained(args.export_dir, torch_dtype="auto")
        .to(device)
        .eval()
    )
    print(
        f"[predict] model loaded vocab={model.config.vocab_size} sid_base={base} dev={device}",
        flush=True,
    )

    # Per-level token-space SID bands (level j atom in [base+j*cb, base+(j+1)*cb-1]).
    # Only needed by --dynamic_beam; the dataset offset codes are already disjoint.
    L, cb = args.max_new_tokens, args.codebook
    lo_tok = torch.tensor([base + j * cb for j in range(L)], device=device)
    hi_tok = torch.tensor([base + (j + 1) * cb - 1 for j in range(L)], device=device)
    if args.dynamic_beam:
        print(
            f"[predict] dynamic beam: widths "
            f"{[args.num_beams * 2 ** (j + 1) for j in range(L)]} "
            f"-> {args.num_beams * 2**L} candidates/row",
            flush=True,
        )

    # Cached template fragments — identical composition to training splice.
    tpl_system = _enc(tok, f"<|im_start|>system\n{args.system_instruction}<|im_end|>\n")
    tpl_user_prefix = _enc(tok, f"<|im_start|>user\n{args.user_prefix_text}")
    tpl_user_suffix = _enc(tok, f"{args.user_suffix_text}<|im_end|>\n")
    tpl_asst_prefix = _enc(tok, "<|im_start|>assistant\n")

    # Rows: user_sequence SIDs from parquet, answer strings from the CSV
    # (the converter wrote every CSV row, so ordering is 1:1).
    paths = sorted(glob.glob(os.path.join(args.test_parquet_dir, "*.parquet")))
    user_rows: List[List[int]] = []
    for p in paths:
        user_rows.extend(
            pq.read_table(p, columns=["user_sequence"])
            .column("user_sequence")
            .to_pylist()
        )
    answers: List[str] = []
    with open(args.test_csv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            answers.append((row.get("answer") or "").strip())
    assert len(user_rows) == len(answers), (len(user_rows), len(answers))

    idxs = list(range(len(user_rows)))[args.rank :: args.world_size]
    if args.max_rows:
        idxs = idxs[: args.max_rows]
    print(f"[predict] rank={args.rank}/{args.world_size} rows={len(idxs)}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    t0 = time.time()
    n_done = 0
    with open(args.out, "w", encoding="utf-8") as out_f:
        for bstart in range(0, len(idxs), args.bsz):
            bidx = idxs[bstart : bstart + args.bsz]
            prompts = []
            for i in bidx:
                u_tok = [s + base - 1 for s in user_rows[i]]
                prompts.append(
                    tpl_system
                    + tpl_user_prefix
                    + u_tok
                    + tpl_user_suffix
                    + tpl_asst_prefix
                )
            T = max(len(p) for p in prompts)
            input_ids = torch.full((len(prompts), T), eos, dtype=torch.long)
            attn = torch.zeros((len(prompts), T), dtype=torch.long)
            for r, p in enumerate(prompts):  # left pad
                input_ids[r, -len(p) :] = torch.tensor(p, dtype=torch.long)
                attn[r, -len(p) :] = 1
            input_ids, attn = input_ids.to(device), attn.to(device)

            with torch.no_grad():
                if args.dynamic_beam:
                    new_tokens = escalating_beam_search(
                        model,
                        input_ids,
                        attn,
                        num_beams=args.num_beams,
                        lo_tok=lo_tok,
                        hi_tok=hi_tok,
                    )  # (B * num_beams*2**L, L)
                else:
                    gen = model.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        num_return_sequences=args.num_return,
                        do_sample=False,
                        pad_token_id=eos,
                    )
                    new_tokens = gen[:, T:]  # (B * num_return, max_new_tokens)
            nret = new_tokens.shape[0] // len(bidx)
            new_tokens = new_tokens.view(len(bidx), nret, -1).tolist()

            for row_i, seqs in zip(bidx, new_tokens):
                texts = []
                for seq in seqs:
                    parts = []
                    for t in seq:
                        if t >= base:
                            parts.append(f"C{t - base}")
                        else:
                            parts.append(tok.decode([t], skip_special_tokens=True))
                    texts.append("".join(parts))
                out_f.write(
                    json.dumps(
                        {"_generated_text_": texts, "answer": answers[row_i]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            n_done += len(bidx)
            if (bstart // args.bsz) % 25 == 0:
                rate = n_done / max(time.time() - t0, 1e-6)
                print(
                    f"[predict] {n_done}/{len(idxs)} rows ({rate:.1f} rows/s)",
                    flush=True,
                )

    print(
        f"[predict] DONE rank={args.rank} rows={n_done} wall={time.time() - t0:.0f}s "
        f"out={args.out}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
