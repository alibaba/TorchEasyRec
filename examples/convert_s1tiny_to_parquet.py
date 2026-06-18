"""Convert algr's ``s1_tiny.csv`` into TorchEasyRec-shaped parquet.

algr's row format::

    system,user,answer
    <CN prompt>,...C4805C8364C16402C4487C12277...,C1517C12109C16399

Each item is encoded as 3 contiguous ``C{i}`` codes (one per RQ-VAE layer).
The ``user`` column holds a CN sentence prefix + the user-history SID codes
+ a CN sentence suffix; the ``answer`` column holds just the target SID
codes for the next item.

This script extracts all ``C\\d+`` matches from each field, converts to
1-indexed SID integers (``Ck → SID = k + 1``), and writes a parquet shard
per N rows with two columns matching ``GenerativeRecLM``'s contract:

    user_sequence : list<int64>
    label         : list<int64>

The chat-template prompt strings are NOT carried over — ``GenerativeRecLM``
re-builds them from cached buffers at splice time, using its own
``system_instruction``. To preserve algr's bit-exact prompt for parity
testing, set ``GenerativeRecLM.system_instruction`` to algr's CN prefix in
the pipeline.config (proto field already exists; see §3 of the design).

Usage on remote::

    /opt/conda/bin/python -m examples.convert_s1tiny_to_parquet \\
        --csv /home/admin/workspace/aop_lab/data/AL-GR-Tiny/train_data/s1_tiny.csv \\
        --out_dir /home/admin/workspace/aop_lab/data/AL-GR-Tiny/train_data_genreclm \\
        --shard_size 200000 --max_rows 0 \\
        --log_every 200000
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

# Allow rows up to ~6 MB of CSV text (the long Chinese prompt + thousand SIDs).
csv.field_size_limit(10 * 1024 * 1024)

# Match every Cxxx in a string. SIDs are non-negative integers; we use a
# bounded ``+`` quantifier so it's anchored on whole tokens.
_SID_RE = re.compile(r"C(\d+)")


def _extract_sids(text: str) -> List[int]:
    """Pull every ``C\\d+`` out of `text` and return as 1-indexed SIDs.

    ``Ck → SID = k + 1`` so that SID range is [1, sum(codebook)] which is what
    ``GenerativeRecLM._splice_input_ids`` expects (SID=1 maps to atom C0 via
    ``token = sid + base - 1`` = ``base + (sid - 1)``).
    """
    return [int(m) + 1 for m in _SID_RE.findall(text)]


def _iter_rows(csv_path: str, max_rows: int) -> Iterator[Tuple[List[int], List[int]]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if max_rows and i >= max_rows:
                return
            user_sids = _extract_sids(row.get("user") or "")
            label_sids = _extract_sids(row.get("answer") or "")
            if not user_sids or not label_sids:
                continue
            yield user_sids, label_sids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument(
        "--shard_size", type=int, default=200_000,
        help="rows per parquet shard",
    )
    ap.add_argument(
        "--max_rows", type=int, default=0,
        help="cap total rows; 0 = whole CSV",
    )
    ap.add_argument("--log_every", type=int, default=100_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    # Two int64 lists, no nulls. ``list_(int64())`` lets pyarrow store these
    # as ListArray(int64) inside the parquet, which TER's
    # ``sequence_raw_feature`` reads directly into a JaggedTensor.
    schema = pa.schema(
        [
            pa.field("user_sequence", pa.list_(pa.int64()), nullable=False),
            pa.field("label",         pa.list_(pa.int64()), nullable=False),
        ]
    )

    t0 = time.time()
    rows_in_shard: List[Tuple[List[int], List[int]]] = []
    shard_idx = 0
    total = 0
    max_sid_seen = 0
    max_user_len = 0
    max_label_len = 0

    def flush(rows, idx):
        if not rows:
            return
        user_col = [r[0] for r in rows]
        label_col = [r[1] for r in rows]
        tbl = pa.table(
            {"user_sequence": user_col, "label": label_col}, schema=schema,
        )
        path = os.path.join(args.out_dir, f"shard-{idx:05d}.parquet")
        pq.write_table(tbl, path, compression="zstd")

    for u, lab in _iter_rows(args.csv, args.max_rows):
        rows_in_shard.append((u, lab))
        total += 1
        max_sid_seen = max(max_sid_seen, max(u), max(lab))
        max_user_len = max(max_user_len, len(u))
        max_label_len = max(max_label_len, len(lab))

        if len(rows_in_shard) >= args.shard_size:
            flush(rows_in_shard, shard_idx)
            rows_in_shard = []
            shard_idx += 1
            print(
                f"[shard {shard_idx-1}] flushed; total={total} "
                f"max_sid={max_sid_seen} max_user_len={max_user_len} "
                f"max_label_len={max_label_len} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )

        if args.log_every and total % args.log_every == 0:
            print(
                f"[progress] rows={total} max_sid={max_sid_seen} "
                f"max_user_len={max_user_len} max_label_len={max_label_len} "
                f"wall={time.time()-t0:.0f}s",
                flush=True,
            )

    if rows_in_shard:
        flush(rows_in_shard, shard_idx)
        shard_idx += 1

    print(
        f"[done] rows_written={total} shards={shard_idx} "
        f"max_sid={max_sid_seen} max_user_len={max_user_len} "
        f"max_label_len={max_label_len} "
        f"wall={time.time()-t0:.1f}s out_dir={args.out_dir}",
        flush=True,
    )
    if max_sid_seen > 65536:
        print(
            f"[WARN] max_sid={max_sid_seen} > 65536 — algr's tokenizer adds "
            f"only 65 536 atoms. Vocab extension in TER must be at least "
            f"{max_sid_seen}.",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
