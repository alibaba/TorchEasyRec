"""Convert s2 test CSV to TER parquet (user_sequence only; label = [0] placeholder).

s2 test CSV answers are item IDs (not SID codes), so label cannot be derived.
The predict script only reads `user_sequence` from parquet, so label is a dummy.

Usage on remote::

    python3 /home/admin/workspace/TorchEasyRec_qwen_smoke/TorchEasyRec/examples/convert_s2test_to_parquet.py \
        --csv /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data/s2_tiny_test.csv \
        --out_dir /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data_genreclm_s2
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

csv.field_size_limit(10 * 1024 * 1024)

_SID_RE = re.compile(r"C(\d+)")


def _extract_user_sids(text: str) -> List[int]:
    return [int(m) + 1 for m in _SID_RE.findall(text)]


def _iter_rows(csv_path: str, max_rows: int) -> Iterator[List[int]]:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            user_sids = _extract_user_sids(row["user"])
            if not user_sids:
                continue
            yield user_sids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=200_000)
    ap.add_argument("--max_rows", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    schema = pa.schema([
        pa.field("user_sequence", pa.list_(pa.int64()), nullable=False),
        pa.field("label",         pa.list_(pa.int64()), nullable=False),
    ])

    t0 = time.time()
    rows_in_shard: List[Tuple[List[int], List[int]]] = []
    shard_idx = 0
    total = 0

    def flush(rows, idx):
        if not rows:
            return
        user_col = [r[0] for r in rows]
        label_col = [r[1] for r in rows]
        tbl = pa.table({"user_sequence": user_col, "label": label_col}, schema=schema)
        pq.write_table(tbl, os.path.join(args.out_dir, f"shard-{idx:05d}.parquet"), compression="zstd")

    for u in _iter_rows(args.csv, args.max_rows):
        rows_in_shard.append((u, [0]))  # dummy label
        total += 1
        if len(rows_in_shard) >= args.shard_size:
            flush(rows_in_shard, shard_idx)
            rows_in_shard = []
            shard_idx += 1

    if rows_in_shard:
        flush(rows_in_shard, shard_idx)
        shard_idx += 1

    print(f"[done] rows_written={total} shards={shard_idx} wall={time.time()-t0:.1f}s out_dir={args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
