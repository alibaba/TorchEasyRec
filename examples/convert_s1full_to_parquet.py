"""Convert full-s1 (``s1_splits/part_*.csv``, 41 shards, ~246 GB) into
TorchEasyRec genrec parquet — the multi-CSV streaming sibling of
``convert_s1tiny_to_parquet.py``.

Identical row logic (extract every ``C\\d+`` -> 1-indexed SID; columns
``user_sequence`` / ``label`` as ``list<int64>``), but iterates a GLOB of CSV
parts with a CONTINUOUS shard index so the 41 files land in one parquet dir.
Streaming (``csv.DictReader`` + shard-buffered flush) so memory stays bounded
regardless of the 246 GB input; the dropped CN-prompt text means the parquet is
a small fraction of the CSV size.

Usage on remote::

    /opt/conda/bin/python -m examples.convert_s1full_to_parquet \\
        --csv_glob '/home/admin/workspace/aop_lab/data/AL-GR-v1/s1_splits/*.csv' \\
        --out_dir /home/admin/workspace/aop_lab/data/AL-GR-v1/train_data_genreclm_s1full \\
        --shard_size 200000 --log_every 1000000
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
import time
from typing import Iterator, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(10 * 1024 * 1024)
_SID_RE = re.compile(r"C(\d+)")


def _extract_sids(text: str) -> List[int]:
    return [int(m) + 1 for m in _SID_RE.findall(text)]


def _iter_rows(
    csv_paths: List[str], max_rows: int
) -> Iterator[Tuple[List[int], List[int]]]:
    n = 0
    for p in csv_paths:
        with open(p, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if max_rows and n >= max_rows:
                    return
                u = _extract_sids(row.get("user") or "")
                lab = _extract_sids(row.get("answer") or "")
                if not u or not lab:
                    continue
                n += 1
                yield u, lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_glob", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=200_000)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=1_000_000)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.csv_glob))
    print(f"{len(paths)} csv parts matched", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("user_sequence", pa.list_(pa.int64()), nullable=False),
            pa.field("label", pa.list_(pa.int64()), nullable=False),
        ]
    )

    def flush(rows: List[Tuple[List[int], List[int]]], idx: int) -> None:
        if not rows:
            return
        tbl = pa.table(
            {"user_sequence": [r[0] for r in rows], "label": [r[1] for r in rows]},
            schema=schema,
        )
        pq.write_table(
            tbl, os.path.join(args.out_dir, f"shard-{idx:05d}.parquet"),
            compression="zstd",
        )

    t0 = time.time()
    rows: List[Tuple[List[int], List[int]]] = []
    idx = 0
    total = 0
    max_sid = 0
    for u, lab in _iter_rows(paths, args.max_rows):
        rows.append((u, lab))
        total += 1
        max_sid = max(max_sid, max(u), max(lab))
        if len(rows) >= args.shard_size:
            flush(rows, idx)
            rows = []
            idx += 1
        if args.log_every and total % args.log_every == 0:
            print(
                f"[progress] rows={total} shards={idx} max_sid={max_sid} "
                f"wall={time.time() - t0:.0f}s",
                flush=True,
            )
    if rows:
        flush(rows, idx)
        idx += 1
    print(
        f"[done] rows={total} shards={idx} max_sid={max_sid} "
        f"wall={time.time() - t0:.0f}s out_dir={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
