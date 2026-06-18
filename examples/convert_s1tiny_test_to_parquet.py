"""Convert algr's ``s1_tiny_test.csv`` into TorchEasyRec-shaped parquet.

The TEST split differs from the train split: the ``answer`` column holds
ground-truth ITEM IDs (e.g. ``AIB2f`` or ``9nXWC;I6kld;...`` —
semicolon-separated when multiple), not SID ``C``-codes. algr evaluates
this split via beam-search generation + ``calc_hr_fast.py`` item mapping.

For TER's periodic CE-loss eval we need ``(user_sequence, label)`` SID rows,
so this script maps the FIRST ground-truth item of each row to its SID
triple via ``item_info/tiny_item_sid_final.csv``:

    sid_triple(item) = [lv1 + 1, lv2 + 8192 + 1, lv3 + 16384 + 1]

(the per-layer offsets follow the dataset's C-token scheme
``C{lv1}C{lv2+8192}C{lv3+16384}``; the +1 converts the 0-indexed C-code to
the 1-indexed SID contract of ``GenerativeRecLM`` — identical to
``convert_s1tiny_to_parquet.py``'s ``Ck → SID = k + 1``.)

Rows whose first answer item is missing from the item map are skipped
(counted and reported).

Usage on remote::

    /opt/conda/bin/python -m examples.convert_s1tiny_test_to_parquet \\
        --csv /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data/s1_tiny_test.csv \\
        --item_sid_csv /home/admin/workspace/aop_lab/data/AL-GR-Tiny/item_info/tiny_item_sid_final.csv \\
        --out_dir /home/admin/workspace/aop_lab/data/AL-GR-Tiny/test_data_genreclm
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Tuple

import pyarrow as pa
import pyarrow.parquet as pq

csv.field_size_limit(10 * 1024 * 1024)

_SID_RE = re.compile(r"C(\d+)")

_LV2_OFFSET = 8192
_LV3_OFFSET = 16384


def _extract_sids(text: str) -> List[int]:
    """``Ck → SID = k + 1`` (1-indexed; matches the train converter)."""
    return [int(m) + 1 for m in _SID_RE.findall(text)]


def _load_item_map(path: str) -> Dict[str, Tuple[int, int, int]]:
    t0 = time.time()
    item_map: Dict[str, Tuple[int, int, int]] = {}
    bad = 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                item_map[row["item_id"]] = (
                    int(row["codebook_lv1"]),
                    int(row["codebook_lv2"]),
                    int(row["codebook_lv3"]),
                )
            except (ValueError, TypeError, KeyError):
                # the 25M-row map contains a few malformed/empty rows
                bad += 1
    print(
        f"[item_map] {len(item_map)} items loaded "
        f"({bad} malformed rows skipped) in {time.time()-t0:.0f}s",
        flush=True,
    )
    return item_map


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--item_sid_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--shard_size", type=int, default=200_000)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    schema = pa.schema(
        [
            pa.field("user_sequence", pa.list_(pa.int64()), nullable=False),
            pa.field("label",         pa.list_(pa.int64()), nullable=False),
        ]
    )

    item_map = _load_item_map(args.item_sid_csv)

    rows: List[Tuple[List[int], List[int]]] = []
    total = read = miss_item = miss_user = 0
    max_sid = 0
    with open(args.csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            read += 1
            user_sids = _extract_sids(row.get("user") or "")
            if not user_sids:
                miss_user += 1
                continue
            first_item = (row.get("answer") or "").split(";")[0].strip()
            lv = item_map.get(first_item)
            if lv is None:
                miss_item += 1
                continue
            label_sids = [
                lv[0] + 1,
                lv[1] + _LV2_OFFSET + 1,
                lv[2] + _LV3_OFFSET + 1,
            ]
            rows.append((user_sids, label_sids))
            total += 1
            max_sid = max(max_sid, max(user_sids), max(label_sids))

    shard_idx = 0
    for i in range(0, len(rows), args.shard_size):
        chunk = rows[i : i + args.shard_size]
        tbl = pa.table(
            {
                "user_sequence": [r[0] for r in chunk],
                "label": [r[1] for r in chunk],
            },
            schema=schema,
        )
        pq.write_table(
            tbl,
            os.path.join(args.out_dir, f"shard-{shard_idx:05d}.parquet"),
            compression="zstd",
        )
        shard_idx += 1

    print(
        f"[done] read={read} rows_written={total} shards={shard_idx} "
        f"skipped_missing_item={miss_item} skipped_no_user_sids={miss_user} "
        f"max_sid={max_sid} out_dir={args.out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
