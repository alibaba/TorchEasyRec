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

r"""Global SID collision / distribution statistics.

Reads the ``codes`` column of a Semantic-ID table produced by ``tzrec.predict``
for a SID model (``codes`` is a ``list<int>`` of length ``n_layers``) and reports,
over all items:

- ``no_collision_rate`` = distinct SID tuples / total items (1.0 == collision-free);
- ``collision_free_item_rate`` = items whose SID is unique / total items;
- ``max_collision`` = size of the most-collided SID bucket;
- ``gini`` / ``entropy`` / ``max_entropy`` / ``entropy_ratio`` of the
  SID-frequency distribution;
- per-layer codebook ``coverage`` / ``dead_codes`` / ``perplexity`` (a separate
  long-format table, one row per layer, via ``--layer_stats_output``).

Unlike :class:`tzrec.metrics.unique_ratio.UniqueRatio` (a per-batch diversity
proxy that is biased by batch size), this is the exact global figure. It is a
single-process tool -- launch it with ``python -m`` (no torchrun / process group).

Example::

    python -m tzrec.tools.sid.sid_quality_report \
        --input_path 'experiments/sid_rqkmeans/predict_output/*.parquet' \
        --codes_field codes --codebook 256,256,256 \
        --summary_output experiments/sid_rqkmeans/summary \
        --layer_stats_output experiments/sid_rqkmeans/layer_stats
"""

import argparse
import math
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import torch

from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger


def compute_gini(counts: npt.ArrayLike) -> float:
    """Gini coefficient of a frequency distribution.

    Args:
        counts (array-like): per-SID occurrence counts.

    Returns:
        float: Gini in [0, 1); 0 == perfectly uniform, ->1 == concentrated.
    """
    c = np.asarray(counts, dtype=np.int64)
    if c.size == 0:
        return 0.0
    # (distinct count value j, how many SIDs have it n_j), sorted by j.
    # np.unique intentionally: torch.unique measured slower than numpy on CPU.
    values, freqs = np.unique(c, return_counts=True)
    n = int(freqs.sum())
    sum_j_nj = int(np.dot(values, freqs))
    if sum_j_nj == 0:
        return 0.0
    total_dot_product = 0
    cumulative_freq_before = 0
    for j, n_j in zip(values.tolist(), freqs.tolist()):
        first_term = n - cumulative_freq_before
        last_term = n - cumulative_freq_before - n_j + 1
        sum_of_b_block = (first_term + last_term) * n_j // 2
        total_dot_product += j * sum_of_b_block
        cumulative_freq_before += n_j
    return float((n + 1) / n - (2 * total_dot_product / (n * sum_j_nj)))


def compute_entropy(counts: npt.ArrayLike) -> float:
    """Shannon entropy (natural log / nats) of a frequency distribution.

    Uses torch: for the large summary call (up to tens of millions of unique-SID
    counts) its multithreaded reduction is ~10x faster than numpy on CPU; the
    per-layer callers pass tiny arrays where it makes no difference but reuse the
    same helper. ``entr`` handles zero-probability entries, so no masking needed.

    Args:
        counts (array-like): per-SID occurrence counts.

    Returns:
        float: entropy in nats; compare against ``log(codebook_capacity)``.
    """
    c = torch.as_tensor(counts, dtype=torch.float64)
    total = c.sum()
    if total == 0:
        return 0.0
    return float(torch.special.entr(c / total).sum())


def parse_codes(codes: pa.Array) -> List[List[Optional[int]]]:
    """Parse the ``codes`` column into one integer list per row.

    Accepts a native Arrow ``list<int>`` column (as emitted by ``tzrec.predict``
    for a SID model) or a comma-delimited string column (e.g. ``"12,45,89"``).

    Args:
        codes (pa.Array): the ``codes`` column of one read batch.

    Returns:
        list: one ``list`` per row -- length ``n_layers`` for a well-formed row,
        ``[]`` for a null/blank/non-integer row, or a list still containing
        ``None`` for a row with a null element. The caller drops all malformed rows.
    """
    if pa.types.is_list(codes.type) or pa.types.is_large_list(codes.type):
        # to_pylist(): a null row -> None; a null element -> None inside the list.
        return [[] if r is None else r for r in codes.to_pylist()]
    out: List[List[Optional[int]]] = []
    for x in codes.cast(pa.string()).to_pylist():
        if x is None:
            out.append([])
            continue
        try:
            out.append([int(v) for v in x.split(",") if v != ""])
        except ValueError:
            out.append([])  # a non-integer token -> malformed row (dropped later)
    return out


def update_layer_hist(
    layer_hist: Optional[List[np.ndarray]],
    arr: np.ndarray,
    codebook: List[int],
) -> List[np.ndarray]:
    """Add one batch of codes to the per-layer code-usage histograms.

    Args:
        layer_hist (list): current per-layer histograms, or ``None`` to allocate.
        arr (np.ndarray): ``(batch, n_layers)`` code matrix, already filtered to
            ``[0, codebook)`` (the caller drops malformed and out-of-range rows).
        codebook (list): per-layer codebook sizes.

    Returns:
        list: the per-layer histograms with this batch added.
    """
    if layer_hist is None:
        layer_hist = [np.zeros(k, dtype=np.int64) for k in codebook]
    for layer, k in enumerate(codebook):
        layer_hist[layer] += np.bincount(arr[:, layer], minlength=k)
    return layer_hist


def build_arr(codes: pa.Array, n_layers: int) -> Tuple[np.ndarray, int]:
    """Parse one batch's codes into a matrix of well-formed rows.

    Keeps only rows with exactly ``n_layers`` non-null integer codes; a null row,
    a null element, a wrong-width row, or a non-integer token is dropped -- so a
    few bad rows never discard their whole batch. The fast zero-copy Arrow path is
    taken only when the entire batch is clean; otherwise it falls back to per-row
    parsing.

    Args:
        codes (pa.Array): the ``codes`` column of one read batch.
        n_layers (int): expected number of codes per row.

    Returns:
        tuple: the ``(n_valid, n_layers)`` int matrix and the number of dropped rows.
    """
    if pa.types.is_list(codes.type) or pa.types.is_large_list(codes.type):
        widths = np.diff(codes.offsets.to_numpy())
        flat = codes.flatten()
        if (
            codes.null_count == 0
            and flat.null_count == 0
            and bool((widths == n_layers).all())
        ):
            return flat.to_numpy(zero_copy_only=False).reshape(-1, n_layers), 0
    rows = parse_codes(codes)
    valid = [r for r in rows if len(r) == n_layers and None not in r]
    arr = (
        np.asarray(valid, dtype=np.int64)
        if valid
        else np.empty((0, n_layers), dtype=np.int64)
    )
    return arr, len(rows) - len(valid)


def _parse_codebook(value: str) -> List[int]:
    """Parse the ``--codebook`` argument into per-layer sizes.

    Accepts an ``int,int,...`` string; whitespace around each size is stripped
    (``int`` does that). Rejects non-integer tokens and non-positive sizes.

    Args:
        value (str): the raw ``--codebook`` value, e.g. ``"256, 256, 256"``.

    Returns:
        list: the per-layer codebook sizes.
    """
    try:
        sizes = [int(x) for x in value.split(",")]
    except ValueError as err:
        raise argparse.ArgumentTypeError(
            f"--codebook must be 'int,int,...', got {value!r}"
        ) from err
    if any(k < 1 for k in sizes):
        raise argparse.ArgumentTypeError(
            f"--codebook sizes must be positive ints, got {value!r}"
        )
    return sizes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global SID collision / distribution statistics."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="SID table with a codes column (tzrec.predict output). "
        "Point at a glob like '.../predict_output/*.parquet', or pass "
        "--reader_type for a bare directory / odps table.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default=None,
        help="path for the summary table (one row over all items; a directory "
        "for csv/parquet writers).",
    )
    parser.add_argument(
        "--layer_stats_output",
        type=str,
        default=None,
        help="optional path for a per-layer stats table (long format: layer, "
        "codebook_size, coverage, dead_codes, perplexity).",
    )
    parser.add_argument(
        "--codes_field",
        type=str,
        default="codes",
        help="codes column name in the input table.",
    )
    parser.add_argument(
        "--codebook",
        type=_parse_codebook,
        required=True,
        help="per-layer codebook sizes as 'int,int,...' (e.g. '256,256,256'); "
        "surrounding spaces are stripped. Required.",
    )
    parser.add_argument(
        "--log_top_sids",
        type=int,
        default=None,
        help="if set to N, log the N most-frequent (most-collided) SIDs; "
        "off by default.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4096,
        help="table read batch size.",
    )
    parser.add_argument(
        "--reader_type",
        type=str,
        default=None,
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        help="input reader type, if it cannot be inferred from --input_path.",
    )
    parser.add_argument(
        "--writer_type",
        type=str,
        default=None,
        choices=["OdpsWriter", "CsvWriter", "ParquetWriter"],
        help="output writer type; defaults to match the reader.",
    )
    parser.add_argument(
        "--odps_data_quota_name",
        type=str,
        default="pay-as-you-go",
        help="maxcompute storage api/tunnel data quota name.",
    )
    args = parser.parse_args()

    reader = create_reader(
        input_path=args.input_path,
        batch_size=args.batch_size,
        selected_cols=[args.codes_field],
        reader_type=args.reader_type,
        quota_name=args.odps_data_quota_name,
    )
    writer_type = args.writer_type or reader.__class__.__name__.replace(
        "Reader", "Writer"
    )

    codebook: List[int] = args.codebook
    n_layers = len(codebook)
    capacity: int = math.prod(codebook)
    if capacity > np.iinfo(np.int64).max:
        parser.error(
            f"--codebook capacity (product = {capacity}) exceeds int64; "
            "collision analysis is not supported at that scale."
        )
    # Mixed-radix weights: id = sum_l code_l * radix[l] bijects distinct in-range
    # SID tuples onto [0, capacity), so counting is one int64 np.unique at the end.
    radix = np.array(
        [math.prod(codebook[layer + 1 :]) for layer in range(n_layers)],
        dtype=np.int64,
    )
    codebook_arr = np.asarray(codebook, dtype=np.int64)  # per-layer sizes, for range

    # Whole-dataset accumulators. Each well-formed, in-range SID tuple is mixed-
    # radix-encoded to one int64 id, so counting is a single np.unique at the end
    # (no per-row Python Counter). all_ids holds ~8 bytes/row until that reduction.
    all_ids: List[np.ndarray] = []
    total = 0
    n_malformed = 0  # rows dropped: null / wrong-width / non-integer codes
    n_oob = 0  # rows dropped: a code outside [0, codebook) (wrong --codebook)
    layer_hist: Optional[List[np.ndarray]] = None  # per-layer code-usage histograms

    for i, data in enumerate(reader.to_batches()):
        codes = data[args.codes_field]
        if i % 100 == 0:
            logger.info(f"scanned {total} rows...")

        arr, dropped = build_arr(codes, n_layers)
        n_malformed += dropped
        if arr.shape[0] == 0:
            continue

        # Drop out-of-range rows rather than clamp them: clamping would merge
        # distinct out-of-range SIDs into fabricated in-range collisions.
        in_range = ((arr >= 0) & (arr < codebook_arr)).all(axis=1)
        batch_oob = int(arr.shape[0] - in_range.sum())
        if batch_oob:
            arr = arr[in_range]
            n_oob += batch_oob
            if arr.shape[0] == 0:
                continue

        total += arr.shape[0]
        all_ids.append(arr @ radix)  # one int64 SID id per row
        layer_hist = update_layer_hist(layer_hist, arr, codebook)

    if n_malformed or n_oob:
        hint = " (check --codebook)" if n_oob else ""
        logger.warning(
            f"skipped {n_malformed} malformed and {n_oob} out-of-range rows; "
            f"stats cover the remaining {total} rows{hint}."
        )
    if not all_ids:
        raise ValueError(
            f"no valid rows read from {args.input_path}; nothing to report."
        )

    # distinct SID ids + their frequencies = the collision/frequency distribution.
    sid_ids, counts = np.unique(np.concatenate(all_ids), return_counts=True)
    unique = len(sid_ids)
    entropy = compute_entropy(counts)
    max_entropy = math.log(capacity)

    # Overall summary: one row of scalar columns (writes natively to any
    # csv/parquet/odps writer -- no list<> columns to trip up the CSV writer).
    summary = OrderedDict(
        [
            ("source", args.input_path),
            ("codebook", ",".join(map(str, codebook))),
            ("total", total),
            ("unique_sid", unique),
            ("no_collision_rate", unique / total),
            ("collision_free_item_rate", int((counts == 1).sum()) / total),
            ("max_collision", int(counts.max())),
            ("gini", compute_gini(counts)),
            ("entropy", entropy),
            ("max_entropy", max_entropy),
            ("entropy_ratio", entropy / max_entropy if max_entropy else float("nan")),
        ]
    )

    logger.info("===== SID collision stats =====")
    for key, value in summary.items():
        logger.info(f"{key} = {value}")
    if args.log_top_sids:
        order = np.argsort(-counts, kind="stable")[: args.log_top_sids]
        # decode ids back to codes: code_l = (id // radix_l) % codebook_l.
        top_codes = (sid_ids[order][:, None] // radix) % np.asarray(codebook)
        top_sids = [
            (",".join(map(str, code)), int(cnt))
            for code, cnt in zip(top_codes.tolist(), counts[order].tolist())
        ]
        logger.info(f"top-{args.log_top_sids} SIDs by frequency = {top_sids}")

    if args.summary_output:
        stats_writer = create_writer(
            args.summary_output, writer_type, quota_name=args.odps_data_quota_name
        )
        stats_writer.write(OrderedDict((k, pa.array([v])) for k, v in summary.items()))
        stats_writer.close()
        logger.info(f"wrote summary to {args.summary_output}")

    # Per-layer codebook utilization: one row PER LAYER (long format). All scalar
    # columns, so it too writes natively to csv/parquet/odps.
    if layer_hist is not None:
        layer_stats: "OrderedDict[str, list]" = OrderedDict(
            layer=[], codebook_size=[], coverage=[], dead_codes=[], perplexity=[]
        )
        for layer, h in enumerate(layer_hist):
            nonzero = int(np.count_nonzero(h))
            coverage = nonzero / len(h)
            dead_codes = len(h) - nonzero
            perplexity = math.exp(compute_entropy(h))
            layer_stats["layer"].append(layer)
            layer_stats["codebook_size"].append(len(h))
            layer_stats["coverage"].append(coverage)
            layer_stats["dead_codes"].append(dead_codes)
            layer_stats["perplexity"].append(perplexity)
            logger.info(
                f"layer {layer}: coverage={coverage:.4f} "
                f"dead_codes={dead_codes} perplexity={perplexity:.4f}"
            )
        if args.layer_stats_output:
            layer_writer = create_writer(
                args.layer_stats_output,
                writer_type,
                quota_name=args.odps_data_quota_name,
            )
            layer_writer.write(
                OrderedDict((k, pa.array(v)) for k, v in layer_stats.items())
            )
            layer_writer.close()
            logger.info(f"wrote per-layer stats to {args.layer_stats_output}")
