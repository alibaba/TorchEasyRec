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
- ``uniquely_identified_item_rate`` = items whose SID is unique / total items;
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
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import torch

from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


def compute_gini(counts: npt.ArrayLike) -> float:
    """Gini coefficient of a frequency distribution.

    Measures inequality among *occupied* SID buckets only (``counts`` holds one
    entry per distinct SID), not codebook utilization: an all-collapsed single
    bucket and a perfectly even distribution both score 0, told apart instead by
    ``no_collision_rate`` / ``max_collision``.

    Args:
        counts (array-like): per-SID occurrence counts.

    Returns:
        float: Gini in [0, 1); 0 == perfectly uniform, ->1 == concentrated.
    """
    c = np.asarray(counts, dtype=np.int64)
    if c.size == 0:
        return 0.0
    # np.unique intentionally; torch.unique is slower on CPU.
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

    Uses torch (``entr`` handles zero-probability entries without masking); its
    multithreaded reduction is ~10x faster than numpy on the large summary call.

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
        return [[] if r is None else r for r in codes.to_pylist()]
    out: List[List[Optional[int]]] = []
    for x in codes.cast(pa.string()).to_pylist():
        if x is None:
            out.append([])
            continue
        try:
            row = [int(v) for v in x.split(",")]
        except ValueError:
            out.append([])
            continue
        # int() accepts >int64 tokens that would later crash np.asarray(int64).
        out.append(row if all(_INT64_MIN <= v <= _INT64_MAX for v in row) else [])
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

    Raises:
        ValueError: if a list ``codes`` column has a non-integer element type
            (e.g. ``list<float>``), which would otherwise crash on the fast path or
            be silently truncated on the fallback.
    """
    if pa.types.is_list(codes.type) or pa.types.is_large_list(codes.type):
        if not pa.types.is_integer(codes.type.value_type):
            raise ValueError(
                f"codes column has non-integer element type "
                f"{codes.type.value_type}; SID codes must be integers."
            )
        widths = np.diff(codes.offsets.to_numpy())
        flat = codes.flatten()
        if (
            codes.null_count == 0
            and flat.null_count == 0
            and bool((widths == n_layers).all())
        ):
            arr = flat.to_numpy(zero_copy_only=False).reshape(-1, n_layers)
            return arr.astype(np.int64, copy=False), 0
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


def _positive_int(value: str) -> int:
    """Parse a strictly-positive integer CLI argument.

    Args:
        value (str): the raw argument value.

    Returns:
        int: the parsed value, guaranteed ``> 0``.
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive int, got {value!r}")
    return ivalue


def run_report(
    batches: Iterable[Dict[str, pa.Array]],
    codes_field: str,
    codebook: List[int],
    source: str = "",
    log_top_sids: Optional[int] = None,
) -> Tuple[
    "OrderedDict[str, object]",
    Optional["OrderedDict[str, list]"],
    Optional[List[Tuple[str, int]]],
]:
    """Scan SID code batches and compute global collision + per-layer stats.

    This is the importable core (argparse / reader / writer wiring stays in
    ``__main__``), so the numeric branches can be unit-tested in-process.

    Args:
        batches (iterable): batches of ``{column: pa.Array}`` (e.g. the output of
            ``reader.to_batches()``); only ``codes_field`` is read.
        codes_field (str): the codes column name in each batch.
        codebook (list): per-layer codebook sizes; ``prod(codebook)`` must fit int64.
        source (str): label recorded in the summary ``source`` column.
        log_top_sids (int, optional): if set, decode and log the N most-collided SIDs.

    Returns:
        tuple: ``(summary, layer_stats, top_sids)`` -- the one-row summary
        ``OrderedDict``; the per-layer ``OrderedDict`` of columns (``None`` if no row
        contributed); and the decoded top-N ``(sid, count)`` list (``None`` unless
        ``log_top_sids`` is set).

    Raises:
        ValueError: if the codebook capacity overflows int64, ``log_top_sids`` is
            non-positive, or no valid rows are read.
    """
    n_layers = len(codebook)
    capacity = math.prod(codebook)
    if capacity > _INT64_MAX:
        raise ValueError(
            f"codebook capacity (product = {capacity}) exceeds int64; "
            "collision analysis is not supported at that scale."
        )
    if log_top_sids is not None and log_top_sids <= 0:
        raise ValueError(f"log_top_sids must be positive, got {log_top_sids}.")
    # Mixed-radix id = sum_l code_l * radix_l bijects tuples onto [0, capacity),
    # so collision counting is a single np.unique over int64 ids.
    radix = np.array(
        [math.prod(codebook[layer + 1 :]) for layer in range(n_layers)],
        dtype=np.int64,
    )
    codebook_arr = np.asarray(codebook, dtype=np.int64)

    all_ids: List[np.ndarray] = []
    total = 0
    n_malformed = 0
    n_oob = 0
    layer_hist: Optional[List[np.ndarray]] = None

    for i, data in enumerate(batches):
        codes = data[codes_field]
        if i % 100 == 0:
            logger.info(f"scanned {total} rows...")

        arr, dropped = build_arr(codes, n_layers)
        n_malformed += dropped
        if arr.shape[0] == 0:
            continue

        # Drop out-of-range rows; clamping would fabricate in-range collisions.
        in_range = ((arr >= 0) & (arr < codebook_arr)).all(axis=1)
        batch_oob = int(arr.shape[0] - in_range.sum())
        if batch_oob:
            arr = arr[in_range]
            n_oob += batch_oob
            if arr.shape[0] == 0:
                continue

        total += arr.shape[0]
        all_ids.append(arr @ radix)
        layer_hist = update_layer_hist(layer_hist, arr, codebook)

    if n_malformed or n_oob:
        hint = " (check codebook)" if n_oob else ""
        logger.warning(
            f"skipped {n_malformed} malformed and {n_oob} out-of-range rows; "
            f"stats cover the remaining {total} rows{hint}."
        )
    if not all_ids:
        raise ValueError(f"no valid rows read from {source!r}; nothing to report.")

    sid_ids, counts = np.unique(np.concatenate(all_ids), return_counts=True)
    unique = len(sid_ids)
    entropy = compute_entropy(counts)
    max_entropy = math.log(capacity)

    # All-scalar columns so the summary writes natively even to the CSV writer.
    summary: "OrderedDict[str, object]" = OrderedDict(
        [
            ("source", source),
            ("codebook", ",".join(map(str, codebook))),
            ("total", total),
            ("unique_sid", unique),
            ("no_collision_rate", unique / total),
            ("uniquely_identified_item_rate", int((counts == 1).sum()) / total),
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

    top_sids: Optional[List[Tuple[str, int]]] = None
    if log_top_sids:
        order = np.argsort(-counts, kind="stable")[:log_top_sids]
        # decode: code_l = (id // radix_l) % K_l.
        top_codes = (sid_ids[order][:, None] // radix) % codebook_arr
        top_sids = [
            (",".join(map(str, code)), int(cnt))
            for code, cnt in zip(top_codes.tolist(), counts[order].tolist())
        ]
        logger.info(f"top-{log_top_sids} SIDs by frequency = {top_sids}")

    layer_stats: Optional["OrderedDict[str, list]"] = None
    if layer_hist is not None:
        layer_stats = OrderedDict(
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
    return summary, layer_stats, top_sids


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
        type=_positive_int,
        default=None,
        help="if set to a positive N, log the N most-frequent (most-collided) "
        "SIDs; off by default.",
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

    # Clean CLI usage error; run_report re-guards for library callers.
    capacity = math.prod(args.codebook)
    if capacity > _INT64_MAX:
        parser.error(
            f"--codebook capacity (product = {capacity}) exceeds int64; "
            "collision analysis is not supported at that scale."
        )

    reader = create_reader(
        input_path=args.input_path,
        batch_size=args.batch_size,
        selected_cols=[args.codes_field],
        reader_type=args.reader_type,
        quota_name=args.odps_data_quota_name,
    )
    if args.codes_field not in reader.schema.names:
        parser.error(
            f"--codes_field {args.codes_field!r} not found in input "
            f"{args.input_path}; check the column name in the predict output."
        )
    writer_type = args.writer_type or reader.__class__.__name__.replace(
        "Reader", "Writer"
    )

    summary, layer_stats, _ = run_report(
        reader.to_batches(),
        args.codes_field,
        args.codebook,
        source=args.input_path,
        log_top_sids=args.log_top_sids,
    )

    if args.summary_output:
        stats_writer = create_writer(
            args.summary_output, writer_type, quota_name=args.odps_data_quota_name
        )
        stats_writer.write(OrderedDict((k, pa.array([v])) for k, v in summary.items()))
        stats_writer.close()
        logger.info(f"wrote summary to {args.summary_output}")

    if args.layer_stats_output and layer_stats is not None:
        layer_writer = create_writer(
            args.layer_stats_output, writer_type, quota_name=args.odps_data_quota_name
        )
        layer_writer.write(
            OrderedDict((k, pa.array(v)) for k, v in layer_stats.items())
        )
        layer_writer.close()
        logger.info(f"wrote per-layer stats to {args.layer_stats_output}")
