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

r"""Evaluate exact global semantic-ID collision and distribution quality.

The tool evaluates one explicitly selected SID field, or compares an original
field with a final field from the same item-aligned input rows. Comparison uses
the common valid-row cohort and reports ``before``, ``after``, and
``after - before`` views.

Example::

    python -m tzrec.tools.sid.evaluate_sid_quality \
        --input_path 'experiments/sid_rqkmeans/predict_output/*.parquet' \
        --codes_field codes --codebook 256,256,256 \
        --summary_output experiments/sid_rqkmeans/summary \
        --layer_stats_output experiments/sid_rqkmeans/layer_stats
"""

import argparse
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger
from tzrec.utils.sid.quality import (
    SidLayerQualityMetrics,
    SidQualityAccumulator,
    SidQualityMetrics,
    SidQualityResult,
    compare_sid_quality,
    valid_code_rows,
)

_INT64_MAX = int(np.iinfo(np.int64).max)
_INT64_MIN = int(np.iinfo(np.int64).min)


@dataclass(frozen=True)
class DecodedCodes:
    """Row-aligned decoded codes and their structural validity mask.

    Attributes:
        values: Integer matrix with a placeholder value for malformed rows.
        valid_rows: Mask selecting rows with the expected width and integer values.
        malformed_rows: Number of rows excluded by ``valid_rows``.
    """

    values: np.ndarray
    valid_rows: np.ndarray
    malformed_rows: int


@dataclass(frozen=True)
class SidQualityEvaluation:
    """Long-format output rows and top SID details for one evaluation.

    Attributes:
        summary_rows: Global metric rows by view.
        layer_rows: Per-layer metric rows by view and layer.
        top_sids: Top SID lists keyed by non-delta view.
    """

    summary_rows: Tuple["OrderedDict[str, object]", ...]
    layer_rows: Tuple["OrderedDict[str, object]", ...]
    top_sids: Mapping[str, Tuple[Tuple[str, int], ...]]


def decode_codes(codes: pa.Array, n_layers: int) -> DecodedCodes:
    """Decode an Arrow SID field while preserving row alignment.

    Native ``list<int>`` and comma-delimited scalar/string fields are accepted.
    Null rows, null elements, non-integer tokens, values outside int64, and rows
    with the wrong number of layers are marked malformed. A list with a
    non-integer Arrow element type is rejected as a schema error.

    Args:
        codes: One Arrow batch column.
        n_layers: Expected number of SID layers.

    Returns:
        Decoded integer values and a row-level structural validity mask.

    Raises:
        ValueError: If ``n_layers`` is not positive or a list element type is
            not integer.
    """
    if n_layers <= 0:
        raise ValueError(f"n_layers must be positive, got {n_layers}.")

    is_list = pa.types.is_list(codes.type) or pa.types.is_large_list(codes.type)
    if is_list and not pa.types.is_integer(codes.type.value_type):
        raise ValueError(
            f"codes column has non-integer element type {codes.type.value_type}; "
            "SID codes must be integers."
        )

    if is_list:
        widths = np.diff(codes.offsets.to_numpy())
        flat = codes.flatten()
        if (
            codes.null_count == 0
            and flat.null_count == 0
            and bool((widths == n_layers).all())
        ):
            flat_values = flat.to_numpy(zero_copy_only=False)
            fits_int64 = flat_values.dtype.kind != "u" or bool(
                (flat_values <= _INT64_MAX).all()
            )
            if fits_int64:
                values = flat_values.reshape(-1, n_layers).astype(np.int64, copy=False)
                return DecodedCodes(
                    values=values,
                    valid_rows=np.ones(len(codes), dtype=np.bool_),
                    malformed_rows=0,
                )
        rows = codes.to_pylist()
    else:
        rows = codes.cast(pa.string()).to_pylist()

    values = np.zeros((len(rows), n_layers), dtype=np.int64)
    valid_rows = np.zeros(len(rows), dtype=np.bool_)
    for row_index, raw_row in enumerate(rows):
        if raw_row is None:
            continue
        if is_list:
            parsed_row = raw_row
        else:
            try:
                parsed_row = [int(value) for value in raw_row.split(",")]
            except ValueError:
                continue
        if (
            len(parsed_row) != n_layers
            or any(value is None for value in parsed_row)
            or not all(_INT64_MIN <= value <= _INT64_MAX for value in parsed_row)
        ):
            continue
        values[row_index] = parsed_row
        valid_rows[row_index] = True
    return DecodedCodes(
        values=values,
        valid_rows=valid_rows,
        malformed_rows=int((~valid_rows).sum()),
    )


def run_evaluation(
    batches: Iterable[Dict[str, pa.Array]],
    codes_field: str,
    codebook: Sequence[int],
    origin_codes_field: Optional[str] = None,
    source: str = "",
    log_top_sids: Optional[int] = None,
    invalid_row_policy: str = "drop",
) -> SidQualityEvaluation:
    """Evaluate one SID field or compare aligned original and final fields.

    Args:
        batches: Input batches. Only explicitly selected SID fields are accessed.
        codes_field: Field evaluated as the single or after view.
        codebook: Positive codebook size for each SID layer.
        origin_codes_field: Optional field evaluated as the before view.
        source: Input label included in output rows.
        log_top_sids: Optional number of most frequent SIDs to retain per view.
        invalid_row_policy: ``drop`` to exclude invalid rows or ``error`` to fail.

    Returns:
        Long-format global/layer rows and top SID details.

    Raises:
        KeyError: If a selected field is missing from a batch.
        ValueError: If field configuration, rows, or policy are invalid.
    """
    if origin_codes_field == codes_field:
        raise ValueError("origin_codes_field and codes_field must be different.")
    if invalid_row_policy not in ("drop", "error"):
        raise ValueError(
            "invalid_row_policy must be either 'drop' or 'error', got "
            f"{invalid_row_policy!r}."
        )

    n_layers = len(codebook)
    after_accumulator = SidQualityAccumulator(codebook, log_top_sids)
    before_accumulator = (
        SidQualityAccumulator(codebook, log_top_sids)
        if origin_codes_field is not None
        else None
    )
    input_rows = 0
    invalid_rows = 0
    malformed_before = 0
    malformed_after = 0
    out_of_range_before = 0
    out_of_range_after = 0

    for batch_index, data in enumerate(batches):
        if codes_field not in data:
            raise KeyError(f"codes field {codes_field!r} is missing from a batch.")
        after_codes = decode_codes(data[codes_field], n_layers)
        batch_rows = len(after_codes.values)
        input_rows += batch_rows
        after_in_range = valid_code_rows(after_codes.values, codebook)
        after_valid = after_codes.valid_rows & after_in_range
        malformed_after += after_codes.malformed_rows
        out_of_range_after += int((after_codes.valid_rows & ~after_in_range).sum())

        before_codes: Optional[DecodedCodes] = None
        if origin_codes_field is not None:
            if origin_codes_field not in data:
                raise KeyError(
                    f"origin codes field {origin_codes_field!r} is missing from "
                    "a batch."
                )
            before_codes = decode_codes(data[origin_codes_field], n_layers)
            if len(before_codes.values) != batch_rows:
                raise ValueError(
                    "origin and final code fields must contain the same number "
                    "of rows in every batch."
                )
            before_in_range = valid_code_rows(before_codes.values, codebook)
            before_valid = before_codes.valid_rows & before_in_range
            malformed_before += before_codes.malformed_rows
            out_of_range_before += int(
                (before_codes.valid_rows & ~before_in_range).sum()
            )
            valid_rows = before_valid & after_valid
        else:
            valid_rows = after_valid

        batch_invalid_rows = batch_rows - int(valid_rows.sum())
        if batch_invalid_rows and invalid_row_policy == "error":
            raise ValueError(
                f"batch {batch_index} contains {batch_invalid_rows} malformed "
                "or out-of-range SID rows."
            )
        invalid_rows += batch_invalid_rows
        if valid_rows.any():
            if before_accumulator is not None and before_codes is not None:
                before_accumulator.update(
                    before_codes.values[valid_rows], assume_in_range=True
                )
            after_accumulator.update(
                after_codes.values[valid_rows], assume_in_range=True
            )

        if batch_index % 100 == 0:
            logger.info(f"scanned {input_rows} rows...")

    if invalid_rows:
        logger.warning(
            "skipped %d invalid rows (before: %d malformed, %d out-of-range; "
            "after: %d malformed, %d out-of-range); stats cover %d rows.",
            invalid_rows,
            malformed_before,
            out_of_range_before,
            malformed_after,
            out_of_range_after,
            input_rows - invalid_rows,
        )
    evaluated_items = input_rows - invalid_rows
    if evaluated_items == 0:
        raise ValueError(f"no valid rows read from {source!r}; nothing to report.")

    results: List[Tuple[str, str, SidQualityResult]] = []
    if before_accumulator is not None and origin_codes_field is not None:
        before_result = before_accumulator.finalize()
        before_accumulator = None
        after_result = after_accumulator.finalize()
        delta_result = compare_sid_quality(before_result, after_result)
        results.extend(
            [
                ("before", origin_codes_field, before_result),
                ("after", codes_field, after_result),
                (
                    "delta",
                    f"{origin_codes_field}->{codes_field}",
                    delta_result,
                ),
            ]
        )
    else:
        results.append(("single", codes_field, after_accumulator.finalize()))

    summary_rows = []
    layer_rows = []
    top_sids: Dict[str, Tuple[Tuple[str, int], ...]] = {}
    for view, sid_field, result in results:
        summary_rows.append(
            _build_summary_row(
                source,
                view,
                sid_field,
                codebook,
                input_rows,
                evaluated_items,
                invalid_rows,
                result.metrics,
            )
        )
        layer_rows.extend(
            _build_layer_rows(
                source,
                view,
                sid_field,
                codebook,
                input_rows,
                evaluated_items,
                invalid_rows,
                result.layer_metrics,
            )
        )
        _log_result(view, result)
        if result.top_sids is not None:
            top_sids[view] = result.top_sids

    return SidQualityEvaluation(
        summary_rows=tuple(summary_rows),
        layer_rows=tuple(layer_rows),
        top_sids=top_sids,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the SID quality command-line parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate global SID collision and distribution quality."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Item-aligned table containing the explicitly selected SID fields.",
    )
    parser.add_argument(
        "--summary_output",
        default=None,
        help="Optional path for long-format global metric rows.",
    )
    parser.add_argument(
        "--layer_stats_output",
        default=None,
        help="Optional path for long-format per-layer metric rows.",
    )
    parser.add_argument(
        "--codes_field",
        default="codes",
        help="SID field evaluated as the single or after view.",
    )
    parser.add_argument(
        "--origin_codes_field",
        default=None,
        help="Optional original SID field; enables before/after/delta comparison.",
    )
    parser.add_argument(
        "--codebook",
        type=_parse_codebook,
        required=True,
        help="Per-layer codebook sizes as 'int,int,...'.",
    )
    parser.add_argument(
        "--invalid_row_policy",
        choices=["drop", "error"],
        default="drop",
        help="Drop malformed/out-of-range rows or fail on the first invalid batch.",
    )
    parser.add_argument(
        "--log_top_sids",
        type=_positive_int,
        default=None,
        help="Optional number of most frequent SIDs to log for each non-delta view.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="Table read batch size."
    )
    parser.add_argument(
        "--reader_type",
        choices=["OdpsReader", "CsvReader", "ParquetReader"],
        default=None,
        help="Input reader type when it cannot be inferred from the input path.",
    )
    parser.add_argument(
        "--writer_type",
        choices=["OdpsWriter", "CsvWriter", "ParquetWriter"],
        default=None,
        help="Output writer type; defaults to the corresponding reader type.",
    )
    parser.add_argument(
        "--odps_data_quota_name",
        default="pay-as-you-go",
        help="MaxCompute storage API or tunnel data quota name.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Evaluate SID quality from command-line arguments.

    Args:
        argv: Optional arguments for in-process callers. Defaults to ``sys.argv``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.origin_codes_field == args.codes_field:
        parser.error("--origin_codes_field and --codes_field must be different.")
    capacity = math.prod(args.codebook)
    if capacity > _INT64_MAX:
        parser.error(
            f"--codebook capacity (product = {capacity}) exceeds int64; "
            "collision analysis is not supported at that scale."
        )

    selected_fields = [args.codes_field]
    if args.origin_codes_field is not None:
        selected_fields.insert(0, args.origin_codes_field)
    reader = create_reader(
        input_path=args.input_path,
        batch_size=args.batch_size,
        selected_cols=selected_fields,
        reader_type=args.reader_type,
        quota_name=args.odps_data_quota_name,
    )
    missing_fields = [
        field for field in selected_fields if field not in reader.schema.names
    ]
    if missing_fields:
        parser.error(
            f"selected SID fields {missing_fields!r} not found in input "
            f"{args.input_path!r}."
        )
    writer_type = args.writer_type or reader.__class__.__name__.replace(
        "Reader", "Writer"
    )

    evaluation = run_evaluation(
        reader.to_batches(),
        codes_field=args.codes_field,
        codebook=args.codebook,
        origin_codes_field=args.origin_codes_field,
        source=args.input_path,
        log_top_sids=args.log_top_sids,
        invalid_row_policy=args.invalid_row_policy,
    )
    if args.summary_output:
        _write_rows(
            args.summary_output,
            writer_type,
            args.odps_data_quota_name,
            evaluation.summary_rows,
        )
        logger.info(f"wrote summary to {args.summary_output}")
    if args.layer_stats_output:
        _write_rows(
            args.layer_stats_output,
            writer_type,
            args.odps_data_quota_name,
            evaluation.layer_rows,
        )
        logger.info(f"wrote per-layer stats to {args.layer_stats_output}")


def _parse_codebook(value: str) -> List[int]:
    """Parse a comma-delimited codebook command-line value."""
    try:
        sizes = [int(token) for token in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"--codebook must be 'int,int,...', got {value!r}"
        ) from error
    if not sizes or any(size < 1 for size in sizes):
        raise argparse.ArgumentTypeError(
            f"--codebook sizes must be positive ints, got {value!r}"
        )
    return sizes


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer command-line value."""
    parsed_value = int(value)
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive int, got {value!r}")
    return parsed_value


def _build_summary_row(
    source: str,
    view: str,
    sid_field: str,
    codebook: Sequence[int],
    input_rows: int,
    evaluated_items: int,
    invalid_pair_rows: int,
    metrics: SidQualityMetrics,
) -> "OrderedDict[str, object]":
    """Build one ordered long-format global metric row."""
    return OrderedDict(
        [
            ("source", source),
            ("view", view),
            ("sid_field", sid_field),
            ("codebook", ",".join(map(str, codebook))),
            ("input_rows", input_rows),
            ("evaluated_items", evaluated_items),
            ("invalid_pair_rows", invalid_pair_rows),
            ("total", metrics.total),
            ("unique_sid", metrics.unique_sid),
            ("no_collision_rate", metrics.no_collision_rate),
            (
                "uniquely_identified_item_rate",
                metrics.uniquely_identified_item_rate,
            ),
            ("max_collision", metrics.max_collision),
            ("gini", metrics.gini),
            ("entropy", metrics.entropy),
            ("max_entropy", metrics.max_entropy),
            ("entropy_ratio", metrics.entropy_ratio),
        ]
    )


def _build_layer_rows(
    source: str,
    view: str,
    sid_field: str,
    codebook: Sequence[int],
    input_rows: int,
    evaluated_items: int,
    invalid_pair_rows: int,
    layer_metrics: Sequence[SidLayerQualityMetrics],
) -> List["OrderedDict[str, object]"]:
    """Build ordered long-format per-layer metric rows for one view."""
    rows = []
    for metrics in layer_metrics:
        rows.append(
            OrderedDict(
                [
                    ("source", source),
                    ("view", view),
                    ("sid_field", sid_field),
                    ("codebook", ",".join(map(str, codebook))),
                    ("input_rows", input_rows),
                    ("evaluated_items", evaluated_items),
                    ("invalid_pair_rows", invalid_pair_rows),
                    ("layer", metrics.layer),
                    ("codebook_size", metrics.codebook_size),
                    ("coverage", metrics.coverage),
                    ("dead_codes", metrics.dead_codes),
                    ("perplexity", metrics.perplexity),
                ]
            )
        )
    return rows


def _rows_to_arrow_columns(
    rows: Sequence["OrderedDict[str, object]"],
) -> "OrderedDict[str, pa.Array]":
    """Convert ordered row dictionaries to Arrow writer columns."""
    if not rows:
        raise ValueError("cannot write an empty SID quality table.")
    keys = tuple(rows[0].keys())
    if any(tuple(row.keys()) != keys for row in rows[1:]):
        raise ValueError("all SID quality rows must use the same ordered schema.")
    return OrderedDict((key, pa.array([row[key] for row in rows])) for key in keys)


def _write_rows(
    output_path: str,
    writer_type: str,
    quota_name: str,
    rows: Sequence["OrderedDict[str, object]"],
) -> None:
    """Write one quality table and always close its writer."""
    writer = create_writer(output_path, writer_type, quota_name=quota_name)
    try:
        writer.write(_rows_to_arrow_columns(rows))
    finally:
        writer.close()


def _log_result(view: str, result: SidQualityResult) -> None:
    """Log one quality result without coupling numerical utilities to logging."""
    logger.info(f"===== SID quality stats ({view}) =====")
    for field, value in result.metrics.__dict__.items():
        logger.info(f"{field} = {value}")
    for metrics in result.layer_metrics:
        logger.info(
            f"layer {metrics.layer}: coverage={metrics.coverage:.4f} "
            f"dead_codes={metrics.dead_codes} "
            f"perplexity={metrics.perplexity:.4f}"
        )
    if result.top_sids is not None:
        logger.info(f"top SIDs by frequency ({view}) = {result.top_sids}")


if __name__ == "__main__":
    main()
