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

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import torch

_INT64_MAX = int(np.iinfo(np.int64).max)


@dataclass(frozen=True)
class SidQualityMetrics:
    """Global quality metrics for a collection of semantic IDs.

    Attributes:
        total: Number of evaluated items.
        unique_sid: Number of occupied SID buckets.
        no_collision_rate: Ratio of occupied SID buckets to items.
        uniquely_identified_item_rate: Ratio of items in singleton buckets.
        max_collision: Size of the largest occupied SID bucket.
        gini: Gini coefficient over occupied bucket sizes.
        entropy: Shannon entropy over occupied bucket probabilities.
        max_entropy: Logarithm of the full codebook capacity.
        entropy_ratio: Entropy divided by maximum entropy.
    """

    total: int
    unique_sid: int
    no_collision_rate: float
    uniquely_identified_item_rate: float
    max_collision: int
    gini: float
    entropy: float
    max_entropy: float
    entropy_ratio: float


@dataclass(frozen=True)
class SidLayerQualityMetrics:
    """Code usage metrics for one semantic-ID layer.

    Attributes:
        layer: Zero-based layer index.
        codebook_size: Number of available codes in the layer.
        coverage: Ratio of codes observed at least once.
        dead_codes: Number of codes that were not observed.
        perplexity: Exponential Shannon entropy of code usage.
    """

    layer: int
    codebook_size: int
    coverage: float
    dead_codes: int
    perplexity: float


@dataclass(frozen=True)
class SidQualityResult:
    """Final global, layer, and optional top-SID quality results.

    Attributes:
        metrics: Global SID quality metrics.
        layer_metrics: Metrics in ascending layer order.
        top_sids: Most frequent ``(comma-delimited SID, count)`` pairs.
    """

    metrics: SidQualityMetrics
    layer_metrics: Tuple[SidLayerQualityMetrics, ...]
    top_sids: Optional[Tuple[Tuple[str, int], ...]] = None


def compute_gini(counts: npt.ArrayLike) -> float:
    """Compute the Gini coefficient of occupied SID bucket sizes.

    Args:
        counts: Per-SID occurrence counts.

    Returns:
        Gini in ``[0, 1)``. Empty or all-zero input returns zero.
    """
    count_array = np.asarray(counts, dtype=np.int64)
    if count_array.size == 0:
        return 0.0
    values, frequencies = np.unique(count_array, return_counts=True)
    num_buckets = int(frequencies.sum())
    weighted_sum = int(np.dot(values, frequencies))
    if weighted_sum == 0:
        return 0.0

    total_dot_product = 0
    cumulative_frequency = 0
    for value, frequency in zip(values.tolist(), frequencies.tolist()):
        first_term = num_buckets - cumulative_frequency
        last_term = num_buckets - cumulative_frequency - frequency + 1
        rank_sum = (first_term + last_term) * frequency // 2
        total_dot_product += value * rank_sum
        cumulative_frequency += frequency
    return float(
        (num_buckets + 1) / num_buckets
        - (2 * total_dot_product / (num_buckets * weighted_sum))
    )


def compute_entropy(counts: npt.ArrayLike) -> float:
    """Compute Shannon entropy in nats for a frequency distribution.

    Args:
        counts: Occurrence counts. Zero-count entries are allowed.

    Returns:
        Shannon entropy using the natural logarithm.
    """
    count_tensor = torch.as_tensor(counts, dtype=torch.float64)
    total = count_tensor.sum()
    if total == 0:
        return 0.0
    return float(torch.special.entr(count_tensor / total).sum())


def valid_code_rows(codes: np.ndarray, codebook: Sequence[int]) -> np.ndarray:
    """Return a mask selecting rows inside every layer's code range.

    Args:
        codes: Two-dimensional integer code matrix.
        codebook: Positive codebook size for each layer.

    Returns:
        Boolean mask with one value per input row.

    Raises:
        ValueError: If the codebook or code matrix is invalid.
    """
    codebook_array = _validate_codebook(codebook)
    code_array = _validate_code_matrix(codes, len(codebook_array))
    return _code_rows_in_range(code_array, codebook_array)


class SidQualityAccumulator:
    """Accumulate exact global semantic-ID statistics across batches.

    Args:
        codebook: Positive codebook size for each SID layer.
        top_sids: Optional number of most frequent SIDs to retain.
    """

    def __init__(self, codebook: Sequence[int], top_sids: Optional[int] = None) -> None:
        codebook_array = _validate_codebook(codebook)
        capacity = math.prod(codebook_array.tolist())
        if capacity > _INT64_MAX:
            raise ValueError(
                f"codebook capacity (product = {capacity}) exceeds int64; "
                "collision analysis is not supported at that scale."
            )
        if top_sids is not None and top_sids <= 0:
            raise ValueError(f"top_sids must be positive, got {top_sids}.")

        self._codebook = codebook_array
        self._capacity = capacity
        self._top_sids = top_sids
        self._radix = np.asarray(
            [
                math.prod(codebook_array[layer + 1 :].tolist())
                for layer in range(len(codebook_array))
            ],
            dtype=np.int64,
        )
        self._id_chunks: List[np.ndarray] = []
        self._layer_hist = [
            np.zeros(int(size), dtype=np.int64) for size in codebook_array
        ]
        self._total = 0
        self._finalized = False

    @property
    def total(self) -> int:
        """Number of codes accumulated so far."""
        return self._total

    def update(self, codes: np.ndarray, *, assume_in_range: bool = False) -> None:
        """Add a batch of valid SID codes.

        Args:
            codes: Two-dimensional integer matrix whose values are in range.
            assume_in_range: Skip the range scan when the caller has already
                applied :func:`valid_code_rows`. Shape and dtype are still
                validated.

        Raises:
            ValueError: If called after finalization or with invalid codes.
        """
        if self._finalized:
            raise ValueError("cannot update a finalized SID quality accumulator.")
        code_array = _validate_code_matrix(codes, len(self._codebook))
        if (
            not assume_in_range
            and not _code_rows_in_range(code_array, self._codebook).all()
        ):
            raise ValueError("codes contain values outside the configured codebook.")
        if code_array.shape[0] == 0:
            return

        code_array = code_array.astype(np.int64, copy=False)
        self._id_chunks.append(code_array @ self._radix)
        for layer, size in enumerate(self._codebook):
            self._layer_hist[layer] += np.bincount(
                code_array[:, layer], minlength=int(size)
            )
        self._total += code_array.shape[0]

    def finalize(self) -> SidQualityResult:
        """Compute final metrics and release accumulated mixed-radix chunks.

        Returns:
            Exact global, layer, and optional top-SID results.

        Raises:
            ValueError: If no codes were added or finalization is repeated.
        """
        if self._finalized:
            raise ValueError("SID quality accumulator has already been finalized.")
        self._finalized = True
        if not self._id_chunks:
            raise ValueError("no valid SID codes were added; nothing to report.")

        sid_id_array = (
            self._id_chunks[0]
            if len(self._id_chunks) == 1
            else np.concatenate(self._id_chunks)
        )
        self._id_chunks = []
        sid_ids, counts = np.unique(sid_id_array, return_counts=True)
        entropy = compute_entropy(counts)
        max_entropy = math.log(self._capacity)
        unique_sid = len(sid_ids)
        metrics = SidQualityMetrics(
            total=self._total,
            unique_sid=unique_sid,
            no_collision_rate=unique_sid / self._total,
            uniquely_identified_item_rate=int((counts == 1).sum()) / self._total,
            max_collision=int(counts.max()),
            gini=compute_gini(counts),
            entropy=entropy,
            max_entropy=max_entropy,
            entropy_ratio=(entropy / max_entropy if max_entropy else float("nan")),
        )
        layer_metrics = tuple(
            _compute_layer_metrics(layer, histogram)
            for layer, histogram in enumerate(self._layer_hist)
        )

        top_sids: Optional[Tuple[Tuple[str, int], ...]] = None
        if self._top_sids is not None:
            order = np.argsort(-counts, kind="stable")[: self._top_sids]
            top_codes = (sid_ids[order][:, None] // self._radix) % self._codebook
            top_sids = tuple(
                (",".join(map(str, code)), int(count))
                for code, count in zip(top_codes.tolist(), counts[order].tolist())
            )
        return SidQualityResult(
            metrics=metrics,
            layer_metrics=layer_metrics,
            top_sids=top_sids,
        )


def compare_sid_quality(
    before: SidQualityResult, after: SidQualityResult
) -> SidQualityResult:
    """Subtract before quality metrics from after quality metrics.

    Args:
        before: Quality result for original SIDs.
        after: Quality result for resolved SIDs over the same item cohort.

    Returns:
        A result whose metrics are ``after - before``. Layer identity and
        codebook size remain metadata, and top SIDs are omitted.

    Raises:
        ValueError: If results do not describe compatible cohorts/codebooks.
    """
    if before.metrics.total != after.metrics.total:
        raise ValueError("before and after results must use the same item cohort.")
    if len(before.layer_metrics) != len(after.layer_metrics):
        raise ValueError("before and after results have different layer counts.")

    layer_deltas = []
    for before_layer, after_layer in zip(before.layer_metrics, after.layer_metrics):
        if (
            before_layer.layer != after_layer.layer
            or before_layer.codebook_size != after_layer.codebook_size
        ):
            raise ValueError("before and after results use different codebooks.")
        layer_deltas.append(
            SidLayerQualityMetrics(
                layer=after_layer.layer,
                codebook_size=after_layer.codebook_size,
                coverage=after_layer.coverage - before_layer.coverage,
                dead_codes=after_layer.dead_codes - before_layer.dead_codes,
                perplexity=after_layer.perplexity - before_layer.perplexity,
            )
        )

    before_metrics = before.metrics
    after_metrics = after.metrics
    return SidQualityResult(
        metrics=SidQualityMetrics(
            total=after_metrics.total - before_metrics.total,
            unique_sid=after_metrics.unique_sid - before_metrics.unique_sid,
            no_collision_rate=(
                after_metrics.no_collision_rate - before_metrics.no_collision_rate
            ),
            uniquely_identified_item_rate=(
                after_metrics.uniquely_identified_item_rate
                - before_metrics.uniquely_identified_item_rate
            ),
            max_collision=(after_metrics.max_collision - before_metrics.max_collision),
            gini=after_metrics.gini - before_metrics.gini,
            entropy=after_metrics.entropy - before_metrics.entropy,
            max_entropy=after_metrics.max_entropy - before_metrics.max_entropy,
            entropy_ratio=(after_metrics.entropy_ratio - before_metrics.entropy_ratio),
        ),
        layer_metrics=tuple(layer_deltas),
    )


def _validate_codebook(codebook: Sequence[int]) -> np.ndarray:
    """Validate and normalize per-layer codebook sizes."""
    codebook_array = np.asarray(codebook)
    if codebook_array.ndim != 1 or codebook_array.size == 0:
        raise ValueError("codebook must contain at least one layer.")
    if codebook_array.dtype.kind not in "iu":
        raise ValueError("codebook sizes must be integers.")
    if (codebook_array < 1).any():
        raise ValueError("codebook sizes must be positive integers.")
    if codebook_array.dtype.kind == "u" and (codebook_array > _INT64_MAX).any():
        raise ValueError("codebook sizes must fit int64.")
    return codebook_array.astype(np.int64, copy=False)


def _validate_code_matrix(codes: np.ndarray, n_layers: int) -> np.ndarray:
    """Validate a two-dimensional integer SID code matrix."""
    code_array = np.asarray(codes)
    if code_array.ndim != 2 or code_array.shape[1] != n_layers:
        raise ValueError(
            f"codes must have shape (N, {n_layers}), got {code_array.shape}."
        )
    if code_array.dtype.kind not in "iu":
        raise ValueError("codes must use an integer dtype.")
    if code_array.dtype.kind == "u" and (code_array > _INT64_MAX).any():
        raise ValueError("codes must fit int64.")
    return code_array


def _code_rows_in_range(
    code_array: np.ndarray, codebook_array: np.ndarray
) -> np.ndarray:
    """Return a range mask for already validated code and codebook arrays."""
    return ((code_array >= 0) & (code_array < codebook_array)).all(axis=1)


def _compute_layer_metrics(layer: int, histogram: np.ndarray) -> SidLayerQualityMetrics:
    """Compute usage metrics from one layer histogram."""
    nonzero = int(np.count_nonzero(histogram))
    codebook_size = len(histogram)
    return SidLayerQualityMetrics(
        layer=layer,
        codebook_size=codebook_size,
        coverage=nonzero / codebook_size,
        dead_codes=codebook_size - nonzero,
        perplexity=math.exp(compute_entropy(histogram)),
    )
