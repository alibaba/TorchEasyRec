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

"""NumPy collision-resolution core for semantic IDs."""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

_MASK64 = (1 << 64) - 1
_SEED = 2026
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15
_FALLBACK_POLICIES = frozenset({"error", "drop", "keep_original"})
_GROUPING_ROW_CHUNK = 1_000_000


@dataclass(frozen=True)
class CollisionResolutionConfig:
    """Configuration for within-band SID collision resolution.

    Args:
        layer_sizes: Cardinality of each SID layer.
        capacity: Maximum number of retained items in one SID bucket.
        fallback_policy: Action when no candidate has a free slot. Supported
            values are ``error``, ``drop``, and ``keep_original``.
    """

    layer_sizes: tuple[int, ...]
    capacity: int
    fallback_policy: str

    def __post_init__(self) -> None:
        if not self.layer_sizes:
            raise ValueError("layer_sizes must contain at least one layer.")
        if any(size <= 0 for size in self.layer_sizes):
            raise ValueError("all layer_sizes must be positive.")
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}.")
        if self.fallback_policy not in _FALLBACK_POLICIES:
            raise ValueError(
                "fallback_policy must be one of error, drop, or keep_original, "
                f"got {self.fallback_policy!r}."
            )


@dataclass(frozen=True)
class CollisionPlan:
    """Compact grouping plan consumed by collision resolution.

    The plan stores full-length last-code, dense bucket-ID, and index arrays,
    plus only the overflow-aligned identity and ordering data needed by
    candidate loading. It deliberately does not retain the full input code
    matrix.
    """

    item_count: int
    original_last_codes: np.ndarray
    origin_bucket_ids: np.ndarray
    initial_slot_indices: np.ndarray
    occupied_sid_keys: np.ndarray
    raw_bucket_counts: np.ndarray
    overflow_rows: np.ndarray
    overflow_item_ids: np.ndarray
    overflow_band_bases: np.ndarray
    overflow_origin_last_codes: np.ndarray
    config: CollisionResolutionConfig

    @property
    def origin_sid_keys(self) -> np.ndarray:
        """Return flattened original SID keys aligned with input rows."""
        return self.occupied_sid_keys[self.origin_bucket_ids]


@dataclass(frozen=True)
class CollisionResolutionStats:
    """Summary statistics for a collision-resolution run."""

    total_items: int
    raw_collision_buckets: int
    final_collision_buckets: int
    relocated_count: int
    unresolved_count: int
    max_final_bucket_size: int

    @property
    def reassigned_count(self) -> int:
        """Return the legacy name for ``relocated_count``."""
        return self.relocated_count

    @property
    def unassigned_count(self) -> int:
        """Return the legacy name for ``unresolved_count``."""
        return self.unresolved_count

    def to_output_dict(self) -> Dict[str, int]:
        """Return diagnostics using the existing output column names."""
        return {
            "total_items": self.total_items,
            "raw_collision_buckets": self.raw_collision_buckets,
            "final_collision_buckets": self.final_collision_buckets,
            "reassigned_count": self.relocated_count,
            "unassigned_count": self.unresolved_count,
            "max_final_bucket_size": self.max_final_bucket_size,
        }


@dataclass(frozen=True)
class CollisionResolutionResult:
    """Resolved last codes, indexes, row retention, and diagnostics."""

    resolved_last_codes: np.ndarray
    slot_indices: np.ndarray
    retained_mask: Optional[np.ndarray]
    unresolved_rows: np.ndarray
    final_occupied_sid_keys: np.ndarray
    final_bucket_counts: np.ndarray
    slots_match_final_sid_keys: bool
    grouping_collected: bool
    stats: CollisionResolutionStats


@dataclass(frozen=True)
class CodebookItemGrouping:
    """CSR-like row grouping for sorted flattened SID keys.

    ``row_order[offsets[i]:offsets[i + 1]]`` contains the original row indices
    for ``sid_keys[i]``. Rows within a bucket follow their one-based slot order.

    Args:
        sid_keys: Sorted flattened SID keys, one per occupied bucket.
        counts: Item counts aligned with ``sid_keys``.
        row_order: Original row indices flattened in SID and slot order.
    """

    sid_keys: np.ndarray
    counts: np.ndarray
    row_order: np.ndarray

    @property
    def offsets(self) -> np.ndarray:
        """Return CSR offsets into ``row_order`` for every SID bucket."""
        offsets = np.empty(self.counts.shape[0] + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(self.counts, dtype=np.int64, out=offsets[1:])
        return offsets

    @property
    def representative_rows(self) -> np.ndarray:
        """Return the first original row index in every occupied bucket."""
        return self.row_order[self.offsets[:-1]]


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Hash a uint64 array with the collision tool's fixed SplitMix64 seed."""
    with np.errstate(over="ignore"):
        mixed = values.astype(np.uint64) + np.uint64(
            (_SEED * _SPLITMIX_INCREMENT) & _MASK64
        )
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return mixed ^ (mixed >> np.uint64(31))


def stable_order_hash(item_ids: np.ndarray) -> np.ndarray:
    """Return the existing order-independent uint64 tie-break hashes.

    Integer IDs are converted directly to uint64. Other ID types retain the
    existing ``pandas.util.hash_array`` folding step before SplitMix64.

    Args:
        item_ids: One-dimensional item ID array.

    Returns:
        A uint64 hash array aligned with ``item_ids``.

    Raises:
        ValueError: If ``item_ids`` is not one-dimensional.
    """
    item_ids = np.asarray(item_ids)
    if item_ids.ndim != 1:
        raise ValueError(f"item_ids must be 1-D, got shape {item_ids.shape}.")
    if np.issubdtype(item_ids.dtype, np.integer):
        base = item_ids.astype(np.uint64)
    else:
        import pandas as pd

        base = pd.util.hash_array(np.asarray(item_ids, dtype=object))
    return _splitmix64(base)


def _band_ids(codes: np.ndarray, layer_sizes: tuple[int, ...]) -> np.ndarray:
    """Return a dense integer ID for each prefix band."""
    row_count, layer_count = codes.shape
    if layer_count == 1:
        return np.zeros(row_count, dtype=np.int64)
    keys = codes[:, 0].astype(np.int64)
    for layer in range(1, layer_count - 1):
        keys *= layer_sizes[layer]
        keys += codes[:, layer]
    return np.unique(keys, return_inverse=True)[1].astype(np.int64, copy=False)


def _within_bucket_rank(
    band_ids: np.ndarray, last_codes: np.ndarray, order_hashes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-row rank, bucket IDs, sorted rows, representatives, and counts."""
    row_count = band_ids.shape[0]
    if row_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy(), empty.copy()
    sorted_rows = np.lexsort((order_hashes, last_codes, band_ids))
    sorted_bands = band_ids[sorted_rows]
    sorted_last_codes = last_codes[sorted_rows]
    is_first = np.empty(row_count, dtype=bool)
    is_first[0] = True
    is_first[1:] = (sorted_bands[1:] != sorted_bands[:-1]) | (
        sorted_last_codes[1:] != sorted_last_codes[:-1]
    )
    first_sorted_rows = np.flatnonzero(is_first)
    bucket_counts = np.diff(np.append(first_sorted_rows, row_count))
    del sorted_bands, sorted_last_codes, is_first
    bucket_ids = np.empty(row_count, dtype=np.int64)
    bucket_ranks = np.empty(row_count, dtype=np.int64)
    for start in range(0, row_count, _GROUPING_ROW_CHUNK):
        end = min(start + _GROUPING_ROW_CHUNK, row_count)
        first_bucket = int(np.searchsorted(first_sorted_rows, start, side="right") - 1)
        last_bucket = int(np.searchsorted(first_sorted_rows, end - 1, side="right") - 1)
        local_starts = first_sorted_rows[first_bucket : last_bucket + 1].copy()
        local_starts[0] = start
        local_counts = np.diff(np.append(local_starts, end))
        sorted_bucket_ids = np.repeat(
            np.arange(first_bucket, last_bucket + 1, dtype=np.int64),
            local_counts,
        )
        rows = sorted_rows[start:end]
        bucket_ids[rows] = sorted_bucket_ids
        bucket_ranks[rows] = (
            np.arange(start, end, dtype=np.int64) - first_sorted_rows[sorted_bucket_ids]
        )
    return (
        bucket_ranks,
        bucket_ids,
        sorted_rows,
        sorted_rows[first_sorted_rows],
        bucket_counts,
    )


def prepare_collision_plan(
    item_ids: np.ndarray,
    codes: np.ndarray,
    config: CollisionResolutionConfig,
) -> CollisionPlan:
    """Group SID buckets and identify overflow rows in stable processing order.

    Args:
        item_ids: One-dimensional item IDs aligned with ``codes``.
        codes: Integer SID matrix with shape ``(N, number_of_layers)``.
        config: Collision capacity, shape, and fallback configuration.

    Returns:
        A compact plan for candidate loading and collision resolution.

    Raises:
        ValueError: If input dimensions, row counts, or layer counts disagree.
    """
    item_ids = np.asarray(item_ids)
    codes = np.asarray(codes)
    if item_ids.ndim != 1:
        raise ValueError(f"item_ids must be 1-D, got shape {item_ids.shape}.")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}.")
    if item_ids.shape[0] != codes.shape[0]:
        raise ValueError(
            "item_ids and codes must have the same row count, got "
            f"{item_ids.shape[0]} and {codes.shape[0]}."
        )
    layer_count = codes.shape[1]
    if layer_count != len(config.layer_sizes):
        raise ValueError(
            f"codes have {layer_count} layers but layer_sizes has "
            f"{len(config.layer_sizes)}."
        )

    item_count = codes.shape[0]
    original_last_codes = codes[:, -1].astype(np.int64, copy=False)
    band_ids = _band_ids(codes, config.layer_sizes)
    order_hashes = stable_order_hash(item_ids)
    (
        bucket_ranks,
        origin_bucket_ids,
        sorted_rows,
        representative_rows,
        raw_bucket_counts,
    ) = _within_bucket_rank(band_ids, original_last_codes, order_hashes)

    overflow_rows = sorted_rows[bucket_ranks[sorted_rows] >= config.capacity]
    last_size = config.layer_sizes[-1]
    occupied_sid_keys = (
        band_ids[representative_rows] * last_size
        + original_last_codes[representative_rows]
    )
    overflow_band_bases = band_ids[overflow_rows] * last_size

    return CollisionPlan(
        item_count=item_count,
        original_last_codes=original_last_codes,
        origin_bucket_ids=origin_bucket_ids,
        initial_slot_indices=bucket_ranks + 1,
        occupied_sid_keys=occupied_sid_keys,
        raw_bucket_counts=raw_bucket_counts,
        overflow_rows=overflow_rows,
        overflow_item_ids=item_ids[overflow_rows].copy(),
        overflow_band_bases=overflow_band_bases,
        overflow_origin_last_codes=original_last_codes[overflow_rows].copy(),
        config=config,
    )


def generate_random_candidate_last_codes(
    item_ids: np.ndarray, num: int, last_size: int
) -> np.ndarray:
    """Generate the existing deterministic full-space random candidate draws.

    Sampling is with replacement and includes each item's original code because
    that code is not an input to this function. The placement step skips an
    origin draw without replacing it, preserving current behavior.

    Args:
        item_ids: One-dimensional IDs for the overflow rows.
        num: Requested number of raw draws per row. As in the current strategy,
            this is capped at ``last_size - 1``.
        last_size: Cardinality of the last SID layer.

    Returns:
        An ``(len(item_ids), min(num, last_size - 1))`` int64 matrix.

    Raises:
        ValueError: If ``num`` is negative or ``last_size`` is smaller than two.
    """
    if last_size < 2:
        raise ValueError("random candidates require last_size >= 2.")
    if num < 0:
        raise ValueError(f"num must be >= 0, got {num}.")
    effective_num = min(num, last_size - 1)
    hashes = stable_order_hash(item_ids)
    draw_indices = np.arange(effective_num, dtype=np.uint64)
    with np.errstate(over="ignore"):
        mixed = _splitmix64(
            hashes[:, None] + draw_indices[None, :] * np.uint64(_SPLITMIX_INCREMENT)
        )
    return (mixed % np.uint64(last_size)).astype(np.int64)


def resolve_sid_collisions(
    plan: CollisionPlan,
    candidate_last_codes: np.ndarray,
    *,
    collect_grouping: bool = True,
) -> CollisionResolutionResult:
    """Greedily relocate overflow rows to their first free candidate slot.

    Candidate rows must be aligned with ``plan.overflow_rows``. Candidate values
    intentionally receive no range validation so negative and out-of-range
    values retain the existing key and modulo behavior.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.
        candidate_last_codes: Ordered candidate matrix with shape ``(M, K)``,
            where ``M`` equals the number of overflow rows.
        collect_grouping: Whether to collect final SID keys and bucket counts
            for :func:`build_resolved_item_grouping`. Disable this in
            diagnostics-only runs to avoid the grouping arrays and sorts.

    Returns:
        Resolved last-layer codes, slot indices, retention mask, unresolved row
        indices, and diagnostics.

    Raises:
        RuntimeError: If a row is unresolved under the ``error`` policy.
        TypeError: If candidate codes do not use an integer dtype.
        ValueError: If candidates are not a row-aligned two-dimensional matrix.
    """
    candidates = np.asarray(candidate_last_codes)
    if candidates.ndim != 2:
        raise ValueError(
            f"candidate_last_codes must be 2-D, got shape {candidates.shape}."
        )
    overflow_count = plan.overflow_rows.shape[0]
    if candidates.shape[0] != overflow_count:
        raise ValueError(
            "candidate_last_codes must be row-aligned with overflow_rows, got "
            f"{candidates.shape[0]} candidate rows for {overflow_count} "
            "overflow rows."
        )
    if not np.issubdtype(candidates.dtype, np.integer):
        raise TypeError("candidate_last_codes must use an integer dtype.")

    capacity = plan.config.capacity
    initial_counts = np.minimum(plan.raw_bucket_counts, capacity)
    slot_counts = dict(zip(plan.occupied_sid_keys.tolist(), initial_counts.tolist()))
    resolved_last_codes = plan.original_last_codes.copy()
    slot_indices = plan.initial_slot_indices.copy()
    relocated_rows = []
    relocated_last_codes = []
    relocated_indices = []
    unresolved_rows = []
    slots_match_final_sid_keys = True
    get_slot_count = slot_counts.get
    last_size = plan.config.layer_sizes[-1]

    overflow_rows = plan.overflow_rows.tolist()
    band_bases = plan.overflow_band_bases.tolist()
    origin_last_codes = plan.overflow_origin_last_codes.tolist()
    for position, row in enumerate(overflow_rows):
        band_base = band_bases[position]
        origin_last_code = origin_last_codes[position]
        for candidate in candidates[position].tolist():
            if candidate == origin_last_code:
                continue
            destination_key = band_base + candidate
            destination_count = get_slot_count(destination_key, 0)
            if destination_count < capacity:
                slot_counts[destination_key] = destination_count + 1
                relocated_rows.append(row)
                relocated_last_codes.append(destination_key % last_size)
                relocated_indices.append(destination_count + 1)
                if candidate < 0 or candidate >= last_size:
                    slots_match_final_sid_keys = False
                break
        else:
            unresolved_rows.append(row)

    if relocated_rows:
        resolved_last_codes[relocated_rows] = relocated_last_codes
        slot_indices[relocated_rows] = relocated_indices

    retained_mask = None
    fallback_policy = plan.config.fallback_policy
    if unresolved_rows and fallback_policy == "error":
        preview = ",".join(str(row) for row in unresolved_rows[:10])
        raise RuntimeError(
            f"{len(unresolved_rows)} items could not be placed within capacity; "
            f"first unresolved row indices: {preview}"
        )
    if unresolved_rows and fallback_policy == "drop":
        retained_mask = np.ones(plan.item_count, dtype=bool)
        retained_mask[unresolved_rows] = False
    elif unresolved_rows and fallback_policy == "keep_original":
        for row in unresolved_rows:
            origin_key = int(plan.occupied_sid_keys[plan.origin_bucket_ids[row]])
            origin_count = get_slot_count(origin_key, 0) + 1
            slot_counts[origin_key] = origin_count
            slot_indices[row] = origin_count

    final_occupied_sid_keys = np.empty(0, dtype=np.int64)
    final_bucket_counts = np.empty(0, dtype=np.int64)
    if collect_grouping:
        occupancy_counts = np.fromiter(slot_counts.values(), dtype=np.int64)
        final_collision_buckets = int((occupancy_counts > capacity).sum())
        max_final_bucket_size = (
            int(occupancy_counts.max()) if occupancy_counts.size else 0
        )
        if slots_match_final_sid_keys:
            final_occupied_sid_keys = np.fromiter(slot_counts, dtype=np.int64)
            occupancy_order = np.argsort(final_occupied_sid_keys, kind="stable")
            final_occupied_sid_keys = final_occupied_sid_keys[occupancy_order]
            final_bucket_counts = occupancy_counts[occupancy_order]
        else:
            retained_rows = (
                np.flatnonzero(retained_mask)
                if retained_mask is not None
                else np.arange(plan.item_count, dtype=np.int64)
            )
            origin_keys = plan.occupied_sid_keys[plan.origin_bucket_ids[retained_rows]]
            final_sid_keys = (
                origin_keys
                - plan.original_last_codes[retained_rows]
                + resolved_last_codes[retained_rows]
            )
            final_occupied_sid_keys, final_bucket_counts = np.unique(
                final_sid_keys, return_counts=True
            )
    else:
        final_collision_buckets = 0
        max_final_bucket_size = 0
        for count in slot_counts.values():
            final_collision_buckets += count > capacity
            max_final_bucket_size = max(max_final_bucket_size, count)
    unresolved_array = np.asarray(unresolved_rows, dtype=np.int64)
    stats = CollisionResolutionStats(
        total_items=plan.item_count,
        raw_collision_buckets=int((plan.raw_bucket_counts > capacity).sum()),
        final_collision_buckets=final_collision_buckets,
        relocated_count=len(relocated_rows),
        unresolved_count=len(unresolved_rows),
        max_final_bucket_size=max_final_bucket_size,
    )
    return CollisionResolutionResult(
        resolved_last_codes=resolved_last_codes,
        slot_indices=slot_indices,
        retained_mask=retained_mask,
        unresolved_rows=unresolved_array,
        final_occupied_sid_keys=final_occupied_sid_keys,
        final_bucket_counts=final_bucket_counts,
        slots_match_final_sid_keys=slots_match_final_sid_keys,
        grouping_collected=collect_grouping,
        stats=stats,
    )


def _scatter_item_grouping(
    sid_keys: np.ndarray,
    counts: np.ndarray,
    row_count: int,
    chunks: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> CodebookItemGrouping:
    """Build a CSR row order from bounded row, bucket-ID, and slot chunks."""
    grouping = CodebookItemGrouping(
        sid_keys=sid_keys,
        counts=counts,
        row_order=np.empty(row_count, dtype=np.int64),
    )
    offsets = grouping.offsets
    if int(offsets[-1]) != row_count:
        raise RuntimeError(
            "bucket counts do not match retained row count: "
            f"{int(offsets[-1])} vs {row_count}."
        )
    written_count = 0
    for rows, bucket_ids, slot_indices in chunks:
        if rows.shape != bucket_ids.shape or rows.shape != slot_indices.shape:
            raise RuntimeError("grouping row, bucket, and slot chunks must align.")
        if rows.size == 0:
            continue
        if np.any(bucket_ids < 0) or np.any(bucket_ids >= counts.shape[0]):
            raise RuntimeError("grouping bucket IDs are outside the SID key array.")
        positions = offsets[bucket_ids] + slot_indices - 1
        if np.any(slot_indices < 1) or np.any(positions >= offsets[bucket_ids + 1]):
            raise RuntimeError("slot indices are outside their SID bucket bounds.")
        grouping.row_order[positions] = rows
        written_count += rows.shape[0]
    if written_count != row_count:
        raise RuntimeError(
            f"grouping chunks contain {written_count} rows, expected {row_count}."
        )
    return grouping


def build_original_item_grouping(plan: CollisionPlan) -> CodebookItemGrouping:
    """Group all original rows by SID and initial one-based slot index.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.

    Returns:
        Sorted original SID keys, bucket counts, and grouped original row order.
    """

    def row_chunks() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for start in range(0, plan.item_count, _GROUPING_ROW_CHUNK):
            end = min(start + _GROUPING_ROW_CHUNK, plan.item_count)
            yield (
                np.arange(start, end, dtype=np.int64),
                plan.origin_bucket_ids[start:end],
                plan.initial_slot_indices[start:end],
            )

    return _scatter_item_grouping(
        plan.occupied_sid_keys,
        plan.raw_bucket_counts,
        plan.item_count,
        row_chunks(),
    )


def build_resolved_item_grouping(
    plan: CollisionPlan, result: CollisionResolutionResult
) -> CodebookItemGrouping:
    """Group retained rows by emitted SID and final one-based slot index.

    Canonical in-range candidates use a linear scatter from final slot indices.
    Malformed candidates can make internal occupancy keys differ from emitted SID
    keys because the legacy resolver applies modulo only to the emitted last code.
    That compatibility path groups by emitted keys and reported slots instead.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.
        result: Collision output from :func:`resolve_sid_collisions`.

    Returns:
        Sorted final SID keys, bucket counts, and grouped retained row order.

    Raises:
        RuntimeError: If grouping metadata was not collected or result bucket
            keys do not cover an emitted SID.
    """
    if not result.grouping_collected:
        raise RuntimeError(
            "resolved item grouping metadata was not collected; call "
            "resolve_sid_collisions with collect_grouping=True."
        )
    if not result.slots_match_final_sid_keys:
        rows = (
            np.flatnonzero(result.retained_mask)
            if result.retained_mask is not None
            else np.arange(plan.item_count, dtype=np.int64)
        )
        origin_keys = plan.occupied_sid_keys[plan.origin_bucket_ids[rows]]
        emitted_sid_keys = (
            origin_keys
            - plan.original_last_codes[rows]
            + result.resolved_last_codes[rows]
        )
        row_order = np.lexsort((rows, result.slot_indices[rows], emitted_sid_keys))
        return CodebookItemGrouping(
            sid_keys=result.final_occupied_sid_keys,
            counts=result.final_bucket_counts,
            row_order=rows[row_order],
        )

    def row_chunks() -> Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        for start in range(0, plan.item_count, _GROUPING_ROW_CHUNK):
            end = min(start + _GROUPING_ROW_CHUNK, plan.item_count)
            if result.retained_mask is None:
                rows = np.arange(start, end, dtype=np.int64)
            else:
                rows = np.flatnonzero(result.retained_mask[start:end])
                rows += start
            if rows.size == 0:
                continue
            emitted_sid_keys = plan.occupied_sid_keys[plan.origin_bucket_ids[rows]]
            emitted_sid_keys -= plan.original_last_codes[rows]
            emitted_sid_keys += result.resolved_last_codes[rows]
            bucket_ids = np.searchsorted(
                result.final_occupied_sid_keys, emitted_sid_keys
            )
            in_bounds = bucket_ids < result.final_occupied_sid_keys.shape[0]
            if not np.all(in_bounds) or not np.all(
                result.final_occupied_sid_keys[bucket_ids] == emitted_sid_keys
            ):
                raise RuntimeError("final bucket keys do not cover every emitted SID.")
            yield rows, bucket_ids, result.slot_indices[rows]

    return _scatter_item_grouping(
        result.final_occupied_sid_keys,
        result.final_bucket_counts,
        int(result.final_bucket_counts.sum()),
        row_chunks(),
    )
