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

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger

_INT64_MAX = int(np.iinfo(np.int64).max)
_MASK64 = (1 << 64) - 1
_SEED = 2026
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class CollisionResolutionConfig:
    """Configuration for within-prefix SID collision resolution.

    ``layer_sizes`` also fixes the SID key layout: a row's bucket key is the
    mixed-radix number of its codes, its prefix id is that number without the
    last code, and its offset codes shift each layer into one contiguous
    vocabulary.

    Overflow items that cannot be placed within capacity stay over capacity,
    so every input item is preserved in the output.

    Args:
        layer_sizes: Cardinality of each SID layer.
        capacity: Maximum number of retained items in one SID bucket.
    """

    layer_sizes: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        if not self.layer_sizes:
            raise ValueError("layer_sizes must contain at least one layer.")
        if any(size <= 0 for size in self.layer_sizes):
            raise ValueError("all layer_sizes must be positive.")
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}.")
        if math.prod(self.layer_sizes) > _INT64_MAX:
            raise ValueError(
                f"prod(layer_sizes) exceeds int64 range: {self.layer_sizes}; "
                "bucket keys would overflow."
            )

    def prefix_ids(self, codes: np.ndarray) -> np.ndarray:
        """Return the mixed-radix prefix id of every SID row (zeros if one layer)."""
        row_count, layer_count = codes.shape
        if layer_count == 1:
            return np.zeros(row_count, dtype=np.int64)
        keys = codes[:, 0].astype(np.int64)
        for layer in range(1, layer_count - 1):
            keys *= self.layer_sizes[layer]
            keys += codes[:, layer]
        return keys

    def bucket_keys(self, codes: np.ndarray) -> np.ndarray:
        """Return the int64 mixed-radix key of every SID row."""
        radix = [
            math.prod(self.layer_sizes[layer + 1 :])
            for layer in range(len(self.layer_sizes))
        ]
        return np.asarray(codes).astype(np.int64, copy=False) @ np.asarray(
            radix, dtype=np.int64
        )

    def offset_codes(self, codes: np.ndarray) -> np.ndarray:
        """Shift each layer's codes into one contiguous vocabulary."""
        return np.asarray(codes) + np.cumsum((0, *self.layer_sizes[:-1]))


def concat_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenate ``arange(start, start + length)`` for every aligned pair."""
    return np.arange(lengths.sum()) + np.repeat(
        starts - np.cumsum(lengths) + lengths, lengths
    )


def lookup_sorted(
    sorted_keys: np.ndarray, keys: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Locate ``keys`` inside an ascending ``sorted_keys``.

    Returns the insertion positions and a mask of exact hits; a position is
    meaningful only where the mask is set, and repeated keys resolve to their
    first occurrence.
    """
    positions = np.searchsorted(sorted_keys, keys)
    found = positions < sorted_keys.shape[0]
    found[found] = sorted_keys[positions[found]] == keys[found]
    return positions, found


def run_starts(ordered_values: np.ndarray) -> np.ndarray:
    """Return start offsets of equal-value runs in a nonempty sorted array."""
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1,
        )
    )


def ranks_within_runs(ordered_values: np.ndarray) -> np.ndarray:
    """Return zero-based ranks within equal-value runs in a sorted array."""
    starts = run_starts(ordered_values)
    return np.arange(ordered_values.shape[0], dtype=np.int64) - np.repeat(
        starts, np.diff(np.append(starts, ordered_values.shape[0]))
    )


@dataclass(frozen=True)
class PriorOccupancy:
    """Per-bucket occupancy of an already-published corpus.

    Args:
        bucket_keys: Occupied bucket keys, int64, strictly ascending.
        bucket_counts: Published item count of each key, aligned with
            ``bucket_keys``.
    """

    bucket_keys: np.ndarray
    bucket_counts: np.ndarray

    @classmethod
    def empty(cls) -> "PriorOccupancy":
        """Return the occupancy of a corpus with no published items."""
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    @property
    def is_empty(self) -> bool:
        """Whether no bucket is occupied."""
        return self.bucket_keys.shape[0] == 0

    def counts_for(self, bucket_keys: np.ndarray) -> np.ndarray:
        """Return prior counts aligned with ``bucket_keys``, zero when absent."""
        counts = np.zeros(bucket_keys.shape[0], dtype=np.int64)
        positions, found = lookup_sorted(self.bucket_keys, bucket_keys)
        counts[found] = self.bucket_counts[positions[found]]
        return counts

    def restrict_to_prefixes(
        self, prefix_ids: np.ndarray, last_size: int
    ) -> "PriorOccupancy":
        """Return the sub-occupancy of buckets lying in the given prefixes.

        One prefix owns the contiguous key range ``[prefix * S, (prefix + 1) * S)``,
        so this is a range gather, not a membership test over every prior key.
        ``prefix_ids`` must be unique and ascending.
        """
        starts = np.searchsorted(self.bucket_keys, prefix_ids * last_size, side="left")
        selected = concat_ranges(
            starts,
            np.searchsorted(self.bucket_keys, (prefix_ids + 1) * last_size, side="left")
            - starts,
        )
        return PriorOccupancy(self.bucket_keys[selected], self.bucket_counts[selected])


@dataclass(frozen=True)
class CollisionPlan:
    """Compact grouping plan consumed by collision resolution.

    The plan stores full-length last codes, dense row-to-bucket and slot index
    arrays, plus only the overflow-aligned identity and prefix data needed by
    candidate loading. It deliberately does not retain the full input code
    matrix.

    Args:
        origin_last_codes: Last-layer code of every input row, int64 shape
            ``(N,)`` in original row order.
        origin_bucket_indices: Per-row position of each input row's origin bucket
            *within* ``bucket_keys`` / ``bucket_counts`` (a row->bucket
            index in ``[0, num_buckets)``, not a key value), int64 shape
            ``(N,)`` in original row order.
        initial_slot_indices: One-based slot of each row within its origin
            bucket in deterministic (hash) order, continuing the published
            occupancy when appending; int64 shape ``(N,)`` in original row
            order.
        bucket_keys: Mixed-radix SID key of every occupied bucket, int64 shape
            ``(num_buckets,)`` indexed by bucket index (the values in
            ``origin_bucket_indices``) and ascending.
        bucket_counts: Number of this plan's rows in every bucket before
            capacity capping, int64 shape ``(num_buckets,)`` aligned with
            ``bucket_keys``; add ``prior_bucket_counts`` for the corpus total.
        overflow_rows: Original row indices ranked at or beyond ``capacity``
            within their bucket (the rows to relocate), int64 sorted by bucket
            key, then order hash, then row index.
        overflow_item_ids: Item IDs of ``overflow_rows``, aligned with it.
        overflow_prefix_ids: Prefix id of each overflow row, int64 aligned with
            ``overflow_rows``; ``prefix_id * last_size + last_code`` is a
            destination bucket key within the same prefix.
        config: Collision capacity and SID shape configuration.
        prior: Occupancy of the already-published corpus. Empty for a full
            resolve; for an append it must cover at least every bucket of every
            prefix affected by the planned rows, because relocation can read any
            bucket inside those prefixes.
        prior_bucket_counts: Published occupancy of every bucket in
            ``bucket_keys``, int64 aligned with it and zero where the bucket
            is new; ``prior_bucket_counts + bucket_counts`` is a bucket's
            uncapped total over the published corpus and these rows.
    """

    origin_last_codes: np.ndarray
    origin_bucket_indices: np.ndarray
    initial_slot_indices: np.ndarray
    bucket_keys: np.ndarray
    bucket_counts: np.ndarray
    overflow_rows: np.ndarray
    overflow_item_ids: np.ndarray
    overflow_prefix_ids: np.ndarray
    config: CollisionResolutionConfig
    prior: PriorOccupancy
    prior_bucket_counts: np.ndarray


@dataclass(frozen=True)
class CollisionResolutionStats:
    """Summary statistics for a collision-resolution run.

    Attributes:
        total_items: Number of rows this run placed.
        raw_collision_buckets: Buckets whose published plus new count exceeds
            capacity before relocation.
        final_collision_buckets: Buckets still over capacity after relocation.
        relocated_count: Overflow rows whose final slot is within capacity.
        unresolved_count: Overflow rows whose final slot exceeds capacity. They
            keep their origin SID, except under the uniform strategy, which may
            still move them.
        max_final_bucket_size: Largest final bucket, published items included.
    """

    total_items: int
    raw_collision_buckets: int
    final_collision_buckets: int
    relocated_count: int
    unresolved_count: int
    max_final_bucket_size: int


@dataclass(frozen=True)
class CollisionResolutionResult:
    """Resolved last codes, slot indexes, and diagnostics."""

    resolved_last_codes: np.ndarray
    slot_indices: np.ndarray
    unresolved_rows: np.ndarray
    final_bucket_keys: np.ndarray
    final_bucket_counts: np.ndarray
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

    @cached_property
    def offsets(self) -> np.ndarray:
        """Return CSR offsets into ``row_order`` for every SID bucket."""
        offsets = np.empty(self.counts.shape[0] + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(self.counts, dtype=np.int64, out=offsets[1:])
        return offsets


class CollisionResolver(ABC):
    """Interface for best-effort SID collision-resolution strategies.

    Args:
        progress_interval: Minimum number of overflow rows between progress
            logs.
    """

    def __init__(self, progress_interval: int = 1_000_000) -> None:
        if progress_interval < 1:
            raise ValueError(
                f"progress_interval must be >= 1, got {progress_interval}."
            )
        self._progress_interval = progress_interval

    @abstractmethod
    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows while preserving every input item.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.
            candidate_last_codes: Optional strategy-specific candidate last
                codes.
            collect_grouping: Whether to retain final bucket metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, and diagnostics.
        """
        raise NotImplementedError

    def _validate_candidate_last_codes(
        self, plan: CollisionPlan, candidate_last_codes: Optional[np.ndarray]
    ) -> np.ndarray:
        """Validate and normalize candidates aligned with overflow rows.

        Args:
            plan: Collision plan defining overflow rows and codebook bounds.
            candidate_last_codes: Ordered candidate matrix to validate. It may
                be omitted only when the plan has no overflow rows.

        Returns:
            The candidate matrix as an int64 NumPy array.

        Raises:
            TypeError: If candidate codes do not use an integer dtype.
            ValueError: If candidates are omitted for a plan with overflow, are
                not a row-aligned two-dimensional matrix, or contain
                out-of-range last-layer indices.
        """
        if candidate_last_codes is None:
            if plan.overflow_rows.size:
                raise ValueError(
                    "candidate_last_codes are required when the collision plan "
                    "has overflow rows."
                )
            candidate_last_codes = np.empty((0, 0), dtype=np.int64)
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

        last_size = plan.config.layer_sizes[-1]
        if candidates.size and (
            int(candidates.min()) < 0 or int(candidates.max()) >= last_size
        ):
            raise ValueError(
                f"candidate_last_codes must be in [0, {last_size}), got values "
                "outside that range."
            )
        return candidates.astype(np.int64, copy=False)

    @staticmethod
    def _initial_occupancy(
        plan: CollisionPlan,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the occupancy before relocation of every affected bucket.

        Relocation only reads or writes buckets in a prefix with an overflow
        row. A plan bucket there holds its published count plus its retained
        rows, capped at capacity but never below the published count; a
        published-only bucket there holds its published count.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.

        Returns:
            A mask of ``plan.bucket_keys`` lying in an overflow prefix, and the
            ascending keys and initial counts of every occupied bucket in those
            prefixes.
        """
        last_size = plan.config.layer_sizes[-1]
        prefix_ids = np.unique(plan.overflow_prefix_ids)
        _, in_overflow_prefix = lookup_sorted(prefix_ids, plan.bucket_keys // last_size)
        affected_prior = plan.prior.restrict_to_prefixes(prefix_ids, last_size)
        prior_counts = plan.prior_bucket_counts[in_overflow_prefix]
        # np.unique keeps the first occurrence, so a plan count wins over the prior.
        affected_keys, first = np.unique(
            np.concatenate(
                (plan.bucket_keys[in_overflow_prefix], affected_prior.bucket_keys)
            ),
            return_index=True,
        )
        affected_counts = np.concatenate(
            (
                np.maximum(
                    prior_counts,
                    np.minimum(
                        prior_counts + plan.bucket_counts[in_overflow_prefix],
                        plan.config.capacity,
                    ),
                ),
                affected_prior.bucket_counts,
            )
        )[first]
        return in_overflow_prefix, affected_keys, affected_counts

    @staticmethod
    def _build_result(
        plan: CollisionPlan,
        resolved_last_codes: np.ndarray,
        slot_indices: np.ndarray,
        unresolved_rows: np.ndarray,
        in_overflow_prefix: np.ndarray,
        affected_keys: np.ndarray,
        affected_counts: np.ndarray,
        collect_grouping: bool,
    ) -> CollisionResolutionResult:
        """Assemble the resolution result and its statistics.

        Args:
            plan: Grouping and overflow plan that was resolved.
            resolved_last_codes: Final last-layer code of every row.
            slot_indices: Final one-based slot of every row.
            unresolved_rows: Overflow rows whose final slot exceeds capacity.
            in_overflow_prefix: Mask of ``plan.bucket_keys`` lying in an
                overflow prefix.
            affected_keys: Unique keys of the occupied buckets in overflow
                prefixes.
            affected_counts: Final counts aligned with ``affected_keys``.
            collect_grouping: Whether to retain final bucket metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, and diagnostics.
        """
        capacity = plan.config.capacity
        combined_counts = plan.prior_bucket_counts + plan.bucket_counts
        unaffected = ~in_overflow_prefix
        # An unaffected bucket has no overflow row, so it is already within capacity.
        final_counts = np.concatenate((combined_counts[unaffected], affected_counts))
        final_bucket_keys = np.empty(0, dtype=np.int64)
        final_bucket_counts = np.empty(0, dtype=np.int64)
        if collect_grouping:
            final_keys = np.concatenate((plan.bucket_keys[unaffected], affected_keys))
            order = np.argsort(final_keys, kind="stable")
            final_bucket_keys = final_keys[order]
            final_bucket_counts = final_counts[order]
        return CollisionResolutionResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_bucket_keys=final_bucket_keys,
            final_bucket_counts=final_bucket_counts,
            grouping_collected=collect_grouping,
            stats=CollisionResolutionStats(
                total_items=plan.origin_last_codes.shape[0],
                raw_collision_buckets=int((combined_counts > capacity).sum()),
                final_collision_buckets=int((affected_counts > capacity).sum()),
                relocated_count=plan.overflow_rows.shape[0] - unresolved_rows.shape[0],
                unresolved_count=unresolved_rows.shape[0],
                max_final_bucket_size=int(final_counts.max(initial=0)),
            ),
        )


class KnnCollisionResolver(CollisionResolver):
    """Resolve collisions first-fit from externally supplied candidate codes."""

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_last_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Greedily relocate overflow rows to their first free candidate slot.

        An overflow row that finds no free candidate slot keeps its origin SID
        (over capacity), so every input row is preserved in the output.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.
            candidate_last_codes: Last-layer candidate matrix with shape
                ``(M, K)``, row-aligned with ``plan.overflow_rows``. It may be
                omitted only when the plan has no overflow rows.
            collect_grouping: Whether to retain final bucket metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, unresolved row indices,
            and diagnostics.

        Raises:
            TypeError: If candidate codes do not use an integer dtype.
            ValueError: If candidates are omitted for a plan with overflow, are
                not a row-aligned two-dimensional matrix, or contain
                out-of-range last-layer indices.
        """
        candidates = self._validate_candidate_last_codes(plan, candidate_last_codes)
        in_overflow_prefix, affected_keys, affected_counts = self._initial_occupancy(
            plan
        )
        overflow_count = plan.overflow_rows.shape[0]
        if overflow_count == 0:
            return self._build_result(
                plan,
                plan.origin_last_codes,
                plan.initial_slot_indices,
                np.empty(0, dtype=np.int64),
                in_overflow_prefix,
                affected_keys,
                affected_counts,
                collect_grouping,
            )

        last_size = plan.config.layer_sizes[-1]
        capacity = plan.config.capacity
        occupancy = dict(zip(affected_keys.tolist(), affected_counts.tolist()))
        get_occupancy = occupancy.get
        resolved_last_codes = plan.origin_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_rows = []
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        for resolved_count, (
            row,
            base_key,
            origin_last_code,
            candidate_row,
        ) in enumerate(
            zip(
                plan.overflow_rows,
                plan.overflow_prefix_ids * last_size,
                plan.origin_last_codes[plan.overflow_rows],
                candidates,
            ),
            start=1,
        ):
            for candidate in candidate_row:
                destination_key = base_key + candidate
                destination_occupancy = get_occupancy(destination_key, 0)
                if destination_occupancy < capacity:
                    occupancy[destination_key] = destination_occupancy + 1
                    resolved_last_codes[row] = candidate
                    slot_indices[row] = destination_occupancy + 1
                    break
            else:
                unresolved_rows.append(row)
                origin_key = base_key + origin_last_code
                origin_occupancy = get_occupancy(origin_key, 0) + 1
                occupancy[origin_key] = origin_occupancy
                slot_indices[row] = origin_occupancy
            progress.log(resolved_count)

        return self._build_result(
            plan,
            resolved_last_codes,
            slot_indices,
            np.asarray(unresolved_rows, dtype=np.int64),
            in_overflow_prefix,
            np.fromiter(occupancy, dtype=np.int64, count=len(occupancy)),
            np.fromiter(occupancy.values(), dtype=np.int64, count=len(occupancy)),
            collect_grouping,
        )


def _splitmix64(values: np.ndarray) -> np.ndarray:
    """Hash a uint64 array with the collision tool's fixed SplitMix64 seed."""
    with np.errstate(over="ignore"):
        mixed = values.astype(np.uint64, copy=False) + np.uint64(
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
        base = item_ids.astype(np.uint64, copy=False)
    else:
        import pandas as pd

        base = pd.util.hash_array(np.asarray(item_ids, dtype=object))
    return _splitmix64(base)


def prepare_collision_plan(
    item_ids: np.ndarray,
    codes: np.ndarray,
    config: CollisionResolutionConfig,
    prior: Optional[PriorOccupancy] = None,
) -> CollisionPlan:
    """Group SID buckets and identify overflow rows in stable processing order.

    With a ``prior``, the rows are treated as an append onto a published
    corpus: a bucket already holding ``P`` items admits only ``capacity - P``
    more, and retained rows are numbered from ``P + 1`` so they continue the
    published slot sequence instead of restarting at one. Published items are
    never rows here, which is what makes them impossible to move.

    Args:
        item_ids: One-dimensional item IDs aligned with ``codes``.
        codes: Integer SID matrix with shape ``(N, number_of_layers)``.
        config: Collision capacity and SID shape configuration.
        prior: Occupancy of the already-published corpus, or ``None`` for a
            full resolve. It must cover every bucket of every prefix these rows
            touch, because relocation can address any bucket inside a prefix.

    Returns:
        A compact plan for candidate loading and collision resolution.

    Raises:
        ValueError: If input dimensions, row counts, or layer counts disagree,
            or if prior bucket keys fall outside the configured key space.
    """
    if prior is None:
        prior = PriorOccupancy.empty()
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
    if codes.size and (
        codes.min() < 0 or np.any(codes >= np.asarray(config.layer_sizes))
    ):
        raise ValueError(
            "codes contain out-of-range values for layer_sizes "
            f"{config.layer_sizes}; check that --codebook matches the model that "
            "produced the SID table."
        )

    if not prior.is_empty:
        if int(prior.bucket_keys[0]) < 0 or int(prior.bucket_keys[-1]) >= math.prod(
            config.layer_sizes
        ):
            raise ValueError(
                "prior bucket keys fall outside the key space of layer_sizes "
                f"{config.layer_sizes}; check that --codebook matches the run "
                "that produced the published SID artifacts."
            )

    row_count = codes.shape[0]
    keys = config.bucket_keys(codes)
    sorted_rows = np.lexsort((stable_order_hash(item_ids), keys))
    bucket_keys, bucket_counts = np.unique(keys[sorted_rows], return_counts=True)
    del keys
    prior_bucket_counts = prior.counts_for(bucket_keys)
    origin_bucket_indices = np.empty(row_count, dtype=np.int64)
    origin_bucket_indices[sorted_rows] = np.repeat(
        np.arange(bucket_keys.shape[0]), bucket_counts
    )
    # Zero-based rank of every sorted row in its bucket, after the prior items.
    ranks = np.arange(row_count)
    ranks -= np.repeat(
        np.cumsum(bucket_counts) - bucket_counts - prior_bucket_counts, bucket_counts
    )
    overflow_rows = sorted_rows[ranks >= config.capacity]
    ranks += 1
    initial_slot_indices = np.empty(row_count, dtype=np.int64)
    initial_slot_indices[sorted_rows] = ranks
    del sorted_rows, ranks

    return CollisionPlan(
        origin_last_codes=codes[:, -1].astype(np.int64, copy=False),
        origin_bucket_indices=origin_bucket_indices,
        initial_slot_indices=initial_slot_indices,
        bucket_keys=bucket_keys,
        bucket_counts=bucket_counts,
        overflow_rows=overflow_rows,
        overflow_item_ids=item_ids[overflow_rows],
        overflow_prefix_ids=bucket_keys[origin_bucket_indices[overflow_rows]]
        // config.layer_sizes[-1],
        config=config,
        prior=prior,
        prior_bucket_counts=prior_bucket_counts,
    )


def _scatter_item_grouping(
    sid_keys: np.ndarray,
    counts: np.ndarray,
    bucket_indices: np.ndarray,
    slot_indices: np.ndarray,
) -> CodebookItemGrouping:
    """Build a CSR row order from each row's bucket and one-based local slot."""
    grouping = CodebookItemGrouping(
        sid_keys=sid_keys,
        counts=counts,
        row_order=np.empty(bucket_indices.size, dtype=np.int64),
    )
    positions = grouping.offsets[bucket_indices] + slot_indices - 1
    grouping.row_order[positions] = np.arange(bucket_indices.size)
    return grouping


def build_origin_item_grouping(plan: CollisionPlan) -> CodebookItemGrouping:
    """Group all rows by origin SID and initial one-based slot index.

    Slot indices are made bucket-local by subtracting the prior occupancy, so
    the grouping covers only this plan's rows even when appending to a corpus.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.

    Returns:
        Sorted origin SID keys, bucket counts, and grouped original row order.
    """
    bucket_indices = plan.origin_bucket_indices
    return _scatter_item_grouping(
        plan.bucket_keys,
        plan.bucket_counts,
        bucket_indices,
        plan.initial_slot_indices - plan.prior_bucket_counts[bucket_indices],
    )


def build_resolved_item_grouping(
    plan: CollisionPlan, result: CollisionResolutionResult
) -> CodebookItemGrouping:
    """Group this plan's rows by resolved SID and bucket-local slot index.

    Only buckets that gained rows appear, and slot indices have the prior
    occupancy subtracted, so the grouping covers only this plan's rows.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.
        result: Collision output from :meth:`CollisionResolver.resolve`.

    Returns:
        Sorted final SID keys, bucket counts, and grouped row order.

    Raises:
        RuntimeError: If grouping metadata was not collected or result bucket
            keys do not cover a resolved SID.
    """
    if not result.grouping_collected:
        raise RuntimeError(
            "resolved item grouping metadata was not collected; call resolve "
            "with collect_grouping=True."
        )

    final_prior_counts = plan.prior.counts_for(result.final_bucket_keys)
    delta_counts = result.final_bucket_counts - final_prior_counts
    gained = delta_counts > 0
    sid_keys = result.final_bucket_keys[gained]
    prior_counts = final_prior_counts[gained]
    resolved_keys = plan.bucket_keys[plan.origin_bucket_indices]
    resolved_keys -= plan.origin_last_codes
    resolved_keys += result.resolved_last_codes
    bucket_indices, found = lookup_sorted(sid_keys, resolved_keys)
    if not np.all(found):
        raise RuntimeError("final bucket keys do not cover every resolved SID.")
    return _scatter_item_grouping(
        sid_keys,
        delta_counts[gained],
        bucket_indices,
        result.slot_indices - prior_counts[bucket_indices],
    )
