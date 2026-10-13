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

_INT64_MAX = int(np.iinfo(np.int64).max)
_MASK64 = (1 << 64) - 1
_SEED = 2026
_SPLITMIX_INCREMENT = 0x9E3779B97F4A7C15


@dataclass(frozen=True)
class CollisionResolutionConfig:
    """Configuration for within-prefix SID collision resolution.

    ``codebook`` also fixes the SID key layout: a row's SID key is the
    mixed-radix number of its codes, its prefix id is that number without the
    last code, and its offset codes shift each layer into one contiguous
    vocabulary.

    Overflow items that cannot be placed within capacity stay over capacity,
    so every input item is preserved in the output.

    Args:
        codebook: Codebook size of each SID layer, matching the model config.
        capacity: Maximum number of retained items in one SID.
    """

    codebook: tuple[int, ...]
    capacity: int

    def __post_init__(self) -> None:
        if not self.codebook:
            raise ValueError("codebook must contain at least one layer.")
        if any(size <= 0 for size in self.codebook):
            raise ValueError("codebook sizes must be positive.")
        if self.capacity < 1:
            raise ValueError(f"capacity must be >= 1, got {self.capacity}.")
        if math.prod(self.codebook) > _INT64_MAX:
            raise ValueError(
                f"prod(codebook) exceeds int64 range: {self.codebook}; "
                "SID keys would overflow."
            )

    def prefix_ids(self, codes: np.ndarray) -> np.ndarray:
        """Return the mixed-radix prefix id of every SID row (zeros if one layer)."""
        row_count, layer_count = codes.shape
        if layer_count == 1:
            return np.zeros(row_count, dtype=np.int64)
        keys = codes[:, 0].astype(np.int64)
        for layer in range(1, layer_count - 1):
            keys *= self.codebook[layer]
            keys += codes[:, layer]
        return keys

    def convert_codes_to_sid_keys(self, codes: np.ndarray) -> np.ndarray:
        """Convert SID codes to one int64 mixed-radix key per row."""
        radix = [
            math.prod(self.codebook[layer + 1 :])
            for layer in range(len(self.codebook))
        ]
        return np.asarray(codes).astype(np.int64, copy=False) @ np.asarray(
            radix, dtype=np.int64
        )

    def offset_codes(self, codes: np.ndarray) -> np.ndarray:
        """Shift each layer's codes into one contiguous vocabulary."""
        return np.asarray(codes) + np.cumsum((0, *self.codebook[:-1]))


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


def get_same_value_start_indices(ordered_values: np.ndarray) -> np.ndarray:
    """Return start indices of consecutive equal values in a sorted array.

    Args:
        ordered_values: Nonempty sorted one-dimensional array.

    Returns:
        An int64 array of start indices, including values that appear only once.

    Examples:
        >>> get_same_value_start_indices(np.array([7, 7, 7, 9, 9, 12]))
        array([0, 3, 5])
    """
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1,
        )
    )


def ranks_within_runs(ordered_values: np.ndarray) -> np.ndarray:
    """Return zero-based ranks within equal-value runs in a sorted array."""
    starts = get_same_value_start_indices(ordered_values)
    return np.arange(ordered_values.shape[0], dtype=np.int64) - np.repeat(
        starts, np.diff(np.append(starts, ordered_values.shape[0]))
    )


@dataclass(frozen=True)
class PriorOccupancy:
    """Per-SID occupancy of an already-published corpus.

    Args:
        sid_keys: Occupied SID keys, int64, strictly ascending.
        item_counts: Published item count of each key, aligned with
            ``sid_keys``.
    """

    sid_keys: np.ndarray
    item_counts: np.ndarray

    @classmethod
    def empty(cls) -> "PriorOccupancy":
        """Return the occupancy of a corpus with no published items."""
        return cls(np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))

    @property
    def is_empty(self) -> bool:
        """Whether no SID is occupied."""
        return self.sid_keys.shape[0] == 0

    def counts_for(self, sid_keys: np.ndarray) -> np.ndarray:
        """Return prior counts aligned with ``sid_keys``, zero when absent."""
        positions, found = lookup_sorted(self.sid_keys, sid_keys)
        counts = np.zeros(sid_keys.shape[0], dtype=np.int64)
        counts[found] = self.item_counts[positions[found]]
        return counts

    def restrict_to_prefixes(
        self, prefix_ids: np.ndarray, last_size: int
    ) -> "PriorOccupancy":
        """Return the sub-occupancy of SIDs lying in the given prefixes.

        One prefix owns the contiguous key range ``[prefix * S, (prefix + 1) * S)``,
        so this is a range gather, not a membership test over every prior key.
        ``prefix_ids`` must be unique and ascending.
        """
        used_code_offsets = np.searchsorted(
            self.sid_keys, prefix_ids * last_size, side="left"
        )
        selected = concat_ranges(
            used_code_offsets,
            np.searchsorted(self.sid_keys, (prefix_ids + 1) * last_size, side="left")
            - used_code_offsets,
        )
        return PriorOccupancy(self.sid_keys[selected], self.item_counts[selected])


@dataclass(frozen=True)
class CollisionPlan:
    """Compact grouping plan consumed by collision resolution.

    The plan stores full-length last codes, dense SID key indices and slot index
    arrays, plus only the overflow-aligned identity and prefix data needed by
    candidate loading. It deliberately does not retain the full input code
    matrix.

    Args:
        origin_last_codes: Last-layer code of every input row, int64 shape
            ``(N,)`` in original row order.
        origin_sid_key_indices: Position of each input row's original SID key
            in ``sid_keys``, int64 shape ``(N,)`` in original row order.
            Values are array indices, not SID keys.
        initial_slot_indices: One-based slot of each row within its origin
            SID in deterministic (hash) order, continuing the published
            occupancy when appending; int64 shape ``(N,)`` in original row
            order.
        sid_keys: Ascending mixed-radix keys of SIDs present in this input,
            int64 shape ``(occupied_sid_count,)``.
        new_item_counts: Number of this plan's rows in every SID before
            capacity capping, int64 shape ``(occupied_sid_count,)`` aligned with
            ``sid_keys``; add ``existing_item_counts`` for the corpus total.
        overflow_rows: Original row indices ranked at or beyond ``capacity``
            within their SID (the rows to relocate), int64 sorted by SID
            key, then order hash, then row index.
        overflow_item_ids: Item IDs of ``overflow_rows``, aligned with it.
        overflow_prefix_ids: Prefix id of each overflow row, int64 aligned with
            ``overflow_rows``; ``prefix_id * last_size + last_code`` is a
            destination SID key within the same prefix.
        config: Collision capacity and SID shape configuration.
        prior: Occupancy of the already-published corpus. Empty for a full
            resolve; for an append it must cover at least every SID of every
            prefix affected by the planned rows, because relocation can read any
            SID inside those prefixes.
        existing_item_counts: Published occupancy of every SID in
            ``sid_keys``, int64 aligned with it and zero where the SID
            is new; ``existing_item_counts + new_item_counts`` is a SID's
            uncapped total over the published corpus and these rows.
    """

    origin_last_codes: np.ndarray
    origin_sid_key_indices: np.ndarray
    initial_slot_indices: np.ndarray
    sid_keys: np.ndarray
    new_item_counts: np.ndarray
    overflow_rows: np.ndarray
    overflow_item_ids: np.ndarray
    overflow_prefix_ids: np.ndarray
    config: CollisionResolutionConfig
    prior: PriorOccupancy
    existing_item_counts: np.ndarray


@dataclass(frozen=True)
class CollisionResolutionStats:
    """Summary statistics for a collision-resolution run.

    Attributes:
        total_items: Number of rows this run placed.
        raw_colliding_sid_count: SIDs whose published plus new count exceeds
            capacity before relocation.
        final_colliding_sid_count: SIDs still over capacity after relocation.
        relocated_count: Overflow rows whose final slot is within capacity.
        unresolved_count: Overflow rows whose final slot exceeds capacity. They
            keep their origin SID, except under the uniform strategy, which may
            still move them.
        max_final_items_per_sid: Largest final item count per SID within this
            run's affected SIDs, including published items.
    """

    total_items: int
    raw_colliding_sid_count: int
    final_colliding_sid_count: int
    relocated_count: int
    unresolved_count: int
    max_final_items_per_sid: int


@dataclass(frozen=True)
class CollisionResolutionResult:
    """Resolved last codes, slot indexes, and diagnostics."""

    resolved_last_codes: np.ndarray
    slot_indices: np.ndarray
    unresolved_rows: np.ndarray
    final_sid_keys: np.ndarray
    final_item_counts: np.ndarray
    grouping_collected: bool
    stats: CollisionResolutionStats


@dataclass(frozen=True)
class CodebookItemGrouping:
    """CSR-like row grouping for sorted flattened SID keys.

    ``row_order[offsets[i]:offsets[i + 1]]`` contains the original row indices
    for ``sid_keys[i]``. Rows within a SID follow their one-based slot order.

    Args:
        sid_keys: Sorted flattened SID keys, one per occupied SID.
        item_counts: Counts of this run's items, aligned with ``sid_keys``.
        row_order: Original row indices flattened in SID and slot order.
    """

    sid_keys: np.ndarray
    item_counts: np.ndarray
    row_order: np.ndarray

    @cached_property
    def offsets(self) -> np.ndarray:
        """Return CSR offsets into ``row_order`` for every SID."""
        offsets = np.empty(self.item_counts.shape[0] + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(self.item_counts, dtype=np.int64, out=offsets[1:])
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
            collect_grouping: Whether to retain final SID metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, and diagnostics.
        """
        raise NotImplementedError

    def _prepare_collision(
        self, plan: CollisionPlan, collect_grouping: bool
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, Optional[CollisionResolutionResult]
    ]:
        """Prepare initial occupancy and finish plans without overflow.

        Args:
            plan: Grouping and overflow plan from ``prepare_collision_plan``.
            collect_grouping: Whether to retain final SID metadata.

        Returns:
            The prefix mask, affected SID keys, initial item counts, and a
            completed result when no overflow exists. The result is ``None``
            when overflow rows still need resolution.
        """
        in_overflow_prefix, affected_keys, affected_counts = self._initial_occupancy(
            plan
        )
        result = None
        if plan.overflow_rows.size == 0:
            result = self._build_result(
                plan,
                plan.origin_last_codes,
                plan.initial_slot_indices,
                np.empty(0, dtype=np.int64),
                in_overflow_prefix,
                affected_keys,
                affected_counts,
                collect_grouping,
            )
        return in_overflow_prefix, affected_keys, affected_counts, result

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

        last_size = plan.config.codebook[-1]
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
        """Return the occupancy before relocation of every affected sid.

        Relocation only reads or writes SIDs in a prefix with an overflow
        row. A plan SID there holds its published count plus its retained
        rows, capped at capacity but never below the published count; a
        published-only SID there holds its published count.

        Args:
            plan: Grouping and overflow plan from
                :func:`prepare_collision_plan`.

        Returns:
            A mask of ``plan.sid_keys`` lying in an overflow prefix, and the
            ascending keys and initial counts of every occupied SID in those
            prefixes.
        """
        last_size = plan.config.codebook[-1]
        prefix_ids = np.unique(plan.overflow_prefix_ids)
        _, in_overflow_prefix = lookup_sorted(prefix_ids, plan.sid_keys // last_size)
        affected_prior = plan.prior.restrict_to_prefixes(prefix_ids, last_size)
        # np.unique keeps the first occurrence, so a plan count wins over the prior.
        affected_keys, first = np.unique(
            np.concatenate(
                (plan.sid_keys[in_overflow_prefix], affected_prior.sid_keys)
            ),
            return_index=True,
        )
        prior_counts = plan.existing_item_counts[in_overflow_prefix]
        affected_counts = np.concatenate(
            (
                np.maximum(
                    prior_counts,
                    np.minimum(
                        prior_counts + plan.new_item_counts[in_overflow_prefix],
                        plan.config.capacity,
                    ),
                ),
                affected_prior.item_counts,
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
            in_overflow_prefix: Mask of ``plan.sid_keys`` lying in an
                overflow prefix.
            affected_keys: Unique keys of the occupied SIDs in overflow
                prefixes.
            affected_counts: Final counts aligned with ``affected_keys``.
            collect_grouping: Whether to retain final SID metadata required
                by :func:`build_resolved_item_grouping`.

        Returns:
            Resolved last-layer codes, slot indices, and diagnostics.
        """
        combined_counts = plan.existing_item_counts + plan.new_item_counts
        unaffected = ~in_overflow_prefix
        # An unaffected SID has no overflow row, so it is already within capacity.
        final_counts = np.concatenate((combined_counts[unaffected], affected_counts))
        final_sid_keys = np.empty(0, dtype=np.int64)
        final_item_counts = np.empty(0, dtype=np.int64)
        if collect_grouping:
            final_keys = np.concatenate((plan.sid_keys[unaffected], affected_keys))
            order = np.argsort(final_keys, kind="stable")
            final_sid_keys = final_keys[order]
            final_item_counts = final_counts[order]
        capacity = plan.config.capacity
        return CollisionResolutionResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_sid_keys=final_sid_keys,
            final_item_counts=final_item_counts,
            grouping_collected=collect_grouping,
            stats=CollisionResolutionStats(
                total_items=plan.origin_last_codes.shape[0],
                raw_colliding_sid_count=int((combined_counts > capacity).sum()),
                final_colliding_sid_count=int((affected_counts > capacity).sum()),
                relocated_count=plan.overflow_rows.shape[0] - unresolved_rows.shape[0],
                unresolved_count=unresolved_rows.shape[0],
                max_final_items_per_sid=int(final_counts.max(initial=0)),
            ),
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
    """Group SIDs and identify overflow rows in stable processing order.

    With a ``prior``, the rows are treated as an append onto a published
    corpus: a SID already holding ``P`` items admits only ``capacity - P``
    more, and retained rows are numbered from ``P + 1`` so they continue the
    published slot sequence instead of restarting at one. Published items are
    never rows here, which is what makes them impossible to move.

    Args:
        item_ids: One-dimensional item IDs aligned with ``codes``.
        codes: Integer SID matrix with shape ``(N, number_of_layers)``.
        config: Collision capacity and SID shape configuration.
        prior: Occupancy of the already-published corpus, or ``None`` for a
            full resolve. It must cover every SID of every prefix these rows
            touch, because relocation can address any SID inside a prefix.

    Returns:
        A compact plan for candidate loading and collision resolution.

    Raises:
        ValueError: If input dimensions, row counts, or layer counts disagree,
            or if prior SID keys fall outside the configured key space.
    """
    if prior is None:
        prior = PriorOccupancy.empty()
    if item_ids.ndim != 1:
        raise ValueError(f"item_ids must be 1-D, got shape {item_ids.shape}.")
    if codes.ndim != 2:
        raise ValueError(f"codes must be 2-D, got shape {codes.shape}.")
    if item_ids.shape[0] != codes.shape[0]:
        raise ValueError(
            "item_ids and codes must have the same row count, got "
            f"{item_ids.shape[0]} and {codes.shape[0]}."
        )
    if codes.shape[1] != len(config.codebook):
        raise ValueError(
            f"codes have {codes.shape[1]} layers but codebook has "
            f"{len(config.codebook)}."
        )
    if codes.size and (
        codes.min() < 0 or np.any(codes >= np.asarray(config.codebook))
    ):
        raise ValueError(
            "codes contain out-of-range values for codebook "
            f"{config.codebook}; check that --codebook matches the model that "
            "produced the SID table."
        )

    if not prior.is_empty:
        if int(prior.sid_keys[0]) < 0 or int(prior.sid_keys[-1]) >= math.prod(
            config.codebook
        ):
            raise ValueError(
                "prior SID keys fall outside the key space of codebook "
                f"{config.codebook}; check that --codebook matches the run "
                "that produced the published SID artifacts."
            )

    keys = config.convert_codes_to_sid_keys(codes)
    sorted_rows = np.lexsort((stable_order_hash(item_ids), keys))
    sid_keys, new_item_counts = np.unique(keys[sorted_rows], return_counts=True)
    del keys
    # Equal SIDs are contiguous in sorted_rows. Repeat each sid_keys index by
    # its new item count, then scatter the indices back to original input rows.
    row_count = codes.shape[0]
    origin_sid_key_indices = np.empty(row_count, dtype=np.int64)
    origin_sid_key_indices[sorted_rows] = np.repeat(
        np.arange(sid_keys.shape[0]), new_item_counts
    )
    # cumsum(counts) - counts gives each SID group's start in sorted_rows.
    # Subtracting start - existing_count gives each new item's zero-based
    # position within its SID: sorted position - group start + existing count.
    # Repeat each group's adjustment by its new item count to align with rows.
    new_item_positions = np.arange(row_count)
    existing_item_counts = prior.counts_for(sid_keys)
    new_item_positions -= np.repeat(
        np.cumsum(new_item_counts) - new_item_counts - existing_item_counts,
        new_item_counts,
    )
    overflow_rows = sorted_rows[new_item_positions >= config.capacity]
    # Convert to one-based slots and restore original input row order.
    new_item_positions += 1
    initial_slot_indices = np.empty(row_count, dtype=np.int64)
    initial_slot_indices[sorted_rows] = new_item_positions
    del sorted_rows, new_item_positions

    return CollisionPlan(
        origin_last_codes=codes[:, -1].astype(np.int64, copy=False),
        origin_sid_key_indices=origin_sid_key_indices,
        initial_slot_indices=initial_slot_indices,
        sid_keys=sid_keys,
        new_item_counts=new_item_counts,
        overflow_rows=overflow_rows,
        overflow_item_ids=item_ids[overflow_rows],
        overflow_prefix_ids=sid_keys[origin_sid_key_indices[overflow_rows]]
        // config.codebook[-1],
        config=config,
        prior=prior,
        existing_item_counts=existing_item_counts,
    )


def _scatter_item_grouping(
    sid_keys: np.ndarray,
    item_counts: np.ndarray,
    sid_key_indices: np.ndarray,
    slot_indices: np.ndarray,
) -> CodebookItemGrouping:
    """Build a CSR row order from each row's SID and one-based local slot."""
    grouping = CodebookItemGrouping(
        sid_keys=sid_keys,
        item_counts=item_counts,
        row_order=np.empty(sid_key_indices.size, dtype=np.int64),
    )
    positions = grouping.offsets[sid_key_indices] + slot_indices - 1
    grouping.row_order[positions] = np.arange(sid_key_indices.size)
    return grouping


def build_origin_item_grouping(plan: CollisionPlan) -> CodebookItemGrouping:
    """Group all rows by origin SID and initial one-based slot index.

    Slot indices are made SID-local by subtracting the prior occupancy, so
    the grouping covers only this plan's rows even when appending to a corpus.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.

    Returns:
        Sorted origin SID keys, SID counts, and grouped original row order.
    """
    sid_key_indices = plan.origin_sid_key_indices
    return _scatter_item_grouping(
        plan.sid_keys,
        plan.new_item_counts,
        sid_key_indices,
        plan.initial_slot_indices - plan.existing_item_counts[sid_key_indices],
    )


def build_resolved_item_grouping(
    plan: CollisionPlan, result: CollisionResolutionResult
) -> CodebookItemGrouping:
    """Group this plan's rows by resolved SID and SID-local slot index.

    Only SIDs that gained rows appear, and slot indices have the prior
    occupancy subtracted, so the grouping covers only this plan's rows.

    Args:
        plan: Grouping and overflow plan from :func:`prepare_collision_plan`.
        result: Collision output from :meth:`CollisionResolver.resolve`.

    Returns:
        Sorted final SID keys, SID counts, and grouped row order.

    Raises:
        RuntimeError: If grouping metadata was not collected or result SID
            keys do not cover a resolved SID.
    """
    if not result.grouping_collected:
        raise RuntimeError(
            "resolved item grouping metadata was not collected; call resolve "
            "with collect_grouping=True."
        )

    resolved_keys = plan.sid_keys[plan.origin_sid_key_indices]
    resolved_keys -= plan.origin_last_codes
    resolved_keys += result.resolved_last_codes
    final_prior_counts = plan.prior.counts_for(result.final_sid_keys)
    delta_counts = result.final_item_counts - final_prior_counts
    gained = delta_counts > 0
    sid_keys = result.final_sid_keys[gained]
    sid_key_indices, found = lookup_sorted(sid_keys, resolved_keys)
    if not np.all(found):
        raise RuntimeError("final SID keys do not cover every resolved SID.")
    prior_counts = final_prior_counts[gained]
    return _scatter_item_grouping(
        sid_keys,
        delta_counts[gained],
        sid_key_indices,
        result.slot_indices - prior_counts[sid_key_indices],
    )
