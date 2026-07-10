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

"""Offline SID codebook collision-prevention tool (vectorized).

Caps every SID bucket at ``--max_items_per_codebook`` and reassigns overflow
items to a free code in the SAME band -- keeping every layer but the last and
varying only the last SID layer, so an item never leaves its ``(prefix)`` band:

- ``--strategy candidate`` walks the item's model-provided candidate SIDs
  (``candidate_codes``), taking the last code of each, best-first;
- ``--strategy random`` draws random last-layer codes (excluding the origin),
  a baseline that ignores semantic proximity.

Items with no free slot fall back per ``--unassigned_policy``. The hot path is
numpy/Arrow-vectorized so a hundred-million-row map fits in one pass; results
are deterministic and independent of input row order. I/O goes
through ``create_reader`` / ``create_writer`` so CSV, Parquet, and ODPS all work;
outputs are written in chunks to stay under Arrow's 2^31 list-offset limit.
"""

import argparse
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

# Register the local reader/writer classes used through create_reader/writer.
import tzrec.datasets.csv_dataset  # noqa: F401
import tzrec.datasets.parquet_dataset  # noqa: F401
from tzrec.datasets.dataset import BaseReader, create_reader, create_writer
from tzrec.utils.logging_util import logger

_MASK64 = (1 << 64) - 1

# Fixed hash seed for the order-independent tie-break. Not a CLI knob: the
# collision rate is seed-invariant, so runs stay reproducible without one.
_SEED = 2026

# CSV-fallback delimiters (Parquet/ODPS carry codes as native list<int64>, so
# these are unused there). Both are non-comma so a codebook cell never collides
# with CSV's own field separator, and distinct so a compact cell parses cleanly.
_CODE_SEP = "|"  # separates the int codes within one SID, e.g. "1|2|3"
_CAND_SEP = ";"  # separates candidate SIDs in a compact cell, e.g. "1|2;3|4"

# Chunk rows per output write: keeps each Arrow array's int32 offsets under 2^31
# (a single 255M-row string/list column would overflow one array).
_WRITE_CHUNK = 20_000_000


@dataclass
class AssignmentStats:
    """Summary statistics for a collision-prevention run."""

    total_items: int
    raw_collision_buckets: int
    final_collision_buckets: int
    reassigned_count: int
    unassigned_count: int
    max_final_bucket_size: int


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized order-independent SplitMix64 hash of a uint64 array.

    A pure function of the input values (not their position), mixed with the
    module ``_SEED``, so it gives a stable tie-break invariant to input row order
    -- unlike a read-order index -- while staying fully vectorized.

    Args:
        x (np.ndarray): uint64 values to hash.

    Returns:
        np.ndarray: uint64 hashes, same shape as ``x``.
    """
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64((_SEED * 0x9E3779B97F4A7C15) & _MASK64)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _order_hash(item_ids: np.ndarray) -> np.ndarray:
    """Per-item order-independent tie-break hash (uint64), vectorized.

    Integer ids hash directly; string/object ids are folded to uint64 via
    ``pandas.util.hash_array`` first. Both then pass through :func:`_splitmix64`.

    Args:
        item_ids (np.ndarray): item id values.

    Returns:
        np.ndarray: uint64 per-item hashes.
    """
    if np.issubdtype(item_ids.dtype, np.integer):
        base = item_ids.astype(np.uint64)
    else:
        import pandas as pd

        base = pd.util.hash_array(np.asarray(item_ids, dtype=object))
    return _splitmix64(base)


class CollisionRunner:
    """Vectorized SID collision-prevention runner over the dataset reader/writer.

    The backend (CSV / Parquet / ODPS) is chosen by the reader/writer type, so
    the same path serves local files and MaxCompute tables.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._input_path = args.input_path
        self._output_path = args.output_path
        self._diagnostics_output_path = args.diagnostics_output_path
        self._reader_type = args.reader_type
        self._writer_type = args.writer_type
        self._batch_size = args.batch_size
        self._item_id_field = args.item_id_field
        self._code_field = args.code_field
        self._candidate_codes_field = args.candidate_codes_field
        self._candidate_depth = args.candidate_depth
        self._max_items_per_codebook = args.max_items_per_codebook
        if self._max_items_per_codebook < 1:
            raise ValueError(
                "--max_items_per_codebook must be >= 1, got "
                f"{self._max_items_per_codebook}."
            )
        self._unassigned_policy = args.unassigned_policy
        self._strategy = args.strategy
        self._codebook = [int(x) for x in args.codebook.split(",") if x.strip()]
        if not self._codebook:
            raise ValueError("--codebook must list at least one per-layer size.")
        self._random_num_candidates = args.random_num_candidates
        self._rate_only = args.rate_only
        self._odps_data_quota_name = args.odps_data_quota_name
        # The writer type (and its is-CSV flag) can depend on the input reader
        # class when --writer_type is unset, so they are finalized once at the
        # first read rather than here; declared now so the attributes always exist.
        self._writer_type_str: Optional[str] = None
        self._is_csv: bool = False

    def run(self) -> AssignmentStats:
        """Read the SID map, assign, and write the reassigned map + stats."""
        item_ids, codes = self._load_codes()
        final_codes, index, keep_mask, stats = self._assign(item_ids, codes)
        if not self._rate_only:
            self._write(item_ids, codes, final_codes, index, keep_mask, stats)
        else:
            logger.info("rate_only: skipping map write")
        logger.info("SID collision prevention finished: %s", stats)
        return stats

    def _make_reader(self, selected_cols: List[str]) -> BaseReader:
        """Open the input reader projecting ``selected_cols``."""
        return create_reader(
            input_path=self._input_path,
            batch_size=self._batch_size,
            selected_cols=selected_cols,
            reader_type=self._reader_type,
            quota_name=self._odps_data_quota_name,
        )

    def _read(self) -> Iterable[Dict[str, pa.Array]]:
        """Stream item-id + code batches, caching the resolved output backend."""
        reader = self._make_reader([self._item_id_field, self._code_field])
        # Writer defaults to matching the input reader (hitrate.py idiom) when
        # --writer_type is unset; ODPS still resolves to OdpsWriter via
        # create_writer's path detection regardless.
        reader_cls_name = reader.__class__.__name__
        self._writer_type_str = self._writer_type or (
            reader_cls_name.replace("Reader", "Writer")
        )
        self._is_csv = self._writer_type_str == "CsvWriter"
        yield from reader.to_batches()

    @staticmethod
    def _codes_matrix(arr: pa.Array) -> np.ndarray:
        """Decode a SID code column into an (N, n_layers) int64 matrix.

        Parquet/ODPS give a ``list<int64>`` cell; CSV gives a ``_CODE_SEP``
        string; a single-layer numeric CSV column may arrive already as ints.
        """
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            n = len(arr)
            flat = (
                arr.flatten()
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )
        elif pa.types.is_integer(arr.type):
            arr_np = arr.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            return arr_np.reshape(-1, 1)
        else:
            parts = pc.split_pattern(arr, _CODE_SEP)
            n = len(parts)
            flat = pc.cast(parts.flatten(), pa.int64()).to_numpy(zero_copy_only=False)
        if n == 0:
            return np.empty((0, 0), dtype=np.int64)
        n_layers = flat.shape[0] // n
        if flat.shape[0] != n * n_layers:
            raise ValueError("ragged SID codes: all items must share n_layers.")
        return flat.reshape(n, n_layers)

    def _load_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Stream the map into an item-id array and an (N, n_layers) code matrix."""
        id_chunks: List[np.ndarray] = []
        code_chunks: List[np.ndarray] = []
        for batch in self._read():
            id_chunks.append(batch[self._item_id_field].to_numpy(zero_copy_only=False))
            code_chunks.append(self._codes_matrix(batch[self._code_field]))
        if not id_chunks:
            raise ValueError("SID input is empty.")
        item_ids = np.concatenate(id_chunks)
        del id_chunks
        codes = np.concatenate(code_chunks, axis=0)
        del code_chunks
        if codes.shape[1] < 1:
            raise ValueError("SID codes must have at least one layer.")
        return item_ids, codes

    def _load_candidate_last(self, overflow_id_arr: np.ndarray) -> Dict[Any, List[int]]:
        """Map each overflow item id to its ordered candidate last-layer codes.

        Reads ``candidate_codes`` (list<list<int64>> for Parquet/ODPS, a
        ``_CAND_SEP``/``_CODE_SEP`` string for CSV) but keeps only the last code
        of each candidate SID and only for overflow items, so memory scales with
        the overflow fraction, not the whole table.
        """
        field = self._candidate_codes_field
        # Sort the overflow ids once; per-batch membership is then a bisect
        # instead of re-sorting the whole (potentially huge) overflow set.
        ov = np.sort(overflow_id_arr)
        cand: Dict[Any, List[int]] = {}
        for batch in self._make_reader([self._item_id_field, field]).to_batches():
            if field not in batch:
                break
            ids = batch[self._item_id_field].to_numpy(zero_copy_only=False)
            pos = np.searchsorted(ov, ids)
            np.clip(pos, 0, len(ov) - 1, out=pos)
            keep = np.where(ov[pos] == ids)[0]
            if keep.size == 0:
                continue
            col = batch[field]
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                self._collect_candidate_lists(ids, col, keep, cand)
            else:
                self._collect_candidate_strings(ids, col, keep, cand)
        return cand

    def _collect_candidate_lists(
        self,
        ids: np.ndarray,
        col: pa.Array,
        keep: np.ndarray,
        cand: Dict[Any, List[int]],
    ) -> None:
        """Vectorized last-code extraction from a list<list<int64>> batch."""
        n = len(col)
        inner = col.flatten()  # list<int64>, one per candidate SID
        if len(inner) == 0:
            return
        k = len(inner) // n
        if len(inner) != n * k:
            raise ValueError("ragged candidate_codes: all items must share topk.")
        flat = (
            inner.flatten().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
        )
        n_layers = flat.shape[0] // (n * k)
        last = flat.reshape(n, k, n_layers)[:, :, n_layers - 1]
        if self._candidate_depth is not None:
            last = last[:, : self._candidate_depth]
        last_lists = last.tolist()  # per-item Python lists: cheap to scan in the loop
        for iid, i in zip(ids[keep].tolist(), keep.tolist()):
            cand[iid] = last_lists[i]

    def _collect_candidate_strings(
        self,
        ids: np.ndarray,
        col: pa.Array,
        keep: np.ndarray,
        cand: Dict[Any, List[int]],
    ) -> None:
        """Per-row last-code extraction from a CSV compact-candidate batch."""
        depth = self._candidate_depth
        values = col.to_pylist()
        for iid, i in zip(ids[keep].tolist(), keep.tolist()):
            text = values[i]
            if not text:
                continue
            last_codes = []
            for part in str(text).split(_CAND_SEP):
                part = part.strip()
                if not part:
                    continue
                codes = [int(p) for p in part.split(_CODE_SEP) if p.strip()]
                if codes:
                    last_codes.append(codes[-1])
            cand[iid] = last_codes[:depth] if depth is not None else last_codes

    def _assign(
        self, item_ids: np.ndarray, codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], AssignmentStats]:
        """Cap buckets and reassign overflow within-band; return the final map."""
        cap = self._max_items_per_codebook
        n, n_layers = codes.shape
        if n_layers != len(self._codebook):
            raise ValueError(
                f"codes have {n_layers} layers but --codebook has "
                f"{len(self._codebook)}."
            )
        last = codes[:, n_layers - 1].astype(np.int64, copy=False)

        band_id = self._band_ids(codes)
        order = _order_hash(item_ids)

        # Rank rows within their (band_id, last) bucket by order-hash: the RATE is
        # invariant to which cap items are kept, but the choice is deterministic.
        # One lexsort yields the rank, each bucket's representative row, and size.
        rank, rep_rows, counts = self._within_bucket_rank(band_id, last, order)
        overflow_order = np.flatnonzero(rank >= cap)
        overflow_order = overflow_order[
            np.lexsort(
                (order[overflow_order], last[overflow_order], band_id[overflow_order])
            )
        ]

        cand = self._candidates(item_ids, overflow_order)
        last_size = self._codebook[-1]

        sid_key = band_id * last_size + last
        slot_count: Dict[int, int] = dict(
            zip(sid_key[rep_rows].tolist(), np.minimum(counts, cap).tolist())
        )

        final_key = sid_key.copy()
        final_index = rank + 1
        reassigned, unassigned = self._place_overflow(
            overflow_order,
            item_ids,
            band_id,
            last,
            cand,
            last_size,
            cap,
            slot_count,
            final_key,
            final_index,
        )
        keep_mask = self._resolve_unassigned(
            unassigned, sid_key, slot_count, final_index, n
        )

        final_codes = codes.copy()
        final_codes[:, n_layers - 1] = final_key % last_size

        stats = self._stats(n, counts, slot_count, cap, reassigned, len(unassigned))
        return final_codes, final_index, keep_mask, stats

    def _band_ids(self, codes: np.ndarray) -> np.ndarray:
        """Dense integer id per distinct ``(prefix)`` band (all layers but last).

        Packs the prefix layers into one int64 key using the declared per-layer
        codebook sizes as radices, then factorizes -- far faster than
        ``np.unique`` over 2-D void rows at scale, and independent of the values a
        given batch happens to contain.
        """
        n, n_layers = codes.shape
        if n_layers == 1:
            return np.zeros(n, dtype=np.int64)
        key = codes[:, 0].astype(np.int64)  # fresh copy: safe to pack in place below
        for j in range(1, n_layers - 1):
            key *= self._codebook[j]
            key += codes[:, j]
        inverse = np.unique(key, return_inverse=True)[1]
        return inverse.astype(np.int64, copy=False).reshape(-1)

    @staticmethod
    def _within_bucket_rank(
        band_id: np.ndarray, last: np.ndarray, order: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rank rows within their ``(band_id, last)`` bucket by order-hash.

        Returns the per-row 0-based rank, each bucket's representative original
        row index, and each bucket's size -- all from a single column lexsort, so
        the grouping is computed once (no packed key, no per-row size scatter).
        """
        n = band_id.shape[0]
        sort_idx = np.lexsort((order, last, band_id))
        sb = band_id[sort_idx]
        sl = last[sort_idx]
        is_first = np.empty(n, dtype=bool)
        is_first[0] = True
        is_first[1:] = (sb[1:] != sb[:-1]) | (sl[1:] != sl[:-1])
        first_sorted = np.flatnonzero(is_first)
        counts = np.diff(np.append(first_sorted, n))
        rank = np.empty(n, dtype=np.int64)
        rank[sort_idx] = np.arange(n) - np.repeat(first_sorted, counts)
        return rank, sort_idx[first_sorted], counts

    def _candidates(
        self,
        item_ids: np.ndarray,
        overflow_order: np.ndarray,
    ) -> Dict[Any, List[int]]:
        """Build per-overflow-item last-code candidate lists."""
        if self._strategy == "random":
            size = self._codebook[-1]
            if size < 2:
                raise ValueError("strategy='random' requires a last codebook >= 2.")
            num = min(self._random_num_candidates, size - 1)
            draws = self._random_draws(overflow_order, item_ids, num, size)
            ov_ids = item_ids[overflow_order].tolist()  # Python-native dict keys
            return {ov_ids[j]: draws[j].tolist() for j in range(len(ov_ids))}
        if overflow_order.size == 0:
            return {}
        cand = self._load_candidate_last(item_ids[overflow_order])
        if not cand:
            raise ValueError(
                "map has overflow items but candidate_codes yielded no candidates."
            )
        return cand

    def _random_draws(
        self, overflow_order: np.ndarray, item_ids: np.ndarray, num: int, size: int
    ) -> np.ndarray:
        """Order-independent random last-layer draws per overflow item, (M, num)."""
        h = _order_hash(item_ids[overflow_order])
        k = np.arange(num, dtype=np.uint64)
        with np.errstate(over="ignore"):
            mixed = _splitmix64(h[:, None] + k[None, :] * np.uint64(0x9E3779B97F4A7C15))
        return (mixed % np.uint64(size)).astype(np.int64)

    def _place_overflow(
        self,
        overflow_order: np.ndarray,
        item_ids: np.ndarray,
        band_id: np.ndarray,
        last: np.ndarray,
        cand: Dict[Any, List[int]],
        last_size: int,
        cap: int,
        slot_count: Dict[int, int],
        final_key: np.ndarray,
        final_index: np.ndarray,
    ) -> Tuple[int, List[int]]:
        """Greedy single pass: place each overflow item in the first free slot.

        Overflow-row inputs are pre-materialized to Python scalars so the loop
        avoids per-iteration 0-d numpy indexing; placements are scattered back
        into ``final_key`` / ``final_index`` vectorized at the end.
        """
        rows = overflow_order.tolist()
        bases = (band_id[overflow_order] * last_size).tolist()
        origin_lasts = last[overflow_order].tolist()
        ids = item_ids[overflow_order].tolist()
        placed_rows: List[int] = []
        placed_key: List[int] = []
        placed_idx: List[int] = []
        unassigned: List[int] = []
        get = slot_count.get
        for k, base in enumerate(bases):
            origin_last = origin_lasts[k]
            for code in cand.get(ids[k], ()):  # ordered best-first
                if code == origin_last:
                    continue
                ck = base + code
                count = get(ck, 0)
                if count < cap:
                    slot_count[ck] = count + 1
                    placed_rows.append(rows[k])
                    placed_key.append(ck)
                    placed_idx.append(count + 1)
                    break
            else:
                unassigned.append(rows[k])
        if placed_rows:
            final_key[placed_rows] = placed_key
            final_index[placed_rows] = placed_idx
        return len(placed_rows), unassigned

    def _resolve_unassigned(
        self,
        unassigned: List[int],
        sid_key: np.ndarray,
        slot_count: Dict[int, int],
        final_index: np.ndarray,
        n: int,
    ) -> Optional[np.ndarray]:
        """Apply --unassigned_policy; return the drop mask (None unless dropping)."""
        if not unassigned:
            return None
        policy = self._unassigned_policy
        if policy == "error":
            preview = ",".join(str(i) for i in unassigned[:10])
            raise RuntimeError(
                f"{len(unassigned)} items could not be assigned within capacity; "
                f"first unassigned row indices: {preview}"
            )
        if policy == "drop":
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[unassigned] = False
            return keep_mask
        for i in unassigned:  # keep_original
            key = int(sid_key[i])
            slot_count[key] = slot_count.get(key, 0) + 1
            final_index[i] = slot_count[key]
        return None

    @staticmethod
    def _stats(
        n: int,
        counts: np.ndarray,
        slot_count: Dict[int, int],
        cap: int,
        reassigned: int,
        unassigned_count: int,
    ) -> AssignmentStats:
        """Summarize raw vs final bucket occupancy.

        Final occupancy is read straight from ``slot_count`` (maintained during
        placement) instead of re-grouping the final keys.
        """
        if slot_count:
            vals = slot_count.values()
            final_collision = sum(1 for v in vals if v > cap)
            max_final = max(vals)
        else:
            final_collision = 0
            max_final = 0
        return AssignmentStats(
            total_items=n,
            raw_collision_buckets=int((counts > cap).sum()),
            final_collision_buckets=final_collision,
            reassigned_count=reassigned,
            unassigned_count=unassigned_count,
            max_final_bucket_size=max_final,
        )

    @staticmethod
    def _item_id_array(values: np.ndarray) -> pa.Array:
        """Encode item ids as int64 when integral, else as strings."""
        if np.issubdtype(values.dtype, np.integer):
            return pa.array(values, type=pa.int64())
        return pa.array(values, type=pa.string())

    def _codes_column(self, codes: np.ndarray) -> pa.Array:
        """Encode an (M, n_layers) matrix as list<int64> (or a CSV string)."""
        m, n_layers = codes.shape
        if self._is_csv:
            cols = [
                pc.cast(pa.array(codes[:, j]), pa.string()) for j in range(n_layers)
            ]
            return pc.binary_join_element_wise(*cols, _CODE_SEP)
        values = pa.array(codes.reshape(-1))
        offsets = pa.array(np.arange(0, (m + 1) * n_layers, n_layers, dtype=np.int32))
        return pa.ListArray.from_arrays(offsets, values)

    def _write(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        final_codes: np.ndarray,
        index: np.ndarray,
        keep_mask: Optional[np.ndarray],
        stats: AssignmentStats,
    ) -> None:
        """Chunked, vectorized write of the reassigned map (and diagnostics)."""
        if keep_mask is not None:
            item_ids = item_ids[keep_mask]
            origin_codes = origin_codes[keep_mask]
            final_codes = final_codes[keep_mask]
            index = index[keep_mask]
        writer = create_writer(
            self._output_path,
            writer_type=self._writer_type_str,
            quota_name=self._odps_data_quota_name,
            world_size=1,
        )
        n = len(item_ids)
        for s in range(0, n, _WRITE_CHUNK):
            e = min(s + _WRITE_CHUNK, n)
            writer.write(
                {
                    "item_id": self._item_id_array(item_ids[s:e]),
                    "origin_codebook": self._codes_column(origin_codes[s:e]),
                    "codebook": self._codes_column(final_codes[s:e]),
                    "index": pa.array(index[s:e], type=pa.int64()),
                }
            )
        writer.close()
        if self._diagnostics_output_path:
            self._write_diagnostics(stats)

    def _write_diagnostics(self, stats: AssignmentStats) -> None:
        """Write the one-row AssignmentStats table to the diagnostics path."""
        writer = create_writer(
            self._diagnostics_output_path,
            writer_type=self._writer_type_str,
            quota_name=self._odps_data_quota_name,
            world_size=1,
        )
        writer.write(
            {k: pa.array([v], type=pa.int64()) for k, v in asdict(stats).items()}
        )
        writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prevent SID codebook collisions (vectorized, within-band)."
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--diagnostics_output_path", default=None)
    parser.add_argument(
        "--reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter", "OdpsWriter"],
        default=None,
        help="Output writer; defaults to matching the input reader "
        "(CsvReader -> CsvWriter, etc.).",
    )
    parser.add_argument("--batch_size", type=int, default=100000)
    parser.add_argument("--item_id_field", default="item_id")
    parser.add_argument("--code_field", default="codes")
    parser.add_argument(
        "--codebook",
        required=True,
        help="Comma-separated per-layer codebook sizes, e.g. '8192,8192,8192'. "
        "Declares n_layers and the last-layer code space used for reassignment.",
    )
    parser.add_argument("--candidate_codes_field", default="candidate_codes")
    parser.add_argument(
        "--candidate_depth",
        type=int,
        default=None,
        help="Cap on candidate last-codes tried per overflow item (default: all).",
    )
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument(
        "--unassigned_policy",
        choices=["error", "drop", "keep_original"],
        default="error",
    )
    parser.add_argument(
        "--strategy",
        choices=["candidate", "random"],
        default="candidate",
        help="Reassignment strategy: 'candidate' uses model candidate_codes; "
        "'random' draws random within-band last-layer codes (no candidates).",
    )
    parser.add_argument(
        "--random_num_candidates",
        type=int,
        default=64,
        help="Random last-layer codes drawn per overflow item for --strategy random.",
    )
    parser.add_argument(
        "--rate_only",
        action="store_true",
        help="Compute + log stats only; skip writing the map (avoids the map-write "
        "cost when only the collision rate is needed).",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    CollisionRunner(parser.parse_args()).run()
