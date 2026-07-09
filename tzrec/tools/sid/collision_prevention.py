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
are deterministic given ``--seed`` and independent of input row order. I/O goes
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
from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import logger

_MASK64 = np.uint64((1 << 64) - 1)

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


def _splitmix64(x: np.ndarray, seed: int) -> np.ndarray:
    """Vectorized order-independent SplitMix64 hash of a uint64 array.

    A pure function of the input values (not their position), so it gives a
    stable, seedable tie-break that is invariant to input row order -- unlike a
    read-order index -- while staying fully vectorized.

    Args:
        x (np.ndarray): uint64 values to hash.
        seed (int): mixing seed.

    Returns:
        np.ndarray: uint64 hashes, same shape as ``x``.
    """
    with np.errstate(over="ignore"):
        z = x.astype(np.uint64) + np.uint64((seed * 0x9E3779B97F4A7C15) & int(_MASK64))
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _order_hash(item_ids: np.ndarray, seed: int) -> np.ndarray:
    """Per-item order-independent tie-break hash (uint64), vectorized.

    Integer ids hash directly; string/object ids are folded to uint64 via
    ``pandas.util.hash_array`` first. Both then pass through :func:`_splitmix64`
    so the ``seed`` is mixed in uniformly.

    Args:
        item_ids (np.ndarray): item id values.
        seed (int): mixing seed.

    Returns:
        np.ndarray: uint64 per-item hashes.
    """
    if np.issubdtype(item_ids.dtype, np.integer):
        base = item_ids.astype(np.uint64)
    else:
        import pandas as pd

        base = pd.util.hash_array(np.asarray(item_ids, dtype=object))
    return _splitmix64(base, seed)


class CollisionRunner:
    """Vectorized SID collision-prevention runner over the dataset reader/writer.

    The backend (CSV / Parquet / ODPS) is chosen by the reader/writer type, so
    the same path serves local files and MaxCompute tables.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        # Class name of the input reader, captured on the first read; used to
        # derive the writer type when --writer_type is unset.
        self._input_reader_cls_name: Optional[str] = None

    def run(self) -> AssignmentStats:
        """Read the SID map, assign, and write the reassigned map + stats."""
        item_ids, codes = self._load_codes()
        final_codes, index, keep_mask, stats = self._assign(item_ids, codes)
        if not self.args.rate_only:
            self._write(item_ids, codes, final_codes, index, keep_mask, stats)
        else:
            logger.info("rate_only: skipping map write")
        logger.info("SID collision prevention finished: %s", stats)
        return stats

    def _read(
        self, selected_cols: Optional[List[str]], capture_reader_cls: bool = False
    ) -> Iterable[Dict[str, pa.Array]]:
        reader = create_reader(
            input_path=self.args.input_path,
            batch_size=self.args.batch_size,
            selected_cols=selected_cols,
            reader_type=self.args.reader_type,
            quota_name=self.args.odps_data_quota_name,
        )
        if capture_reader_cls:
            self._input_reader_cls_name = reader.__class__.__name__
        yield from reader.to_batches()

    @staticmethod
    def _codes_matrix(arr: pa.Array) -> np.ndarray:
        """Decode a SID code column into an (N, n_layers) int64 matrix.

        Parquet/ODPS give a ``list<int64>`` cell; CSV gives a ``_CODE_SEP``
        string; a single-layer numeric CSV column may arrive already as ints.
        """
        if pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type):
            n = len(arr)
            flat = arr.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        elif pa.types.is_integer(arr.type):
            return arr.to_numpy(zero_copy_only=False).astype(np.int64).reshape(-1, 1)
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
        for batch in self._read(
            [self.args.item_id_field, self.args.code_field], capture_reader_cls=True
        ):
            id_chunks.append(
                batch[self.args.item_id_field].to_numpy(zero_copy_only=False)
            )
            code_chunks.append(self._codes_matrix(batch[self.args.code_field]))
        if not id_chunks:
            raise ValueError("SID input is empty.")
        item_ids = np.concatenate(id_chunks)
        codes = np.concatenate(code_chunks, axis=0)
        if codes.shape[1] < 1:
            raise ValueError("SID codes must have at least one layer.")
        return item_ids, codes

    def _load_candidate_last(
        self, overflow_id_arr: np.ndarray
    ) -> Dict[Any, np.ndarray]:
        """Map each overflow item id to its ordered candidate last-layer codes.

        Reads ``candidate_codes`` (list<list<int64>> for Parquet/ODPS, a
        ``_CAND_SEP``/``_CODE_SEP`` string for CSV) but keeps only the last code
        of each candidate SID and only for overflow items, so memory scales with
        the overflow fraction, not the whole table.
        """
        depth = self.args.candidate_depth
        field = self.args.candidate_codes_field
        cand: Dict[Any, np.ndarray] = {}
        for batch in self._read([self.args.item_id_field, field]):
            if field not in batch:
                break
            ids = batch[self.args.item_id_field].to_numpy(zero_copy_only=False)
            keep = np.where(np.isin(ids, overflow_id_arr))[0]
            if keep.size == 0:
                continue
            col = batch[field]
            if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
                self._collect_candidate_lists(ids, col, keep, depth, cand)
            else:
                self._collect_candidate_strings(ids, col, keep, depth, cand)
        return cand

    @staticmethod
    def _collect_candidate_lists(
        ids: np.ndarray,
        col: pa.Array,
        keep: np.ndarray,
        depth: Optional[int],
        cand: Dict[Any, np.ndarray],
    ) -> None:
        """Vectorized last-code extraction from a list<list<int64>> batch."""
        n = len(col)
        inner = col.flatten()  # list<int64>, one per candidate SID
        if len(inner) == 0:
            return
        k = len(inner) // n
        if len(inner) != n * k:
            raise ValueError("ragged candidate_codes: all items must share topk.")
        flat = inner.flatten().to_numpy(zero_copy_only=False).astype(np.int64)
        n_layers = flat.shape[0] // (n * k)
        last = flat.reshape(n, k, n_layers)[:, :, n_layers - 1]
        if depth is not None:
            last = last[:, :depth]
        for i in keep.tolist():
            cand[ids[i]] = last[i]

    def _collect_candidate_strings(
        self,
        ids: np.ndarray,
        col: pa.Array,
        keep: np.ndarray,
        depth: Optional[int],
        cand: Dict[Any, np.ndarray],
    ) -> None:
        """Per-row last-code extraction from a CSV compact-candidate batch."""
        values = col.to_pylist()
        for i in keep.tolist():
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
            cand[ids[i]] = np.asarray(
                last_codes[:depth] if depth else last_codes, dtype=np.int64
            )

    def _assign(
        self, item_ids: np.ndarray, codes: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], AssignmentStats]:
        """Cap buckets and reassign overflow within-band; return the final map."""
        cap = self.args.max_items_per_codebook
        if cap < 1:
            raise ValueError(f"max_items_per_codebook must be >= 1, got {cap}")
        n, n_layers = codes.shape
        last = codes[:, n_layers - 1].astype(np.int64)

        band_id = self._band_ids(codes)
        order = _order_hash(item_ids, self.args.seed)

        # rank within bucket by (bucket, order-hash); collision RATE is invariant
        # to which cap items are kept, but the choice is made deterministic here.
        group_radix = int(last.max()) + 1 if n else 1
        group_key = band_id * group_radix + last
        rank, counts_by_row = self._within_bucket_rank(group_key, order)
        overflow_mask = rank >= cap
        overflow_order = np.where(overflow_mask)[0]
        overflow_order = overflow_order[
            np.lexsort((order[overflow_order], group_key[overflow_order]))
        ]

        cand, last_size = self._candidates(item_ids, overflow_order)
        last_size = max(last_size, group_radix)

        sid_key = band_id * last_size + last
        slot_count: Dict[int, int] = {}
        first = np.unique(group_key, return_index=True)[1]
        for i in first.tolist():
            slot_count[int(band_id[i]) * last_size + int(last[i])] = int(
                min(counts_by_row[i], cap)
            )

        final_key = sid_key.copy()
        final_index = (rank + 1).astype(np.int64)
        reassigned, unassigned = self._place_overflow(
            overflow_order,
            item_ids,
            band_id,
            last,
            sid_key,
            cand,
            last_size,
            cap,
            slot_count,
            final_key,
            final_index,
        )

        keep_mask: Optional[np.ndarray] = None
        if self.args.unassigned_policy == "drop" and unassigned:
            keep_mask = np.ones(n, dtype=bool)
            keep_mask[unassigned] = False

        final_last = (final_key % last_size).astype(np.int64)
        final_codes = codes.copy()
        final_codes[:, n_layers - 1] = final_last

        stats = self._stats(
            n,
            counts_by_row,
            first,
            final_key,
            keep_mask,
            cap,
            reassigned,
            len(unassigned),
        )
        return final_codes, final_index, keep_mask, stats

    @staticmethod
    def _band_ids(codes: np.ndarray) -> np.ndarray:
        """Dense integer id per distinct ``(prefix)`` band (all layers but last)."""
        n, n_layers = codes.shape
        if n_layers == 1:
            return np.zeros(n, dtype=np.int64)
        _, band_id = np.unique(codes[:, : n_layers - 1], axis=0, return_inverse=True)
        return band_id.astype(np.int64).reshape(-1)

    @staticmethod
    def _within_bucket_rank(
        group_key: np.ndarray, order: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Per-row within-bucket rank (0-based) and its bucket size."""
        n = group_key.shape[0]
        sort_idx = np.lexsort((order, group_key))
        sk = group_key[sort_idx]
        _, first, counts = np.unique(sk, return_index=True, return_counts=True)
        rank_sorted = np.arange(n) - np.repeat(first, counts)
        counts_sorted = np.repeat(counts, counts)
        rank = np.empty(n, dtype=np.int64)
        size = np.empty(n, dtype=np.int64)
        rank[sort_idx] = rank_sorted
        size[sort_idx] = counts_sorted
        return rank, size

    def _candidates(
        self,
        item_ids: np.ndarray,
        overflow_order: np.ndarray,
    ) -> Tuple[Dict[Any, np.ndarray], int]:
        """Build per-overflow-item last-code candidate lists and the code space."""
        if self.args.strategy == "random":
            size = self.args.random_last_layer_size
            if size is None or size < 2:
                raise ValueError(
                    "strategy='random' requires --random_last_layer_size >= 2."
                )
            num = min(self.args.random_num_candidates, size - 1)
            draws = self._random_draws(overflow_order, item_ids, num, size)
            cand = {item_ids[i]: draws[j] for j, i in enumerate(overflow_order)}
            return cand, size
        if overflow_order.size == 0:
            return {}, 1
        cand = self._load_candidate_last(item_ids[overflow_order])
        if not cand:
            raise ValueError(
                "map has overflow items but candidate_codes yielded no candidates."
            )
        max_last = max((int(a.max()) for a in cand.values() if a.size), default=0)
        return cand, max_last + 1

    def _random_draws(
        self, overflow_order: np.ndarray, item_ids: np.ndarray, num: int, size: int
    ) -> np.ndarray:
        """Order-independent random last-layer draws per overflow item, (M, num)."""
        h = _order_hash(item_ids[overflow_order], self.args.seed)
        k = np.arange(num, dtype=np.uint64)
        with np.errstate(over="ignore"):
            mixed = _splitmix64(
                h[:, None] + k[None, :] * np.uint64(0x9E3779B97F4A7C15), self.args.seed
            )
        return (mixed % np.uint64(size)).astype(np.int64)

    def _place_overflow(
        self,
        overflow_order: np.ndarray,
        item_ids: np.ndarray,
        band_id: np.ndarray,
        last: np.ndarray,
        sid_key: np.ndarray,
        cand: Dict[Any, np.ndarray],
        last_size: int,
        cap: int,
        slot_count: Dict[int, int],
        final_key: np.ndarray,
        final_index: np.ndarray,
    ) -> Tuple[int, List[int]]:
        """Greedy single pass: place each overflow item in the first free slot."""
        reassigned = 0
        unassigned: List[int] = []
        for i in overflow_order:
            base = int(band_id[i]) * last_size
            origin_last = int(last[i])
            placed = False
            for code in cand.get(item_ids[i], ()):  # ordered best-first
                code = int(code)
                if code == origin_last:
                    continue
                ck = base + code
                count = slot_count.get(ck, 0)
                if count < cap:
                    slot_count[ck] = count + 1
                    final_key[i] = ck
                    final_index[i] = count + 1
                    reassigned += 1
                    placed = True
                    break
            if not placed:
                unassigned.append(int(i))
        self._resolve_unassigned(unassigned, sid_key, cap, slot_count, final_index)
        return reassigned, unassigned

    def _resolve_unassigned(
        self,
        unassigned: List[int],
        sid_key: np.ndarray,
        cap: int,
        slot_count: Dict[int, int],
        final_index: np.ndarray,
    ) -> None:
        """Apply --unassigned_policy to items that found no free slot."""
        if not unassigned:
            return
        policy = self.args.unassigned_policy
        if policy == "error":
            preview = ",".join(str(i) for i in unassigned[:10])
            raise RuntimeError(
                f"{len(unassigned)} items could not be assigned within capacity; "
                f"first unassigned row indices: {preview}"
            )
        if policy == "keep_original":
            for i in unassigned:
                key = int(sid_key[i])
                slot_count[key] = slot_count.get(key, 0) + 1
                final_index[i] = slot_count[key]

    @staticmethod
    def _stats(
        n: int,
        counts_by_row: np.ndarray,
        first: np.ndarray,
        final_key: np.ndarray,
        keep_mask: Optional[np.ndarray],
        cap: int,
        reassigned: int,
        unassigned_count: int,
    ) -> AssignmentStats:
        """Summarize raw vs final bucket occupancy."""
        raw_counts = counts_by_row[first]
        kept_key = final_key if keep_mask is None else final_key[keep_mask]
        if kept_key.size:
            _, fc = np.unique(kept_key, return_counts=True)
            final_collision = int((fc > cap).sum())
            max_final = int(fc.max())
        else:
            final_collision = 0
            max_final = 0
        return AssignmentStats(
            total_items=n,
            raw_collision_buckets=int((raw_counts > cap).sum()),
            final_collision_buckets=final_collision,
            reassigned_count=reassigned,
            unassigned_count=unassigned_count,
            max_final_bucket_size=max_final,
        )

    def _writer_type(self) -> Optional[str]:
        # Derive the writer from the input reader (hitrate.py idiom) when unset,
        # so the output backend matches the input's. ODPS output still resolves
        # to OdpsWriter via create_writer's path detection regardless.
        if self.args.writer_type:
            return self.args.writer_type
        if self._input_reader_cls_name:
            return self._input_reader_cls_name.replace("Reader", "Writer")
        return None

    def _is_csv_output(self) -> bool:
        return self._writer_type() == "CsvWriter"

    @staticmethod
    def _item_id_array(values: np.ndarray) -> pa.Array:
        if np.issubdtype(values.dtype, np.integer):
            return pa.array(values, type=pa.int64())
        return pa.array([str(v) for v in values], type=pa.string())

    def _codes_column(self, codes: np.ndarray) -> pa.Array:
        """Encode an (M, n_layers) matrix as list<int64> (or a CSV string)."""
        m, n_layers = codes.shape
        if self._is_csv_output():
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
            self.args.output_path,
            writer_type=self._writer_type(),
            quota_name=self.args.odps_data_quota_name,
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
        if self.args.diagnostics_output_path:
            self._write_diagnostics(stats)

    def _write_diagnostics(self, stats: AssignmentStats) -> None:
        writer = create_writer(
            self.args.diagnostics_output_path,
            writer_type=self._writer_type(),
            quota_name=self.args.odps_data_quota_name,
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
    parser.add_argument("--candidate_codes_field", default="candidate_codes")
    parser.add_argument(
        "--candidate_depth",
        type=int,
        default=None,
        help="Cap on candidate last-codes tried per overflow item (default: all).",
    )
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2026)
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
        "--random_last_layer_size",
        type=int,
        default=None,
        help="Code-space size of the last SID layer; required for --strategy random.",
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
