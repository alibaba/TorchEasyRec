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

"""Sparse lookup for offline predict of distributed embedding exports.

A distributed embedding export (``USE_DISTRIBUTED_EMBEDDING=1``) keeps the
embedding tables out of ``scripted_model.pt``: ``sparse/`` holds the tables as
npz shards, and the dense graph takes the looked-up embeddings as inputs, laid
out by ``dense_meta.json``. This module looks the tables up offline with
NVIDIA NV-Embeddings (NVE) hierarchical layers through pynve, as online serving
does, including rowwise INT8 (``DIST_QUANT=INT8``) tables, which NVE dequantizes
on lookup.

Tables with the same storage dtype, dim and default row are fused into one host
parameter server (an nvhm hash table) and one NVE layer per GPU, and each
(group, pooling) is looked up in one call, since every NVE lookup pays a fixed
cost (a GPU cache probe plus a host round trip). Static and dynamic tables are
never fused together. The host table is shared by every local GPU; each GPU only
adds its own cache.

In a fused key, the high ``B`` bits after the sign bit hold the table's member
index in its group, and the low ``P = 63 - B`` bits hold the key folded to ``P``
bits and xor-ed with a per-member salt. ``B`` is the bit length of the group
size, so ``INT64_MAX``, the miss key, is never a member key. Keys of different
tables never collide; a table whose exported keys collide after folding is
served on its own, with its raw keys, so every exported key resolves exactly.
"""

import contextlib
import glob
import itertools
import json
import math
import os
import time
import zipfile
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from tzrec.datasets.utils import Batch, batch_to_dict
from tzrec.utils import npz_util
from tzrec.utils.logging_util import logger

_LOAD_BATCH_ROWS = 1 << 20
# NVE uses key -1 as its empty-slot marker, so -1 ids look up this miss key.
_MISSING_KEY = torch.iinfo(torch.int64).max
_MIN_GPU_CACHE_BYTES = 1 << 20
# GPU memory left to the dense forward, lookup scratch and the allocator; the
# measured peak of a 561-feature MMoE predict was ~4% of a 72 GB GPU.
_GPU_RESERVE_FRACTION = 0.2
# NVE memory of a layer besides its cache: ~81 MiB measured on pynve 26.10 (the
# modify buffer at the default max_modify_size) plus per-stream lookup buffers.
_LAYER_OVERHEAD_BYTES = 96 << 20
# NOTE: at most 11 member bits, so the payload never drops below 52 bits.
_MAX_GROUP_MEMBERS = (1 << 11) - 1
_SALT_MULTIPLIER = 0x9E3779B97F4A7C15


def _gpu_cache_ratio(devices: List[torch.device], storage_bytes: List[int]) -> float:
    """Fraction of each group's stored bytes cached on every GPU.

    ``DIST_EMB_GPU_CACHE_RATIO`` unset or ``auto`` sizes the caches from the free
    memory of the most constrained GPU, less a reserve for the dense forward and
    the layers' own buffers, rounded down to 4 decimals; a number in (0, 1] is
    used as is.

    Args:
        devices: the GPUs, each holding one cache per group.
        storage_bytes: the stored bytes of each group.

    Returns:
        The ratio the caches are allocated with.
    """
    value = os.environ.get("DIST_EMB_GPU_CACHE_RATIO", "auto")
    if value != "auto":
        ratio = float(value)
        if not 0 < ratio <= 1:
            raise ValueError(
                f"DIST_EMB_GPU_CACHE_RATIO must be auto or in (0, 1], got {value}"
            )
        return ratio
    layer_bytes = len(storage_bytes) * (_LAYER_OVERHEAD_BYTES + _MIN_GPU_CACHE_BYTES)
    budget = min(
        free - _GPU_RESERVE_FRACTION * total - layer_bytes
        for free, total in (torch.cuda.mem_get_info(d) for d in devices)
    )
    total_bytes = sum(storage_bytes)
    ratio = min(1.0, max(budget, 0) / max(total_bytes, 1))
    ratio = math.floor(ratio * 1e4) / 1e4
    logger.info(
        f"DIST_EMB_GPU_CACHE_RATIO=auto resolved to {ratio} (GPU cache budget "
        f"{budget / 2**30:.2f} GiB on the most constrained of {len(devices)} GPUs, "
        f"{total_bytes / 2**30:.2f} GiB of embeddings); set a lower "
        "DIST_EMB_GPU_CACHE_RATIO if GPU memory runs out."
    )
    if ratio == 0:
        logger.warning(
            "no GPU memory left for embedding caches: each gets the "
            f"{_MIN_GPU_CACHE_BYTES} bytes minimum and lookups read the host tables."
        )
    return ratio


def _remove_suffix(feat_name_impl: str) -> str:
    """Strip the ``__ebc`` / ``__ec`` / ``__lengths`` suffix of a dense input."""
    for suffix in ("__ebc", "__ec", "__lengths"):
        if feat_name_impl.endswith(suffix):
            return feat_name_impl[: -len(suffix)]
    return feat_name_impl


def _input_name(feat_name_impl: str) -> str:
    """Feature name in the parsed batch, without ``@<table>`` of shared features."""
    return _remove_suffix(feat_name_impl).split("@")[0]


def _key_tag(member_idx: int, payload_bits: int) -> int:
    """Member index in the high bits plus the member salt in the payload bits."""
    salt = ((member_idx + 1) * _SALT_MULTIPLIER) & ((1 << payload_bits) - 1)
    return (member_idx << payload_bits) | salt


def _fold_keys(keys: torch.Tensor, payload_bits: int) -> torch.Tensor:
    """Fold non-negative int64 keys into their low ``payload_bits`` bits."""
    return (keys ^ (keys >> payload_bits)) & ((1 << payload_bits) - 1)


def _encode_keys(
    keys: torch.Tensor, tags: torch.Tensor, payload_bits: int
) -> torch.Tensor:
    """Fused keys of ``keys`` given each key's member tag; misses map to INT64_MAX."""
    missing = (keys == -1) | (keys == _MISSING_KEY)
    return torch.where(missing, _MISSING_KEY, _fold_keys(keys, payload_bits) ^ tags)


class _Group(NamedTuple):
    """Tables fused into one host parameter server and one layer per device."""

    dtype: str
    dim: int
    default_row: Optional[np.ndarray]
    members: List[str]
    # member index of each table; isolated tables keep their slot unused.
    member_ids: List[int]
    # payload bits of the fused key; None serves the single member's raw keys.
    payload_bits: Optional[int]


class _Bucket(NamedTuple):
    """One NVE lookup per batch: the features of a (group, pooling)."""

    group_idx: int
    pooling: str
    # features in lookup order, grouped by the KJT they come from.
    feats: List[str]
    # (KJT index, first row in that KJT's permuted rows, number of rows)
    segments: List[Tuple[int, int, int]]
    # per-feature member tag and use_weight.
    tags: List[int]
    use_weight: List[bool]


class _PlanTensors(NamedTuple):
    """A plan's index tensors on one device."""

    # per KJT
    permutes: List[torch.Tensor]
    # per bucket; use_weight is None when no feature of the bucket is weighted.
    tags: List[torch.Tensor]
    use_weight: List[Optional[torch.Tensor]]
    # dense group -> its columns in the concatenated pooled lookups.
    group_cols: Dict[str, torch.Tensor]


class _Plan(NamedTuple):
    """How batches of one KJT key layout are looked up."""

    # per KJT: the key index of each permuted row.
    permute_rows: List[List[int]]
    buckets: List[_Bucket]
    # sequence ``<feat>__lengths`` -> (KJT index, key index)
    seq_lengths: Dict[str, Tuple[int, int]]
    tensors: Dict[torch.device, _PlanTensors]


class DistributedEmbeddingLookup:
    """Looks up the sparse inputs of a distributed embedding export with NVE.

    Args:
        export_dir: distributed embedding export directory.
        devices: CUDA devices to serve lookups on; all of them share one host
            copy of every table.
    """

    def __init__(self, export_dir: str, devices: List[torch.device]) -> None:
        import pynve.nve as nve
        from pynve.torch import nve_dtypes, nve_layers, nve_ps

        if not devices or any(d.type != "cuda" for d in devices):
            raise ValueError(
                "predict of a distributed embedding export requires CUDA devices, "
                f"got {devices}."
            )
        sparse_dir = os.path.join(export_dir, "sparse")
        with open(os.path.join(export_dir, "dense_meta.json")) as f:
            self._dense_meta: Dict[str, List[str]] = json.load(f)
        with open(os.path.join(sparse_dir, "sparse_embedding.json")) as f:
            emb_meta: Dict[str, Dict] = json.load(f)
        with open(os.path.join(sparse_dir, "sparse_features.json")) as f:
            self._feat_meta: Dict[str, Dict] = json.load(f)
        self._plans: Dict[Tuple[Tuple[str, ...], ...], _Plan] = {}

        used_feats = []
        for group, feats in self._dense_meta.items():
            for feat in feats:
                if group == "sequence__ec" and feat.endswith("__lengths"):
                    continue
                used_feats.append(feat)
        self._used_feats = list(dict.fromkeys(used_feats))

        storage_types = {
            "float32": torch.float32,
            "float16": torch.float16,
            "QUint8RowwiseF16": nve.DataType_t.QUint8RowwiseF16,
        }
        with contextlib.ExitStack() as stack:
            # npz entry name -> the shards holding it; each shard is opened once.
            entries: Dict[str, List[zipfile.ZipFile]] = {}
            for path in sorted(
                glob.glob(os.path.join(sparse_dir, "sparse_embeddings-*-of-*.npz"))
                + glob.glob(
                    os.path.join(sparse_dir, "sparse_dynamic_embedding-*-of-*.npz")
                )
            ):
                zf = stack.enter_context(zipfile.ZipFile(path))
                for name in zf.namelist():
                    entries.setdefault(name[: -len(".npy")], []).append(zf)

            candidates: Dict[Tuple, List[str]] = {}
            default_rows: Dict[str, Optional[np.ndarray]] = {}
            for emb_name in sorted(
                {self._feat_meta[f]["embedding_name"] for f in self._used_feats}
            ):
                meta = emb_meta[emb_name]
                if meta["dtype"] not in storage_types:
                    raise ValueError(
                        f"unsupported dtype {meta['dtype']} of embedding {emb_name}."
                    )
                default_row = None
                default_name = (
                    meta.get("default_value_name") if meta["is_dynamic"] else None
                )
                if default_name:
                    default_row = next(
                        npz_util.iter_npz_entry(
                            entries[default_name][0], default_name, 1
                        )
                    )[0].copy()
                default_rows[emb_name] = default_row
                key = (
                    meta["is_dynamic"],
                    meta["dtype"],
                    meta["dimension"],
                    b"" if default_row is None else default_row.tobytes(),
                )
                candidates.setdefault(key, []).append(emb_name)

            groups: List[_Group] = []
            for (_, dtype, dim, _), names in candidates.items():
                for start in range(0, len(names), _MAX_GROUP_MEMBERS):
                    members = names[start : start + _MAX_GROUP_MEMBERS]
                    default_row = default_rows[members[0]]
                    payload_bits = 63 - len(members).bit_length()
                    failed = self._fuse_members(
                        groups, members, payload_bits, emb_meta, entries, default_row
                    )
                    # a fold collision depends only on the payload width, so retry
                    # once, together, only when fewer members leave a wider payload.
                    retry_bits = 63 - len(failed).bit_length()
                    if len(failed) > 1 and retry_bits > payload_bits:
                        begin, retried = time.time(), len(failed)
                        failed = self._fuse_members(
                            groups, failed, retry_bits, emb_meta, entries, default_row
                        )
                        logger.info(
                            f"retried fusing {retried} dynamic embeddings at "
                            f"{retry_bits} bits: {retried - len(failed)} fused, "
                            f"{len(failed)} unfused, in {time.time() - begin:.1f}s."
                        )
                    elif len(failed) > 1:
                        logger.info(
                            f"skip retrying {len(failed)} dynamic embeddings: "
                            f"{retry_bits} payload bits are no wider than "
                            f"{payload_bits}."
                        )
                    for name in failed:
                        logger.info(f"serve dynamic embedding {name} unfused.")
                        groups.append(
                            _Group(dtype, dim, default_rows[name], [name], [0], None)
                        )

            group_rows = [
                sum(emb_meta[e]["shape"][0] for e in group.members) for group in groups
            ]
            storage_bytes = [
                rows * nve_dtypes.stored_row_bytes(g.dim, storage_types[g.dtype])
                for rows, g in zip(group_rows, groups)
            ]
            ratio = _gpu_cache_ratio(devices, storage_bytes)
            # table name -> (group index, member tag)
            self._table_group: Dict[str, Tuple[int, int]] = {}
            self._groups = groups
            self._layers: Dict[torch.device, List[torch.nn.Module]] = {
                d: [] for d in devices
            }
            for group_idx, group in enumerate(groups):
                storage_type = storage_types[group.dtype]
                num_rows = group_rows[group_idx]
                ps = nve_ps.NVEParameterServer(
                    0, group.dim, storage_type, initial_size=max(num_rows, 1024)
                )
                for member_idx, emb_name in zip(group.member_ids, group.members):
                    if group.payload_bits is None:
                        tag = 0
                    else:
                        tag = _key_tag(member_idx, group.payload_bits)
                    self._table_group[emb_name] = (group_idx, tag)
                    if emb_meta[emb_name]["is_dynamic"]:
                        rows = self._load_dynamic(
                            ps, emb_meta[emb_name], entries, group, tag
                        )
                    else:
                        rows = self._load_static(
                            ps, entries.get(emb_name, []), emb_name, group, tag
                        )
                    if rows != emb_meta[emb_name]["shape"][0]:
                        raise ValueError(
                            f"{emb_name}: {rows} rows in the npz shards, but "
                            "sparse_embedding.json declares "
                            f"{emb_meta[emb_name]['shape'][0]}."
                        )
                config = {}
                if group.default_row is not None:
                    config["default_embedding"] = torch.from_numpy(group.default_row)
                gpu_cache_size = max(
                    int(ratio * storage_bytes[group_idx]), _MIN_GPU_CACHE_BYTES
                )
                for device in devices:
                    # SUM is the default mode; MEAN and NONE lookups pass their mode,
                    # so one layer, with one GPU cache, serves every pooling.
                    self._layers[device].append(
                        nve_layers.NVEmbeddingBag(
                            num_rows,
                            group.dim,
                            storage_type,
                            nve_layers.LayerType.Hierarchical,
                            "sum",
                            gpu_cache_size=gpu_cache_size,
                            storage=ps,
                            optimize_for_training=False,
                            device=device,
                            config=dict(config),
                        )
                    )
                logger.info(
                    f"loaded sparse embeddings {group.members}: rows={num_rows}, "
                    f"dim={group.dim}, dtype={group.dtype}, "
                    f"fused_key_payload_bits={group.payload_bits}, "
                    f"gpu_cache_bytes={gpu_cache_size}."
                )

        # features looked up together, per (group, pooling).
        self._buckets: Dict[Tuple[int, str], List[str]] = {}
        for feat in self._used_feats:
            meta = self._feat_meta[feat]
            group_idx, _ = self._table_group[meta["embedding_name"]]
            self._buckets.setdefault((group_idx, meta["pooling"]), []).append(feat)

    @classmethod
    def _fuse_members(
        cls,
        groups: List[_Group],
        members: List[str],
        payload_bits: int,
        emb_meta: Dict[str, Dict],
        entries: Dict[str, List[zipfile.ZipFile]],
        default_row: Optional[np.ndarray],
    ) -> List[str]:
        """Fuse ``members`` into one group at ``payload_bits`` and append it.

        Members are numbered by their position in ``members``; those whose keys
        collide after folding are left out, keeping the others' numbers. A lone
        member keeps its raw keys and is not checked.

        Returns:
            The members left out, in order.
        """
        meta0 = emb_meta[members[0]]
        if len(members) == 1:
            # a lone table needs no member tag, so its raw keys are exact.
            groups.append(
                _Group(
                    meta0["dtype"], meta0["dimension"], default_row, members, [0], None
                )
            )
            return []
        fused, fused_ids, failed = [], [], []
        for member_idx, name in enumerate(members):
            meta = emb_meta[name]
            if meta["is_dynamic"] and cls._has_fold_collision(
                meta, entries.get(meta["key_name"], []), payload_bits
            ):
                logger.info(
                    f"keys of dynamic embedding {name} collide after folding to "
                    f"{payload_bits} bits."
                )
                failed.append(name)
            else:
                fused.append(name)
                fused_ids.append(member_idx)
        if fused:
            groups.append(
                _Group(
                    meta0["dtype"],
                    meta0["dimension"],
                    default_row,
                    fused,
                    fused_ids,
                    payload_bits,
                )
            )
        return failed

    @staticmethod
    def _has_fold_collision(
        meta: Dict, shards: List[zipfile.ZipFile], payload_bits: int
    ) -> bool:
        """Whether two exported keys of a dynamic table fold to the same value.

        The folded keys are collected and sorted in place, 8 bytes per row, one
        table at a time. Keys below 2^payload_bits fold to themselves.
        """
        folded = np.empty(meta["shape"][0], dtype=np.int64)
        num_keys, max_key = 0, 0
        for zf in shards:
            for keys in npz_util.iter_npz_entry(zf, meta["key_name"], _LOAD_BATCH_ROWS):
                if len(keys) == 0:
                    continue
                if num_keys + len(keys) > len(folded):
                    raise ValueError(
                        f"{meta['key_name']}: more keys in the npz shards than the "
                        f"{len(folded)} rows sparse_embedding.json declares."
                    )
                max_key = max(max_key, int(keys.max()))
                folded[num_keys : num_keys + len(keys)] = _fold_keys(
                    torch.from_numpy(keys), payload_bits
                ).numpy()
                num_keys += len(keys)
        if max_key < (1 << payload_bits):
            return False
        folded = folded[:num_keys]
        folded.sort()
        for start in range(0, len(folded) - 1, _LOAD_BATCH_ROWS):
            chunk = folded[start : start + _LOAD_BATCH_ROWS + 1]
            if np.any(chunk[1:] == chunk[:-1]):
                return True
        return False

    @classmethod
    def _load_dynamic(
        cls,
        ps: object,
        meta: Dict,
        entries: Dict[str, List[zipfile.ZipFile]],
        group: _Group,
        tag: int,
    ) -> int:
        """Insert the key/value rows of a dynamic table; return the row count."""
        rows = 0
        for zf in entries.get(meta["key_name"], []):
            for keys, values in zip(
                npz_util.iter_npz_entry(zf, meta["key_name"], _LOAD_BATCH_ROWS),
                npz_util.iter_npz_entry(zf, meta["value_name"], _LOAD_BATCH_ROWS),
                strict=True,
            ):
                ps.insert(cls._stored_keys(keys, group, tag), torch.from_numpy(values))
                rows += len(keys)
        return rows

    @classmethod
    def _load_static(
        cls,
        ps: object,
        shards: List[zipfile.ZipFile],
        emb_name: str,
        group: _Group,
        tag: int,
    ) -> int:
        """Insert the rows of a static table keyed by row index; return the count."""
        offset = 0
        for zf in shards:
            for values in npz_util.iter_npz_entry(zf, emb_name, _LOAD_BATCH_ROWS):
                keys = np.arange(offset, offset + len(values), dtype=np.int64)
                ps.insert(cls._stored_keys(keys, group, tag), torch.from_numpy(values))
                offset += len(values)
        return offset

    @staticmethod
    def _stored_keys(keys: np.ndarray, group: _Group, tag: int) -> torch.Tensor:
        """Host parameter server keys of exported keys."""
        keys_t = torch.from_numpy(keys)
        if group.payload_bits is None:
            return keys_t
        return _fold_keys(keys_t, group.payload_bits) ^ tag

    def _plan(self, batch: Batch) -> _Plan:
        """The lookup plan of the batch's KJT key layout."""
        layout = tuple(tuple(kjt.keys()) for kjt in batch.sparse_features.values())
        plan = self._plans.get(layout)
        if plan is not None:
            return plan
        where: Dict[str, Tuple[int, int]] = {}
        for i, keys in enumerate(layout):
            for j, key in enumerate(keys):
                where.setdefault(key, (i, j))
        missing = [f for f in self._used_feats if _input_name(f) not in where]
        if missing:
            raise ValueError(f"sparse inputs of {missing} are not in the batch.")

        permute_rows: List[List[int]] = [[] for _ in layout]
        buckets, cols = [], {}
        pooled_cols = 0
        for (group_idx, pooling), feats in self._buckets.items():
            dim = self._groups[group_idx].dim
            feats = sorted(feats, key=lambda f: where[_input_name(f)][0])
            segments = []
            for i, kjt_feats in itertools.groupby(
                feats, key=lambda f: where[_input_name(f)][0]
            ):
                rows = [where[_input_name(f)][1] for f in kjt_feats]
                segments.append((i, len(permute_rows[i]), len(rows)))
                permute_rows[i].extend(rows)
            tags = [
                self._table_group[self._feat_meta[f]["embedding_name"]][1]
                for f in feats
            ]
            use_weight = [self._feat_meta[f]["use_weight"] for f in feats]
            buckets.append(
                _Bucket(group_idx, pooling, feats, segments, tags, use_weight)
            )
            if pooling != "NONE":
                for f in feats:
                    cols[f] = range(pooled_cols, pooled_cols + dim)
                    pooled_cols += dim

        group_cols, seq_lengths = {}, {}
        for group, feats in self._dense_meta.items():
            if group != "sequence__ec":
                group_cols[group] = [c for f in feats for c in cols[f]]
                continue
            for feat in feats:
                if feat.endswith("__lengths"):
                    seq_lengths[feat] = where[_input_name(feat)]
        tensors = {
            d: _PlanTensors(
                [torch.tensor(r, dtype=torch.int32, device=d) for r in permute_rows],
                [torch.tensor(b.tags, device=d) for b in buckets],
                [
                    torch.tensor(b.use_weight, device=d) if any(b.use_weight) else None
                    for b in buckets
                ],
                {g: torch.tensor(c, device=d) for g, c in group_cols.items()},
            )
            for d in self._layers
        }
        plan = _Plan(permute_rows, buckets, seq_lengths, tensors)
        self._plans[layout] = plan
        return plan

    def __call__(
        self, batch: Batch, batch_size: Optional[torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Build the dense graph's inputs of a batch.

        Every sparse KJT is copied to ``device`` once and permuted so that each
        (group, pooling) is one slice; no per-feature sparse tensor is built. The
        dense graph never reads raw sparse inputs: export replaces the sparse
        lookups with ``dense_meta`` inputs and drops what only fed them.

        Args:
            batch: the batch to look up.
            batch_size: the ``batch_size`` input of tiled models, None for none.
            device: the device to look up on, one of the constructor devices.

        Returns:
            The dense graph's inputs on ``device``: one concatenated embedding
            tensor per ``dense_meta`` group, ``<feat>`` / ``<feat>__lengths`` of
            every sequence feature, and the batch's non-sparse inputs.
        """
        plan = self._plan(batch)
        tensors = plan.tensors[device]
        layers = self._layers[device]
        stride = 0
        lengths, permuted = [], []
        for kjt, rows, permute in zip(
            batch.sparse_features.values(), plan.permute_rows, tensors.permutes
        ):
            keys, weights = kjt.keys(), kjt.weights_or_none()
            stride = kjt.lengths().numel() // max(len(keys), 1)
            kjt_lengths = kjt.lengths().to(device, torch.int64, non_blocking=True)
            kjt_lengths = kjt_lengths.view(len(keys), stride)
            lengths.append(kjt_lengths)
            if not rows:
                permuted.append(None)
                continue
            length_per_key = kjt.length_per_key()
            sizes = [length_per_key[j] for j in rows]
            permuted.append(
                torch.ops.fbgemm.permute_2D_sparse_data(
                    permute,
                    kjt_lengths,
                    kjt.values().to(device, torch.int64, non_blocking=True),
                    None
                    if weights is None
                    else weights.to(device, torch.float32, non_blocking=True),
                    sum(sizes),
                )
                + ([0, *itertools.accumulate(sizes)],)
            )

        out: Dict[str, torch.Tensor] = {}
        blocks = []
        for bucket, tags, use_weight in zip(
            plan.buckets, tensors.tags, tensors.use_weight
        ):
            group = self._groups[bucket.group_idx]
            values, bucket_lengths, weights, counts = [], [], [], []
            for i, row, num_rows in bucket.segments:
                kjt_lengths, kjt_values, kjt_weights, value_offsets = permuted[i]
                begin, end = value_offsets[row], value_offsets[row + num_rows]
                values.append(kjt_values[begin:end])
                bucket_lengths.append(kjt_lengths[row : row + num_rows])
                weights.append(
                    torch.ones(end - begin, device=device)
                    if kjt_weights is None
                    else kjt_weights[begin:end]
                )
                counts += [
                    value_offsets[r + 1] - value_offsets[r]
                    for r in range(row, row + num_rows)
                ]
            keys = torch.cat(values) if len(values) > 1 else values[0]
            feat_lengths = torch.cat(bucket_lengths)
            num_keys = sum(counts)
            if num_keys == 0:
                # NVE rejects an empty lookup; every bag is empty, so zeros.
                if bucket.pooling == "NONE":
                    for feat in bucket.feats:
                        out[_remove_suffix(feat)] = torch.zeros(
                            0, group.dim, device=device
                        )
                else:
                    blocks.append(
                        torch.zeros(
                            stride, len(bucket.feats) * group.dim, device=device
                        )
                    )
                continue
            if group.payload_bits is None:
                keys = torch.where(keys == -1, _MISSING_KEY, keys)
            else:
                key_tags = torch.repeat_interleave(
                    tags, feat_lengths.sum(1), output_size=num_keys
                )
                keys = _encode_keys(keys, key_tags, group.payload_bits)
            layer = layers[bucket.group_idx]
            if bucket.pooling == "NONE":
                # dequantized rows come back in fp16, the dense graph is fp32.
                emb = layer(keys, mode="concat").float()
                for feat, e in zip(bucket.feats, emb.split(counts)):
                    out[_remove_suffix(feat)] = e
                continue
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(
                feat_lengths.flatten()
            )
            bag_weights = None
            if use_weight is not None:
                bag_weights = torch.cat(weights) if len(weights) > 1 else weights[0]
                if not all(bucket.use_weight):
                    weighted = torch.repeat_interleave(
                        use_weight, feat_lengths.sum(1), output_size=num_keys
                    )
                    bag_weights = torch.where(weighted, bag_weights, 1.0)
            emb = layer(keys, offsets, bag_weights, mode=bucket.pooling.lower()).float()
            # bags are feature-major; lay them out as [batch, feature * dim].
            blocks.append(
                emb.view(len(bucket.feats), stride, group.dim)
                .transpose(0, 1)
                .reshape(stride, -1)
            )

        if blocks:
            pooled = torch.cat(blocks, dim=1) if len(blocks) > 1 else blocks[0]
            for group, cols in tensors.group_cols.items():
                out[group] = pooled.index_select(1, cols)
        for feat, (i, j) in plan.seq_lengths.items():
            out[feat] = lengths[i][j].int()
        rest = batch_to_dict(
            batch.dense_features,
            {},
            batch.sequence_mulval_lengths,
            batch.sequence_dense_features,
            batch.additional_infos,
            batch_size,
            torch.int64,
        )
        for k, v in rest.items():
            out[k] = v.to(device, non_blocking=True)
        return out


class DistributedEmbeddingModel:
    """Predicts a distributed embedding export on a Batch.

    The sparse features are looked up with ``DistributedEmbeddingLookup``, whose
    host tables every device shares, and the dense ``scripted_model.pt`` runs on
    the looked-up inputs, one replica per device.

    Args:
        export_dir: distributed embedding export directory.
        devices: CUDA devices to predict on.
    """

    def __init__(self, export_dir: str, devices: List[torch.device]) -> None:
        self._lookup = DistributedEmbeddingLookup(export_dir, devices)
        self._dense: Dict[torch.device, torch.jit.ScriptModule] = {}
        for device in devices:
            dense = torch.jit.load(
                os.path.join(export_dir, "scripted_model.pt"), map_location=device
            )
            dense.eval()
            self._dense[device] = dense

    def __call__(
        self, batch: Batch, batch_size: Optional[torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Predict a batch.

        Args:
            batch: the batch to predict.
            batch_size: the ``batch_size`` input of tiled models, None for none.
            device: the device to predict on, one of the constructor devices.

        Returns:
            The dense model's predictions.
        """
        return self._dense[device](self._lookup(batch, batch_size, device), device)
