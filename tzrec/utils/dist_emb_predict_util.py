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

import glob
import json
import os
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch

from tzrec.utils import npz_util
from tzrec.utils.logging_util import logger

_LOAD_BATCH_ROWS = 1 << 20
# NVE uses key -1 as its empty-slot marker, so -1 ids look up this miss key.
_MISSING_KEY = torch.iinfo(torch.int64).max
_MIN_GPU_CACHE_BYTES = 1 << 20
_MAX_GROUP_MEMBERS = (1 << 11) - 1
_SALT_MULTIPLIER = 0x9E3779B97F4A7C15


def _gpu_cache_ratio() -> float:
    """Fraction of each table's stored bytes cached on every GPU."""
    ratio = float(os.environ.get("DIST_EMB_GPU_CACHE_RATIO", "1.0"))
    if not 0 < ratio <= 1:
        raise ValueError(f"DIST_EMB_GPU_CACHE_RATIO must be in (0, 1], got {ratio}")
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


class DistributedEmbeddingLookup:
    """Looks up the sparse inputs of a distributed embedding export with NVE.

    Args:
        export_dir: distributed embedding export directory.
        devices: CUDA devices to serve lookups on; all of them share one host
            copy of every table.
    """

    def __init__(self, export_dir: str, devices: List[torch.device]) -> None:
        import pynve.nve as nve
        from pynve.torch import nve_layers, nve_ps

        if any(d.type != "cuda" for d in devices):
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
        static_npz = sorted(
            glob.glob(os.path.join(sparse_dir, "sparse_embeddings-*-of-*.npz"))
        )
        dynamic_npz = sorted(
            glob.glob(os.path.join(sparse_dir, "sparse_dynamic_embedding-*-of-*.npz"))
        )

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
                        self._npz_with(dynamic_npz, default_name), default_name, 1
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
        for (is_dynamic, dtype, dim, _), names in candidates.items():
            for start in range(0, len(names), _MAX_GROUP_MEMBERS):
                members = names[start : start + _MAX_GROUP_MEMBERS]
                payload_bits = 63 - len(members).bit_length()
                fused, fused_ids = [], []
                for member_idx, name in enumerate(members):
                    if is_dynamic and self._has_fold_collision(
                        emb_meta[name], dynamic_npz, payload_bits
                    ):
                        logger.info(
                            f"keys of dynamic embedding {name} collide after "
                            f"folding to {payload_bits} bits, serve it unfused."
                        )
                        groups.append(
                            _Group(dtype, dim, default_rows[name], [name], [0], None)
                        )
                    else:
                        fused.append(name)
                        fused_ids.append(member_idx)
                if fused:
                    groups.append(
                        _Group(
                            dtype,
                            dim,
                            default_rows[members[0]],
                            fused,
                            fused_ids,
                            payload_bits,
                        )
                    )

        ratio = _gpu_cache_ratio()
        # table name -> (group index, member tag)
        self._table_group: Dict[str, Tuple[int, int]] = {}
        self._groups = groups
        self._layers: Dict[torch.device, List[torch.nn.Module]] = {
            d: [] for d in devices
        }
        for group_idx, group in enumerate(groups):
            storage_type = storage_types[group.dtype]
            num_rows = sum(emb_meta[e]["shape"][0] for e in group.members)
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
                    self._load_dynamic(ps, emb_meta[emb_name], dynamic_npz, group, tag)
                else:
                    self._load_static(ps, emb_name, static_npz, group, tag)
            config = {}
            if group.default_row is not None:
                config["default_embedding"] = torch.from_numpy(group.default_row)
            gpu_cache_size = max(
                int(ratio * num_rows * ps.row_bytes), _MIN_GPU_CACHE_BYTES
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

        # features looked up together, per (group, pooling). A weighted MEAN is a
        # SUM over weights normalized by their bag sum, as training does.
        self._buckets: Dict[Tuple[int, str], List[str]] = {}
        for feat in self._used_feats:
            meta = self._feat_meta[feat]
            pooling = meta["pooling"]
            if pooling == "MEAN" and meta["use_weight"]:
                pooling = "SUM"
            group_idx, _ = self._table_group[meta["embedding_name"]]
            self._buckets.setdefault((group_idx, pooling), []).append(feat)
        self._tags: Dict[torch.device, Dict[Tuple[int, str], torch.Tensor]] = {}
        for device in devices:
            self._tags[device] = {
                bucket: torch.tensor(
                    [
                        self._table_group[self._feat_meta[f]["embedding_name"]][1]
                        for f in feats
                    ],
                    device=device,
                )
                for bucket, feats in self._buckets.items()
            }

    @staticmethod
    def _npz_with(npz_files: List[str], name: str) -> str:
        """Return the npz shard holding entry ``name``."""
        for path in npz_files:
            if npz_util.has_npz_entry(path, name):
                return path
        raise FileNotFoundError(f"npz entry {name} not found in {npz_files}.")

    @staticmethod
    def _has_fold_collision(
        meta: Dict, npz_files: List[str], payload_bits: int
    ) -> bool:
        """Whether two exported keys of a dynamic table fold to the same value.

        The folded keys are collected and sorted in place, 8 bytes per row, one
        table at a time. Keys below 2^payload_bits fold to themselves.
        """
        folded = np.empty(meta["shape"][0], dtype=np.int64)
        num_keys, max_key = 0, 0
        for path in npz_files:
            if not npz_util.has_npz_entry(path, meta["key_name"]):
                continue
            for keys in npz_util.iter_npz_entry(
                path, meta["key_name"], _LOAD_BATCH_ROWS
            ):
                if len(keys) == 0:
                    continue
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
        cls, ps: object, meta: Dict, npz_files: List[str], group: _Group, tag: int
    ) -> None:
        """Insert the key/value rows of a dynamic table from every shard."""
        for path in npz_files:
            if not npz_util.has_npz_entry(path, meta["key_name"]):
                continue
            for keys, values in zip(
                npz_util.iter_npz_entry(path, meta["key_name"], _LOAD_BATCH_ROWS),
                npz_util.iter_npz_entry(path, meta["value_name"], _LOAD_BATCH_ROWS),
            ):
                ps.insert(cls._stored_keys(keys, group, tag), torch.from_numpy(values))

    @classmethod
    def _load_static(
        cls, ps: object, emb_name: str, npz_files: List[str], group: _Group, tag: int
    ) -> None:
        """Insert the rows of a static table keyed by their row index."""
        offset = 0
        for path in npz_files:
            if not npz_util.has_npz_entry(path, emb_name):
                continue
            for values in npz_util.iter_npz_entry(path, emb_name, _LOAD_BATCH_ROWS):
                keys = np.arange(offset, offset + len(values), dtype=np.int64)
                ps.insert(cls._stored_keys(keys, group, tag), torch.from_numpy(values))
                offset += len(values)

    @staticmethod
    def _stored_keys(keys: np.ndarray, group: _Group, tag: int) -> torch.Tensor:
        """Host parameter server keys of exported keys."""
        keys_t = torch.from_numpy(keys)
        if group.payload_bits is None:
            return keys_t
        return _fold_keys(keys_t, group.payload_bits) ^ tag

    def __call__(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Add the dense graph's embedding inputs to a parsed batch.

        Args:
            data: parsed batch, as from ``Batch.to_dict(sparse_dtype=torch.int64)``.
            device: the device to look up on, one of the constructor devices.

        Returns:
            The batch moved to ``device``, plus one concatenated embedding tensor
            per ``dense_meta`` group and ``<feat>`` / ``<feat>__lengths`` inputs
            for every sequence feature.
        """
        out = {k: v.to(device, non_blocking=True) for k, v in data.items()}
        layers = self._layers[device]
        embs: Dict[str, torch.Tensor] = {}
        for (group_idx, pooling), feats in self._buckets.items():
            names = [_input_name(f) for f in feats]
            values = [out[f"{n}.values"] for n in names]
            counts = [v.numel() for v in values]
            values_cat = torch.cat(values) if len(values) > 1 else values[0]
            if values_cat.numel() == 0:
                # NVE rejects an empty lookup; every bag is empty, so zeros.
                dim = self._groups[group_idx].dim
                for feat, n in zip(feats, names):
                    rows = 0 if pooling == "NONE" else out[f"{n}.lengths"].numel()
                    embs[feat] = torch.zeros(rows, dim, device=device)
                continue
            payload_bits = self._groups[group_idx].payload_bits
            if payload_bits is None:
                keys = torch.where(values_cat == -1, _MISSING_KEY, values_cat)
            else:
                tags = torch.repeat_interleave(
                    self._tags[device][(group_idx, pooling)],
                    torch.tensor(counts, device=device),
                    output_size=values_cat.numel(),
                )
                keys = _encode_keys(values_cat, tags, payload_bits)
            layer = layers[group_idx]
            if pooling == "NONE":
                # dequantized rows come back in fp16, the dense graph is fp32.
                emb = layer(keys, mode="concat").float()
                for feat, e in zip(feats, emb.split(counts)):
                    embs[feat] = e
                continue
            lengths = [out[f"{n}.lengths"] for n in names]
            lengths_cat = torch.cat(lengths)
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths_cat)
            if pooling == "MEAN":
                emb = layer(keys, offsets, mode="mean").float()
            else:
                weights = None
                if any(self._feat_meta[f]["use_weight"] for f in feats):
                    weights = torch.cat(
                        [
                            out[f"{n}.weights"].float()
                            if self._feat_meta[f]["use_weight"]
                            else torch.ones(c, device=device)
                            for f, n, c in zip(feats, names, counts)
                        ]
                    )
                    mean_feats = [
                        self._feat_meta[f]["pooling"] == "MEAN" for f in feats
                    ]
                    if any(mean_feats):
                        is_mean = torch.repeat_interleave(
                            torch.tensor(mean_feats, device=device),
                            torch.tensor(counts, device=device),
                            output_size=weights.numel(),
                        )
                        weights = torch.where(
                            is_mean,
                            self._normalize_weights(weights, lengths_cat),
                            weights,
                        )
                emb = layer(keys, offsets, weights).float()
            for feat, e in zip(feats, emb.split([x.numel() for x in lengths])):
                embs[feat] = e

        for group, feats in self._dense_meta.items():
            if group == "sequence__ec":
                for feat in feats:
                    if feat.endswith("__lengths"):
                        out[feat] = out[f"{_input_name(feat)}.lengths"].int()
                    else:
                        out[_remove_suffix(feat)] = embs[feat]
            else:
                out[group] = torch.cat([embs[feat] for feat in feats], dim=1)
        return out

    @staticmethod
    def _normalize_weights(
        weights: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Divide weights by their bag's sum (1 for a zero sum), as training does."""
        weight_sum = torch.segment_reduce(weights, "sum", lengths=lengths)
        weight_sum = torch.where(
            weight_sum == 0, torch.ones_like(weight_sum), weight_sum
        )
        return weights / torch.repeat_interleave(
            weight_sum, lengths, output_size=weights.numel()
        )
