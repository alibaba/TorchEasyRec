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
out by ``dense_meta.json``. The online processor serves the tables with NVIDIA
NV-Embeddings (NVE) hierarchical layers; this module does the same lookup
offline through pynve, so offline predict matches online serving, including
rowwise INT8 (``DIST_QUANT=INT8``) tables, which NVE dequantizes on lookup.

Each table is loaded once into a host parameter server (an nvhm hash table)
that the layers on every local GPU share; each GPU only adds its own cache.
"""

import glob
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch

from tzrec.utils import npz_util
from tzrec.utils.logging_util import logger

_LOAD_BATCH_ROWS = 1 << 20
# NVE uses key -1 as its empty-slot marker; the processor serves it as a miss.
_MISSING_KEY = torch.iinfo(torch.int64).max
_MIN_GPU_CACHE_BYTES = 1 << 20


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
            self._feat_meta: Dict[str, Dict[str, str]] = json.load(f)

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
        ratio = _gpu_cache_ratio()
        # one layer per (table, pooled or not); MEAN is a SUM divided by the bag
        # length, matching torchrec for weighted features too, where NVE's
        # weighted mean would divide by the weight sum instead.
        self._layers: Dict[torch.device, Dict[Tuple[str, bool], torch.nn.Module]] = {
            d: {} for d in devices
        }
        layer_keys = {
            (
                self._feat_meta[f]["embedding_name"],
                self._feat_meta[f]["pooling"] != "NONE",
            )
            for f in self._used_feats
        }
        for emb_name in sorted({emb for emb, _ in layer_keys}):
            meta = emb_meta[emb_name]
            if meta["dtype"] not in storage_types:
                raise ValueError(
                    f"unsupported dtype {meta['dtype']} of embedding {emb_name}."
                )
            storage_type = storage_types[meta["dtype"]]
            dim = meta["dimension"]
            num_rows = meta["shape"][0]
            ps = nve_ps.NVEParameterServer(
                0, dim, storage_type, initial_size=max(num_rows, 1024)
            )
            config = {}
            if meta["is_dynamic"]:
                self._load_dynamic(ps, meta, dynamic_npz)
                default_name = meta.get("default_value_name")
                if default_name:
                    default_row = next(
                        npz_util.iter_npz_entry(
                            self._npz_with(dynamic_npz, default_name), default_name, 1
                        )
                    )[0]
                    config["default_embedding"] = torch.from_numpy(default_row.copy())
            else:
                self._load_static(ps, emb_name, static_npz)
            gpu_cache_size = max(
                int(ratio * num_rows * ps.row_bytes), _MIN_GPU_CACHE_BYTES
            )
            for device in devices:
                for pooled in sorted(p for e, p in layer_keys if e == emb_name):
                    kwargs = dict(
                        gpu_cache_size=gpu_cache_size,
                        storage=ps,
                        optimize_for_training=False,
                        device=device,
                        config=dict(config),
                    )
                    if not pooled:
                        layer = nve_layers.NVEmbedding(
                            num_rows,
                            dim,
                            storage_type,
                            nve_layers.LayerType.Hierarchical,
                            **kwargs,
                        )
                    else:
                        layer = nve_layers.NVEmbeddingBag(
                            num_rows,
                            dim,
                            storage_type,
                            nve_layers.LayerType.Hierarchical,
                            "sum",
                            **kwargs,
                        )
                    self._layers[device][(emb_name, pooled)] = layer
            logger.info(
                f"loaded sparse embedding {emb_name}: rows={num_rows}, dim={dim}, "
                f"dtype={meta['dtype']}, gpu_cache_bytes={gpu_cache_size}."
            )

    @staticmethod
    def _npz_with(npz_files: List[str], name: str) -> str:
        """Return the npz shard holding entry ``name``."""
        for path in npz_files:
            if npz_util.has_npz_entry(path, name):
                return path
        raise FileNotFoundError(f"npz entry {name} not found in {npz_files}.")

    @staticmethod
    def _load_dynamic(ps: object, meta: Dict, npz_files: List[str]) -> None:
        """Insert the key/value rows of a dynamic table from every shard."""
        for path in npz_files:
            if not npz_util.has_npz_entry(path, meta["key_name"]):
                continue
            for keys, values in zip(
                npz_util.iter_npz_entry(path, meta["key_name"], _LOAD_BATCH_ROWS),
                npz_util.iter_npz_entry(path, meta["value_name"], _LOAD_BATCH_ROWS),
            ):
                ps.insert(torch.from_numpy(keys), torch.from_numpy(values))

    @staticmethod
    def _load_static(ps: object, emb_name: str, npz_files: List[str]) -> None:
        """Insert the rows of a static table keyed by their row index."""
        offset = 0
        for path in npz_files:
            if not npz_util.has_npz_entry(path, emb_name):
                continue
            for values in npz_util.iter_npz_entry(path, emb_name, _LOAD_BATCH_ROWS):
                keys = np.arange(offset, offset + len(values), dtype=np.int64)
                ps.insert(torch.from_numpy(keys), torch.from_numpy(values))
                offset += len(values)

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
        for feat in self._used_feats:
            name = _remove_suffix(feat)
            meta = self._feat_meta[feat]
            values = out[f"{name}.values"]
            keys = torch.where(values == -1, _MISSING_KEY, values)
            pooling = meta["pooling"]
            layer = layers[(meta["embedding_name"], pooling != "NONE")]
            if pooling == "NONE":
                # dequantized rows come back in fp16, the dense graph is fp32.
                embs[feat] = layer(keys).float()
                continue
            lengths = out[f"{name}.lengths"]
            offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
            emb = layer(keys, offsets, out.get(f"{name}.weights")).float()
            if pooling == "MEAN":
                emb = emb / lengths.clamp(min=1).unsqueeze(1)
            embs[feat] = emb

        for group, feats in self._dense_meta.items():
            if group == "sequence__ec":
                for feat in feats:
                    name = _remove_suffix(feat)
                    if feat.endswith("__lengths"):
                        out[feat] = out[f"{name}.lengths"].int()
                    else:
                        out[name] = embs[feat]
            else:
                out[group] = torch.cat([embs[feat] for feat in feats], dim=1)
        return out
