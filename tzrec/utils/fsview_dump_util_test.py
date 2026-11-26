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

import json
import os
import shutil
import socket
import unittest
from typing import Any, Dict, List, Optional, Sequence
from unittest import mock

import numpy as np
import pyarrow as pa
import torch
from parameterized import parameterized

from tzrec.protos import pipeline_pb2, train_pb2
from tzrec.utils import config_util, fsview_dump_util, npz_util, quant_util
from tzrec.utils.feature_store_uploader import (
    FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
    FEATURE_STORE_EMBEDDING_TYPE_UINT8,
)
from tzrec.utils.fsview_dump_util import (
    CURRENT_JSON,
    META_JSON,
    READY_FILE,
    SPARSE_FS_DUMP_DIR,
    VERSIONS_DIR,
    dump_fsview,
)
from tzrec.utils.test_util import make_test_dir, parameterized_name_func


class _FakeEKKVStream:
    def __init__(self, batches: Sequence[pa.RecordBatch], next_ts: int) -> None:
        self._batches = list(batches)
        self._idx = 0
        self._next_ts = next_ts
        self._complete = False

    def __enter__(self) -> "_FakeEKKVStream":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        return False

    def __iter__(self) -> "_FakeEKKVStream":
        return self

    def __next__(self) -> pa.RecordBatch:
        if self._idx >= len(self._batches):
            self._complete = True
            raise StopIteration
        batch = self._batches[self._idx]
        self._idx += 1
        return batch

    @property
    def next_ts(self) -> int:
        if not self._complete:
            raise RuntimeError("next_ts requested before EOF")
        return self._next_ts


class _FakeDynamicEmbeddingView:
    def __init__(
        self,
        shards: Dict[int, List[pa.RecordBatch]],
        embedding_field_type: str = FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
        additional_fields: Optional[Dict[str, str]] = None,
        next_ts_override: Optional[int] = None,
        server_now_ms: Optional[int] = None,
    ) -> None:
        self._shards = shards
        self.pk_field = "embedding_name"
        self.sk_field = "key_id"
        self.embedding_field = "embedding"
        self.embedding_field_type = embedding_field_type
        self.additional_fields = dict(additional_fields or {})
        self.next_ts_override = next_ts_override
        self.server_now_ms = server_now_ms
        self.calls: List[Dict[str, Any]] = []

    def get_shard_count(self) -> int:
        return len(self._shards)

    def iterate_features(
        self,
        ts: int,
        version: str = "default",
        end_ts: Optional[int] = None,
        shard_id: Optional[int] = None,
        output_fields: Optional[List[str]] = None,
    ) -> _FakeEKKVStream:
        if ts > 0 and shard_id is None and end_ts is None:
            server_ts = (
                self.server_now_ms
                if self.server_now_ms is not None
                else max(1, fsview_dump_util.time.time_ns() // 1_000_000)
            )
            return _FakeEKKVStream([], server_ts)
        assert ts == 0
        assert shard_id is not None
        self.calls.append(
            {
                "ts": ts,
                "version": version,
                "end_ts": end_ts,
                "shard_id": shard_id,
                "output_fields": list(output_fields) if output_fields else None,
            }
        )
        batches = self._shards[shard_id]
        if output_fields is not None:
            batches = [
                pa.RecordBatch.from_arrays(
                    [b.column(name) for name in output_fields],
                    names=output_fields,
                )
                for b in batches
            ]
        next_ts = (
            self.next_ts_override
            if self.next_ts_override is not None
            else int(end_ts or 0)
        )
        return _FakeEKKVStream(batches, next_ts)


def _write_pipeline_config(
    path: str,
    version: str = "ol_v1",
    quant_type: int = train_pb2.DELTA_EMBEDDING_QUANT_NONE,
    with_fs: bool = True,
) -> None:
    cfg = pipeline_pb2.EasyRecConfig()
    cfg.train_input_path = "dummy"
    cfg.eval_input_path = "dummy"
    cfg.model_dir = os.path.dirname(path)
    cfg.train_config.sparse_optimizer.adam_optimizer.lr = 0.001
    cfg.train_config.dense_optimizer.adam_optimizer.lr = 0.001
    if with_fs:
        delta = cfg.train_config.delta_embedding_dump_config
        delta.paired_dump_interval_steps = 100
        delta.quant_type = quant_type
        fs = delta.feature_store_config
        fs.region = "cn-test"
        fs.project_name = "proj_test"
        fs.feature_view_name = "fv_test"
        fs.version = version
        fs.max_workers = 2
    config_util.save_message(cfg, path)


def _make_arrow_batch(
    pks: List[str],
    sks: List[int],
    embeddings: np.ndarray,
    scores: Optional[List[Optional[int]]] = None,
) -> pa.RecordBatch:
    n, dim = embeddings.shape
    val_type = pa.uint8() if embeddings.dtype == np.uint8 else pa.float32()
    emb_col = pa.ListArray.from_arrays(
        pa.array(np.arange(0, (n + 1) * dim, dim, dtype=np.int32)),
        pa.array(embeddings.reshape(-1), type=val_type),
    )
    arrays: List[pa.Array] = [
        pa.array(pks, type=pa.string()),
        pa.array([str(k) for k in sks], type=pa.string()),
        emb_col,
    ]
    names = ["embedding_name", "key_id", "embedding"]
    if scores is not None:
        arrays.append(pa.array(scores, type=pa.int64()))
        names.append("score")
    return pa.RecordBatch.from_arrays(arrays, names=names)


def _create_fp32_baseline(model_dir: str, include_zch_default: bool = False) -> None:
    sparse_dir = os.path.join(model_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    static_w = np.arange(16, dtype=np.float32).reshape(4, 4)
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"),
        {"ebc.embedding_bags.stat_tbl": static_w},
    )

    dyn_a_0 = {
        "keys": np.array([1, 2, 3], dtype=np.int64),
        "values": np.array(
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0], [3.0, 3.0, 3.0, 3.0]],
            dtype=np.float32,
        ),
        "scores": np.array([10, 20, 30], dtype=np.int64),
    }
    dyn_a_1 = {
        "keys": np.array([4, 5], dtype=np.int64),
        "values": np.array(
            [[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0]], dtype=np.float32
        ),
        "scores": np.array([40, 50], dtype=np.int64),
    }
    dyn_b_0 = {
        "keys": np.array([100], dtype=np.int64),
        "values": np.array([[100.0, 100.0]], dtype=np.float32),
        "scores": np.array([7], dtype=np.int64),
    }
    dyn_b_1 = {
        "keys": np.empty(0, dtype=np.int64),
        "values": np.empty((0, 2), dtype=np.float32),
        "scores": np.empty(0, dtype=np.int64),
    }
    part0: Dict[str, Any] = {
        "ebc.embedding_bags.dyn_a.keys": dyn_a_0["keys"],
        "ebc.embedding_bags.dyn_a.values": dyn_a_0["values"],
        "ebc.embedding_bags.dyn_a.scores": dyn_a_0["scores"],
        "ec.embeddings.dyn_b.keys": dyn_b_0["keys"],
        "ec.embeddings.dyn_b.values": dyn_b_0["values"],
        "ec.embeddings.dyn_b.scores": dyn_b_0["scores"],
    }
    if include_zch_default:
        part0["ec.embeddings.dyn_b.default_value"] = np.array(
            [[9.0, 9.0]], dtype=np.float32
        )
    part1: Dict[str, Any] = {
        "ebc.embedding_bags.dyn_a.keys": dyn_a_1["keys"],
        "ebc.embedding_bags.dyn_a.values": dyn_a_1["values"],
        "ebc.embedding_bags.dyn_a.scores": dyn_a_1["scores"],
        "ec.embeddings.dyn_b.keys": dyn_b_1["keys"],
        "ec.embeddings.dyn_b.values": dyn_b_1["values"],
        "ec.embeddings.dyn_b.scores": dyn_b_1["scores"],
    }
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-02.npz"), part0
    )
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_dynamic_embedding-01-of-02.npz"), part1
    )

    emb_meta: Dict[str, Any] = {
        "ebc.embedding_bags.stat_tbl": {
            "feat_name_impl": ["stat_f__ebc"],
            "dense": False,
            "is_dynamic": False,
            "dimension": 4,
            "dtype": "float32",
            "memory": 64,
            "shape": [4, 4],
        },
        "ebc.embedding_bags.dyn_a": {
            "feat_name_impl": ["dyn_a_f__ebc"],
            "dense": False,
            "is_dynamic": True,
            "dimension": 4,
            "dtype": "float32",
            "memory": 80,
            "shape": [5, 4],
            "key_dtype": "int64",
            "score_dtype": "int64",
            "key_name": "ebc.embedding_bags.dyn_a.keys",
            "value_name": "ebc.embedding_bags.dyn_a.values",
            "score_name": "ebc.embedding_bags.dyn_a.scores",
        },
        "ec.embeddings.dyn_b": {
            "feat_name_impl": ["dyn_b_f__ec"],
            "dense": False,
            "is_dynamic": True,
            "dimension": 2,
            "dtype": "float32",
            "memory": 8,
            "shape": [1, 2],
            "key_dtype": "int64",
            "score_dtype": "int64",
            "key_name": "ec.embeddings.dyn_b.keys",
            "value_name": "ec.embeddings.dyn_b.values",
            "score_name": "ec.embeddings.dyn_b.scores",
        },
    }
    if include_zch_default:
        emb_meta["ec.embeddings.dyn_b"]["default_value_name"] = (
            "ec.embeddings.dyn_b.default_value"
        )
    feat_meta = {
        "stat_f__ebc": {
            "embedding_name": "ebc.embedding_bags.stat_tbl",
            "pooling": "SUM",
            "use_weight": False,
        },
        "dyn_a_f__ebc": {
            "embedding_name": "ebc.embedding_bags.dyn_a",
            "pooling": "MEAN",
            "use_weight": False,
        },
        "dyn_b_f__ec": {
            "embedding_name": "ec.embeddings.dyn_b",
            "pooling": "NONE",
            "use_weight": False,
        },
    }
    for name in ("sparse_embedding.json", "sparse_embeddings-00-of-01.json"):
        with open(os.path.join(sparse_dir, name), "w") as f:
            json.dump(emb_meta, f, indent=4)
    with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
        json.dump(feat_meta, f, indent=4)


def _collect_dynamic_entries(
    version_dir: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    npz_paths = sorted(
        os.path.join(version_dir, name)
        for name in os.listdir(version_dir)
        if name.startswith("sparse_dynamic_embedding-") and name.endswith(".npz")
    )
    out: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for path in npz_paths:
        with np.load(path) as data:
            for key in data.files:
                if key.endswith(".default_value"):
                    continue
                table, suffix = key.rsplit(".", maxsplit=1)
                out.setdefault(table, {}).setdefault(suffix, []).append(data[key])
    merged: Dict[str, Dict[str, np.ndarray]] = {}
    for table, parts in out.items():
        merged[table] = {
            suffix: np.concatenate(arrs, axis=0) for suffix, arrs in parts.items()
        }
    return merged


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run_distributed_worker(
    rank: int,
    world_size: int,
    port: int,
    config_path: str,
    model_dir: str,
) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    # Build identical fake view inside each spawned process.
    shards = {
        0: [
            _make_arrow_batch(
                ["ebc.embedding_bags.stat_tbl", "ebc.embedding_bags.dyn_a"],
                [1, 2],
                np.array([[9.0, 9.0, 9.0, 9.0], [22.0, 22.0, 22.0, 22.0]], np.float32),
                scores=[None, 220],
            )
        ],
        1: [
            _make_arrow_batch(
                ["ebc.embedding_bags.dyn_a", "ebc.embedding_bags.dyn_a"],
                [4, 9],
                np.array([[0.0, 0.0, 0.0, 0.0], [99.0, 99.0, 99.0, 99.0]], np.float32),
                scores=[0, 990],
            )
        ],
        2: [
            _make_arrow_batch(
                ["ebc.embedding_bags.stat_tbl"],
                [3],
                np.array([[0.0, 0.0, 0.0, 0.0]], np.float32),
                scores=[None],
            )
        ],
    }
    view = _FakeDynamicEmbeddingView(
        shards,
        embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
        additional_fields={"score": "INT64"},
    )
    with (
        mock.patch.object(
            fsview_dump_util, "_open_feature_view", return_value=(view, True)
        ),
        mock.patch(
            "tzrec.utils.fsview_dump_util.time.time_ns", return_value=5_000_000_000
        ),
    ):
        dump_fsview(config_path, model_dir, keep_versions=0)


class FsviewDumpUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir("fsview_dump_")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_fp32_zero_shuffle_merge_with_score_and_zch(self) -> None:
        model_dir = os.path.join(self.test_dir, "model")
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _create_fp32_baseline(model_dir, include_zch_default=True)
        _write_pipeline_config(config_path)

        # M=2 baseline dynamic shards, S=3 FeatureDB shards -> P=3 output shards.
        shards = {
            0: [
                _make_arrow_batch(
                    [
                        "ebc.embedding_bags.stat_tbl",
                        "ebc.embedding_bags.dyn_a",
                        "ebc.embedding_bags.dyn_a",
                    ],
                    [1, 2, 2],
                    np.array(
                        [
                            [11.0, 11.0, 11.0, 11.0],
                            [21.0, 21.0, 21.0, 21.0],
                            [22.0, 22.0, 22.0, 22.0],
                        ],
                        dtype=np.float32,
                    ),
                    scores=[None, 200, 205],
                )
            ],
            1: [
                _make_arrow_batch(
                    ["ebc.embedding_bags.dyn_a", "ebc.embedding_bags.dyn_a"],
                    # key 3 is deleted by tombstone; key 10 is newly inserted.
                    [3, 10],
                    np.array(
                        [
                            [0.0, 0.0, 0.0, 0.0],
                            [10.0, 10.0, 10.0, 10.0],
                        ],
                        dtype=np.float32,
                    ),
                    scores=[999, 101],
                ),
                # Static table all-zero update is NOT treated as tombstone.
                _make_arrow_batch(
                    ["ebc.embedding_bags.stat_tbl"],
                    [3],
                    np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    scores=[None],
                ),
            ],
            2: [
                _make_arrow_batch(
                    ["ec.embeddings.dyn_b"],
                    [200],
                    np.array([[200.0, 201.0]], dtype=np.float32),
                    scores=[None],
                )
            ],
        }
        view = _FakeDynamicEmbeddingView(
            shards,
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
            additional_fields={"score": "INT64"},
        )

        with (
            mock.patch.object(
                fsview_dump_util, "_open_feature_view", return_value=(view, True)
            ),
            mock.patch(
                "tzrec.utils.fsview_dump_util.time.time_ns",
                return_value=1_764_000_000_123_000_000,
            ),
        ):
            payload = dump_fsview(config_path, model_dir)

        assert payload is not None
        self.assertEqual(payload["end_ts"], 1_764_000_000_123)
        self.assertEqual(payload["fs_version"], "ol_v1")
        self.assertTrue(payload["has_score"])

        dump_root = os.path.join(model_dir, SPARSE_FS_DUMP_DIR)
        version_dir = os.path.join(dump_root, VERSIONS_DIR, payload["version"])
        self.assertTrue(os.path.isfile(os.path.join(version_dir, READY_FILE)))

        with open(os.path.join(dump_root, CURRENT_JSON)) as f:
            self.assertEqual(json.load(f), payload)
        with open(os.path.join(version_dir, META_JSON)) as f:
            meta = json.load(f)
            self.assertEqual(meta["start_ts"], 0)
            self.assertEqual(meta["end_ts"], 1_764_000_000_123)
            self.assertEqual(meta["num_dynamic_parts"], 3)
            self.assertEqual(meta["shard_count"], 3)

        # Every output shard must contain the full key/value/score entry set.
        expected_parts = [
            "sparse_dynamic_embedding-00-of-03.npz",
            "sparse_dynamic_embedding-01-of-03.npz",
            "sparse_dynamic_embedding-02-of-03.npz",
        ]
        for fname in expected_parts:
            with np.load(os.path.join(version_dir, fname)) as data:
                for tbl in ("ebc.embedding_bags.dyn_a", "ec.embeddings.dyn_b"):
                    self.assertIn(f"{tbl}.keys", data.files)
                    self.assertIn(f"{tbl}.values", data.files)
                    self.assertIn(f"{tbl}.scores", data.files)
                if fname.endswith("00-of-03.npz"):
                    np.testing.assert_allclose(
                        data["ec.embeddings.dyn_b.default_value"],
                        np.array([[9.0, 9.0]], dtype=np.float32),
                    )

        # Static table: row 0 & 2 preserved, row 1 overwritten, row 3 zeroed.
        with np.load(os.path.join(version_dir, "sparse_embeddings-00-of-01.npz")) as st:
            stat = st["ebc.embedding_bags.stat_tbl"]
            expected_stat = np.arange(16, dtype=np.float32).reshape(4, 4)
            expected_stat[1] = 11.0
            expected_stat[3] = 0.0
            np.testing.assert_allclose(stat, expected_stat)

        # dyn_a has keys {1, 4, 5 (base), 2 (updated), 10 (new)}; key 3 deleted.
        merged_dyn = _collect_dynamic_entries(version_dir)
        dyn_a_rows = {
            int(k): (v, int(s))
            for k, v, s in zip(
                merged_dyn["ebc.embedding_bags.dyn_a"]["keys"],
                merged_dyn["ebc.embedding_bags.dyn_a"]["values"],
                merged_dyn["ebc.embedding_bags.dyn_a"]["scores"],
            )
        }
        self.assertEqual(set(dyn_a_rows.keys()), {1, 2, 4, 5, 10})
        self.assertEqual(dyn_a_rows[1][1], 10)
        self.assertEqual(dyn_a_rows[2][1], 205)
        np.testing.assert_allclose(dyn_a_rows[2][0], [22.0, 22.0, 22.0, 22.0])
        self.assertEqual(dyn_a_rows[4][1], 40)
        self.assertEqual(dyn_a_rows[10][1], 101)

        dyn_b_rows = {
            int(k): (v, int(s))
            for k, v, s in zip(
                merged_dyn["ec.embeddings.dyn_b"]["keys"],
                merged_dyn["ec.embeddings.dyn_b"]["values"],
                merged_dyn["ec.embeddings.dyn_b"]["scores"],
            )
        }
        self.assertEqual(set(dyn_b_rows.keys()), {100, 200})
        self.assertEqual(dyn_b_rows[100][1], 7)
        self.assertEqual(dyn_b_rows[200][1], 0)

        # Updated metadata reflects merged counts (dyn_a: 5, dyn_b: 2).
        with open(os.path.join(version_dir, "sparse_embedding.json")) as f:
            emb_meta = json.load(f)
        self.assertEqual(emb_meta["ebc.embedding_bags.dyn_a"]["shape"], [5, 4])
        self.assertEqual(emb_meta["ebc.embedding_bags.dyn_a"]["memory"], 5 * 16)
        self.assertEqual(emb_meta["ec.embeddings.dyn_b"]["shape"], [2, 2])
        self.assertEqual(emb_meta["ec.embeddings.dyn_b"]["memory"], 2 * 8)

    def test_quantized_int8_and_extreme_int64_scores(self) -> None:
        model_dir = os.path.join(self.test_dir, "model_q")
        sparse_dir = os.path.join(model_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _write_pipeline_config(
            config_path,
            version="ol_q",
            quant_type=train_pb2.DELTA_EMBEDDING_QUANT_INT8,
        )

        # Dim=4 -> row_bytes = 8 (4 uint8 codes + 2 fp16 scale + 2 fp16 offset).
        base_vals = quant_util.distributed_quantize_embeddings(
            np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], np.float32),
            emb_dim=4,
            emb_name="ebc.embedding_bags.dyn_q",
            quant_format=quant_util.DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0],
        )
        npz_util.savez_streaming(
            os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"), {}
        )
        npz_util.savez_streaming(
            os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz"),
            {
                "ebc.embedding_bags.dyn_q.keys": np.array([10, 20], dtype=np.int64),
                "ebc.embedding_bags.dyn_q.values": base_vals,
                "ebc.embedding_bags.dyn_q.scores": np.array([11, 22], dtype=np.int64),
            },
        )
        emb_meta = {
            "ebc.embedding_bags.dyn_q": {
                "feat_name_impl": ["q_f__ebc"],
                "dense": False,
                "is_dynamic": True,
                "dimension": 4,
                "dtype": "QUint8RowwiseF16",
                "storage_dtype": "uint8",
                "storage_shape": [2, 8],
                "row_bytes": 8,
                "memory": 16,
                "shape": [2, 4],
                "quant": {
                    "enabled": True,
                    "format": "QUint8RowwiseF16",
                    "scale_offset_dtype": "float16",
                    "output_dtype": "float16",
                },
                "key_dtype": "int64",
                "score_dtype": "int64",
                "key_name": "ebc.embedding_bags.dyn_q.keys",
                "value_name": "ebc.embedding_bags.dyn_q.values",
                "score_name": "ebc.embedding_bags.dyn_q.scores",
            }
        }
        for name in ("sparse_embedding.json", "sparse_embeddings-00-of-01.json"):
            with open(os.path.join(sparse_dir, name), "w") as f:
                json.dump(emb_meta, f)
        with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
            json.dump(
                {
                    "q_f__ebc": {
                        "embedding_name": "ebc.embedding_bags.dyn_q",
                        "pooling": "SUM",
                        "use_weight": False,
                    }
                },
                f,
            )

        def _make_q_row(codes: List[int], scale: float, offset: float) -> np.ndarray:
            arr = np.empty(8, dtype=np.uint8)
            arr[:4] = np.array(codes, dtype=np.uint8)
            arr[4:6] = np.array([scale], dtype=np.float16).view(np.uint8)
            arr[6:8] = np.array([offset], dtype=np.float16).view(np.uint8)
            return arr

        # key 10: quantized tombstone (codes=0, offset=0, scale=1.0 -> deleted).
        # key 30: codes=0, offset=0.5 -> NOT tombstone, kept with max int64 score.
        # key 40: valid quantized row with min int64 score.
        # key 50: valid quantized row with null score (-> 0).
        q_rows = np.stack(
            [
                _make_q_row([0, 0, 0, 0], 1.0, 0.0),
                _make_q_row([0, 0, 0, 0], 1.0, 0.5),
                _make_q_row([1, 2, 3, 4], 0.25, -1.0),
                _make_q_row([5, 6, 7, 8], 0.5, 0.0),
            ],
            axis=0,
        )
        max_i64 = 2**63 - 1
        min_i64 = -(2**63)
        view = _FakeDynamicEmbeddingView(
            {
                0: [
                    _make_arrow_batch(
                        ["ebc.embedding_bags.dyn_q"] * 4,
                        [10, 30, 40, 50],
                        q_rows,
                        scores=[123, max_i64, min_i64, None],
                    )
                ]
            },
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_UINT8,
            additional_fields={"score": "INT64"},
        )
        with (
            mock.patch.object(
                fsview_dump_util, "_open_feature_view", return_value=(view, True)
            ),
            mock.patch(
                "tzrec.utils.fsview_dump_util.time.time_ns", return_value=2_000_000_000
            ),
        ):
            payload = dump_fsview(config_path, model_dir)

        assert payload is not None
        version_dir = os.path.join(
            model_dir, SPARSE_FS_DUMP_DIR, VERSIONS_DIR, payload["version"]
        )
        merged = _collect_dynamic_entries(version_dir)["ebc.embedding_bags.dyn_q"]
        by_key = {
            int(k): (v, int(s))
            for k, v, s in zip(merged["keys"], merged["values"], merged["scores"])
        }
        # Exact bit-level preservation of extreme int64 scores (no float64 rounding).
        self.assertEqual(set(by_key.keys()), {20, 30, 40, 50})
        self.assertEqual(by_key[20][1], 22)
        self.assertEqual(by_key[30][1], max_i64)
        self.assertEqual(by_key[40][1], min_i64)
        self.assertEqual(by_key[50][1], 0)
        np.testing.assert_array_equal(by_key[30][0], q_rows[1])

        with open(os.path.join(version_dir, "sparse_embedding.json")) as f:
            meta = json.load(f)["ebc.embedding_bags.dyn_q"]
        self.assertEqual(meta["shape"], [4, 4])
        self.assertEqual(meta["storage_shape"], [4, 8])
        self.assertEqual(meta["memory"], 32)
        self.assertEqual(meta["row_bytes"], 8)

    def test_legacy_view_without_score_and_retention(self) -> None:
        model_dir = os.path.join(self.test_dir, "model_legacy")
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _create_fp32_baseline(model_dir)
        _write_pipeline_config(config_path)

        server_ts_iter = iter([1000, 2000, 3000])
        view = _FakeDynamicEmbeddingView(
            {
                0: [
                    _make_arrow_batch(
                        ["ebc.embedding_bags.dyn_a"],
                        [77],
                        np.array([[7.0, 7.0, 7.0, 7.0]], np.float32),
                    )
                ]
            },
            embedding_field_type=FEATURE_STORE_EMBEDDING_TYPE_FLOAT,
            additional_fields={},
        )
        orig_iter = view.iterate_features

        def _seq_iterate(ts: int, *args: Any, **kwargs: Any) -> _FakeEKKVStream:
            if (
                ts > 0
                and kwargs.get("shard_id") is None
                and kwargs.get("end_ts") is None
            ):
                return _FakeEKKVStream([], next(server_ts_iter))
            return orig_iter(ts, *args, **kwargs)

        view.iterate_features = _seq_iterate  # type: ignore[method-assign]
        with mock.patch.object(
            fsview_dump_util, "_open_feature_view", return_value=(view, False)
        ):
            p1 = dump_fsview(config_path, model_dir, keep_versions=2)
            p2 = dump_fsview(config_path, model_dir, keep_versions=2)
            p3 = dump_fsview(config_path, model_dir, keep_versions=2)

        assert p1 is not None and p2 is not None and p3 is not None
        self.assertLess(p1["version"], p2["version"])
        self.assertLess(p2["version"], p3["version"])
        self.assertFalse(p3["has_score"])
        self.assertEqual(
            view.calls[0]["output_fields"],
            ["embedding_name", "key_id", "embedding"],
        )

        versions_root = os.path.join(model_dir, SPARSE_FS_DUMP_DIR, VERSIONS_DIR)
        remaining = sorted(os.listdir(versions_root))
        self.assertEqual(remaining, [p2["version"], p3["version"]])

        merged = _collect_dynamic_entries(os.path.join(versions_root, p3["version"]))
        scores_by_key = {
            int(k): int(s)
            for k, s in zip(
                merged["ebc.embedding_bags.dyn_a"]["keys"],
                merged["ebc.embedding_bags.dyn_a"]["scores"],
            )
        }
        # Baseline scores preserved; newly pulled key outranks max baseline score (50).
        self.assertEqual(scores_by_key[1], 10)
        self.assertEqual(scores_by_key[77], 51)

    def test_fp16_baseline_server_clock_skew_and_incomplete_retention(self) -> None:
        model_dir = os.path.join(self.test_dir, "model_fp16")
        sparse_dir = os.path.join(model_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _write_pipeline_config(config_path)

        npz_util.savez_streaming(
            os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"),
            {
                "ebc.embedding_bags.stat_fp16": np.array(
                    [[1.0, 2.0], [3.0, 4.0]], dtype=np.float16
                )
            },
        )
        npz_util.savez_streaming(
            os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz"),
            {
                "ebc.embedding_bags.dyn_fp16.keys": np.array([10, 20], dtype=np.int64),
                "ebc.embedding_bags.dyn_fp16.values": np.array(
                    [[1.5, 2.5], [3.5, 4.5]], dtype=np.float16
                ),
                "ebc.embedding_bags.dyn_fp16.scores": np.array([5, 8], dtype=np.int64),
            },
        )
        emb_meta = {
            "ebc.embedding_bags.stat_fp16": {
                "feat_name_impl": ["sf__ebc"],
                "dense": False,
                "is_dynamic": False,
                "dimension": 2,
                "dtype": "float16",
                "memory": 8,
                "shape": [2, 2],
            },
            "ebc.embedding_bags.dyn_fp16": {
                "feat_name_impl": ["df__ebc"],
                "dense": False,
                "is_dynamic": True,
                "dimension": 2,
                "dtype": "float16",
                "memory": 8,
                "shape": [2, 2],
                "key_dtype": "int64",
                "score_dtype": "int64",
                "key_name": "ebc.embedding_bags.dyn_fp16.keys",
                "value_name": "ebc.embedding_bags.dyn_fp16.values",
                "score_name": "ebc.embedding_bags.dyn_fp16.scores",
            },
        }
        for name in ("sparse_embedding.json", "sparse_embeddings-00-of-01.json"):
            with open(os.path.join(sparse_dir, name), "w") as f:
                json.dump(emb_meta, f)
        with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
            json.dump(
                {
                    "sf__ebc": {"embedding_name": "ebc.embedding_bags.stat_fp16"},
                    "df__ebc": {"embedding_name": "ebc.embedding_bags.dyn_fp16"},
                },
                f,
            )

        # Subnormal FP32 (1e-8) underflows to 0.0 in FP16, yet must NOT be treated
        # as a deletion tombstone; key 10 ([0.0, 0.0] in FP32) IS a tombstone.
        shards = {
            0: [
                _make_arrow_batch(
                    [
                        "ebc.embedding_bags.stat_fp16",
                        "ebc.embedding_bags.dyn_fp16",
                        "ebc.embedding_bags.dyn_fp16",
                    ],
                    [1, 10, 30],
                    np.array(
                        [[6.5, 7.5], [0.0, 0.0], [1e-8, 0.0]],
                        dtype=np.float32,
                    ),
                )
            ]
        }
        # Client clock is 45s AHEAD of FeatureDB server clock (100_000 ms).
        view = _FakeDynamicEmbeddingView(shards, server_now_ms=100_000)
        with (
            mock.patch.object(
                fsview_dump_util, "_open_feature_view", return_value=(view, False)
            ),
            mock.patch(
                "tzrec.utils.fsview_dump_util.time.time_ns",
                return_value=145_000_000_000,
            ),
        ):
            p1 = dump_fsview(config_path, model_dir, keep_versions=1)

        assert p1 is not None
        self.assertEqual(p1["end_ts"], 100_000)
        v1_dir = os.path.join(
            model_dir, SPARSE_FS_DUMP_DIR, VERSIONS_DIR, p1["version"]
        )
        with np.load(os.path.join(v1_dir, "sparse_embeddings-00-of-01.npz")) as st:
            arr = st["ebc.embedding_bags.stat_fp16"]
            self.assertEqual(arr.dtype, np.float16)
            np.testing.assert_allclose(arr[1], [6.5, 7.5])
        merged = _collect_dynamic_entries(v1_dir)["ebc.embedding_bags.dyn_fp16"]
        self.assertEqual(merged["values"].dtype, np.float16)
        self.assertEqual(set(merged["keys"].tolist()), {20, 30})
        with open(os.path.join(v1_dir, "sparse_embedding.json")) as f:
            dmeta = json.load(f)["ebc.embedding_bags.dyn_fp16"]
        self.assertEqual(dmeta["memory"], 2 * 4)

        # Plant two incomplete newer directories without READY; pruning with
        # keep_versions=1 must NOT count them or delete the valid READY v1!
        versions_root = os.path.join(model_dir, SPARSE_FS_DUMP_DIR, VERSIONS_DIR)
        for broken in ("20990101000001", "20990101000002"):
            os.makedirs(os.path.join(versions_root, broken), exist_ok=True)
        fsview_dump_util._prune_old_versions(
            os.path.join(model_dir, SPARSE_FS_DUMP_DIR),
            versions_root,
            keep_versions=1,
            active_version="other_version",
            stale_incomplete_secs=999999,
        )
        self.assertTrue(os.path.isdir(v1_dir))

    @parameterized.expand(
        [
            ("bad_next_ts", {"next_ts_override": 999}, "next_ts"),
            ("unknown_pk", {"pk": "unknown.table"}, "unknown table"),
            ("reserved_key", {"sk": -1}, "reserved key"),
            ("nan_value", {"val": float("nan")}, "NaN or Inf"),
        ],
        name_func=parameterized_name_func,
    )
    def test_fail_closed_does_not_publish_current_json(
        self, _name: str, case_cfg: Dict[str, Any], err_regex: str
    ) -> None:
        model_dir = os.path.join(self.test_dir, "model_fail")
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _create_fp32_baseline(model_dir)
        _write_pipeline_config(config_path)

        val = case_cfg.get("val", 1.0)
        batch = _make_arrow_batch(
            [case_cfg.get("pk", "ebc.embedding_bags.dyn_a")],
            [case_cfg.get("sk", 10)],
            np.array([[val, 1.0, 1.0, 1.0]], dtype=np.float32),
        )
        view = _FakeDynamicEmbeddingView(
            {0: [batch]},
            next_ts_override=case_cfg.get("next_ts_override"),
        )
        with (
            mock.patch.object(
                fsview_dump_util, "_open_feature_view", return_value=(view, False)
            ),
            mock.patch(
                "tzrec.utils.fsview_dump_util.time.time_ns", return_value=5_000_000_000
            ),
            self.assertRaisesRegex(Exception, err_regex),
        ):
            dump_fsview(config_path, model_dir)

        self.assertFalse(
            os.path.exists(os.path.join(model_dir, SPARSE_FS_DUMP_DIR, CURRENT_JSON))
        )

    def test_multi_rank_gloo_dump(self) -> None:
        model_dir = os.path.join(self.test_dir, "model_dist")
        config_path = os.path.join(self.test_dir, "pipeline.config")
        _create_fp32_baseline(model_dir)
        _write_pipeline_config(config_path)

        port = _find_free_port()
        torch.multiprocessing.spawn(
            _run_distributed_worker,
            args=(2, port, config_path, model_dir),
            nprocs=2,
            join=True,
        )

        dump_root = os.path.join(model_dir, SPARSE_FS_DUMP_DIR)
        with open(os.path.join(dump_root, CURRENT_JSON)) as f:
            current = json.load(f)
        self.assertEqual(current["end_ts"], 5000)
        version_dir = os.path.join(dump_root, VERSIONS_DIR, current["version"])
        self.assertTrue(os.path.isfile(os.path.join(version_dir, READY_FILE)))

        with np.load(os.path.join(version_dir, "sparse_embeddings-00-of-01.npz")) as st:
            stat = st["ebc.embedding_bags.stat_tbl"]
            np.testing.assert_allclose(stat[1], [9.0, 9.0, 9.0, 9.0])
            np.testing.assert_allclose(stat[3], [0.0, 0.0, 0.0, 0.0])

        merged = _collect_dynamic_entries(version_dir)
        dyn_a = {
            int(k): int(s)
            for k, s in zip(
                merged["ebc.embedding_bags.dyn_a"]["keys"],
                merged["ebc.embedding_bags.dyn_a"]["scores"],
            )
        }
        # key 4 tombstoned on rank 1 -> dropped from baseline; key 9 added.
        self.assertEqual(set(dyn_a.keys()), {1, 2, 3, 5, 9})
        self.assertEqual(dyn_a[2], 220)
        self.assertEqual(dyn_a[9], 990)

    def test_oss_uri_staging_and_remote_publish(self) -> None:
        remote_backing = os.path.join(self.test_dir, "fake_oss_bucket", "exp_model")
        _create_fp32_baseline(remote_backing)
        remote_cfg_dir = os.path.join(self.test_dir, "fake_oss_cfg")
        os.makedirs(remote_cfg_dir, exist_ok=True)
        local_cfg = os.path.join(remote_cfg_dir, "pipeline.config")
        _write_pipeline_config(local_cfg)

        class _LocalBackedRemoteFS:
            def _map(self, uri: str) -> str:
                assert uri.startswith("oss://test-bucket/")
                rel = uri[len("oss://test-bucket/") :]
                return os.path.join(self.root, rel)

            def __init__(self, root: str) -> None:
                self.root = root

            def exists(self, uri: str) -> bool:
                return os.path.exists(self._map(uri))

            def ls(self, uri: str, detail: bool = False) -> List[str]:
                p = self._map(uri)
                return [f"{uri.rstrip('/')}/{name}" for name in sorted(os.listdir(p))]

            def makedirs(self, uri: str, exist_ok: bool = False) -> None:
                os.makedirs(self._map(uri), exist_ok=exist_ok)

            def open(self, uri: str, mode: str = "rb") -> Any:
                p = self._map(uri)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                return open(p, mode)

            def rm(self, uri: str, recursive: bool = False) -> None:
                p = self._map(uri)
                if recursive and os.path.isdir(p):
                    shutil.rmtree(p)
                else:
                    os.remove(p)

        fake_fs = _LocalBackedRemoteFS(self.test_dir)

        def _fake_url_to_fs(path: str) -> Any:
            if isinstance(path, str) and path.startswith("oss://"):
                return fake_fs, path
            return None, path

        view = _FakeDynamicEmbeddingView(
            {
                0: [
                    _make_arrow_batch(
                        ["ebc.embedding_bags.dyn_a"],
                        [88],
                        np.array([[8.0, 8.0, 8.0, 8.0]], dtype=np.float32),
                        scores=[880],
                    )
                ]
            },
            additional_fields={"score": "INT64"},
            server_now_ms=777_000,
        )
        with (
            mock.patch.object(
                fsview_dump_util.filesystem_util,
                "url_to_fs",
                side_effect=_fake_url_to_fs,
            ),
            mock.patch.object(
                fsview_dump_util, "_open_feature_view", return_value=(view, True)
            ),
        ):
            payload = dump_fsview(
                "oss://test-bucket/fake_oss_cfg/pipeline.config",
                "oss://test-bucket/fake_oss_bucket/exp_model",
                keep_versions=1,
            )

        assert payload is not None
        self.assertEqual(payload["end_ts"], 777_000)
        remote_ver_dir = os.path.join(
            remote_backing, SPARSE_FS_DUMP_DIR, VERSIONS_DIR, payload["version"]
        )
        self.assertTrue(os.path.isfile(os.path.join(remote_ver_dir, READY_FILE)))
        with open(os.path.join(remote_backing, SPARSE_FS_DUMP_DIR, CURRENT_JSON)) as f:
            self.assertEqual(json.load(f)["version"], payload["version"])


if __name__ == "__main__":
    unittest.main()
