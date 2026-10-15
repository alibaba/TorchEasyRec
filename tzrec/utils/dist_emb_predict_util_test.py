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
import unittest

import numpy as np
import torch
from parameterized import param, parameterized

from tzrec.utils import npz_util, quant_util
from tzrec.utils.dist_emb_predict_util import DistributedEmbeddingLookup
from tzrec.utils.test_util import (
    make_test_dir,
    mark_ci_scope,
    nv_gpu_unavailable,
    parameterized_name_func,
)

_QUANT = quant_util.DISTRIBUTED_SPARSE_SUPPORTED_QUANT_FORMATS[0]
_MISS_KEY = 987654321


class _Table:
    """One exported table and the float rows a lookup should return."""

    def __init__(self, name, num_rows, dim, quant, dynamic, default=False):
        rng = np.random.default_rng(len(name))
        self.name = name
        self.dim = dim
        values = rng.standard_normal((num_rows, dim)).astype(np.float32)
        self.keys = (
            rng.choice(1 << 40, num_rows, replace=False).astype(np.int64)
            if dynamic
            else np.arange(num_rows, dtype=np.int64)
        )
        self.meta = {"dense": False, "is_dynamic": dynamic, "dimension": dim}
        self.meta["shape"] = [num_rows, dim]
        if quant:
            self.stored = quant_util.distributed_quantize_embeddings(
                values, dim, name, _QUANT
            )
            # lookups dequantize to fp16 and are cast to fp32
            self.rows = (
                quant_util.dequantize_quint8_rowwise_f16(self.stored, dim)
                .astype(np.float16)
                .astype(np.float32)
            )
            self.meta.update(dtype=_QUANT, row_bytes=dim + 4)
        else:
            self.stored = values
            self.rows = values
            self.meta["dtype"] = "float32"
        self.miss = np.zeros(dim, dtype=np.float32)
        self.stored_default = None
        if default:
            self.stored_default = self.stored[:1] + 1 if not quant else self.stored[:1]
            self.miss = (
                self.rows[0] if quant else self.stored_default[0].astype(np.float32)
            )
        if dynamic:
            self.meta.update(
                key_dtype="int64",
                score_dtype="int64",
                key_name=f"{name}.keys",
                value_name=f"{name}.values",
                score_name=f"{name}.scores",
            )
            if default:
                self.meta["default_value_name"] = f"{name}.default_value"

    def lookup(self, key):
        """Expected embedding of one key."""
        idx = np.flatnonzero(self.keys == key)
        return self.rows[idx[0]] if len(idx) else self.miss


def _write_export(export_dir, tables, features, dense_meta):
    sparse_dir = os.path.join(export_dir, "sparse")
    os.makedirs(sparse_dir)
    static, dynamic = {}, {}
    for t in tables:
        if t.meta["is_dynamic"]:
            dynamic[f"{t.name}.keys"] = t.keys
            dynamic[f"{t.name}.values"] = t.stored
            dynamic[f"{t.name}.scores"] = np.zeros_like(t.keys)
            if t.stored_default is not None:
                dynamic[f"{t.name}.default_value"] = t.stored_default
        else:
            static[t.name] = t.stored
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_embeddings-00-of-01.npz"), static
    )
    npz_util.savez_streaming(
        os.path.join(sparse_dir, "sparse_dynamic_embedding-00-of-01.npz"), dynamic
    )
    with open(os.path.join(sparse_dir, "sparse_embedding.json"), "w") as f:
        json.dump({t.name: t.meta for t in tables}, f)
    with open(os.path.join(sparse_dir, "sparse_features.json"), "w") as f:
        json.dump(features, f)
    with open(os.path.join(export_dir, "dense_meta.json"), "w") as f:
        json.dump(dense_meta, f)


@mark_ci_scope("gpu")
@unittest.skipIf(*nv_gpu_unavailable)
class DistributedEmbeddingLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.export_dir = make_test_dir()

    def tearDown(self) -> None:
        shutil.rmtree(self.export_dir, ignore_errors=True)

    @parameterized.expand(
        [
            param("quant", quant=True),
            param("fp32", quant=False),
        ],
        name_func=parameterized_name_func,
    )
    def test_lookup_matches_reference(self, _name, quant) -> None:
        # dyn, dyn2 and static merge into one host table (same dtype, dim and
        # default row); zch keeps its own for its default row, small for its dim.
        dyn = _Table("model.ebc.dyn", 50, 8, quant, dynamic=True)
        dyn2 = _Table("model.ebc.dyn2", 40, 8, quant, dynamic=True)
        zch = _Table("model.ebc.zch", 30, 8, quant, dynamic=True, default=True)
        static = _Table("model.ebc.static", 20, 8, quant, dynamic=False)
        small = _Table("model.ebc.small", 10, 4, quant, dynamic=False)
        tables = [dyn, dyn2, zch, static, small]
        features = {
            "a__ebc": {"embedding_name": dyn.name, "pooling": "SUM"},
            "b__ebc": {"embedding_name": zch.name, "pooling": "MEAN"},
            "c__ebc": {"embedding_name": static.name, "pooling": "SUM"},
            "d__ebc": {"embedding_name": dyn2.name, "pooling": "MEAN"},
            "e__ebc": {"embedding_name": dyn.name, "pooling": "SUM"},
            "f__ebc": {"embedding_name": small.name, "pooling": "SUM"},
            "seq__ec": {"embedding_name": dyn.name, "pooling": "NONE"},
            "seq2__ec": {"embedding_name": dyn2.name, "pooling": "NONE"},
        }
        deep = ["a__ebc", "b__ebc", "c__ebc", "d__ebc", "e__ebc", "f__ebc"]
        dense_meta = {
            "sequence__ec": ["seq__ec", "seq__lengths", "seq2__ec", "seq2__lengths"],
            "deep": deep,
        }
        _write_export(self.export_dir, tables, features, dense_meta)

        bags = {
            "a": ([dyn.keys[0], -1, dyn.keys[3]], [], [_MISS_KEY, dyn.keys[7]]),
            "b": ([zch.keys[1], _MISS_KEY], [zch.keys[2]], []),
            "c": ([0, 19], [5], [3, 3, 3]),
            # a key of dyn looked up in dyn2 must miss.
            "d": ([dyn2.keys[4], dyn.keys[4]], [dyn2.keys[0]], [-1]),
            "e": ([dyn.keys[9]], [dyn.keys[9], dyn.keys[1]], []),
            "f": ([1], [], [9, 0]),
            "seq": ([dyn.keys[1], _MISS_KEY], [], [dyn.keys[2]]),
            "seq2": ([dyn2.keys[1]], [dyn.keys[1], -1], [dyn2.keys[3]]),
        }
        # weighted features: "b" (MEAN divides the weighted sum by the bag length,
        # as torchrec does) and "c" (SUM), each fused with unweighted features.
        weights = {
            "b": ([0.5, 2.0], [3.0], []),
            "c": ([1.0, -1.0], [0.25], [2.0, 1.0, 0.5]),
        }
        data = {"label": torch.zeros(3)}
        for name, rows in bags.items():
            data[f"{name}.values"] = torch.tensor(
                [k for r in rows for k in r], dtype=torch.int64
            )
            data[f"{name}.lengths"] = torch.tensor([len(r) for r in rows])
        for name, rows in weights.items():
            data[f"{name}.weights"] = torch.tensor([w for r in rows for w in r])

        device = torch.device("cuda:0")
        lookup = DistributedEmbeddingLookup(self.export_dir, [device])
        # merged dim-8 group (pooled + sequence), zch, small
        self.assertEqual(len(lookup._layers[device]), 4)
        for _ in range(2):
            out = lookup(data, device)

            def pooled(table, rows, mean, row_weights=None):
                res = []
                for i, r in enumerate(rows):
                    w = row_weights[i] if row_weights else [1.0] * len(r)
                    embs = [wi * table.lookup(k) for k, wi in zip(r, w)]
                    v = np.sum(embs, 0) if embs else np.zeros(table.dim)
                    res.append(v / len(embs) if mean and embs else v)
                return np.stack(res).astype(np.float32)

            expected_deep = np.concatenate(
                [
                    pooled(dyn, bags["a"], False),
                    pooled(zch, bags["b"], True, weights["b"]),
                    pooled(static, bags["c"], False, weights["c"]),
                    pooled(dyn2, bags["d"], True),
                    pooled(dyn, bags["e"], False),
                    pooled(small, bags["f"], False),
                ],
                axis=1,
            )
            tol = 2e-3 if quant else 1e-5
            self.assertEqual(out["deep"].device, device)
            self.assertEqual(out["deep"].dtype, torch.float32)
            np.testing.assert_allclose(
                out["deep"].cpu().numpy(), expected_deep, rtol=tol, atol=tol
            )
            expected_seq = np.stack([dyn.lookup(k) for r in bags["seq"] for k in r])
            np.testing.assert_allclose(
                out["seq"].cpu().numpy(), expected_seq, rtol=tol, atol=tol
            )
            expected_seq2 = np.stack([dyn2.lookup(k) for r in bags["seq2"] for k in r])
            np.testing.assert_allclose(
                out["seq2"].cpu().numpy(), expected_seq2, rtol=tol, atol=tol
            )
            self.assertEqual(out["seq__lengths"].dtype, torch.int32)
            self.assertEqual(out["seq__lengths"].tolist(), [2, 0, 1])
            self.assertEqual(out["label"].device, device)

    @unittest.skipIf(torch.cuda.device_count() < 2, "requires 2 GPUs")
    def test_devices_share_tables(self) -> None:
        dyn = _Table("model.ebc.dyn", 50, 8, True, dynamic=True)
        features = {"a__ebc": {"embedding_name": dyn.name, "pooling": "SUM"}}
        _write_export(
            self.export_dir, [dyn], features, {"sequence__ec": [], "deep": ["a__ebc"]}
        )
        data = {
            "a.values": torch.tensor(dyn.keys[:4], dtype=torch.int64),
            "a.lengths": torch.tensor([3, 1]),
        }
        devices = [torch.device("cuda:0"), torch.device("cuda:1")]
        lookup = DistributedEmbeddingLookup(self.export_dir, devices)
        outs = [lookup(data, d)["deep"] for d in devices]
        self.assertEqual(outs[1].device, devices[1])
        torch.testing.assert_close(outs[0].cpu(), outs[1].cpu())


if __name__ == "__main__":
    unittest.main()
