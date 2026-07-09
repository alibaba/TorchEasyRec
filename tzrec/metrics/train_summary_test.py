# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]


def _stub_pkg(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = mod
    return mod


def _load_module(name: str, rel_path: str) -> types.ModuleType:
    path = _ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_stub_pkg("tzrec")
_stub_pkg("tzrec.protos")
_stub_pkg("tzrec.metrics")
summary_pb2 = _load_module(
    "tzrec.protos.summary_pb2", "tzrec/protos/summary_pb2.py"
)
train_summary = _load_module(
    "tzrec.metrics.train_summary", "tzrec/metrics/train_summary.py"
)

BASE_DATA_GROUP = train_summary.BASE_DATA_GROUP
TrainSummaryModule = train_summary.TrainSummaryModule
build_feature_mask = train_summary.build_feature_mask
resolve_pred_tensor = train_summary.resolve_pred_tensor


class _FakeDense:
    def values(self) -> torch.Tensor:
        return self._values

    def __init__(self, values: torch.Tensor) -> None:
        self._values = values


class _FakeKeyedTensor:
    def __init__(self, tensors: dict) -> None:
        self._tensors = tensors

    def to_dict(self) -> dict:
        return self._tensors


def _make_batch(labels, dense=None):
    batch = types.SimpleNamespace()
    batch.labels = labels
    batch.dense_features = {}
    batch.sparse_features = {}
    if dense is not None:
        batch.dense_features[BASE_DATA_GROUP] = _FakeKeyedTensor(dense)
    return batch


class TrainSummaryTest(unittest.TestCase):
    def test_resolve_pred_logits(self) -> None:
        preds = {"logits": torch.tensor([0.0, 10.0])}
        out = resolve_pred_tensor(preds, "logits")
        self.assertAlmostEqual(out[0].item(), 0.5, places=4)
        self.assertGreater(out[1].item(), 0.99)

    def test_pcoc_summary(self) -> None:
        cfg = summary_pb2.ModelSummaries()
        cfg.pcoc.epsilon = 1e-7
        module = TrainSummaryModule(cfg)
        predictions = {"probs": torch.tensor([0.8, 0.4, 0.6, 0.2])}
        batch = _make_batch({"label": torch.tensor([1, 0, 1, 0], dtype=torch.float32)})
        module.update(predictions, batch, "label")
        values = module.compute()
        self.assertAlmostEqual(values["summary/pcoc"].item(), 1.0, places=4)

    def test_scalars_numeric_feature_slice(self) -> None:
        cfg = summary_pb2.ModelSummaries()
        cfg.scalars.name = "raw_1_val"
        cfg.scalars.feature_name = "raw_1"
        cfg.scalars.feature_value = "1005"
        module = TrainSummaryModule(cfg)
        predictions = {"probs": torch.tensor([0.9, 0.1])}
        batch = _make_batch(
            {"label": torch.tensor([1, 0], dtype=torch.float32)},
            dense={"raw_1": _FakeDense(torch.tensor([[1005.0], [1002.0]]))},
        )
        mask = build_feature_mask(batch, "raw_1", "1005")
        self.assertTrue(mask[0].item())
        self.assertFalse(mask[1].item())
        module.update(predictions, batch, "label")
        values = module.compute()
        self.assertAlmostEqual(values["summary/raw_1_val"].item(), 0.9, places=4)


if __name__ == "__main__":
    unittest.main()
