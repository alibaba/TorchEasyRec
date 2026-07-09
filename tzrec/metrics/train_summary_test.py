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
import tempfile
import types
import unittest
from pathlib import Path

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.tensorboard import SummaryWriter

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
build_train_summary_modules = train_summary.build_train_summary_modules
resolve_pred_tensor = train_summary.resolve_pred_tensor


def _mock_config_summaries():
    """summaries_set from multi_tower_din_mock.config."""
    pcoc_cfg = summary_pb2.ModelSummaries()
    pcoc_cfg.pcoc.pred_name = "probs"
    pcoc_cfg.pcoc.label_name = "clk"
    pcoc_cfg.pcoc.epsilon = 1e-7

    scalars_cfg = summary_pb2.ModelSummaries()
    scalars_cfg.scalars.name = "id_1_bucket_pred"
    scalars_cfg.scalars.pred_name = "probs"
    scalars_cfg.scalars.feature_name = "id_1"
    scalars_cfg.scalars.feature_value = "1005"
    return [pcoc_cfg, scalars_cfg]


class _FakeDense:
    def values(self) -> torch.Tensor:
        return self._values

    def __init__(self, values: torch.Tensor) -> None:
        self._values = values


class _FakeSparse:
    def __init__(self, padded: torch.Tensor) -> None:
        self._padded = padded

    def to_padded_dense(self, max_len: int) -> torch.Tensor:
        return self._padded


class _FakeKeyedTensor:
    def __init__(self, tensors: dict) -> None:
        self._tensors = tensors

    def to_dict(self) -> dict:
        return self._tensors


def _make_batch(labels, dense=None, sparse=None):
    batch = types.SimpleNamespace()
    batch.labels = labels
    batch.dense_features = {}
    batch.sparse_features = {}
    if dense is not None:
        batch.dense_features[BASE_DATA_GROUP] = _FakeKeyedTensor(dense)
    if sparse is not None:
        batch.sparse_features[BASE_DATA_GROUP] = _FakeKeyedTensor(sparse)
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

    def test_scalars_sparse_feature_slice(self) -> None:
        """Sparse id feature slice, aligned with multi_tower_din_mock.config."""
        cfg = summary_pb2.ModelSummaries()
        cfg.scalars.name = "id_1_bucket_pred"
        cfg.scalars.feature_name = "id_1"
        cfg.scalars.feature_value = "1005"
        module = TrainSummaryModule(cfg)
        predictions = {"probs": torch.tensor([0.7, 0.3, 0.5])}
        batch = _make_batch(
            {"clk": torch.tensor([1, 0, 1], dtype=torch.float32)},
            sparse={"id_1": _FakeSparse(torch.tensor([[1005], [1002], [1005]]))},
        )
        mask = build_feature_mask(batch, "id_1", "1005")
        self.assertEqual(mask.tolist(), [True, False, True])
        module.update(predictions, batch, "clk")
        values = module.compute()
        self.assertAlmostEqual(values["summary/id_1_bucket_pred"].item(), 0.6, places=4)

    def test_multi_batch_running_mean(self) -> None:
        """Accumulate summaries across batches between log steps."""
        cfg = summary_pb2.ModelSummaries()
        cfg.pcoc.epsilon = 1e-7
        module = TrainSummaryModule(cfg)
        batch1 = _make_batch(
            {"label": torch.tensor([1, 0], dtype=torch.float32)},
        )
        batch2 = _make_batch(
            {"label": torch.tensor([0, 0], dtype=torch.float32)},
        )
        module.update({"probs": torch.tensor([0.8, 0.2])}, batch1, "label")
        module.update({"probs": torch.tensor([0.4, 0.6])}, batch2, "label")
        values = module.compute()
        self.assertAlmostEqual(values["summary/predicted_ctr"].item(), 0.5, places=4)
        self.assertAlmostEqual(values["summary/observed_ctr"].item(), 0.25, places=4)
        self.assertAlmostEqual(values["summary/pcoc"].item(), 2.0, places=4)

    def test_mini_train_tensorboard_integration(self) -> None:
        """Mini-batch train loop: config -> accumulate -> TensorBoard write."""
        modules = build_train_summary_modules(_mock_config_summaries())
        mini_batches = [
            (
                {"probs": torch.tensor([0.6, 0.4])},
                _make_batch(
                    {"clk": torch.tensor([1, 0], dtype=torch.float32)},
                    sparse={"id_1": _FakeSparse(torch.tensor([[1005], [1002]]))},
                ),
            ),
            (
                {"probs": torch.tensor([0.8, 0.2])},
                _make_batch(
                    {"clk": torch.tensor([1, 1], dtype=torch.float32)},
                    sparse={"id_1": _FakeSparse(torch.tensor([[1005], [1005]]))},
                ),
            ),
            (
                {"probs": torch.tensor([0.5])},
                _make_batch(
                    {"clk": torch.tensor([0], dtype=torch.float32)},
                    sparse={"id_1": _FakeSparse(torch.tensor([[1002]]))},
                ),
            ),
        ]
        for predictions, batch in mini_batches:
            for module in modules:
                label_name = (
                    module.summary_cfg.pcoc.label_name
                    if module.summary_cfg.WhichOneof("summary") == "pcoc"
                    else "clk"
                )
                module.update(predictions, batch, label_name)

        summaries: dict = {}
        for module in modules:
            summaries.update(module.compute())

        self.assertAlmostEqual(
            summaries["summary/predicted_ctr"].item(), 0.5, places=4
        )
        self.assertAlmostEqual(
            summaries["summary/observed_ctr"].item(), 0.6, places=4
        )
        self.assertAlmostEqual(summaries["summary/pcoc"].item(), 0.8333, places=3)
        self.assertAlmostEqual(
            summaries["summary/id_1_bucket_pred"].item(), 0.5333, places=3
        )

        with tempfile.TemporaryDirectory() as log_dir:
            writer = SummaryWriter(log_dir)
            step = 10
            for tag, value in summaries.items():
                writer.add_scalar(tag, value, step)
            writer.close()

            event_acc = EventAccumulator(log_dir)
            event_acc.Reload()
            written_tags = set(event_acc.Tags().get("scalars", []))
            self.assertTrue(
                {
                    "summary/pcoc",
                    "summary/predicted_ctr",
                    "summary/observed_ctr",
                    "summary/id_1_bucket_pred",
                }.issubset(written_tags)
            )
            for tag in written_tags:
                if tag.startswith("summary/"):
                    self.assertAlmostEqual(
                        event_acc.Scalars(tag)[0].value,
                        summaries[tag].item(),
                        places=4,
                    )


if __name__ == "__main__":
    unittest.main()
