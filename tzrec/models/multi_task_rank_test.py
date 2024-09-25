# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from typing import Dict, List

import torch
from parameterized import parameterized
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import TrainWrapper
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.protos import loss_pb2, metric_pb2, model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.protos.tower_pb2 import TaskTower
from tzrec.utils.test_util import TestGraphType, create_test_model


class _TestMultiTaskRankModel(MultiTaskRank):
    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)
        self._task_tower_cfgs = self._model_config.task_towers

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        dense_feat_kt = batch.dense_features[BASE_DATA_GROUP]
        outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            y = dense_feat_kt.values()
            outputs[task_tower_cfg.tower_name] = y + i
        return self._multi_task_output_to_prediction(outputs)


class MultiTaskRankTest(unittest.TestCase):
    @parameterized.expand([[TestGraphType.NORMAL], [TestGraphType.FX_TRACE]])
    def test_multi_task_rank_model(self, graph_type):
        model_config = model_pb2.ModelConfig(
            simple_multi_task=multi_task_rank_pb2.SimpleMultiTask(
                task_towers=[
                    TaskTower(
                        tower_name="t1",
                        label_name="label1",
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                        metrics=[metric_pb2.MetricConfig(auc=metric_pb2.AUC())],
                    ),
                    TaskTower(
                        tower_name="t2",
                        label_name="label2",
                        losses=[
                            loss_pb2.LossConfig(
                                binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                            )
                        ],
                        metrics=[metric_pb2.MetricConfig(auc=metric_pb2.AUC())],
                    ),
                ]
            )
        )
        model = _TestMultiTaskRankModel(
            model_config=model_config, features=[], labels=["label1", "label2"]
        )
        model = TrainWrapper(model)
        model = create_test_model(model, graph_type)

        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        label1 = torch.tensor([0, 1])
        label2 = torch.tensor([1, 0])
        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={},
            labels={"label1": label1, "label2": label2},
        )
        total_loss, (losses, predictions, batch) = model(batch)

        if graph_type == TestGraphType.NORMAL:
            model.model.update_metric(predictions, batch)
            metric_result = model.model.compute_metric()

        expected_total_loss = torch.tensor(1.5784)
        expected_losses = {
            "binary_cross_entropy_t1": torch.tensor(0.6762),
            "binary_cross_entropy_t2": torch.tensor(0.9021),
        }
        expected_logits = {
            "logits_t1": torch.tensor([0.2000, 0.3000]),
            "logits_t2": torch.tensor([1.2000, 1.3000]),
        }
        expected_probs = {
            "probs_t1": torch.tensor([0.5498, 0.5744]),
            "probs_t2": torch.tensor([0.7685, 0.7858]),
        }
        expected_metrics = {"auc_t1": torch.tensor(1.0), "auc_t2": torch.tensor(0.0)}
        torch.testing.assert_close(
            total_loss, expected_total_loss, rtol=1e-4, atol=1e-4
        )
        for tower_name in ["t1", "t2"]:
            torch.testing.assert_close(
                losses[f"binary_cross_entropy_{tower_name}"],
                expected_losses[f"binary_cross_entropy_{tower_name}"],
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                predictions[f"logits_{tower_name}"],
                expected_logits[f"logits_{tower_name}"],
                rtol=1e-4,
                atol=1e-4,
            )
            torch.testing.assert_close(
                predictions[f"probs_{tower_name}"],
                expected_probs[f"probs_{tower_name}"],
                rtol=1e-4,
                atol=1e-4,
            )
            if graph_type == TestGraphType.NORMAL:
                torch.testing.assert_close(
                    metric_result[f"auc_{tower_name}"],
                    expected_metrics[f"auc_{tower_name}"],
                    rtol=1e-4,
                    atol=1e-4,
                )


if __name__ == "__main__":
    unittest.main()
