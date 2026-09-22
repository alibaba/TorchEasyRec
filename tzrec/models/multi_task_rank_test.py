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
from typing import Dict, List, Optional

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.loss.listwise_rank_loss import ListwiseRankLoss
from tzrec.models.model import TrainWrapper
from tzrec.models.multi_task_rank import MultiTaskRank
from tzrec.protos import loss_pb2, metric_pb2, model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models import multi_task_rank_pb2
from tzrec.protos.tower_pb2 import TaskTower
from tzrec.utils.test_util import TestGraphType, create_test_model


class _TestMultiTaskRankModel(MultiTaskRank):
    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights)
        self._task_tower_cfgs = self._model_config.task_towers

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        dense_feat_kt = batch.dense_features[BASE_DATA_GROUP]
        outputs = {}
        for i, task_tower_cfg in enumerate(self._task_tower_cfgs):
            y = dense_feat_kt.values()
            outputs[task_tower_cfg.tower_name] = y + i
        return self._multi_task_output_to_prediction(outputs)


class MultiTaskRankTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, 1.0, True],
            [TestGraphType.FX_TRACE, 1.0, True],
            [TestGraphType.NORMAL, 1.0, False],
            [TestGraphType.FX_TRACE, 1.0, False],
            [TestGraphType.NORMAL, 2.0, True],
            [TestGraphType.FX_TRACE, 2.0, True],
            [TestGraphType.NORMAL, 2.0, False],
            [TestGraphType.FX_TRACE, 2.0, False],
        ]
    )
    def test_multi_task_rank_model(self, graph_type, t2_loss_weight, task_space):
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
                        weight=t2_loss_weight,
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
        if task_space:
            second_task = model_config.simple_multi_task.task_towers[1]
            second_task.task_space_indicator_label = "label1"
            second_task.in_task_space_weight = 1
            second_task.out_task_space_weight = 0
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

        if task_space:
            expected_total_loss = torch.tensor(0.6762 + 1.5410 * t2_loss_weight)
            expected_losses = {
                "binary_cross_entropy_t1": torch.tensor(0.6762),
                "binary_cross_entropy_t2": torch.tensor(1.5410 * t2_loss_weight),
            }
        else:
            expected_total_loss = torch.tensor(0.6762 + 0.9021 * t2_loss_weight)
            expected_losses = {
                "binary_cross_entropy_t1": torch.tensor(0.6762),
                "binary_cross_entropy_t2": torch.tensor(0.9021 * t2_loss_weight),
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
                    # pyrefly: ignore[unbound-name]
                    metric_result[f"auc_{tower_name}"],
                    expected_metrics[f"auc_{tower_name}"],
                    rtol=1e-4,
                    atol=1e-4,
                )

    def test_listwise_loss_with_tower_weight(self):
        """A scalar tower weight is not a per-sample weight.

        It must neither trip the listwise guard at construction nor change
        the reduction the point-wise sibling sees; a genuine per-sample
        weight on the same tower is still refused.
        """

        def model_config(**tower_kwargs):
            return model_pb2.ModelConfig(
                simple_multi_task=multi_task_rank_pb2.SimpleMultiTask(
                    task_towers=[
                        TaskTower(
                            tower_name="t1",
                            label_name="label1",
                            weight=0.5,
                            losses=[
                                loss_pb2.LossConfig(
                                    binary_cross_entropy=loss_pb2.BinaryCrossEntropy()
                                ),
                                loss_pb2.LossConfig(
                                    listwise_rank_loss=loss_pb2.ListwiseRankLoss(
                                        session_name="id_a",
                                        learnable_temperature=False,
                                    )
                                ),
                            ],
                            **tower_kwargs,
                        )
                    ]
                )
            )

        model = TrainWrapper(
            _TestMultiTaskRankModel(
                model_config=model_config(), features=[], labels=["label1"]
            )
        )
        ids = torch.tensor([1, 2, 1, 2, 1, 3])
        torch.manual_seed(0)
        logits = torch.randn(6)
        label = torch.tensor([0, 1, 1, 1, 0, 0])
        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: KeyedTensor.from_tensor_list(
                    keys=["int_a"], tensors=[logits.unsqueeze(1)]
                )
            },
            sparse_features={
                BASE_DATA_GROUP: KeyedJaggedTensor.from_lengths_sync(
                    keys=["id_a"], values=ids, lengths=torch.ones(6, dtype=torch.int64)
                )
            },
            labels={"label1": label},
        )
        _, (losses, _, _) = model(batch)

        _, index, lengths = torch.unique(ids, return_inverse=True, return_counts=True)
        expected_listwise = ListwiseRankLoss(learnable_temperature=False)(
            logits, label, lengths, index
        )
        expected_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, label.float()
        )
        torch.testing.assert_close(
            losses["listwise_rank_loss_t1"], 0.5 * expected_listwise
        )
        torch.testing.assert_close(
            losses["binary_cross_entropy_t1"], 0.5 * expected_bce
        )

        with self.assertRaisesRegex(ValueError, "per-sample weights"):
            TrainWrapper(
                _TestMultiTaskRankModel(
                    model_config=model_config(sample_weight_name="w"),
                    features=[],
                    labels=["label1"],
                    sample_weights=["w"],
                )
            )


if __name__ == "__main__":
    unittest.main()
