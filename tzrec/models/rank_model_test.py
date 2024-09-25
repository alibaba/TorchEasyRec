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
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import TrainWrapper
from tzrec.models.rank_model import RankModel
from tzrec.protos import loss_pb2, metric_pb2, model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.test_util import TestGraphType, create_test_model


class _TestRankModel(RankModel):
    def __init__(
        self, model_config: ModelConfig, features: List[BaseFeature], labels: List[str]
    ) -> None:
        super().__init__(model_config, features, labels)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        dense_feat_kt = batch.dense_features[BASE_DATA_GROUP]
        y = dense_feat_kt.values()
        return self._output_to_prediction(y)


class RankModelTest(unittest.TestCase):
    @parameterized.expand([[TestGraphType.NORMAL], [TestGraphType.FX_TRACE]])
    def test_rank_model(self, graph_type):
        model_config = model_pb2.ModelConfig(
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
            metrics=[
                metric_pb2.MetricConfig(auc=metric_pb2.AUC()),
                metric_pb2.MetricConfig(
                    grouped_auc=metric_pb2.GroupedAUC(grouping_key="id_a")
                ),
            ],
        )
        model = _TestRankModel(model_config=model_config, features=[], labels=["label"])
        model = TrainWrapper(model)
        model = create_test_model(model, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["id_a"], values=torch.tensor([1, 1]), lengths=torch.tensor([1, 1])
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        label = torch.tensor([0, 1])
        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={"label": label},
        )
        total_loss, (losses, predictions, batch) = model(batch)

        expected_total_loss = torch.tensor(0.6762)
        expected_logits = torch.tensor([0.2000, 0.3000])
        expected_probs = torch.tensor([0.5498, 0.5744])
        expected_auc = torch.tensor(1.0)
        expected_gauc = torch.tensor(1.0)
        torch.testing.assert_close(
            total_loss, expected_total_loss, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            losses["binary_cross_entropy"],
            expected_total_loss,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            predictions["logits"], expected_logits, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            predictions["probs"], expected_probs, rtol=1e-4, atol=1e-4
        )

        if graph_type == TestGraphType.NORMAL:
            model.model.update_metric(predictions, batch)
            metric_result = model.model.compute_metric()
            torch.testing.assert_close(
                metric_result["auc"], expected_auc, rtol=1e-4, atol=1e-4
            )
            torch.testing.assert_close(
                metric_result["grouped_auc"], expected_gauc, rtol=1e-4, atol=1e-4
            )


if __name__ == "__main__":
    unittest.main()
