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
from typing import Any, Dict, List

import torch
from parameterized import parameterized
from torchrec import KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, NEG_DATA_GROUP, Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.match_model import MatchModel
from tzrec.models.model import TrainWrapper
from tzrec.protos import loss_pb2, metric_pb2, model_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.utils.test_util import TestGraphType, create_test_model


class _TestMatchModel(MatchModel):
    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        in_batch_negative: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, **kwargs)
        self._in_batch_negative = in_batch_negative

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        dense_feat_kt = batch.dense_features[BASE_DATA_GROUP]
        dense_neg_kt = batch.dense_features[NEG_DATA_GROUP]
        simi = self.sim(
            dense_feat_kt.values(),
            dense_neg_kt.values(),
            hard_neg_indices=None,
        )
        return {"similarity": simi}


class MatchModelTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
        ]
    )
    def test_match_model(self, graph_type, in_batch_neg=False):
        model_config = model_pb2.ModelConfig(
            losses=[
                loss_pb2.LossConfig(
                    softmax_cross_entropy=loss_pb2.SoftmaxCrossEntropy()
                )
            ],
            metrics=[
                metric_pb2.MetricConfig(recall_at_k=metric_pb2.RecallAtK(top_k=2))
            ],
        )
        model = _TestMatchModel(
            model_config=model_config,
            features=[],
            labels=["label"],
            in_batch_negative=in_batch_neg,
            sampler_type="negative_sampler",
        )
        model = TrainWrapper(model)
        model = create_test_model(model, graph_type)

        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )
        dense_feature_neg = KeyedTensor.from_tensor_list(
            keys=["int_b"], tensors=[torch.tensor([[0.2], [0.3], [0.4]])]
        )
        label = torch.tensor([1, 1])
        batch = Batch(
            dense_features={
                BASE_DATA_GROUP: dense_feature,
                NEG_DATA_GROUP: dense_feature_neg,
            },
            sparse_features={},
            labels={"label": label},
        )
        total_loss, (losses, predictions, batch) = model(batch)

        if in_batch_neg:
            expected_total_loss = torch.tensor(1.1088)
            expected_similarity = torch.tensor(
                [[0.0400, 0.0600, 0.0800], [0.0600, 0.0900, 0.1200]]
            )
            expected_recall = torch.tensor(0.5)
        else:
            expected_total_loss = torch.tensor(0.71080)
            expected_similarity = torch.tensor([[0.0400, 0.0800], [0.0900, 0.1200]])
            expected_recall = torch.tensor(1.0)
        torch.testing.assert_close(
            total_loss, expected_total_loss, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            losses["softmax_cross_entropy"],
            expected_total_loss,
            rtol=1e-4,
            atol=1e-4,
        )
        torch.testing.assert_close(
            predictions["similarity"], expected_similarity, rtol=1e-4, atol=1e-4
        )
        if graph_type == TestGraphType.NORMAL:
            model.model.update_metric(predictions, batch)
            metric_result = model.model.compute_metric()
            torch.testing.assert_close(
                metric_result["recall@2"],
                expected_recall,
                rtol=1e-4,
                atol=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
