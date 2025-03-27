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

import torch
from parameterized import parameterized
from torchrec import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.features.feature import create_features
from tzrec.models.dlrm import DLRM
from tzrec.protos import feature_pb2, loss_pb2, model_pb2, module_pb2
from tzrec.protos.models import rank_model_pb2
from tzrec.utils.test_util import TestGraphType, create_test_model, init_parameters


class DLRMTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False],
            [TestGraphType.FX_TRACE, False],
            [TestGraphType.JIT_SCRIPT, False],
            [TestGraphType.NORMAL, True],
            [TestGraphType.FX_TRACE, True],
            [TestGraphType.JIT_SCRIPT, True],
        ]
    )
    def test_dlrm(self, graph_type, arch_with_sparse) -> None:
        feature_cfgs = [
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_a", embedding_dim=8, num_buckets=100
                )
            ),
            feature_pb2.FeatureConfig(
                id_feature=feature_pb2.IdFeature(
                    feature_name="cat_b", embedding_dim=8, num_buckets=1000
                )
            ),
            feature_pb2.FeatureConfig(
                raw_feature=feature_pb2.RawFeature(feature_name="int_a")
            ),
        ]
        features = create_features(feature_cfgs)
        feature_groups = [
            model_pb2.FeatureGroupConfig(
                group_name="dense",
                feature_names=["int_a"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
            model_pb2.FeatureGroupConfig(
                group_name="sparse",
                feature_names=["cat_a", "cat_b"],
                group_type=model_pb2.FeatureGroupType.DEEP,
            ),
        ]
        model_config = model_pb2.ModelConfig(
            feature_groups=feature_groups,
            dlrm=rank_model_pb2.DLRM(
                dense_mlp=module_pb2.MLP(hidden_units=[2, 8]),
                final=module_pb2.MLP(hidden_units=[8, 4]),
                arch_with_sparse=arch_with_sparse,
            ),
            losses=[
                loss_pb2.LossConfig(binary_cross_entropy=loss_pb2.BinaryCrossEntropy())
            ],
        )
        dlrm = DLRM(model_config=model_config, features=features, labels=["label"])
        init_parameters(dlrm, device=torch.device("cpu"))
        dlrm = create_test_model(dlrm, graph_type)

        sparse_feature = KeyedJaggedTensor.from_lengths_sync(
            keys=["cat_a", "cat_b"],
            values=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
            lengths=torch.tensor([1, 2, 1, 3]),
        )
        dense_feature = KeyedTensor.from_tensor_list(
            keys=["int_a"], tensors=[torch.tensor([[0.2], [0.3]])]
        )

        batch = Batch(
            dense_features={BASE_DATA_GROUP: dense_feature},
            sparse_features={BASE_DATA_GROUP: sparse_feature},
            labels={},
        )
        if graph_type == TestGraphType.JIT_SCRIPT:
            predictions = dlrm(batch.to_dict())
        else:
            predictions = dlrm(batch)
        self.assertEqual(predictions["logits"].size(), (2,))
        self.assertEqual(predictions["probs"].size(), (2,))


if __name__ == "__main__":
    unittest.main()
