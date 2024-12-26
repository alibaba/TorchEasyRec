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
from torchrec.sparse.jagged_tensor import KeyedTensor

from tzrec.modules.dense_embedding_collection import (
    AutoDisEmbeddingConfig,
    DenseEmbeddingCollection,
    MLPDenseEmbeddingConfig,
)
from tzrec.utils.test_util import TestGraphType, create_test_module


class DenseEmbeddingCollectionTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mlp_collection(self, graph_type) -> None:
        emb_dense_configs = [
            MLPDenseEmbeddingConfig(16, ["dense_1"]),
            MLPDenseEmbeddingConfig(16, ["dense_2"]),
            MLPDenseEmbeddingConfig(8, ["dense_3"]),
        ]

        emb_collection = DenseEmbeddingCollection(emb_dense_configs)
        emb_collection = create_test_module(emb_collection, graph_type)

        batch_size = 4
        dense_feature = KeyedTensor(
            keys=["dense_1", "dense_2", "dense_3"],
            length_per_key=[1, 1, 1],
            values=torch.concat(
                [
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                ],
                dim=1,
            ),
            key_dim=1,
        )

        result = emb_collection(dense_feature)
        self.assertEqual(result.values().size(), (batch_size, 40))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_autodis_collection(self, graph_type) -> None:
        emb_dense_configs = [
            AutoDisEmbeddingConfig(16, 3, 0.1, 0.8, ["dense_1"]),
            AutoDisEmbeddingConfig(16, 3, 0.1, 0.8, ["dense_2"]),
            AutoDisEmbeddingConfig(8, 3, 0.1, 0.8, ["dense_3"]),
            AutoDisEmbeddingConfig(8, 3, 0.2, 0.8, ["dense_4"]),
        ]

        emb_collection = DenseEmbeddingCollection(emb_dense_configs)
        emb_collection = create_test_module(emb_collection, graph_type)

        batch_size = 4
        dense_feature = KeyedTensor(
            keys=["dense_1", "dense_2", "dense_3", "dense_4"],
            length_per_key=[1, 1, 1, 1],
            values=torch.concat(
                [
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                ],
                dim=1,
            ),
            key_dim=1,
        )

        result = emb_collection(dense_feature)
        self.assertEqual(result.values().size(), (batch_size, 48))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mlp_and_autodis_collection(self, graph_type) -> None:
        emb_dense_configs = [
            AutoDisEmbeddingConfig(16, 3, 0.1, 0.8, ["dense_1"]),
            AutoDisEmbeddingConfig(16, 3, 0.1, 0.8, ["dense_2"]),
            AutoDisEmbeddingConfig(8, 3, 0.1, 0.8, ["dense_3"]),
            AutoDisEmbeddingConfig(8, 3, 0.2, 0.8, ["dense_4"]),
            MLPDenseEmbeddingConfig(16, ["dense_5"]),
        ]

        emb_collection = DenseEmbeddingCollection(emb_dense_configs)
        emb_collection = create_test_module(emb_collection, graph_type)

        batch_size = 4
        dense_feature = KeyedTensor(
            keys=["dense_1", "dense_2", "dense_3", "dense_4", "dense_5"],
            length_per_key=[1, 1, 1, 1, 1],
            values=torch.concat(
                [
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                    torch.randn(batch_size, 1),
                ],
                dim=1,
            ),
            key_dim=1,
        )

        result = emb_collection(dense_feature)
        self.assertEqual(result.values().size(), (batch_size, 64))


if __name__ == "__main__":
    unittest.main()
