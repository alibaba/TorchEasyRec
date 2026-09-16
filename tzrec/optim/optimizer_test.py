# Copyright (c) 2025, Alibaba Group;
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
from fbgemm_gpu.split_embedding_configs import EmbOptimType
from fbgemm_gpu.split_table_batched_embeddings_ops_common import (
    ComputeDevice,
    EmbeddingLocation,
)
from fbgemm_gpu.split_table_batched_embeddings_ops_training import (
    SplitTableBatchedEmbeddingBagsCodegen,
)
from parameterized import param, parameterized
from torchrec.optim import KeyedOptimizerWrapper

from tzrec.optim.optimizer import TZRecOptimizer, set_sparse_init_accumulator_value
from tzrec.utils.test_util import parameterized_name_func


class TZRecOptimizerTest(unittest.TestCase):
    def test_optimizer(self):
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=0.001)
        )
        optimizer = TZRecOptimizer(keyed_optimizer)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.zero_grad()
        self.assertEqual(param_1.grad, None)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.step()
        torch.testing.assert_close(param_1, torch.tensor([0.9990, 1.9980]))

    def test_optimizer_with_ga(self):
        param_1 = torch.tensor([1.0, 2.0], requires_grad=True)
        keyed_optimizer = KeyedOptimizerWrapper(
            {"param_1": param_1}, lambda params: torch.optim.SGD(params, lr=0.001)
        )
        optimizer = TZRecOptimizer(keyed_optimizer, gradient_accumulation_steps=2)
        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.zero_grad()
        self.assertEqual(param_1.grad, None)

        param_1.grad = torch.tensor([1.0, 2.0])
        optimizer.step()  # do not update
        torch.testing.assert_close(param_1, torch.tensor([1.0, 2.0]))
        optimizer.zero_grad()  # do not zero_grad
        torch.testing.assert_close(param_1.grad, torch.tensor([1.0, 2.0]))

        param_1.grad += torch.tensor([1.0, 2.0])
        optimizer.step()
        torch.testing.assert_close(param_1, torch.tensor([0.9980, 1.9960]))
        optimizer.zero_grad()
        torch.testing.assert_close(param_1.grad, None)


class SparseInitAccumulatorValueTest(unittest.TestCase):
    """The apply_split_helper patch that fills FBGEMM's momentum1 state."""

    _NUM_EMBEDDINGS = 16
    _EMBEDDING_DIM = 8

    def tearDown(self):
        set_sparse_init_accumulator_value(0.0)

    def _build_tbe(self, optimizer, value):
        set_sparse_init_accumulator_value(value)
        return SplitTableBatchedEmbeddingBagsCodegen(
            embedding_specs=[
                (
                    self._NUM_EMBEDDINGS,
                    self._EMBEDDING_DIM,
                    EmbeddingLocation.HOST,
                    ComputeDevice.CPU,
                )
            ],
            optimizer=optimizer,
            learning_rate=0.01,
        )

    @parameterized.expand(
        [
            param(
                "adagrad",
                optimizer=EmbOptimType.EXACT_ADAGRAD,
                # adagrad keeps one accumulator per embedding element.
                state_numel=_NUM_EMBEDDINGS * _EMBEDDING_DIM,
            ),
            param(
                "rowwise_adagrad",
                optimizer=EmbOptimType.EXACT_ROWWISE_ADAGRAD,
                state_numel=_NUM_EMBEDDINGS,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_momentum1_init_value(self, name, optimizer, state_numel):
        emb = self._build_tbe(optimizer, 0.1)
        self.assertEqual(emb.momentum1_host.numel(), state_numel)
        torch.testing.assert_close(emb.momentum1_host, torch.full((state_numel,), 0.1))
        # Only momentum1 is filled, the embedding weights keep their own init.
        torch.testing.assert_close(
            emb.weights_host.detach(),
            torch.zeros(self._NUM_EMBEDDINGS * self._EMBEDDING_DIM),
        )

        emb = self._build_tbe(optimizer, 0.0)
        torch.testing.assert_close(emb.momentum1_host, torch.zeros(state_numel))


if __name__ == "__main__":
    unittest.main()
