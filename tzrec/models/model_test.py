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

import io
import unittest
from typing import Dict, Tuple

import torch
from torch import nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, KeyedTensor

from tzrec.datasets.utils import Batch
from tzrec.models.model import CombinedModelWrapper, packed_predict_model


class _Sparse(nn.Module):
    def forward(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        out: Dict[str, torch.Tensor] = {}
        for k, v in data.items():
            out[k] = v.float().sum().reshape(1) * float(len(k))
        return out, out


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sparse = _Sparse()

    def forward(
        self, data: Dict[str, torch.Tensor], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        return self.sparse(data, device)[0]


class _Dense(nn.Module):
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {"p": torch.cat(list(data.values()))}


def _reload(module: nn.Module) -> torch.jit.ScriptModule:
    buf = io.BytesIO()
    torch.jit.save(torch.jit.script(module), buf)
    buf.seek(0)
    return torch.jit.load(buf)


class PackedPredictModelTest(unittest.TestCase):
    def _batch(self) -> Batch:
        torch.manual_seed(0)
        lengths = torch.randint(0, 4, (3 * 5,), dtype=torch.int32)
        n = int(lengths.sum())
        return Batch(
            dense_features={
                "deep": KeyedTensor(
                    keys=["x", "y"], length_per_key=[1, 3], values=torch.rand(5, 4)
                )
            },
            sparse_features={
                "deep": KeyedJaggedTensor(
                    keys=["a", "b", "c"],
                    values=torch.randint(0, 100, (n,), dtype=torch.int32),
                    lengths=lengths,
                    weights=torch.rand(n),
                )
            },
            labels={"clk": torch.rand(5)},
        )

    def test_scripted_model(self):
        batch, device = self._batch(), torch.device("cpu")
        model = _reload(_Model())
        expected = model(batch.to_dict(sparse_dtype=torch.int64), device)
        actual = packed_predict_model(model)(batch.to_packed(), device)
        self.assertEqual(list(actual), list(expected))
        for k, v in expected.items():
            torch.testing.assert_close(actual[k], v, rtol=0, atol=0, msg=k)

    def test_two_stage_model(self):
        batch, device = self._batch(), torch.device("cpu")
        model = CombinedModelWrapper(_reload(_Sparse()), _Dense())
        expected = model(batch.to_dict(sparse_dtype=torch.int64), device)
        actual = packed_predict_model(model)(batch.to_packed(), device)
        torch.testing.assert_close(actual["p"], expected["p"], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
