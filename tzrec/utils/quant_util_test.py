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

import unittest

import numpy as np

from tzrec.utils import quant_util


def _dequant_quint8_rowwise_f16(values: np.ndarray, emb_dim: int) -> np.ndarray:
    q = values[:, :emb_dim].astype(np.float32)
    scale = np.ascontiguousarray(values[:, emb_dim : emb_dim + 2]).view(np.float16)
    offset = np.ascontiguousarray(values[:, emb_dim + 2 : emb_dim + 4]).view(np.float16)
    dequant = q * scale.astype(np.float32).reshape(-1, 1)
    dequant += offset.astype(np.float32).reshape(-1, 1)
    return dequant.astype(np.float16).astype(np.float32)


class QuantUtilTest(unittest.TestCase):
    def test_distributed_quantize_embeddings(self) -> None:
        values = np.array([[-2.0, 2.0], [-1.0, 1.0]], dtype=np.float32)

        quantized = quant_util.distributed_quantize_embeddings(
            values,
            emb_dim=2,
            emb_name="test_emb",
            quant_format="QUint8RowwiseF16",
        )

        self.assertEqual(quantized.dtype, np.uint8)
        self.assertEqual(quantized.shape, (2, 6))
        np.testing.assert_allclose(
            _dequant_quint8_rowwise_f16(quantized, emb_dim=2),
            values,
            atol=5e-3,
        )

    def test_distributed_quantize_embeddings_rejects_unsupported_format(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValueError, "Unsupported distributed sparse quant format"
        ):
            quant_util.distributed_quantize_embeddings(
                np.ones((1, 2), dtype=np.float32),
                emb_dim=2,
                emb_name="test_emb",
                quant_format="FP16",
            )


if __name__ == "__main__":
    unittest.main()
