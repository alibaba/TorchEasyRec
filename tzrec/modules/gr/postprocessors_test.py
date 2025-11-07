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

from tzrec.modules.gr.postprocessors import (
    L2NormPostprocessor,
    LayerNormPostprocessor,
    TimestampLayerNormPostprocessor,
)


class PostprocessorTest(unittest.TestCase):
    def test_l2norm_postprocessor(self):
        postprocessor = L2NormPostprocessor()
        seq_embeddings = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32
        )
        result = postprocessor(seq_embeddings, None)
        torch.testing.assert_close(
            result,
            torch.tensor(
                [
                    [[0.4472, 0.8944], [0.6000, 0.8000]],
                    [[0.6402, 0.7682], [0.6585, 0.7526]],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_layernorm_postprocessor(self):
        postprocessor = LayerNormPostprocessor(embedding_dim=2)
        seq_embeddings = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32
        )
        result = postprocessor(seq_embeddings, None)
        torch.testing.assert_close(
            result,
            torch.tensor(
                [
                    [[-0.9980, 0.9980], [-0.9980, 0.9980]],
                    [[-0.9980, 0.9980], [-0.9980, 0.9980]],
                ]
            ),
            rtol=1e-4,
            atol=1e-4,
        )

    def test_timestamp_layernorm_postprocessor(self):
        postprocessor = TimestampLayerNormPostprocessor(
            embedding_dim=2,
            time_duration_period_units=[60 * 60, 24 * 60 * 60],
            time_duration_units_per_period=[24, 7],
        )
        seq_embeddings = torch.tensor(
            [[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]],
            dtype=torch.float32,
        )
        seq_timestamps = torch.tensor(
            [[1, 2], [3, 4]],
            dtype=torch.float32,
        )
        result = postprocessor(seq_embeddings, seq_timestamps)
        self.assertEqual(result.size(), (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
