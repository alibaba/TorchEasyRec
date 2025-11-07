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
from parameterized import parameterized

from tzrec.modules.gr.content_encoder import (
    MLPContentEncoder,
    PadContentEncoder,
    SliceContentEncoder,
)
from tzrec.utils.test_util import TestGraphType, gpu_unavailable


class ContentEncoderTest(unittest.TestCase):
    @unittest.skipIf(*gpu_unavailable)
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
        ]
    )
    def test_slice_content_encoder(self, graph_type) -> None:
        device = torch.device("cuda")
        uih_embedding_dim = 32
        target_embedding_dim = 64
        encoder = SliceContentEncoder(
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            is_inference=False,
        ).to(device)

        max_uih_len = 4
        max_targets = 2
        uih_offsets = [0, 4, 6]
        target_offsets = [0, 2, 3]
        total_uih_len = 6
        total_targets = 3
        content_embeddings = encoder(
            uih_embeddings=torch.rand(total_uih_len, uih_embedding_dim, device=device),
            target_embeddings=torch.rand(
                total_targets, target_embedding_dim, device=device
            ),
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            uih_offsets=torch.tensor(uih_offsets, device=device),
            target_offsets=torch.tensor(target_offsets, device=device),
            total_uih_len=total_uih_len,
            total_targets=total_targets,
        )
        self.assertEqual(content_embeddings.size(), (9, 32))

    @unittest.skipIf(*gpu_unavailable)
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
        ]
    )
    def test_pad_content_encoder(self, graph_type) -> None:
        device = torch.device("cuda")
        uih_embedding_dim = 32
        target_embedding_dim = 64
        encoder = PadContentEncoder(
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            is_inference=False,
        ).to(device)

        max_uih_len = 4
        max_targets = 2
        uih_offsets = [0, 4, 6]
        target_offsets = [0, 2, 3]
        total_uih_len = 6
        total_targets = 3
        content_embeddings = encoder(
            uih_embeddings=torch.rand(total_uih_len, uih_embedding_dim, device=device),
            target_embeddings=torch.rand(
                total_targets, target_embedding_dim, device=device
            ),
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            uih_offsets=torch.tensor(uih_offsets, device=device),
            target_offsets=torch.tensor(target_offsets, device=device),
            total_uih_len=total_uih_len,
            total_targets=total_targets,
        )
        self.assertEqual(content_embeddings.size(), (9, 64))
        if graph_type == TestGraphType.NORMAL:
            content_embeddings.sum().backward()

    @unittest.skipIf(*gpu_unavailable)
    @parameterized.expand(
        [
            [TestGraphType.NORMAL],
            [TestGraphType.FX_TRACE],
        ]
    )
    def test_mlp_content_encoder(self, graph_type) -> None:
        device = torch.device("cuda")
        uih_embedding_dim = 32
        target_embedding_dim = 64
        encoder = MLPContentEncoder(
            uih_embedding_dim=uih_embedding_dim,
            target_embedding_dim=target_embedding_dim,
            uih_mlp=dict(hidden_units=[128]),
            target_mlp=dict(hidden_units=[128]),
            is_inference=False,
        ).to(device)

        max_uih_len = 4
        max_targets = 2
        uih_offsets = [0, 4, 6]
        target_offsets = [0, 2, 3]
        total_uih_len = 6
        total_targets = 3
        content_embeddings = encoder(
            uih_embeddings=torch.rand(total_uih_len, uih_embedding_dim, device=device),
            target_embeddings=torch.rand(
                total_targets, target_embedding_dim, device=device
            ),
            max_uih_len=max_uih_len,
            max_targets=max_targets,
            uih_offsets=torch.tensor(uih_offsets, device=device),
            target_offsets=torch.tensor(target_offsets, device=device),
            total_uih_len=total_uih_len,
            total_targets=total_targets,
        )
        self.assertEqual(content_embeddings.size(), (9, 128))
        if graph_type == TestGraphType.NORMAL:
            content_embeddings.sum().backward()


if __name__ == "__main__":
    unittest.main()
