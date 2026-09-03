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

import torch
from torch import nn

from tzrec.modules.prompt_projection import PromptProjection
from tzrec.protos.prompt_pb2 import PromptProjection as PromptProjectionConfig


class PromptProjectionTest(unittest.TestCase):
    def test_bodyless_config_is_a_plain_linear(self) -> None:
        proj = PromptProjection(
            PromptProjectionConfig(bias=False), in_dim=12, hidden_size=8
        )
        self.assertIsNone(proj.body)
        self.assertIsInstance(proj.head, nn.Linear)
        self.assertIsNone(proj.head.bias)
        with torch.no_grad():
            proj.head.weight.fill_(-1.0)
        self.assertTrue(bool((proj(torch.ones(1, 12)) < 0).all()))
        self.assertEqual(proj(torch.randn(5, 12)).shape, (5, 8))

    def test_mlp_body_feeds_a_bare_head(self) -> None:
        config = PromptProjectionConfig()
        config.mlp.hidden_units.extend([16, 6])
        proj = PromptProjection(config, in_dim=12, hidden_size=8)

        self.assertIsNotNone(proj.body)
        # the head maps the MLP's output width, not the slot's input width
        self.assertEqual(proj.head.in_features, 6)
        self.assertEqual(proj.head.out_features, 8)
        self.assertEqual(proj(torch.randn(5, 12)).shape, (5, 8))

    def test_rejects_an_explicitly_empty_mlp(self) -> None:
        config = PromptProjectionConfig()
        config.mlp.SetInParent()

        with self.assertRaisesRegex(ValueError, "hidden_units must not be empty"):
            PromptProjection(config, in_dim=12, hidden_size=8)


if __name__ == "__main__":
    unittest.main()
