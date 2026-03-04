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

from tzrec.modules.personalized_net import EPNet, GateNU, PPNet
from tzrec.utils.test_util import TestGraphType, create_test_module


class GateNUTest(unittest.TestCase):
    """Test GateNU module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_gatenu(self, graph_type) -> None:
        """Test GateNU forward pass."""
        input_dim = 32
        hidden_dim = 16
        output_dim = 8

        gatenu = GateNU(
            input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, gamma=2.0
        )
        gatenu = create_test_module(gatenu, graph_type)

        batch_size = 4
        x = torch.randn(batch_size, input_dim)

        output = gatenu(x)

        self.assertEqual(output.shape, (batch_size, output_dim))
        # Check that output is in [0, gamma] range due to sigmoid activation
        self.assertTrue(torch.all(output >= 0))
        self.assertTrue(torch.all(output <= 2.0))


class EPNetTest(unittest.TestCase):
    """Test EPNet module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_epnet(self, graph_type) -> None:
        """Test EPNet forward pass."""
        domain_dim = 16
        embedding_dim = 32
        epnet = EPNet(
            main_dim=embedding_dim,
            domain_dim=domain_dim,
            hidden_dim=8,
        )
        epnet = create_test_module(epnet, graph_type)
        batch_size = 4
        domain_emb = torch.randn(batch_size, domain_dim)
        main_emb = torch.randn(batch_size, embedding_dim)
        personalized_emb = epnet(main_emb, domain_emb)
        self.assertEqual(personalized_emb.shape, (batch_size, embedding_dim))


class PPNetTest(unittest.TestCase):
    """Test PPNet module."""

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_ppnet_forward(self, graph_type) -> None:
        """Test PPNet forward pass."""
        main_feature = 32
        uia_feature = 16
        num_task = 2

        ppnet = PPNet(
            main_feature=main_feature,
            uia_feature=uia_feature,
            num_task=num_task,
            hidden_units=[16, 8],
        )
        ppnet = create_test_module(ppnet, graph_type)
        batch_size = 4
        main_emb = torch.randn(batch_size, main_feature)
        uia_emb = torch.randn(batch_size, uia_feature)

        task_outputs = ppnet(main_emb, uia_emb)

        self.assertEqual(len(task_outputs), num_task)
        for output in task_outputs:
            self.assertEqual(
                output.shape, (batch_size, 8)
            )  # Output dim from hidden_units


if __name__ == "__main__":
    unittest.main()
