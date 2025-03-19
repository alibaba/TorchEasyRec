# Copyright (c) 2024-2025, Alibaba Group;
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

from tzrec.modules.sequence import (
    DINEncoder,
    HSTUEncoder,
    MultiWindowDINEncoder,
    PoolingEncoder,
    SimpleAttention,
    create_seq_encoder,
)
from tzrec.protos import module_pb2, seq_encoder_pb2
from tzrec.utils.test_util import TestGraphType, create_test_module


class DINEncoderTest(unittest.TestCase):
    @parameterized.expand(
        [
            [TestGraphType.NORMAL, False, False],
            [TestGraphType.FX_TRACE, False, False],
            [TestGraphType.JIT_SCRIPT, False, False],
            [TestGraphType.NORMAL, True, False],
            [TestGraphType.FX_TRACE, True, False],
            [TestGraphType.JIT_SCRIPT, True, False],
            [TestGraphType.NORMAL, False, True],
            [TestGraphType.FX_TRACE, False, True],
            [TestGraphType.JIT_SCRIPT, False, True],
        ]
    )
    def test_din_encoder(self, graph_type, use_bn, use_dice) -> None:
        din = DINEncoder(
            query_dim=16,
            sequence_dim=16,
            input="click_seq",
            attn_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="Dice" if use_dice else "nn.ReLU",
                use_bn=use_bn,
                dropout_ratio=0.9,
            ),
        )
        self.assertEqual(din.output_dim(), 16)
        din = create_test_module(din, graph_type)
        embedded = {
            "click_seq.query": torch.randn(4, 16),
            "click_seq.sequence": torch.randn(4, 10, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = din(embedded)
        self.assertEqual(result.size(), (4, 16))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_din_encoder_padding(self, graph_type) -> None:
        din = DINEncoder(
            query_dim=12,
            sequence_dim=16,
            input="click_seq",
            attn_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
        )
        self.assertEqual(din.output_dim(), 16)
        din = create_test_module(din, graph_type)
        embedded = {
            "click_seq.query": torch.randn(4, 12),
            "click_seq.sequence": torch.randn(4, 10, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = din(embedded)
        self.assertEqual(result.size(), (4, 16))


class HSTUEncoderTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_hstu_encoder(self, graph_type) -> None:
        hstu = HSTUEncoder(
            sequence_dim=16,
            input="click_seq",
            max_seq_length=10,
            attn_dim=16,
            linear_dim=16,
        )
        self.assertEqual(hstu.output_dim(), 16)
        hstu = create_test_module(hstu, graph_type)
        embedded = {
            "click_seq.sequence": torch.randn(4, 10, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = hstu(embedded)
        if hstu.training:
            self.assertEqual(result.size(), (14, 16))
        else:
            self.assertEqual(result.size(), (4, 16))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_hstu_encoder_padding(self, graph_type) -> None:
        hstu = HSTUEncoder(
            sequence_dim=16,
            input="click_seq",
            max_seq_length=10,
            attn_dim=16,
            linear_dim=16,
        )
        self.assertEqual(hstu.output_dim(), 16)
        hstu = create_test_module(hstu, graph_type)
        embedded = {
            "click_seq.sequence": torch.randn(4, 10, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = hstu(embedded)
        self.assertEqual(result.size(), (4, 16))


class SimpleAttentionTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_simple_attention(self, graph_type) -> None:
        attn = SimpleAttention(
            16,
            16,
            input="click_seq",
        )
        self.assertEqual(attn.output_dim(), 16)
        attn = create_test_module(attn, graph_type)
        embedded = {
            "click_seq.query": torch.randn(4, 16),
            "click_seq.sequence": torch.randn(4, 10, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = attn(embedded)
        self.assertEqual(result.size(), (4, 16))


class PoolingEncoderTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_mean_pooling(self, graph_type) -> None:
        attn = PoolingEncoder(16, input="click_seq", pooling_type="mean")
        self.assertEqual(attn.output_dim(), 16)
        attn = create_test_module(attn, graph_type)
        sequence_length = torch.tensor([2, 3, 4, 5])
        sequence_mask = torch.arange(10).unsqueeze(0) < sequence_length.unsqueeze(1)
        embedded = {
            "click_seq.sequence": torch.ones([4, 10, 16]) * sequence_mask.unsqueeze(2),
            "click_seq.sequence_length": sequence_length,
        }
        result = attn(embedded)
        torch.testing.assert_close(result, torch.ones(4, 16))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_sum_pooling(self, graph_type) -> None:
        attn = PoolingEncoder(16, input="click_seq", pooling_type="sum")
        self.assertEqual(attn.output_dim(), 16)
        attn = create_test_module(attn, graph_type)
        sequence_length = torch.tensor([2, 3, 4, 5])
        sequence_mask = torch.arange(10).unsqueeze(0) < sequence_length.unsqueeze(1)
        embedded = {
            "click_seq.sequence": torch.ones([4, 10, 16]) * sequence_mask.unsqueeze(2),
            "click_seq.sequence_length": sequence_length,
        }
        result = attn(embedded)
        torch.testing.assert_close(
            result, torch.ones(4, 16) * sequence_length.unsqueeze(1)
        )


class MultiWindowDINEncoderTest(unittest.TestCase):
    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_multiwindow_din_encoder(self, graph_type) -> None:
        multiwindow_din = MultiWindowDINEncoder(
            sequence_dim=12,
            query_dim=12,
            input="click_seq",
            windows_len=[1, 1, 2, 2, 5, 10],
            attn_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
        )
        multiwindow_din = create_test_module(multiwindow_din, graph_type)
        embedded = {
            "click_seq.query": torch.randn(4, 12),
            "click_seq.sequence": torch.randn(4, 21, 12),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = multiwindow_din(embedded)
        self.assertEqual(result.size(), (4, 7 * 12))

    @parameterized.expand(
        [[TestGraphType.NORMAL], [TestGraphType.FX_TRACE], [TestGraphType.JIT_SCRIPT]]
    )
    def test_multiwindow_din_encoder_padding(self, graph_type) -> None:
        multiwindow_din = MultiWindowDINEncoder(
            sequence_dim=16,
            query_dim=12,
            input="click_seq",
            windows_len=[1, 1, 2, 2, 5, 10],
            attn_mlp=dict(
                hidden_units=[8, 4, 2],
                activation="nn.ReLU",
                use_bn=False,
                dropout_ratio=0.9,
            ),
        )
        multiwindow_din = create_test_module(multiwindow_din, graph_type)
        embedded = {
            "click_seq.query": torch.randn(4, 12),
            "click_seq.sequence": torch.randn(4, 21, 16),
            "click_seq.sequence_length": torch.tensor([2, 3, 4, 5]),
        }
        result = multiwindow_din(embedded)
        self.assertEqual(result.size(), (4, 7 * 16))


class CreateSequenceTest(unittest.TestCase):
    def test_create_seq_encoder(self) -> None:
        din_encoder = seq_encoder_pb2.DINEncoder(
            name="test", input="all", attn_mlp=module_pb2.MLP(hidden_units=[128, 20])
        )
        config = seq_encoder_pb2.SeqEncoderConfig(din_encoder=din_encoder)
        group_total_dim = {"all.query": 12, "all.sequence": 16}
        encoder = create_seq_encoder(config, group_total_dim)
        self.assertEqual(encoder.__class__, DINEncoder)
        simple_attn = seq_encoder_pb2.SimpleAttention(
            name="test",
            input="all",
        )
        config = seq_encoder_pb2.SeqEncoderConfig(simple_attention=simple_attn)
        group_total_dim = {"all.query": 16, "all.sequence": 16}
        encoder = create_seq_encoder(config, group_total_dim)
        self.assertEqual(encoder.__class__, SimpleAttention)


if __name__ == "__main__":
    unittest.main()
