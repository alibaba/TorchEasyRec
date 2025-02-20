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

import torch
from torch import nn


class CapsuleLayer(nn.Module):
    """Capsule layer."""

    def __init__(self, capsule_config, input_dim, is_training, *args, **kwargs) -> None:
        """Capsule layer."""
        super().__init__(*args, **kwargs)
        # max_seq_len: max behaviour sequence length(history length)
        self._max_seq_len = capsule_config.max_seq_len
        # max_k: max high capsule number
        self._max_k = capsule_config.max_k
        # high_dim: high capsule vector dimension
        self._high_dim = capsule_config.high_dim
        # low_dim: low capsule vector dimension
        self._low_dim = input_dim
        # number of Expectation-Maximization iterations
        self._num_iters = capsule_config.num_iters
        # routing_logits_scale
        self._routing_logits_scale = capsule_config.routing_logits_scale
        # routing_logits_stddev
        self._routing_logits_stddev = capsule_config.routing_logits_stddev
        # squash power
        self._squash_pow = capsule_config.squash_pow
        # scale ratio
        # self._scale_ratio = capsule_config.scale_ratio
        # self._const_caps_num = capsule_config.const_caps_num
        self._is_training = is_training

        self.bilinear_matrix = nn.Parameter(
            torch.randn(self._low_dim, self._high_dim)
        )  # [ld, hd]

    def squash(self, inputs):  # double check
        """Squash inputs over the last dimension."""
        input_norm = torch.linalg.norm(inputs, dim=-1, keepdim=True)
        input_norm_eps = torch.max(input_norm, torch.tensor(1e-7))
        scale_factor = input_norm_eps**2 / ((1 + input_norm_eps**2) * input_norm_eps)
        return scale_factor * inputs

    def dyanmic_routing(self, inputs, num_iters):
        """Dynamic routing algorithm.

        Args:
            inputs: [batch_size, seq_len, low_dim]
            num_iters: number of iterations
        return:
            [batch_size, seq_len, high_dim]
        """
        batch_size, seq_len, _ = inputs.size()
        routing_logits = torch.randn(
            batch_size, self._max_k, seq_len, dtype=torch.float32
        )
        routing_logits = (
            routing_logits * self._routing_logits_stddev + self._routing_logits_scale
        )

        y_vec = torch.einsum("bsl, lh -> bsh", inputs, self.bilinear_matrix)
        for _ in range(num_iters):
            routing_logits = torch.nn.functional.softmax(
                routing_logits, dim=-1
            )  # [b, k, s]
            z_vec = torch.einsum("bks, bsh -> bkh", routing_logits, y_vec)
            u_vec = self.squash(z_vec)
            routing_logits = routing_logits + torch.einsum(
                "bkh, bsh -> bks", u_vec, y_vec
            )
        return u_vec

    def forward(self, inputs):
        """Forward method.

        Args:
            inputs: [batch_size, seq_len, low_dim]

        Return:
            [batch_size, max_k, high_dim]
        """
        user_interests = self.dyanmic_routing(inputs, self._num_iters)
        return user_interests
