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


def sequence_mask(lengths, max_len=None, device=None):
    """Create a boolean mask from sequence lengths.

    Args:
    lengths (Tensor): 1-D tensor containing actual sequence lengths.
    max_len (int, optional): max length。If None, max_len is the maximum
                            value in lengths.
    device: device of the mask.

    Returns:
    mask (Tensor): boolean mask with shape [len(lengths), max_len],
                   the first lengths[i] elements of i-th row are True,
                   and the rest are False.
    """
    if max_len is None:
        max_len = lengths.max()
    mask = torch.arange(0, max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask.to(device)


class CapsuleLayer(nn.Module):
    """Capsule layer."""

    def __init__(self, capsule_config, input_dim, *args, **kwargs) -> None:
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
        # number of dynamic routing iterations
        self._num_iters = capsule_config.num_iters
        # routing_logits_scale
        self._routing_logits_scale = capsule_config.routing_logits_scale
        # routing_logits_stddev
        self._routing_logits_stddev = capsule_config.routing_logits_stddev
        # squash power
        self._squash_pow = capsule_config.squash_pow
        # scale ratio
        # self._scale_ratio = capsule_config.scale_ratio
        self._const_caps_num = capsule_config.const_caps_num

        self.bilinear_matrix = nn.Parameter(
            torch.randn(self._low_dim, self._high_dim)
        )  # [ld, hd]

    def squash(self, inputs):
        """Squash inputs over the last dimension."""
        input_norm = torch.linalg.norm(inputs, dim=-1, keepdim=True)
        input_norm_eps = torch.max(input_norm, torch.tensor(1e-7))
        scale_factor = input_norm_eps**2 / ((1 + input_norm_eps**2) * input_norm_eps)
        return scale_factor * inputs

    def dyanmic_routing(self, inputs, seq_mask, capsule_mask, num_iters):
        """Dynamic routing algorithm.

        Args:
            inputs: Tensor, shape: [batch_size, max_seq_len, low_dim]
            num_iters: int, number of iterations
            seq_mask: Tensor, shape: [batch_size, max_seq_len]
            capsule_mask: Tensor, shape: [batch_size, max_k]

        Return:
            [batch_size, seq_len, high_dim]
        """
        batch_size, max_seq_len, _ = inputs.size()
        routing_logits = torch.randn(
            batch_size, max_seq_len, self._max_k, dtype=torch.float32
        ).to(inputs.device)
        routing_logits = (
            routing_logits * self._routing_logits_stddev + self._routing_logits_scale
        )

        capsule_mask = capsule_mask.unsqueeze(1)  # [bs, 1, max_k]
        capsule_mask_thresh = (capsule_mask.float() * 2 - 1) * 1e32

        low_capsule_vec = torch.einsum("bsl, lh -> bsh", inputs, self.bilinear_matrix)
        for _ in range(num_iters):
            routing_logits = torch.minimum(routing_logits, capsule_mask_thresh)
            routing_logits = torch.nn.functional.softmax(
                routing_logits, dim=1
            )  # [b, s, k]
            routing_logits = routing_logits * seq_mask.unsqueeze(2).float()

            high_capsule_vec = torch.einsum(
                "bsk, bsh -> bkh", routing_logits, low_capsule_vec
            )
            high_capsule_vec = self.squash(high_capsule_vec)
            routing_logits = routing_logits + torch.einsum(
                "bkh, bsh -> bsk", high_capsule_vec, low_capsule_vec
            )
        return high_capsule_vec

    def forward(self, inputs, seq_len):
        """Forward method.

        Args:
            inputs: [batch_size, seq_len, low_dim]
            seq_len: [batch_size]

        Return:
            [batch_size, max_k, high_dim]
        """
        bs, s, low_dim = inputs.size()
        device = inputs.device
        if s < self._max_seq_len:  # padding
            inputs = torch.cat(
                [inputs, torch.zeros(bs, self._max_seq_len - s, low_dim).to(device)],
                dim=1,
            )  # [bs, max_seq_len, low_dim]
        else:  # truncating
            inputs = inputs[:, : self._max_seq_len, :]  # [bs, max_seq_len, low_dim]
        seq_mask = sequence_mask(seq_len, self._max_seq_len, device)

        if self._const_caps_num:
            n_high_capsules = (
                torch.zeros_like(seq_len, dtype=torch.float32) + self._max_k
            )  # [bs,]
            n_high_capsules = n_high_capsules.to(device)
        else:
            n_high_capsules = torch.maximum(
                torch.Tensor([1]).to(seq_len.device),
                torch.minimum(
                    torch.Tensor([self._max_k]).to(seq_len.device),
                    torch.log2(seq_len.float()),
                ),
            ).to(device)  # [bs,]

        capsule_mask = sequence_mask(n_high_capsules, self._max_k, device)

        user_interests = self.dyanmic_routing(
            inputs, seq_mask, capsule_mask, self._num_iters
        )
        return user_interests
