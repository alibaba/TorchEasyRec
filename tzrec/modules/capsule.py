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

from typing import Any, Optional, Tuple

import torch
from torch import nn

from tzrec.protos.module_pb2 import B2ICapsule


def sequence_mask(lengths: torch.Tensor, max_len: Optional[int] = None) -> torch.Tensor:
    """Create a boolean mask from sequence lengths.

    Args:
    lengths (Tensor): 1-D tensor containing actual sequence lengths.
    max_len (int, optional): max length. If None, max_len is the maximum
                            value in lengths.

    Returns:
    mask (Tensor): boolean mask with shape [len(lengths), max_len],
                   the first lengths[i] elements of i-th row are True,
                   and the rest are False.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    mask = torch.arange(0, max_len).to(lengths.device)
    # symbolic tracing trick
    zeros_padding = torch.zeros_like(lengths).unsqueeze(1).tile(1, max_len)
    mask = mask + zeros_padding  # broadcasting
    mask = mask < lengths.unsqueeze(1)
    return mask


@torch.fx.wrap
def _init_routing_logits(x: torch.Tensor, k: int) -> torch.Tensor:
    return torch.randn(
        x.size()[:-1] + (k,),
        device=x.device,
        dtype=x.dtype,
    )


class CapsuleLayer(nn.Module):
    """Capsule layer.

    Args:
        capsule_config (B2ICapsule): capsule config.
        input_dim (int): input dimension.

    """

    def __init__(
        self, capsule_config: B2ICapsule, input_dim: int, *args: Any, **kwargs: Any
    ) -> None:
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

    def squash(self, inputs: torch.Tensor) -> torch.Tensor:
        """Squash inputs over the last dimension.

        Args:
            inputs: Tensor, shape: [batch_size, max_k, high_dim]

        Return:
            Tensor, shape: [batch_size, max_k, high_dim]
        """
        input_norm = torch.linalg.norm(inputs, dim=-1, keepdim=True)
        input_norm_eps = torch.max(input_norm, torch.tensor(1e-7))
        scale_factor = (
            torch.pow(
                torch.square(input_norm_eps) / (1 + torch.square(input_norm_eps)),
                self._squash_pow,
            )
            / input_norm_eps
        )
        return scale_factor * inputs

    def dynamic_routing(
        self,
        inputs: torch.Tensor,
        seq_mask: torch.Tensor,
        capsule_mask: torch.Tensor,
        num_iters: int,
    ) -> torch.Tensor:
        """Dynamic routing algorithm.

        Args:
            inputs: Tensor, shape: [batch_size, max_seq_len, low_dim]
            seq_mask: Tensor, shape: [batch_size, max_seq_len]
            capsule_mask: Tensor, shape: [batch_size, max_k]
            num_iters: int, number of iterations

        Return:
            [batch_size, max_k, high_dim]
        """
        routing_logits = _init_routing_logits(inputs, self._max_k)
        routing_logits = routing_logits.detach()

        routing_logits = routing_logits * self._routing_logits_stddev

        capsule_mask = capsule_mask.unsqueeze(1)  # [bs, 1, max_k]
        capsule_mask_thresh = (capsule_mask.float() * 2 - 1) * 1e32

        low_capsule_vec = torch.einsum("bsl, lh -> bsh", inputs, self.bilinear_matrix)
        low_capsule_vec_detach = low_capsule_vec.detach()
        low_capsule_vec_detach_norm = torch.nn.functional.normalize(
            low_capsule_vec_detach, p=2.0, dim=-1
        )

        assert num_iters > 0, "num_iters should be greater than 0"
        high_capsule_vec = torch.Tensor([0])
        for iter in range(num_iters):
            routing_logits = torch.minimum(routing_logits, capsule_mask_thresh)
            routing_logits = torch.nn.functional.softmax(
                routing_logits * self._routing_logits_scale, dim=2
            )  # [b, s, k]
            routing_logits = routing_logits * seq_mask.unsqueeze(2).float()

            if iter + 1 < num_iters:
                high_capsule_vec = torch.einsum(
                    "bsh,bsk->bkh", low_capsule_vec_detach, routing_logits
                )
                routing_logits = routing_logits + torch.einsum(
                    "bkh, bsh -> bsk", high_capsule_vec, low_capsule_vec_detach_norm
                )

            else:
                high_capsule_vec = torch.einsum(
                    "bsh,bsk->bkh", low_capsule_vec, routing_logits
                )
                high_capsule_vec = self.squash(high_capsule_vec)

        return high_capsule_vec

    def forward(
        self, inputs: torch.Tensor, seq_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward method.

        Args:
            inputs: [batch_size, seq_len, low_dim]
            seq_len: [batch_size]

        Return:
            [batch_size, max_k, high_dim]
        """
        _, s, _ = inputs.shape
        device = inputs.device

        # truncating or padding to the input sequence,
        # avoid using if-else statement since the symbolic
        # traced variables are not allowed in control flow
        padding_tensor = torch.zeros_like(inputs)[:, 0:1, :].to(device)
        padding_tensor = padding_tensor.tile(1, self._max_seq_len, 1)
        inputs = inputs[:, : self._max_seq_len, :]
        inputs = torch.cat([inputs, padding_tensor[:, s:, :]], dim=1)

        seq_mask = sequence_mask(seq_len, self._max_seq_len)
        seq_mask = seq_mask.to(device)
        inputs = inputs * seq_mask.unsqueeze(-1).float()
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

        capsule_mask = sequence_mask(n_high_capsules, self._max_k)
        capsule_mask = capsule_mask.to(device)

        user_interests = self.dynamic_routing(
            inputs, seq_mask, capsule_mask, self._num_iters
        )
        user_interests = user_interests * capsule_mask.unsqueeze(-1).float()
        return user_interests, capsule_mask
