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

"""Shared-prefix KV storage that preserves Hugging Face's FA2 attention path."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache


class FlashAttentionBeamCache(Cache):
    """Keep immutable prompt KV once and fork only the short beam suffix.

    Args:
        num_layers: number of full-attention layers in the backbone.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__(layers=[])
        self.num_layers = num_layers
        self._prefix_keys: Dict[int, torch.Tensor] = {}
        self._prefix_values: Dict[int, torch.Tensor] = {}
        self._tail_keys: Dict[int, torch.Tensor] = {}
        self._tail_values: Dict[int, torch.Tensor] = {}
        self._root_rows: Optional[torch.Tensor] = None

    @property
    def is_sliding(self) -> List[bool]:
        """Describe full attention to the Hugging Face mask builder."""
        return [False] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the padded prefix length plus this layer's appended tokens."""
        if layer_idx not in self._prefix_keys:
            return 0
        tail = self._tail_keys.get(layer_idx)
        return self._prefix_keys[layer_idx].shape[2] + (
            tail.shape[2] if tail is not None else 0
        )

    def get_query_offset(self, layer_idx: int = 0) -> int:
        """Match the ordinary dynamic cache's logical query offset."""
        return self.get_seq_length(layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        """Preserve the original mask extent, including left padding."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def _materialize(self, prefix: torch.Tensor, suffix: torch.Tensor) -> torch.Tensor:
        """Construct only the current attention layer's dense KV input.

        Args:
            prefix: immutable prompt KV, shaped ``(B, Hkv, P, D)``.
            suffix: current beam tails, shaped ``(Bbeam, Hkv, T, D)``.

        Returns:
            Dense KV in the original HF layout, released after this layer.
        """
        assert self._root_rows is not None
        prompt_width = prefix.shape[2]
        full = suffix.new_empty(
            (
                suffix.shape[0],
                suffix.shape[1],
                prompt_width + suffix.shape[2],
                suffix.shape[3],
            )
        )
        torch.index_select(prefix, 0, self._root_rows, out=full[:, :, :prompt_width])
        full[:, :, prompt_width:].copy_(suffix)
        return full

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args: Any,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retain prompt storage and append only the small per-beam suffix.

        Args:
            key_states: new rotated keys, shaped ``(Bbeam, Hkv, T, D)``.
            value_states: new values with the FA2 compute dtype.
            layer_idx: attention layer number.
            *args: unused cache protocol arguments.
            **kwargs: unused cache protocol arguments.

        Returns:
            Original tensors at prefill, dense KV for the current decode layer
            otherwise. Attention continues through the unmodified FA2 kernel.
        """
        if value_states.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("FlashAttention beam cache requires FP16 or BF16 compute")
        key = key_states.to(value_states.dtype)
        if layer_idx not in self._prefix_keys:
            self._prefix_keys[layer_idx] = key
            self._prefix_values[layer_idx] = value_states
            if self._root_rows is None:
                self._root_rows = torch.arange(
                    key_states.shape[0], device=key_states.device
                )
            return key_states, value_states
        if layer_idx in self._tail_keys:
            key = torch.cat([self._tail_keys[layer_idx], key], dim=2)
            value_states = torch.cat(
                [self._tail_values[layer_idx], value_states], dim=2
            )
        self._tail_keys[layer_idx] = key
        self._tail_values[layer_idx] = value_states
        return (
            self._materialize(self._prefix_keys[layer_idx], key),
            self._materialize(self._prefix_values[layer_idx], value_states),
        )

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reindex root owners and fork suffix KV without duplicating prompt KV.

        Args:
            beam_idx: selected parent rows, including repetitions for branching.
        """
        assert self._root_rows is not None
        self._root_rows = self._root_rows.index_select(0, beam_idx)
        for layer_idx in self._tail_keys:
            self._tail_keys[layer_idx] = self._tail_keys[layer_idx].index_select(
                0, beam_idx
            )
            self._tail_values[layer_idx] = self._tail_values[layer_idx].index_select(
                0, beam_idx
            )


def create_flash_attention_beam_cache(model: nn.Module) -> FlashAttentionBeamCache:
    """Create inference storage without replacing any attention-module forward.

    Args:
        model: eval-mode Qwen2/Qwen3 causal LM using full FlashAttention 2.

    Returns:
        A request-local cache compatible with the normal model forward.
    """
    if model.config.model_type not in ("qwen2", "qwen3"):
        raise ValueError("Shared FA2 beam decode supports Qwen2 and Qwen3")
    if model.config._attn_implementation != "flash_attention_2":
        raise ValueError("Shared FA2 beam decode requires flash_attention_2")
    attentions = [layer.self_attn for layer in model.model.layers]
    if model.training or any(a.sliding_window is not None for a in attentions):
        raise ValueError("Shared FA2 beam decode requires eval-mode full attention")
    return FlashAttentionBeamCache(len(attentions))
