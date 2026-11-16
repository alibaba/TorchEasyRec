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

"""Shared prompt KV storage that preserves Hugging Face's FA2 attention path."""

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers.cache_utils import Cache


class SharedPromptBeamCache(Cache):
    """Keep immutable prompt KV once and fork only the short beam suffix.

    Args:
        num_layers: number of full-attention layers in the backbone.
    """

    def __init__(self, num_layers: int) -> None:
        super().__init__(layers=[])
        self.num_layers = num_layers
        self._prompt_keys: Dict[int, torch.Tensor] = {}
        self._prompt_values: Dict[int, torch.Tensor] = {}
        self._beam_keys: Dict[int, torch.Tensor] = {}
        self._beam_values: Dict[int, torch.Tensor] = {}
        self._beam_to_prompt: Optional[torch.Tensor] = None

    @property
    def is_sliding(self) -> List[bool]:
        """Describe full attention to the Hugging Face mask builder."""
        return [False] * self.num_layers

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the padded prompt length plus this layer's appended tokens."""
        if layer_idx not in self._prompt_keys:
            return 0
        generated = self._beam_keys.get(layer_idx)
        return self._prompt_keys[layer_idx].shape[2] + (
            generated.shape[2] if generated is not None else 0
        )

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> Tuple[int, int]:
        """Preserve the original mask extent, including left padding."""
        return self.get_seq_length(layer_idx) + query_length, 0

    def _build_beam_kv(
        self, prompt: torch.Tensor, generated: torch.Tensor
    ) -> torch.Tensor:
        """Construct only the current attention layer's dense KV input.

        Args:
            prompt: immutable prompt KV, shaped ``(B, Hkv, P, D)``.
            generated: per-beam generated KV, shaped ``(Bbeam, Hkv, T, D)``.

        Returns:
            Dense KV in the original HF layout, released after this layer.
        """
        assert self._beam_to_prompt is not None
        prompt_width = prompt.shape[2]
        full = generated.new_empty(
            (
                generated.shape[0],
                generated.shape[1],
                prompt_width + generated.shape[2],
                generated.shape[3],
            )
        )
        torch.index_select(
            prompt, 0, self._beam_to_prompt, out=full[:, :, :prompt_width]
        )
        full[:, :, prompt_width:].copy_(generated)
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
            raise ValueError("Shared prompt beam cache requires FP16 or BF16 compute")
        key = key_states.to(value_states.dtype)
        if layer_idx not in self._prompt_keys:
            self._prompt_keys[layer_idx] = key
            self._prompt_values[layer_idx] = value_states
            if self._beam_to_prompt is None:
                self._beam_to_prompt = torch.arange(
                    key_states.shape[0], device=key_states.device
                )
            return key_states, value_states
        if layer_idx in self._beam_keys:
            key = torch.cat([self._beam_keys[layer_idx], key], dim=2)
            value_states = torch.cat(
                [self._beam_values[layer_idx], value_states], dim=2
            )
        self._beam_keys[layer_idx] = key
        self._beam_values[layer_idx] = value_states
        return (
            self._build_beam_kv(self._prompt_keys[layer_idx], key),
            self._build_beam_kv(self._prompt_values[layer_idx], value_states),
        )

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        """Reindex prompt owners and fork suffix KV without duplicating prompt KV.

        Args:
            beam_idx: selected parent rows, including repetitions for branching.
        """
        assert self._beam_to_prompt is not None
        self._beam_to_prompt = self._beam_to_prompt.index_select(0, beam_idx)
        for layer_idx in self._beam_keys:
            self._beam_keys[layer_idx] = self._beam_keys[layer_idx].index_select(
                0, beam_idx
            )
            self._beam_values[layer_idx] = self._beam_values[layer_idx].index_select(
                0, beam_idx
            )


def create_shared_prompt_beam_cache(
    model: nn.Module,
) -> Optional[SharedPromptBeamCache]:
    """Select shared KV storage for full FlashAttention 2 prediction.

    Args:
        model: eval-mode causal LM using the Hugging Face Cache protocol.

    Returns:
        A request-local shared cache for full FlashAttention 2, or None to use
        the backbone's ordinary cache.
    """
    if model.config._attn_implementation != "flash_attention_2":
        return None
    attentions = [layer.self_attn for layer in model.model.layers]
    if any(getattr(a, "sliding_window", None) is not None for a in attentions):
        return None
    return SharedPromptBeamCache(len(attentions))
