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

"""Fixtures shared by the generative-rec tests across models/, modules/, utils/."""

from typing import Any

import torch


def create_tiny_causal_lm(
    vocab_size: int,
    seed: int = 0,
    tie_word_embeddings: bool = False,
    max_position_embeddings: int = 64,
) -> Any:
    """A 2-layer Qwen2 causal LM cheap enough to build inside a unit test.

    Seeded so two builds agree, and in ``eval()`` so dropout cannot make a decode
    non-deterministic.

    Args:
        vocab_size (int): rows in the embedding table.
        seed (int): torch seed the random init draws from.
        tie_word_embeddings (bool): tie ``lm_head`` to the input embedding.
        max_position_embeddings (int): longest sequence the backbone accepts.

    Returns:
        an eval-mode ``Qwen2ForCausalLM``.
    """
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=tie_word_embeddings,
    )
    return Qwen2ForCausalLM(config).eval()
