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

from typing import Any, Dict, Sequence

import numpy as np
from google.protobuf import text_format
from tokenizers import Tokenizer, models, pre_tokenizers

from tzrec.features.feature import BaseFeature, FgMode, create_features
from tzrec.prompt.assembler import PromptAssembler
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos import feature_pb2


def create_prompt_tokenizer(path: str, words: Sequence[str]) -> str:
    """Write a word-level tokenizer used by prompt tests.

    Args:
        path: destination JSON path.
        words: vocabulary entries in token-id order.

    Returns:
        The destination path.
    """
    tokenizer = Tokenizer(
        models.WordLevel(
            vocab={word: i for i, word in enumerate(words)}, unk_token="<unk>"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(path)
    return path


def create_prompt_feature(text: str) -> BaseFeature:
    """Create one prompt test feature from text-format protobuf.

    Args:
        text: text-format ``FeatureConfig``.

    Returns:
        The created feature.
    """
    config = feature_pb2.FeatureConfig()
    text_format.Merge(text, config)
    return create_features([config], fg_mode=FgMode.FG_NONE)[0]


def offset_sid_codes(codes: Sequence[Any], codebook: Sequence[int]) -> np.ndarray:
    """Shift local SID codes into the flattened per-level space.

    Args:
        codes: local codes grouped by SID item.
        codebook: vocabulary size for each SID level.

    Returns:
        Flat offset codes in item-major order.
    """
    offsets = np.cumsum([0, *codebook[:-1]])
    return (np.asarray(codes).reshape(-1, len(codebook)) + offsets).reshape(-1)


def assemble_into(
    compiled_prompt: CompiledPrompt,
    parsed_features: Dict[str, "np.ndarray"],
) -> Dict[str, np.ndarray]:
    """Assemble one parsed batch with a temporary assembler.

    Args:
        compiled_prompt: the compiled prompt.
        parsed_features: ``{feature}.values`` / ``{feature}.lengths`` as the
            data parser emits them.

    Returns:
        The assembled streams keyed for ``additional_infos``.
    """
    return PromptAssembler(
        compiled_prompt.prompt_plan, compiled_prompt.sid_space
    ).assemble_batch(parsed_features)
