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

"""Writes the prompt contract beside the weights, and checks it on restore.

A checkpoint that cannot describe its own vocabulary is a checkpoint serving
has to be told about out of band, which is where offline/online skew comes
from. ``ModulePlan`` is deliberately absent: it is model-only and rebuilt from
config at every ``__init__``.
"""

import dataclasses
import json
import os
import shutil
from enum import Enum
from typing import Any, Dict, Optional

from tzrec.prompt.plan import CompiledPrompt
from tzrec.utils.logging_util import logger

PROMPT_DIR = "prompt"
_SID_SPACE = "sid_space.json"
_PROMPT_PLAN = "prompt_plan.json"
_HASHES = "prompt_hashes.json"
_TOKENIZER = "tokenizer"


def _plain(value: Any) -> Any:
    """Render a compiled artifact as JSON-safe values."""
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)
        }
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


def save_prompt_assets(prompt: CompiledPrompt, target_dir: str) -> None:
    """Write the prompt contract into a checkpoint or export directory.

    Args:
        prompt: the compiled prompt.
        target_dir: the checkpoint or export directory.
    """
    out = os.path.join(target_dir, PROMPT_DIR)
    os.makedirs(out, exist_ok=True)

    with open(os.path.join(out, _SID_SPACE), "w") as f:
        json.dump(_plain(prompt.sid_space), f, indent=2)
    with open(os.path.join(out, _PROMPT_PLAN), "w") as f:
        json.dump(_plain(prompt.prompt_plan), f, indent=2)
    with open(os.path.join(out, _HASHES), "w") as f:
        json.dump(
            {"vocab_hash": prompt.vocab_hash, "plan_hash": prompt.plan_hash},
            f,
            indent=2,
        )

    if prompt.tokenizer_dir and os.path.isdir(prompt.tokenizer_dir):
        shutil.copytree(
            prompt.tokenizer_dir, os.path.join(out, _TOKENIZER), dirs_exist_ok=True
        )


def read_prompt_hashes(source_dir: str) -> Optional[Dict[str, str]]:
    """Read the hashes a checkpoint recorded, or None when it has none."""
    path = os.path.join(source_dir, PROMPT_DIR, _HASHES)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def check_prompt_assets(prompt: Optional[CompiledPrompt], ckpt_dir: str) -> None:
    """Compare a compiled prompt against what a checkpoint recorded.

    A ``vocab_hash`` mismatch is fatal: the decode bands would point at token
    ranges the weights never learned, which produces plausible output rather
    than an error. A ``plan_hash`` mismatch only reshapes the prompt, so it
    warns.

    Args:
        prompt: the freshly compiled prompt, or None when the pipeline declares
            no prompt_config.
        ckpt_dir: the checkpoint being restored.
    """
    if prompt is None:
        return
    recorded = read_prompt_hashes(ckpt_dir)
    if recorded is None:
        logger.warning(
            f"checkpoint [{ckpt_dir}] records no prompt assets, so its "
            f"vocabulary cannot be checked against the current prompt_config."
        )
        return

    if recorded.get("vocab_hash") != prompt.vocab_hash:
        raise ValueError(
            f"prompt vocabulary does not match checkpoint [{ckpt_dir}]: the "
            f"checkpoint was trained against {recorded.get('vocab_hash')} but "
            f"prompt_config now compiles to {prompt.vocab_hash}. The SID space "
            f"or the tokenizer changed, so the decode bands no longer address "
            f"the rows these weights learned."
        )
    if recorded.get("plan_hash") != prompt.plan_hash:
        logger.warning(
            f"prompt plan differs from checkpoint [{ckpt_dir}]: the vocabulary "
            f"matches, so the weights are usable, but the template, slots or "
            f"projections changed."
        )
