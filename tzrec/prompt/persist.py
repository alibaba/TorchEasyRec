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

"""Checks a checkpoint's prompt contract on restore, and publishes it on export.

The digests ride in the HF export metadata the checkpoint already carries, so
there is no second file that can drift from the weights beside it. Export
additionally writes ``prompt/prompt.json``: what a serving runtime reads, and
only that. The plan itself is deliberately not published there -- it reaches
serving compiled into the front-end artifact, and a copy a runtime could
interpret would invite the second assembler this design exists to prevent.
"""

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

from tzrec.constant import HF_EXPORT_META_FILENAME
from tzrec.prompt.types import CompiledPrompt, SlotSeg, Static
from tzrec.utils.logging_util import logger

PROMPT_DIR = "prompt"
PROMPT_CONTRACT_FILENAME = "prompt.json"
TOKENIZER_DIR = "tokenizer"


def _decode_schedule(compiled_prompt: CompiledPrompt) -> List[Dict[str, int]]:
    """One entry per response position: a forced token or the band it is masked to.

    Derived from the response segments, so a runtime gets the schedule without
    gaining the ability to interpret a plan.
    """
    sid_space = compiled_prompt.sid_space
    schedule: List[Dict[str, int]] = []
    for seg in compiled_prompt.prompt_plan.response_segments:
        if isinstance(seg, Static):
            schedule.extend({"token_id": int(t)} for t in seg.token_ids)
            continue
        assert isinstance(seg, SlotSeg) and sid_space is not None
        width = seg.width.num_positions or 0
        for index in range(width):
            level = index % sid_space.num_levels
            schedule.append(
                {
                    "level": level,
                    "band_lo": sid_space.band_lo[level],
                    "band_hi": sid_space.band_hi[level],
                }
            )
    return schedule


def write_serving_contract(
    compiled_prompt: CompiledPrompt, export_dir: str, frontend: Dict[str, Any]
) -> str:
    """Write ``prompt/prompt.json``, everything a serving runtime reads.

    Args:
        compiled_prompt: the compiled prompt.
        export_dir: the HuggingFace export directory.
        frontend: how the exported front-end is laid out and what it reads.

    Returns:
        The path written.
    """
    plan = compiled_prompt.prompt_plan
    out = os.path.join(export_dir, PROMPT_DIR)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, PROMPT_CONTRACT_FILENAME)
    with open(path, "w") as f:
        json.dump(
            {
                "sid_space": dataclasses.asdict(compiled_prompt.sid_space)
                if compiled_prompt.sid_space is not None
                else None,
                "decode": _decode_schedule(compiled_prompt),
                "static_prefix_len": plan.static_prefix_len,
                "max_length": plan.max_length,
                "max_total_length": plan.max_total_length,
                "vocab_hash": compiled_prompt.vocab_hash,
                "plan_hash": compiled_prompt.plan_hash,
                "frontend": frontend,
            },
            f,
            indent=2,
        )
    return path


def read_prompt_digests(source_dir: str) -> Optional[Dict[str, str]]:
    """Read the digests a checkpoint recorded, or None when it has none."""
    path = os.path.join(source_dir, HF_EXPORT_META_FILENAME)
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        recorded = json.load(f)
    if "vocab_hash" not in recorded:
        return None
    return recorded


def check_prompt_assets(
    compiled_prompt: Optional[CompiledPrompt], ckpt_dir: str
) -> None:
    """Compare a compiled prompt against what a checkpoint recorded.

    A ``vocab_hash`` mismatch is fatal: the decode bands would point at token
    ranges the weights never learned, which produces plausible output rather
    than an error. A ``plan_hash`` mismatch only reshapes the prompt, so it
    warns. Absent digests are fatal too -- restoring unchecked is the one case
    the guard exists to prevent.

    Args:
        compiled_prompt: the freshly compiled prompt, or None when the pipeline
            declares no prompt_config.
        ckpt_dir: the checkpoint being restored.

    Raises:
        ValueError: if the checkpoint records no digests, or its ``vocab_hash``
            disagrees with the compiled prompt.
    """
    if compiled_prompt is None:
        return
    recorded = read_prompt_digests(ckpt_dir)
    if recorded is None:
        raise ValueError(
            f"checkpoint [{ckpt_dir}] records no prompt digests, so its "
            f"vocabulary cannot be checked against the current prompt_config. "
            f"Restoring unchecked risks decode bands that address rows these "
            f"weights never learned, so this is fatal rather than a warning."
        )

    if recorded.get("vocab_hash") != compiled_prompt.vocab_hash:
        raise ValueError(
            f"prompt vocabulary does not match checkpoint [{ckpt_dir}]: the "
            f"checkpoint was trained against {recorded.get('vocab_hash')} but "
            f"prompt_config now compiles to {compiled_prompt.vocab_hash}. The "
            f"SID space or the tokenizer changed, so the decode bands no longer "
            f"address the rows these weights learned."
        )
    if recorded.get("plan_hash") != compiled_prompt.plan_hash:
        logger.warning(
            f"prompt plan differs from checkpoint [{ckpt_dir}]: the vocabulary "
            f"matches, so the weights are usable, but the template, slots or "
            f"projections changed."
        )
