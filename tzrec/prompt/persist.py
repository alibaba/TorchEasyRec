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

"""Publishes the SID space beside an export.

Export writes ``prompt/prompt.json`` with the resolved SID space: the token
base, the per-level bands and the bundle identity a serving side needs to
build its constraint index and to refuse one built from another bundle. The
plan itself is deliberately not published -- it reaches serving compiled into
the front-end, and a copy a runtime could interpret would invite the second
assembler this design exists to prevent.
"""

import dataclasses
import json
import os

from tzrec.prompt.types import CompiledPrompt

PROMPT_DIR = "prompt"
PROMPT_CONTRACT_FILENAME = "prompt.json"
TOKENIZER_DIR = "tokenizer"


def write_serving_contract(compiled_prompt: CompiledPrompt, export_dir: str) -> str:
    """Write ``prompt/prompt.json``: the resolved SID space.

    Args:
        compiled_prompt: the compiled prompt.
        export_dir: the export directory.

    Returns:
        The path written.
    """
    out = os.path.join(export_dir, PROMPT_DIR)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, PROMPT_CONTRACT_FILENAME)
    with open(path, "w") as f:
        json.dump(
            {"sid_space": dataclasses.asdict(compiled_prompt.sid_space)}, f, indent=2
        )
    return path
