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

from tzrec.prompt.types import ResolvedSidSpace

PROMPT_DIR = "prompt"
PROMPT_CONTRACT_FILENAME = "prompt.json"
TOKENIZER_DIR = "tokenizer"


def read_bundle_uuid(manifest_path: str) -> str:
    """The identity of the SID bundle a manifest describes, empty without one.

    Args:
        manifest_path: ``sid_space.manifest_path``, possibly unset.

    Returns:
        The manifest's ``bundle_uuid``, or ``""`` when there is no manifest or
        it records none.
    """
    if not manifest_path:
        return ""
    with open(manifest_path, "r") as f:
        return str(json.load(f).get("bundle_uuid", ""))


def write_serving_contract(
    sid_space: ResolvedSidSpace, bundle_uuid: str, export_dir: str
) -> str:
    """Write ``prompt/prompt.json``: the resolved SID space and the bundle identity.

    Args:
        sid_space: the resolved SID token space.
        bundle_uuid: the bundle the codebook was read from, so an index builder
            can refuse a catalog from another bundle; empty when unknown.
        export_dir: the export directory.

    Returns:
        The path written.
    """
    out = os.path.join(export_dir, PROMPT_DIR)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, PROMPT_CONTRACT_FILENAME)
    with open(path, "w") as f:
        json.dump(
            {"sid_space": dataclasses.asdict(sid_space), "bundle_uuid": bundle_uuid},
            f,
            indent=2,
        )
    return path
