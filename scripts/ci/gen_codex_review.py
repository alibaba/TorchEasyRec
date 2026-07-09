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

"""Generate Codex review inputs from the .claude/ review files.

Keeps .claude/commands/review-pr.md and .claude/agents/*.md as the single
source of truth for AI code review: converts each Claude agent markdown into
a Codex subagent TOML (<out_dir>/.codex/agents/<name>.toml) and emits the
review prompt for `codex exec`. Used by .github/workflows/code_review.yml.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def split_front_matter(text: str) -> Tuple[Dict[str, str], str]:
    """Split markdown into front-matter fields and body.

    Args:
        text (str): markdown text, optionally starting with a `---` block.

    Returns:
        A (fields, body) tuple, fields parsed as flat `key: value` pairs.
    """
    fields = {}
    body = text
    if text.startswith("---"):
        _, front, body = text.split("---", 2)
        for line in front.splitlines():
            key, sep, value = line.partition(":")
            if sep:
                fields[key.strip()] = value.strip()
    return fields, body.strip() + "\n"


def gen_agent_toml(agent_md: Path, agents_dir: Path) -> None:
    """Convert one Claude agent markdown into a Codex subagent TOML.

    Args:
        agent_md (Path): path to a .claude/agents/*.md file.
        agents_dir (Path): output .codex/agents directory.
    """
    fields, body = split_front_matter(agent_md.read_text(encoding="utf-8"))
    name = fields.get("name", agent_md.stem)
    # json.dumps escaping is valid for TOML basic strings.
    lines = [
        f"name = {json.dumps(name)}",
        f"description = {json.dumps(fields.get('description', ''))}",
        # Claude agents are read-only (tools: Glob, Grep, Read).
        'sandbox_mode = "read-only"',
        f"developer_instructions = {json.dumps(body)}",
    ]
    (agents_dir / f"{name}.toml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Generate Codex subagent TOMLs and the review prompt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--claude-dir", required=True, help="trusted .claude dir")
    parser.add_argument("--out-dir", required=True, help="dir to write .codex/agents")
    parser.add_argument("--repo", required=True, help="owner/repo of the PR")
    parser.add_argument("--pr-number", required=True, help="PR number to review")
    parser.add_argument("--prompt-out", required=True, help="review prompt file")
    args = parser.parse_args()

    claude_dir = Path(args.claude_dir)
    agents_dir = Path(args.out_dir) / ".codex" / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for agent_md in sorted((claude_dir / "agents").glob("*.md")):
        gen_agent_toml(agent_md, agents_dir)

    _, prompt_body = split_front_matter(
        (claude_dir / "commands" / "review-pr.md").read_text(encoding="utf-8")
    )
    Path(args.prompt_out).write_text(
        f"REPO: {args.repo} PR_NUMBER: {args.pr_number}\n\n{prompt_body}",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
