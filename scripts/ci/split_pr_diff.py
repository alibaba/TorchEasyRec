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

"""Split a pull request's diff into the per-file inputs an AI review reads.

The reviewers hold Read but no shell, and the PR checkout is shallow, so the
whole change reaches them as files. One combined diff is not enough: Read caps
what it returns, and a diff is path-ordered, so a large PR silently arrives
truncated at its alphabetical tail. Writing one diff per changed file lets each
reviewer read its own area whole. Used by .github/workflows/code_review.yml.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple

_FILE_START = re.compile(rb"(?m)^diff --git ")
# In fallback order: a rename names the new path, and only the header survives
# for a binary, mode-only or pure-rename entry that carries no hunks.
_RENAME_TO = re.compile(rb"(?m)^rename to (.+)$")
_NEW_PATH = re.compile(rb"(?m)^\+\+\+ b/(.+)$")
_OLD_PATH = re.compile(rb"(?m)^--- a/(.+)$")


def chunk_path(chunk: bytes) -> str:
    """Return the post-change path a single-file diff chunk describes.

    Args:
        chunk (bytes): one `diff --git` entry, header included.

    Returns:
        The path as it exists in the head tree, or the pre-change path for a
        deletion.
    """
    for pattern in (_RENAME_TO, _NEW_PATH, _OLD_PATH):
        match = pattern.search(chunk)
        if match:
            return match.group(1).decode()
    header = chunk.split(b"\n", 1)[0][len(b"diff --git ") :].decode()
    _, _, new_path = header.partition(" b/")
    return new_path or header


def split_diff(diff: bytes) -> List[Tuple[str, bytes]]:
    """Split a unified diff into one self-contained chunk per changed file.

    Args:
        diff (bytes): the whole diff, as `gh pr diff` writes it.

    Returns:
        A (path, chunk) list in diff order; the chunks concatenate back to
        `diff`.

    Raises:
        ValueError: the diff holds no entries, or does not begin with one.
    """
    starts = [match.start() for match in _FILE_START.finditer(diff)]
    if not starts:
        raise ValueError("the diff holds no 'diff --git' entries")
    if starts[0] != 0:
        raise ValueError("the diff has content before its first 'diff --git'")
    bounds = starts + [len(diff)]
    chunks = [diff[bounds[i] : bounds[i + 1]] for i in range(len(starts))]
    if b"".join(chunks) != diff:
        raise ValueError("splitting lost bytes; the chunks do not rebuild the diff")
    return [(chunk_path(chunk), chunk) for chunk in chunks]


def write_review_inputs(diff: bytes, out_dir: Path) -> List[Tuple[str, bytes]]:
    """Write one diff per changed file, plus the change map, under `out_dir`.

    Args:
        diff (bytes): the whole diff.
        out_dir (Path): review-inputs directory, already created.

    Returns:
        The (path, chunk) list that was written.
    """
    entries = split_diff(diff)
    for path, chunk in entries:
        target = out_dir / "files" / f"{path}.diff"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(chunk)

    lines = [f"{len(diff.splitlines())} lines, {len(entries)} files", ""]
    lines += [f"{len(chunk.splitlines()):>6}  {path}" for path, chunk in entries]
    (out_dir / "stat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return entries


def main() -> None:
    """Write the review inputs for one pull request diff."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diff", required=True, help="the whole PR diff")
    parser.add_argument("--out-dir", required=True, help="review-inputs directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.is_symlink():
        # A PR can commit one, and writing through it escapes the checkout.
        raise ValueError(f"{out_dir} is a symlink")

    entries = write_review_inputs(Path(args.diff).read_bytes(), out_dir)
    print(f"wrote {len(entries)} per-file diffs to {out_dir / 'files'}")


if __name__ == "__main__":
    main()
