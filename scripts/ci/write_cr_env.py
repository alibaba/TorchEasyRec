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

"""Record where installed dependency sources live, for an AI code review.

A reviewer that wants to check an API against the version actually in use has
to read the installed package, which lives outside the PR checkout. Left to
guess, it probes a handful of plausible conda layouts, misses, and gives up.
This writes the answer to a file instead, and prints the root the review is
granted read access to. Used by .github/workflows/code_review.yml.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Recorded per environment; enough to match one against requirements/runtime.txt.
_INTERESTING = ("torch", "torchrec", "fbgemm_gpu", "transformers", "torchmetrics")


def conda_base(conda: Path) -> Optional[Path]:
    """Return the conda installation root that owns a `conda` executable.

    Args:
        conda (Path): path to the conda executable, which may not exist.

    Returns:
        The installation root, or None when `conda` is not there or what it
        resolves to is not a conda installation.
    """
    if not conda.exists():
        return None
    # resolve() first: `which conda` can be a cross-root symlink such as
    # /usr/bin/conda -> /opt/conda/bin/conda, and a lexical parent.parent would
    # hand /usr to --add-dir. <base>/bin and <base>/condabin both resolve here.
    base = conda.resolve().parent.parent
    return base if (base / "conda-meta").is_dir() else None


def env_prefixes(base: Path) -> List[Path]:
    """List the environment prefixes belonging to a conda installation.

    Only prefixes under `base` are returned. Conda's global registry
    (~/.conda/environments.txt) can name environments from other installations,
    but the review is granted read access to `base` alone, so listing those
    would advertise paths it cannot open.

    Args:
        base (Path): the conda installation root.

    Returns:
        The base and its named environments, in a stable order.
    """
    found = [base] + sorted(base.glob("envs/*"))
    return [p for p in found if p.is_dir()]


def installed_versions(site_packages: Path) -> Dict[str, str]:
    """Read package versions from dist-info directory names.

    Nothing is imported and no interpreter is started, so a broken environment
    is reported rather than raising.

    Args:
        site_packages (Path): a site-packages directory.

    Returns:
        Version by package name, for the packages worth recording.
    """
    versions = {}
    for entry in site_packages.glob("*.dist-info"):
        name, _, version = entry.name[: -len(".dist-info")].rpartition("-")
        key = name.lower().replace("-", "_")
        if key in _INTERESTING:
            versions[key] = version
    return versions


def describe(base: Optional[Path], conda: Path) -> str:
    """Build the manifest the reviewers read.

    Args:
        base (Path, optional): the conda installation root, if there is one.
        conda (Path): where conda was looked for, for the negative message.

    Returns:
        The manifest text.
    """
    entries = []
    # conda ships lib/python3.1 as a symlink to lib/python3.11, so resolve
    # before de-duplicating or every environment is listed twice.
    seen = set()
    for prefix in env_prefixes(base) if base else []:
        for site_packages in sorted(prefix.glob("lib/python*/site-packages")):
            resolved = site_packages.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            versions = installed_versions(resolved)
            if "torch" in versions:
                listed = ", ".join(f"{k} {v}" for k, v in sorted(versions.items()))
                entries.append(f"{resolved}\n    {listed}")
    if not entries:
        reason = (
            f"no conda at {conda}"
            if base is None
            else f"no environment under {base} has torch installed"
        )
        return (
            f"No Python environment on this runner ({reason}).\n"
            "Installed dependency sources are not available here; do not search "
            "for them.\n"
        )
    return (
        "Installed dependency sources on this runner. Read these to check an API\n"
        "against the version actually in use; match the environment to\n"
        "requirements/runtime.txt.\n\n" + "\n".join(entries) + "\n"
    )


def main() -> None:
    """Write the manifest, and print the root to grant read access to."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conda", required=True, help="conda executable to look for")
    parser.add_argument("--out", required=True, help="manifest to write")
    args = parser.parse_args()

    conda = Path(args.conda)
    base = conda_base(conda)
    manifest = describe(base, conda)
    Path(args.out).write_text(manifest, encoding="utf-8")

    print(manifest, file=sys.stderr)
    # stdout is consumed by the workflow as the --add-dir root; keep it bare.
    print(base if base else "")


if __name__ == "__main__":
    main()
