#!/bin/bash
# Install trusted agent files into a PR checkout for AI code review
# (GHSA-f9x3-9rgg-92p7): strip PR-authored instruction files at every depth
# (nested CLAUDE.md/AGENTS.md auto-load; .agents can carry skills), restore
# trusted base-branch copies, overlay optional generated files, and commit
# the swap so git status/diff inside the review shows only the PR.
#
# Usage: use_trusted_agent_files.sh TRUSTED_DIR PR_DIR [OVERLAY_DIR]
set -euo pipefail

trusted="$1"
pr="$2"
overlay="${3:-}"

find "$pr" \( -name .claude -o -name .codex -o -name .agents \
  -o -name CLAUDE.md -o -name CLAUDE.local.md -o -name AGENTS.md \) -prune -exec rm -rf {} +
cp -r "$trusted/.claude" "$pr/.claude"
for f in CLAUDE.md AGENTS.md; do
  if [ -f "$trusted/$f" ]; then
    cp "$trusted/$f" "$pr/$f"
  fi
done
if [ -n "$overlay" ]; then
  cp -r "$overlay/." "$pr/"
fi

git -C "$pr" add -A
git -C "$pr" -c user.name=ci -c user.email=ci@localhost \
  commit -q -m "ci: trusted review setup"
