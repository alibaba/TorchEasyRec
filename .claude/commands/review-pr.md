---
allowed-tools: Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*)
description: Review a pull request
---

Fetch the PR context with `gh pr view`. The PR's full unified diff is
already saved as `pr.diff` at the root of the working directory — read it
from there instead of fetching it again.
Review statically — do not run tests or builds.

Perform a comprehensive code review using subagents for key areas:

- code-quality-reviewer
- performance-reviewer
- test-coverage-reviewer
- documentation-accuracy-reviewer
- security-code-reviewer

Give every subagent the path to `pr.diff` and tell it to read that file
first. Subagents start with an empty context and have only Glob/Grep/Read —
no shell, no `gh`, and no git history to diff against — so that file is
their only view of what changed. Do not paraphrase the change or paste diff
text in its place, and never name a tool an agent does not have.

Instruct each to only provide noteworthy feedback. Once they finish, review
the feedback and post only the feedback that you also deem noteworthy.

The subagents run in the background and report back as notifications, and this
review runs headless: when your turn ends with nothing left to do, the run is
over and any agent still working is killed. So:

- Do not end a turn just to say you are waiting. If you are woken while some
  agents are still pending, keep doing real work in that same turn — continue
  reviewing the diff yourself with Read/Grep — so the run stays alive until the
  rest report back.
- Post your findings before the run can end. Never finish having posted
  nothing; if some agents have still not reported by the time you have to
  wrap up, post what you do have and say which reviews did not finish.
- Always post at least a short top-level summary, even when nothing is
  noteworthy, so a completed review is distinguishable from a review that
  never ran.

Use the `create_inline_comment` tool of the `github_inline_comment` MCP server for inline comments on specific lines.
Use top-level comments for general observations or praise.
Use `gh pr comment` for top-level summary comments.
Do NOT use `gh api` — it is not available.
Keep feedback concise.
