---
allowed-tools: Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*)
description: Review a pull request
---

Perform a comprehensive code review using subagents for key areas:

- code-quality-reviewer
- performance-reviewer
- test-coverage-reviewer
- documentation-accuracy-reviewer
- security-code-reviewer

Instruct each to only provide noteworthy feedback. Once they finish, review
the feedback and post only the feedback that you also deem noteworthy.

Use `mcp__github_inline_comment__create_inline_comment` for inline comments on specific lines.
Use top-level comments for general observations or praise.
Use `gh pr comment` for top-level summary comments.
Do NOT use `gh api` — it is not available.
Keep feedback concise.
