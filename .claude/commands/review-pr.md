---
allowed-tools: Bash(gh pr comment:*),Bash(gh pr diff:*),Bash(gh pr view:*),Bash(gh api repos:*)
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

Provide feedback using inline comments for specific issues via `gh api repos/.../pulls/.../comments`.
Use top-level comments for general observations or praise.
Keep feedback concise.
