---
name: code-review
description: Detailed code review. Use when asked to review code, PR, or check quality.
allowed-tools: Read, Grep, Glob, Bash
---
# Code Review

## Process:
1. Run git diff HEAD~1 or git diff --cached to see changes
2. Read ALL changed files completely
3. Analyze each file:

### Checklist:
- Logic errors, edge cases, off-by-one
- Type safety (missing hints, wrong types)
- Security (injection, auth, secrets exposure)
- Performance (N+1 queries, unnecessary loops, missing indexes)
- Error handling (bare except, swallowed errors)
- Test coverage (are new paths tested?)
- DRY violations (duplicated logic)

## Output format:
- 🔴 CRITICAL: [file:line] — description + fix
- 🟡 IMPORTANT: [file:line] — description + fix
- 🟢 MINOR: [file:line] — description
- Final verdict: ✅ Approve / ⚠️ Request Changes / 🔴 Block