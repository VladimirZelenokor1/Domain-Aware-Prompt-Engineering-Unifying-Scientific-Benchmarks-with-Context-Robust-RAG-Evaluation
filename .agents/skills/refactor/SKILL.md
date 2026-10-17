---
name: refactor
description: Safe code refactoring with test protection. Use when asked to refactor, restructure, or clean up code.
allowed-tools: Read, Grep, Glob, Bash
---
# Safe Refactoring

## Before starting:
1. Run pytest — ALL tests must pass (baseline)
2. Run git stash or confirm working tree is clean
3. Identify all callers: rg "function_name" --type py

## Process:
1. Plan changes — list files to modify
2. Show plan to user for approval
3. Make ONE change at a time
4. After EACH change — run pytest immediately
5. If tests fail — revert and try different approach

## Common patterns:
- Extract function: identify repeated code → extract → replace all call sites → test
- Rename: find all usages with rg → rename → test
- Move: update all imports → move → test
- Split file: identify logical groups → create new modules → move → update imports → test

## Rules:
- NEVER refactor and add features in the same step
- NEVER skip test run between changes
- Keep diffs small and reviewable