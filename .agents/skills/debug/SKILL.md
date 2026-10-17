---
name: debug
description: Systematic debugging of Python errors. Use when asked to debug, fix error, or investigate a traceback.
allowed-tools: Read, Grep, Glob, Bash
---
# Systematic Debugging

## Step 1: Reproduce
- Get the exact error message / traceback
- Identify the failing file and line
- Run the failing command to confirm

## Step 2: Understand
- Read the failing function and its callers
- Check recent changes: git log --oneline -10 + git diff HEAD~3
- Search for similar patterns: rg "function_name" --type py
- Check types and data flow into the failing point

## Step 3: Hypothesize
- List 2-3 most likely causes
- Rank by probability
- State what evidence would confirm each

## Step 4: Fix
- Fix the root cause, not the symptom
- Add a test that catches this specific bug
- Run full test suite to check for regressions

## Step 5: Verify
- Run the originally failing command
- Run related tests
- Show the diff