---
name: tdd
description: Test-Driven Development workflow. Use when asked to implement with TDD or write tests first.
allowed-tools: Read, Grep, Glob, Bash
---
# TDD Workflow

## Cycle: Red → Green → Refactor

### 1. RED — Write failing test first
- Create test file if not exists: test_<module>.py
- Write test for the expected behavior
- Run: pytest <test_file> -x — must FAIL
- If it passes — test is wrong, fix it

### 2. GREEN — Minimal implementation
- Write the MINIMUM code to pass the test
- No extra features, no premature optimization
- Run: pytest <test_file> -x — must PASS

### 3. REFACTOR — Clean up
- Remove duplication
- Improve naming
- Extract common logic
- Run: pytest <test_file> — must still PASS

## Rules:
- Never write implementation before test
- One test at a time
- Each test should test ONE behavior
- Show test output at each stage (red/green)