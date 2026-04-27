# Retrievability filter: checkpoint / resume support

**Target file:** `scripts/retrievability_filter.py`
**Status:** specification for a coding agent.
**Motivation:** the previous run took 12 h to reach 14,114 / 25,353 (55.7 %), then died without writing anything. No resume is possible today. Expected full run is ~21 h. A single failure costs the entire run.

This spec adds append-only JSONL checkpointing so the run is safe to interrupt at any point.

---

## 1. Requirements

### Functional
1. On startup, the script must detect prior progress and continue from the next unprocessed question.
2. After every question (or small batch of questions), state must be persisted to disk such that a `kill -9` leaves the checkpoint consistent.
3. The final output `data/sciknoweval/main_test_retrievable.json` must be byte-identical (modulo key order) whether produced in one shot or from a resumed run.
4. `--force` must allow a clean restart (wipes the checkpoint directory).
5. `--eval-only` (existing mode) is unaffected.

### Non-functional
- No more than ~1 % runtime overhead from checkpointing.
- Checkpoint files stay small: ≤ a few hundred MB even at full scale.
- Must be safe to run under WSL2 (Windows filesystem may reject `os.fsync` on some paths, so use flush + close where appropriate, not hard fsync).
- No external dependencies.

---

## 2. Storage layout

Create a per-split checkpoint directory next to the test file:

```
data/sciknoweval/
  main_test.json
  .retrievability_cache/
    main_test/
      progress.jsonl          # append-only, one line per processed question
      stats.json               # cumulative stats snapshot (overwritten atomically)
      leakage_cache.jsonl      # append-only, one line per question with leaked=True/False
      manifest.json            # config snapshot; invalidates cache if mismatched
```

- `progress.jsonl` record shape:

  ```json
  {"question_id": "ske-main_test-00000",
   "retrievable": true,
   "leaked": false,
   "domain": "Biology",
   "level": "L2",
   "type": "mcq-4-choices"}
  ```

- `manifest.json` shape — any change in these keys invalidates the checkpoint:

  ```json
  {"split_file": "data/sciknoweval/main_test.json",
   "split_size": 25353,
   "split_sha256": "…",         // sha256 of the split file
   "retriever_version": "2026-03-16",
   "skip_leakage": false,
   "top_k": 10,
   "threshold": 0.3,
   "seed": 42,
   "schema_version": 1}
  ```

  `retriever_version` = the FAISS manifest `build_date`, so reindexing invalidates the cache.

- `stats.json` is what `run_retrievability_filter` returns today, but updated on every flush.

`question_id` format: must match the one produced by `scripts/run_inference.load_dataset`, i.e. `f"ske-main_test-{idx:05d}"`, so downstream scripts can cross-reference.

---

## 3. Required code changes in `retrievability_filter.py`

### 3.1 Add a `Checkpoint` class

```python
class Checkpoint:
    """Append-only JSONL checkpoint for retrievability filtering."""

    def __init__(self, cache_dir: Path, manifest: dict) -> None: ...
    def open(self) -> None: ...            # creates dir, validates manifest
    def processed_ids(self) -> set[str]: ...# returns question_ids already done
    def write(self, record: dict) -> None: ...  # append to progress.jsonl + flush
    def write_stats(self, stats: dict) -> None: ...  # atomic replace stats.json
    def close(self) -> None: ...
```

Key behaviours:
- Opens `progress.jsonl` in append mode with `buffering=1` (line-buffered).
- After each write, `file.flush()` (no fsync; WSL2-safe).
- Truncates any trailing incomplete line on open (if the last line fails to parse, drop it and log a warning).
- On open, if `manifest.json` mismatches the supplied `manifest`, refuses to resume unless `--force` was passed.
- `processed_ids()` scans `progress.jsonl` once on open and caches the set.

### 3.2 Rewrite `run_retrievability_filter`

```python
def run_retrievability_filter(
    test_path: Path,
    retriever,
    output_path: Path,
    *,
    skip_leakage: bool = False,
    force: bool = False,
    flush_every: int = 50,
) -> dict:
```

Changes:
1. Assign `question_id` exactly like `run_inference.load_dataset` (add it at load time, not downstream).
2. Compute `manifest` and construct `Checkpoint(cache_dir, manifest)`.
3. If `force`, clear `cache_dir` before opening the checkpoint.
4. Load already-processed `question_ids` from the checkpoint; skip those in the main loop.
5. Initialise stats by replaying `progress.jsonl` (so resume picks up `stats.json` correctly even if the last `stats.json` write failed).
6. In the loop, after every `flush_every` processed questions, call `checkpoint.write_stats(stats)`.
7. At the end, materialise `main_test_retrievable.json` by joining `progress.jsonl` (the authoritative log) with the original questions. Sort by `question_id` so output is deterministic regardless of resume order.
8. Print the same stats summary as before.

### 3.3 CLI

Add to `parse_args`:

```python
parser.add_argument("--force", action="store_true",
                    help="Clear any existing checkpoint before running.")
parser.add_argument("--flush-every", type=int, default=50,
                    help="Flush stats.json every N questions (default: 50).")
parser.add_argument("--split", default=str(MAIN_TEST_PATH),
                    help="Path to the split file (default: main_test.json).")
```

All three flags must be passed through to `run_retrievability_filter`. Keep `--eval-only` and `--skip-leakage` behaviour untouched.

### 3.4 Logging

- Log the resume state once on startup, e.g.
  `"Resuming from checkpoint: 14114 questions already processed (55.7%)"`.
- Log a one-line heartbeat every `flush_every` questions with running totals.
- If a `manifest` mismatch is detected, log the diff and exit 2 unless `--force` is set.

---

## 4. Testing (TDD)

Create `tests/test_retrievability_checkpoint.py` with these tests:

1. `test_fresh_start_creates_cache`: first run on a toy split creates `progress.jsonl` and `manifest.json`, and final JSON matches expected contents.
2. `test_resume_skips_processed`: manually pre-populate `progress.jsonl` with 5 records; run; verify only the remaining records are processed (monkey-patch retriever to count calls).
3. `test_atomic_crash_recovery`: write `progress.jsonl` with a valid line + a truncated tail line; confirm open() drops the tail, resume continues cleanly.
4. `test_manifest_mismatch_aborts`: change `split_sha256` in `manifest.json`; expect the script to exit 2 without clobbering data.
5. `test_force_clears_cache`: `--force` on a populated cache directory removes `progress.jsonl` and re-runs from 0.
6. `test_final_output_is_deterministic`: running end-to-end twice (once in one shot, once split in two halves with Ctrl+C between) produces an identical `main_test_retrievable.json` (key-order sorted).

Use a MockRetriever returning canned passages so tests stay fast and hermetic. No GPU required.

---

## 5. Acceptance criteria

- `python scripts/retrievability_filter.py --skip-leakage` runs; if killed and restarted, it resumes where it left off with no re-processing.
- Old behaviour (no cache, one-shot) is reproducible with `--force`.
- `pytest tests/test_retrievability_checkpoint.py` passes.
- `ruff check scripts/retrievability_filter.py tests/test_retrievability_checkpoint.py` passes.
- No new top-level dependencies.

---

## 6. Out of scope (do NOT change in this PR)

- The filtering logic itself (`check_retrievability_relaxed`, `check_leakage`).
- The retriever or embedding model.
- The dev-set `--eval-only` recall pipeline.
- Splitting the run into parallel workers. (Future work.)
