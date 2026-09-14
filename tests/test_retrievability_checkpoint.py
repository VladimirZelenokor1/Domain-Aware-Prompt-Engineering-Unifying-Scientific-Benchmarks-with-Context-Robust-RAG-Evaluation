"""Tests for the append-only checkpoint in scripts/retrievability_filter.py.

These tests do NOT touch GPU, network, or real model weights. A small
``MockRetriever`` below returns canned passages so every branch of
``run_retrievability_filter`` is exercised in-process.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``scripts/`` importable without triggering ``retriever.py`` (which
# imports torch and Pyserini at module level). We insert scripts on the
# path and then import the target module directly.
SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import retrievability_filter as rf  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class MockRetriever:
    """Minimal retriever double used to count calls and serve canned text.

    The real ``Retriever`` loads FAISS, BGE embeddings, and a reranker; none
    of that is available in the test environment. This double reproduces the
    attributes/methods actually touched by ``run_retrievability_filter``.
    """

    def __init__(self, passage_text: str = "a b c d e f g") -> None:
        self.passage_text = passage_text
        self.calls: list[str] = []
        # ``retrievability_filter._retriever_version`` reads this attribute.
        # Pointing at a non-existent path yields a stable "unknown" version.
        self._faiss_index_path = "/tmp/nonexistent-faiss-index.bin"

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> list[dict]:
        self.calls.append(query)
        return [
            {
                "chunk_id": f"mock-{i}",
                "text": self.passage_text,
                "score": 1.0 - i * 0.01,
                "rank": i + 1,
            }
            for i in range(min(top_k, 3))
        ]

    def _get_embedding_model(self):
        # None is a valid signal to check_retrievability_relaxed -> fallback
        # to keyword overlap (no model required).
        return None


def _make_question(idx: int, *, answer_text: str, keyword: str) -> dict:
    """Factory: build a minimal MCQ question with overlapping keywords."""
    return {
        "question": f"What is the role of {keyword} in system {idx}?",
        "type": "mcq-4-choices",
        "answerKey": "A",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": [answer_text, "wrong1", "wrong2", "wrong3"],
        },
        "domain": "Biology",
        "details": {"level": "L2"},
    }


@pytest.fixture
def split_file(tmp_path: Path) -> Path:
    """Create a toy split with 10 questions that all score as retrievable."""
    questions = [
        _make_question(
            i,
            answer_text=f"alpha beta gamma delta epsilon payload{i}",
            keyword=f"payload{i}",
        )
        for i in range(10)
    ]
    split = tmp_path / "sciknoweval" / "toy_split.json"
    split.parent.mkdir(parents=True)
    split.write_text(json.dumps(questions), encoding="utf-8")
    return split


@pytest.fixture
def retriever() -> MockRetriever:
    # Passage text contains all the keywords so relaxed matching says True.
    return MockRetriever(
        passage_text=(
            "alpha beta gamma delta epsilon role system "
            + " ".join(f"payload{i}" for i in range(10))
        )
    )


def _cache_dir(split_path: Path) -> Path:
    return split_path.parent / ".retrievability_cache" / split_path.stem


def _run(split: Path, ret: MockRetriever, **kwargs) -> dict:
    output = split.parent / f"{split.stem}_retrievable.json"
    return rf.run_retrievability_filter(
        split,
        ret,
        output,
        skip_leakage=kwargs.pop("skip_leakage", True),
        force=kwargs.pop("force", False),
        flush_every=kwargs.pop("flush_every", 50),
    )


# ---------------------------------------------------------------------------
# 1. Fresh start creates cache + final JSON
# ---------------------------------------------------------------------------


def test_fresh_start_creates_cache(split_file: Path, retriever: MockRetriever) -> None:
    stats = _run(split_file, retriever)

    cache = _cache_dir(split_file)
    assert (cache / "progress.jsonl").exists()
    assert (cache / "manifest.json").exists()
    assert (cache / "stats.json").exists()

    # progress.jsonl has exactly one record per question.
    lines = (cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10
    parsed = [json.loads(line) for line in lines]
    assert {r["question_id"] for r in parsed} == {
        f"ske-toy-split-{i:05d}" for i in range(10)
    }

    # Final JSON exists and is sorted by question_id.
    out_path = split_file.parent / f"{split_file.stem}_retrievable.json"
    out = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(out) == stats["retrievable"]
    qids = [q["question_id"] for q in out]
    assert qids == sorted(qids)


# ---------------------------------------------------------------------------
# 2. Resume skips already-processed questions
# ---------------------------------------------------------------------------


def test_resume_skips_processed(split_file: Path, retriever: MockRetriever) -> None:
    cache = _cache_dir(split_file)
    cache.mkdir(parents=True)

    # Pre-populate checkpoint: first 5 questions are already "done".
    # The manifest MUST match what run_retrievability_filter will compute,
    # otherwise the run would exit 2.
    total = 10
    manifest = rf._build_manifest(split_file, total, retriever, skip_leakage=True)
    (cache / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )

    pre_records = [
        {
            "question_id": f"ske-toy-split-{i:05d}",
            "has_answer": True,
            "retrievable": True,
            "leaked": False,
            "domain": "Biology",
            "level": "L2",
            "type": "mcq-4-choices",
        }
        for i in range(5)
    ]
    (cache / "progress.jsonl").write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in pre_records) + "\n",
        encoding="utf-8",
    )

    _run(split_file, retriever)

    # The retriever must have been called only for the remaining 5 questions.
    assert len(retriever.calls) == 5

    # progress.jsonl now has all 10 records.
    lines = (cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


# ---------------------------------------------------------------------------
# 3. Atomic crash recovery: drop malformed trailing line
# ---------------------------------------------------------------------------


def test_atomic_crash_recovery(split_file: Path, retriever: MockRetriever) -> None:
    cache = _cache_dir(split_file)
    cache.mkdir(parents=True)

    manifest = rf._build_manifest(split_file, 10, retriever, skip_leakage=True)
    (cache / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )

    good_record = {
        "question_id": "ske-toy-split-00000",
        "has_answer": True,
        "retrievable": True,
        "leaked": False,
        "domain": "Biology",
        "level": "L2",
        "type": "mcq-4-choices",
    }
    good_line = json.dumps(good_record, sort_keys=True)
    # Write one good line plus a truncated tail line (no trailing newline).
    (cache / "progress.jsonl").write_text(
        good_line + "\n" + '{"question_id": "ske-toy-split-00001", "has_an',
        encoding="utf-8",
    )

    _run(split_file, retriever)

    # After recovery we should have processed 9 new questions (the tail was
    # dropped, so only index 0 was previously recorded).
    assert len(retriever.calls) == 9

    lines = (cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 10


# ---------------------------------------------------------------------------
# 4. Manifest mismatch aborts with exit code 2
# ---------------------------------------------------------------------------


def test_manifest_mismatch_aborts(split_file: Path, retriever: MockRetriever) -> None:
    cache = _cache_dir(split_file)
    cache.mkdir(parents=True)

    manifest = rf._build_manifest(split_file, 10, retriever, skip_leakage=True)
    # Corrupt the split_sha256 so validation fails.
    manifest["split_sha256"] = "0" * 64
    (cache / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8"
    )
    # Also pre-write a record so we can assert it was NOT clobbered.
    original_progress = '{"question_id": "ske-toy-split-00000"}\n'
    (cache / "progress.jsonl").write_text(original_progress, encoding="utf-8")

    with pytest.raises(SystemExit) as excinfo:
        _run(split_file, retriever)
    assert excinfo.value.code == 2

    # Data must be untouched.
    assert (cache / "progress.jsonl").read_text(encoding="utf-8") == original_progress


# ---------------------------------------------------------------------------
# 5. --force wipes the cache and re-runs from zero
# ---------------------------------------------------------------------------


def test_force_clears_cache(split_file: Path, retriever: MockRetriever) -> None:
    cache = _cache_dir(split_file)
    cache.mkdir(parents=True)

    # Seed with junk from a "different" manifest: normally this would abort,
    # but --force must wipe first.
    (cache / "manifest.json").write_text(
        json.dumps({"schema_version": 999}), encoding="utf-8"
    )
    (cache / "progress.jsonl").write_text(
        '{"question_id": "stale-id"}\n', encoding="utf-8"
    )
    (cache / "stats.json").write_text("{}", encoding="utf-8")

    _run(split_file, retriever, force=True)

    # All 10 were re-processed from scratch.
    assert len(retriever.calls) == 10

    # The stale id is gone.
    lines = (cache / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    qids = {json.loads(line)["question_id"] for line in lines}
    assert "stale-id" not in qids
    assert len(qids) == 10


# ---------------------------------------------------------------------------
# 6. End-to-end determinism across one-shot vs. split-in-two runs
# ---------------------------------------------------------------------------


def test_final_output_is_deterministic(
    tmp_path: Path, retriever: MockRetriever
) -> None:
    # Build two identical split files in separate directories so their caches
    # and outputs do not collide.
    questions = [
        _make_question(
            i,
            answer_text=f"alpha beta gamma delta epsilon payload{i}",
            keyword=f"payload{i}",
        )
        for i in range(10)
    ]
    payload = json.dumps(questions)

    split_a = tmp_path / "run_a" / "sciknoweval" / "toy_split.json"
    split_a.parent.mkdir(parents=True)
    split_a.write_text(payload, encoding="utf-8")

    split_b = tmp_path / "run_b" / "sciknoweval" / "toy_split.json"
    split_b.parent.mkdir(parents=True)
    split_b.write_text(payload, encoding="utf-8")

    # Run A: one shot.
    ret_a = MockRetriever(retriever.passage_text)
    _run(split_a, ret_a)
    out_a = json.loads(
        (split_a.parent / f"{split_a.stem}_retrievable.json").read_text(
            encoding="utf-8"
        )
    )

    # Run B: simulate a crash after 5 questions, then resume.
    class FlakyRetriever(MockRetriever):
        """Retriever that aborts after N successful calls."""

        def __init__(self, passage_text: str, fail_after: int) -> None:
            super().__init__(passage_text)
            self._fail_after = fail_after

        def retrieve_hybrid(self, query: str, top_k: int = 10) -> list[dict]:
            if len(self.calls) >= self._fail_after:
                raise RuntimeError("simulated crash")
            return super().retrieve_hybrid(query, top_k)

    flaky = FlakyRetriever(retriever.passage_text, fail_after=5)
    with pytest.raises(RuntimeError, match="simulated crash"):
        _run(split_b, flaky)

    # Resume with a healthy retriever.
    ret_b_second = MockRetriever(retriever.passage_text)
    _run(split_b, ret_b_second)
    out_b = json.loads(
        (split_b.parent / f"{split_b.stem}_retrievable.json").read_text(
            encoding="utf-8"
        )
    )

    # Sort keys for a canonical comparison (dict key order is already
    # deterministic for json.dumps with sort_keys=True, but the records
    # themselves must match).
    assert json.dumps(out_a, sort_keys=True) == json.dumps(out_b, sort_keys=True)
