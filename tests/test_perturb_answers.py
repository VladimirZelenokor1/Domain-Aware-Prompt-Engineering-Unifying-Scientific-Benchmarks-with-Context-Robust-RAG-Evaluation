"""Tests for ``scripts/perturb_answers`` (draft 3.6.4 answer-level Class 2).

The thesis 3.6.4 operationalises Class-2 (semantic-degradation) perturbations on
the model OUTPUT: padding the answer with filler and truncating its content, then
re-judging the SAME question. These tests pin the pure string transforms and the
record rewrite (only the answer fields change; everything else is preserved).

All tests are hermetic: in-memory strings/dicts, no corpus, no judge, no GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from perturb_answers import pad_answer, perturb_record, truncate_answer  # noqa: E402


def test_truncate_keeps_first_half_of_words() -> None:
    ans = "one two three four five six"
    out = truncate_answer(ans)
    assert out == "one two three"
    assert len(out.split()) < len(ans.split())


def test_truncate_keeps_at_least_one_word() -> None:
    assert truncate_answer("alpha") == "alpha"
    assert truncate_answer("alpha beta") == "alpha"


def test_truncate_empty_is_empty() -> None:
    assert truncate_answer("") == ""
    assert truncate_answer("   ") == ""


def test_pad_is_longer_and_contains_original() -> None:
    ans = "The catalyst lowers the activation energy."
    out = pad_answer(ans)
    assert out.startswith(ans)
    assert len(out) > len(ans)


def test_pad_is_deterministic() -> None:
    ans = "Photosynthesis occurs in the chloroplast."
    assert pad_answer(ans) == pad_answer(ans)


def test_perturb_record_rewrites_answer_fields_only() -> None:
    rec = {
        "question_id": "ske-main-test-00007",
        "question": "What is the boiling point of water?",
        "parsed": {"answer": "100 degrees Celsius at sea level", "extra": "keep"},
        "raw_response": "100 degrees Celsius at sea level",
        "model": "qwen2.5-7b",
    }
    out = perturb_record(rec, "trunc")
    assert out is not None
    # answer fields changed
    assert out["parsed"]["answer"] == truncate_answer(rec["parsed"]["answer"])
    assert out["raw_response"] == out["parsed"]["answer"]
    # everything else preserved
    assert out["question_id"] == rec["question_id"]
    assert out["question"] == rec["question"]
    assert out["parsed"]["extra"] == "keep"
    # original record untouched (no in-place mutation)
    assert rec["parsed"]["answer"] == "100 degrees Celsius at sea level"


def test_perturb_record_pad_uses_raw_response_fallback() -> None:
    rec = {
        "question_id": "q-1",
        "parsed": {"answer": None},
        "raw_response": "a concise factual answer",
    }
    out = perturb_record(rec, "pad")
    assert out is not None
    assert out["parsed"]["answer"].startswith("a concise factual answer")


def test_perturb_record_skips_empty_answer() -> None:
    rec = {"question_id": "q-2", "parsed": {"answer": ""}, "raw_response": ""}
    assert perturb_record(rec, "trunc") is None
    assert perturb_record(rec, "pad") is None
