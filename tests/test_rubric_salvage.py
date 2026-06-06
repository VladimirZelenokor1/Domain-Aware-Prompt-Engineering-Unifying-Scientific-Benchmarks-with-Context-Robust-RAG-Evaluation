"""Tests for ``run_judge.parse_rubric_response`` truncation salvage.

A long answer (e.g. the padded Class-2 perturbation) can make the judge's
rationale exhaust max_tokens, so the rubric JSON is cut off before it closes.
The scored ``rubric`` integer is emitted first and is still present; it must be
salvaged rather than collapsed to a spurious 0.

Hermetic: pure string parsing, no vLLM (imported lazily inside run_judge).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_judge import parse_rubric_response  # noqa: E402


def test_valid_json_parses_normally() -> None:
    raw = '{"rubric": 4, "rationale": "good answer", "self_confidence": 0.8}'
    out = parse_rubric_response(raw)
    assert out["rubric"] == 4
    assert out["self_confidence"] == 0.8


def test_truncated_rationale_salvages_rubric() -> None:
    # JSON cut off mid-rationale (no closing brace/quote).
    raw = '{"rubric": 4, "rationale": "The model answer is mostly correct but lacks'
    out = parse_rubric_response(raw)
    assert out["rubric"] == 4  # salvaged, not 0
    assert "SALVAGED" in out["rationale"]


def test_truncated_keeps_self_confidence_if_present() -> None:
    raw = '{"rubric": 3, "self_confidence": 0.7, "rationale": "cut off here'
    out = parse_rubric_response(raw)
    assert out["rubric"] == 3
    assert out["self_confidence"] == 0.7


def test_salvaged_rubric_is_clamped() -> None:
    raw = '{"rubric": 9, "rationale": "broken'
    out = parse_rubric_response(raw)
    assert out["rubric"] == 5  # clamped to max


def test_no_rubric_field_falls_back_to_zero() -> None:
    raw = "complete garbage with no json at all"
    out = parse_rubric_response(raw)
    assert out["rubric"] == 0
    assert "PARSE_FAILED" in out["rationale"]
