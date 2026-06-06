"""Tests for ``scripts/qasper_evidence_grounding`` (B-4 Track-B span grounding).

QASPER gold ``evidence`` spans give a ground-truth check on whether a model
answer is grounded, without human annotation. These pin the pure helpers: the
evidence loader (question_id -> non-empty spans) and the grounded decision
(answer entailed by at least one span at the NLI threshold).

Hermetic: tmp JSON + plain lists, no NLI model, no GPU.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from qasper_evidence_grounding import is_grounded, load_evidence  # noqa: E402


def test_is_grounded_true_when_any_span_entails() -> None:
    assert is_grounded([0.2, 0.6, 0.1]) is True


def test_is_grounded_false_when_all_below_threshold() -> None:
    assert is_grounded([0.2, 0.49, 0.1]) is False


def test_is_grounded_empty_is_false() -> None:
    assert is_grounded([]) is False


def test_is_grounded_custom_threshold() -> None:
    assert is_grounded([0.4], threshold=0.3) is True
    assert is_grounded([0.4], threshold=0.5) is False


def test_load_evidence_maps_qid_to_nonempty_spans(tmp_path: Path) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            [
                {"question_id": "q1", "evidence": ["span A", "  ", "span B"]},
                {"question_id": "q2", "evidence": []},
                {"question_id": "q3", "evidence": ["only span"]},
            ]
        ),
        encoding="utf-8",
    )
    ev = load_evidence(sample)
    assert ev["q1"] == ["span A", "span B"]  # blank dropped
    assert "q2" not in ev  # no usable evidence -> absent
    assert ev["q3"] == ["only span"]
