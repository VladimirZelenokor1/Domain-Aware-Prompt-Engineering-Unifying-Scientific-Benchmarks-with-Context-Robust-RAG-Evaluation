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

from qasper_evidence_grounding import (  # noqa: E402
    is_grounded,
    load_evidence,
    load_evidence_from_raw,
)


def test_is_grounded_true_when_any_span_entails() -> None:
    assert is_grounded([0.2, 0.6, 0.1]) is True


def test_is_grounded_false_when_all_below_threshold() -> None:
    assert is_grounded([0.2, 0.49, 0.1]) is False


def test_is_grounded_empty_is_false() -> None:
    assert is_grounded([]) is False


def test_is_grounded_custom_threshold() -> None:
    assert is_grounded([0.4], threshold=0.3) is True
    assert is_grounded([0.4], threshold=0.5) is False


def test_load_evidence_maps_normalised_question_to_nonempty_spans(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.json"
    sample.write_text(
        json.dumps(
            [
                {"question": "What  is X?", "evidence": ["span A", "  ", "span B"]},
                {"question": "Empty one?", "evidence": []},
                {"question": "Another Q", "evidence": ["only span"]},
            ]
        ),
        encoding="utf-8",
    )
    ev = load_evidence(sample)
    # keyed by normalised question text (collapsed whitespace + lowercase)
    assert ev["what is x?"] == ["span A", "span B"]  # blank dropped
    assert "empty one?" not in ev  # no usable evidence -> absent
    assert ev["another q"] == ["only span"]


def test_load_evidence_from_raw_extracts_first_annotator_spans(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "qasper-test-v0.3.json").write_text(
        json.dumps(
            {
                "paperX": {
                    "qas": [
                        {
                            "question": "What  is Y?",
                            "answers": [
                                {"answer": {"evidence": ["para 1", " ", "para 2"]}},
                                {"answer": {"evidence": ["second annotator"]}},
                            ],
                        },
                        {
                            "question": "No evidence?",
                            "answers": [{"answer": {"evidence": []}}],
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    ev = load_evidence_from_raw(raw)
    # first annotator only, normalised key, blanks dropped
    assert ev["what is y?"] == ["para 1", "para 2"]
    assert "no evidence?" not in ev
