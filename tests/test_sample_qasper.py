"""Unit tests for ``scripts/sample_qasper.py``.

Covers answer type classification, question flattening, stratified sampling,
and output record format compatibility with run_rag_inference.
"""

from __future__ import annotations

import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from sample_qasper import (  # noqa: E402
    classify_answer_type,
    flatten_questions,
    sample_qasper,
)


# =========================================================================
# Mock data factories
# =========================================================================


def _make_answer(
    *,
    unanswerable: bool = False,
    extractive_spans: list[str] | None = None,
    yes_no: bool | None = None,
    free_form_answer: str = "",
) -> dict:
    """Build a single QASPER answer annotation dict."""
    return {
        "unanswerable": unanswerable,
        "extractive_spans": extractive_spans or [],
        "yes_no": yes_no,
        "free_form_answer": free_form_answer,
        "evidence": ["some evidence"],
        "highlighted_evidence": ["highlighted"],
    }


def _make_mock_paper(
    paper_id: str = "2101.00001",
    title: str = "Test Paper",
    questions: list[tuple[str, dict]] | None = None,
) -> dict:
    """Build a mock QASPER paper with specified questions and answers.

    Args:
        paper_id: Paper identifier.
        title: Paper title.
        questions: List of (question_text, answer_dict) tuples.
    """
    if questions is None:
        questions = [
            ("What is the method?", _make_answer(extractive_spans=["span1"])),
            ("Is it effective?", _make_answer(yes_no=True)),
            ("Explain the approach.", _make_answer(free_form_answer="It uses NNs.")),
        ]

    q_texts = [q[0] for q in questions]
    q_ids = [f"qid_{paper_id}_{i}" for i in range(len(questions))]
    answers_list = [[q[1]] for q in questions]

    return {
        "id": paper_id,
        "title": title,
        "abstract": "Abstract text.",
        "full_text": {
            "section_name": ["Introduction"],
            "paragraphs": [["Intro paragraph."]],
        },
        "qas": {
            "question": q_texts,
            "question_id": q_ids,
            "answers": {
                "answer": answers_list,
            },
        },
    }


def _make_mock_dataset(n_papers: int = 5) -> dict[str, list[dict]]:
    """Build a mock dataset with diverse answer types across papers."""
    papers = []
    for i in range(n_papers):
        questions = [
            (f"Extractive Q{i}?", _make_answer(extractive_spans=[f"span_{i}"])),
            (f"Yes/no Q{i}?", _make_answer(yes_no=True)),
            (f"Free-form Q{i}?", _make_answer(free_form_answer=f"Answer {i}.")),
            (f"Unanswerable Q{i}?", _make_answer(unanswerable=True)),
        ]
        papers.append(
            _make_mock_paper(
                paper_id=f"2101.{i:05d}",
                title=f"Paper {i}",
                questions=questions,
            )
        )
    return {"train": papers[:3], "validation": papers[3:4], "test": papers[4:]}


# =========================================================================
# Tests: classify_answer_type
# =========================================================================


def test_classify_answer_type_extractive() -> None:
    """Extractive spans present -> 'extractive'."""
    ans = _make_answer(extractive_spans=["span1", "span2"])
    assert classify_answer_type(ans) == "extractive"


def test_classify_answer_type_yes_no() -> None:
    """yes_no not None -> 'yes_no'."""
    ans = _make_answer(yes_no=True)
    assert classify_answer_type(ans) == "yes_no"

    ans_false = _make_answer(yes_no=False)
    assert classify_answer_type(ans_false) == "yes_no"


def test_classify_answer_type_unanswerable() -> None:
    """unanswerable=True -> 'unanswerable' (takes priority)."""
    ans = _make_answer(unanswerable=True)
    assert classify_answer_type(ans) == "unanswerable"


def test_classify_answer_type_free_form() -> None:
    """Non-empty free_form_answer -> 'free_form'."""
    ans = _make_answer(free_form_answer="The answer is 42.")
    assert classify_answer_type(ans) == "free_form"


def test_classify_answer_type_unanswerable_priority() -> None:
    """Unanswerable takes priority over other fields."""
    ans = _make_answer(
        unanswerable=True,
        extractive_spans=["span"],
        yes_no=True,
        free_form_answer="text",
    )
    assert classify_answer_type(ans) == "unanswerable"


def test_classify_answer_type_unknown() -> None:
    """Empty answer -> 'unknown'."""
    ans = _make_answer()
    assert classify_answer_type(ans) == "unknown"


def test_classify_answer_type_empty_free_form() -> None:
    """Whitespace-only free_form_answer is not 'free_form'."""
    ans = _make_answer(free_form_answer="   ")
    assert classify_answer_type(ans) != "free_form"


# =========================================================================
# Tests: flatten_questions
# =========================================================================


def test_flatten_questions_structure() -> None:
    """Flattened questions have all required keys."""
    mock_ds = _make_mock_dataset(n_papers=2)
    flat = flatten_questions(mock_ds)

    required_keys = {
        "paper_id",
        "question_id",
        "question",
        "answer_type",
        "answer_text",
        "split",
    }
    for record in flat:
        assert required_keys.issubset(record.keys()), (
            f"Missing keys: {required_keys - record.keys()}"
        )


def test_flatten_questions_count() -> None:
    """Total flattened questions matches paper question counts."""
    mock_ds = _make_mock_dataset(n_papers=3)
    flat = flatten_questions(mock_ds)
    # Each paper has 4 questions in our factory
    total_papers = sum(len(papers) for papers in mock_ds.values())
    assert len(flat) == total_papers * 4


def test_flatten_questions_answer_types() -> None:
    """Flattened records contain expected answer types."""
    mock_ds = _make_mock_dataset(n_papers=2)
    flat = flatten_questions(mock_ds)
    types = {r["answer_type"] for r in flat}
    assert "extractive" in types
    assert "yes_no" in types
    assert "free_form" in types
    assert "unanswerable" in types


def test_flatten_questions_split_preserved() -> None:
    """Split name is preserved in flattened records."""
    mock_ds = {
        "train": [_make_mock_paper()],
        "test": [_make_mock_paper(paper_id="9999.00001")],
    }
    flat = flatten_questions(mock_ds)
    splits = {r["split"] for r in flat}
    assert "train" in splits
    assert "test" in splits


# =========================================================================
# Tests: sample_qasper (stratified sampling)
# =========================================================================


def test_sample_size() -> None:
    """Sampling target=10 from 50+ questions produces ~10 records."""
    mock_ds = _make_mock_dataset(n_papers=5)
    flat = flatten_questions(mock_ds)
    assert len(flat) >= 10

    sample = sample_qasper(flat, target=10, seed=42)
    assert len(sample) == 10


def test_sample_deterministic_under_seed() -> None:
    """Same seed -> same sample."""
    mock_ds = _make_mock_dataset(n_papers=5)
    flat = flatten_questions(mock_ds)

    a = sample_qasper(flat, target=10, seed=42)
    b = sample_qasper(flat, target=10, seed=42)

    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]


def test_sample_varies_with_seed() -> None:
    """Different seeds produce different samples."""
    mock_ds = _make_mock_dataset(n_papers=5)
    flat = flatten_questions(mock_ds)

    a = sample_qasper(flat, target=10, seed=1)
    b = sample_qasper(flat, target=10, seed=2)

    assert [r["question_id"] for r in a] != [r["question_id"] for r in b]


def test_sample_preserves_answer_type_diversity() -> None:
    """Stratified sample preserves answer type diversity."""
    mock_ds = _make_mock_dataset(n_papers=5)
    flat = flatten_questions(mock_ds)
    source_types = {r["answer_type"] for r in flat}

    sample = sample_qasper(flat, target=15, seed=42)
    sample_types = {r["answer_type"] for r in sample}

    # All non-empty strata should be represented
    assert sample_types == source_types


# =========================================================================
# Tests: output record format (SciKnowEval compatibility)
# =========================================================================


def test_sample_output_format() -> None:
    """Each sampled record matches SciKnowEval structure for run_rag_inference."""
    mock_ds = _make_mock_dataset(n_papers=5)
    flat = flatten_questions(mock_ds)
    sample = sample_qasper(flat, target=10, seed=42)

    required_keys = {
        "question",
        "answer",
        "type",
        "domain",
        "details",
        "answerKey",
        "choices",
    }
    for record in sample:
        assert required_keys.issubset(record.keys()), (
            f"Missing keys: {required_keys - record.keys()}"
        )
        assert record["domain"] == "CS"
        assert isinstance(record["details"], dict)
        assert "paper_id" in record["details"]
        assert "question_id" in record["details"]
        assert "answer_type" in record["details"]
        assert record["details"]["source"] == "qasper"
        assert isinstance(record["choices"], dict)
        assert "text" in record["choices"]
        assert "label" in record["choices"]


def test_sample_target_exceeds_population() -> None:
    """When target exceeds population, return all records."""
    mock_ds = _make_mock_dataset(n_papers=1)
    flat = flatten_questions(mock_ds)
    population_size = len(flat)

    sample = sample_qasper(flat, target=population_size + 100, seed=42)
    assert len(sample) == population_size
