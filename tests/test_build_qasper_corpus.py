"""Unit tests for ``scripts/build_qasper_corpus.py``.

Covers corpus building from mock QASPER data:
* chunk generation with correct keys and metadata
* papers manifest structure
* corpus JSONL format validation
* abstract-only and section-based chunking
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_qasper_corpus import build_qasper_corpus  # noqa: E402


# =========================================================================
# Mock QASPER paper factory
# =========================================================================


def _make_mock_paper(
    paper_id: str = "2101.00001",
    title: str = "Test Paper Title",
    abstract: str = "This is the abstract of a test paper about machine learning.",
    sections: list[tuple[str, list[str]]] | None = None,
    n_questions: int = 3,
) -> dict:
    """Build a mock QASPER paper record matching HuggingFace dataset structure."""
    if sections is None:
        sections = [
            ("Introduction", ["This is the introduction paragraph."]),
            ("Methods", ["We propose a novel method.", "Details of implementation."]),
        ]

    section_names = [s[0] for s in sections]
    paragraphs = [s[1] for s in sections]

    questions = [f"Question {i}?" for i in range(n_questions)]
    question_ids = [f"qid_{paper_id}_{i}" for i in range(n_questions)]

    # Build answer structure matching QASPER HF format
    answers_list: list[list[dict]] = []
    for _ in range(n_questions):
        answers_list.append(
            [
                {
                    "unanswerable": False,
                    "extractive_spans": ["some span"],
                    "yes_no": None,
                    "free_form_answer": "",
                    "evidence": ["evidence text"],
                    "highlighted_evidence": ["highlighted"],
                }
            ]
        )

    return {
        "id": paper_id,
        "title": title,
        "abstract": abstract,
        "full_text": {
            "section_name": section_names,
            "paragraphs": paragraphs,
        },
        "qas": {
            "question": questions,
            "question_id": question_ids,
            "answers": {
                "answer": answers_list,
            },
        },
    }


def _make_mock_dataset(
    papers_per_split: dict[str, int] | None = None,
) -> dict[str, list[dict]]:
    """Build a mock dataset dict keyed by split name."""
    if papers_per_split is None:
        papers_per_split = {"train": 2, "validation": 1, "test": 1}

    ds: dict[str, list[dict]] = {}
    paper_counter = 0
    for split_name, count in papers_per_split.items():
        papers = []
        for i in range(count):
            paper_counter += 1
            papers.append(
                _make_mock_paper(
                    paper_id=f"2101.{paper_counter:05d}",
                    title=f"Paper {paper_counter} in {split_name}",
                    n_questions=2 + i,
                )
            )
        ds[split_name] = papers
    return ds


# =========================================================================
# Tests: chunk generation
# =========================================================================


def test_chunk_single_paper(tmp_path: Path) -> None:
    """Chunking a single paper produces chunks with correct keys."""
    mock_ds = _make_mock_dataset({"train": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        stats = build_qasper_corpus(tmp_path)

    corpus_path = tmp_path / "corpus.jsonl"
    assert corpus_path.exists()

    chunks = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))

    assert len(chunks) > 0

    required_keys = {"chunk_id", "text", "source", "source_id"}
    for chunk in chunks:
        assert required_keys.issubset(chunk.keys()), (
            f"Missing keys: {required_keys - chunk.keys()}"
        )
        assert chunk["source"] == "qasper"
        assert len(chunk["chunk_id"]) == 16
        assert len(chunk["text"]) > 0


def test_chunk_includes_paper_id_and_section(tmp_path: Path) -> None:
    """Section-based chunks carry paper_id and section metadata."""
    mock_ds = _make_mock_dataset({"train": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        build_qasper_corpus(tmp_path)

    corpus_path = tmp_path / "corpus.jsonl"
    section_chunks = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            chunk = json.loads(line)
            if "paper_id" in chunk:
                section_chunks.append(chunk)

    assert len(section_chunks) > 0
    for chunk in section_chunks:
        assert "paper_id" in chunk
        assert "section" in chunk
        assert isinstance(chunk["section"], str)


def test_chunk_abstract_produces_chunks(tmp_path: Path) -> None:
    """Abstract text is chunked (source_id contains '_abstract')."""
    mock_ds = _make_mock_dataset({"train": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        build_qasper_corpus(tmp_path)

    corpus_path = tmp_path / "corpus.jsonl"
    abstract_chunks = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            chunk = json.loads(line)
            if "_abstract" in chunk.get("source_id", ""):
                abstract_chunks.append(chunk)

    assert len(abstract_chunks) > 0


def test_empty_paragraph_skipped(tmp_path: Path) -> None:
    """Empty paragraphs do not produce chunks."""
    paper = _make_mock_paper(
        sections=[
            ("Intro", ["Real content here."]),
            ("Empty", ["", "  ", ""]),
        ],
    )
    mock_ds = {"train": [paper]}

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        stats = build_qasper_corpus(tmp_path)

    corpus_path = tmp_path / "corpus.jsonl"
    chunks = []
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            chunks.append(json.loads(line))

    # Should only have abstract chunk(s) + "Intro" chunk(s), no "Empty" chunks
    source_ids = [c["source_id"] for c in chunks]
    empty_section_ids = [sid for sid in source_ids if "_s1_" in sid]
    assert len(empty_section_ids) == 0


# =========================================================================
# Tests: papers manifest
# =========================================================================


def test_papers_manifest_structure(tmp_path: Path) -> None:
    """papers.json has required fields for each paper."""
    mock_ds = _make_mock_dataset({"train": 2, "validation": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        stats = build_qasper_corpus(tmp_path)

    papers_path = tmp_path / "papers.json"
    assert papers_path.exists()

    with open(papers_path, encoding="utf-8") as fh:
        papers = json.load(fh)

    assert len(papers) == 3  # 2 train + 1 validation

    required_keys = {"paper_id", "title", "split", "n_questions"}
    for paper in papers:
        assert required_keys.issubset(paper.keys())
        assert isinstance(paper["paper_id"], str)
        assert isinstance(paper["title"], str)
        assert paper["split"] in {"train", "validation", "test"}
        assert isinstance(paper["n_questions"], int)
        assert paper["n_questions"] > 0


def test_papers_manifest_split_assignment(tmp_path: Path) -> None:
    """Each paper's split matches the dataset split it came from."""
    mock_ds = _make_mock_dataset({"train": 1, "test": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        build_qasper_corpus(tmp_path)

    with open(tmp_path / "papers.json", encoding="utf-8") as fh:
        papers = json.load(fh)

    splits = {p["split"] for p in papers}
    assert "train" in splits
    assert "test" in splits


# =========================================================================
# Tests: corpus JSONL format
# =========================================================================


def test_corpus_jsonl_format(tmp_path: Path) -> None:
    """Each line in corpus.jsonl is valid JSON with required fields."""
    mock_ds = _make_mock_dataset({"train": 2})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        build_qasper_corpus(tmp_path)

    corpus_path = tmp_path / "corpus.jsonl"
    line_count = 0
    with open(corpus_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)  # must not raise
            assert "chunk_id" in record
            assert "text" in record
            assert record["source"] == "qasper"
            line_count += 1

    assert line_count > 0


# =========================================================================
# Tests: stats return value
# =========================================================================


def test_stats_return_value(tmp_path: Path) -> None:
    """build_qasper_corpus returns stats dict with paper_count and chunk_count."""
    mock_ds = _make_mock_dataset({"train": 2, "validation": 1, "test": 1})

    with patch("build_qasper_corpus.load_dataset", return_value=mock_ds):
        stats = build_qasper_corpus(tmp_path)

    assert "paper_count" in stats
    assert "chunk_count" in stats
    assert stats["paper_count"] == 4
    assert stats["chunk_count"] > 0
