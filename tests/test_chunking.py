"""Tests for scripts/chunking.py - shared chunking and domain utilities."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from io import StringIO

import pytest
import tiktoken

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from chunking import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    chunk_text,
    classify_domain,
    generate_chunk_id,
    init_tokenizer,
    load_domain_keywords,
    write_chunks_jsonl,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tokenizer() -> tiktoken.Encoding:
    return init_tokenizer()


@pytest.fixture
def long_text() -> str:
    """Text guaranteed to produce multiple chunks (~800 tokens)."""
    return " ".join(["The quick brown fox jumps over the lazy dog."] * 100)


@pytest.fixture
def short_text() -> str:
    """Text that fits in a single chunk."""
    return "Proteins are large biomolecules consisting of amino acid chains."


# ---------------------------------------------------------------------------
# init_tokenizer
# ---------------------------------------------------------------------------

def test_init_tokenizer() -> None:
    tok = init_tokenizer()
    assert tok.name == "cl100k_base"


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_basic(tokenizer: tiktoken.Encoding, long_text: str) -> None:
    chunks = chunk_text(long_text, "test", "doc1", tokenizer)
    assert len(chunks) > 1
    for c in chunks:
        assert set(c.keys()) == {"chunk_id", "text", "source", "source_id"}
        assert c["source"] == "test"
        assert c["source_id"] == "doc1"
        assert len(c["chunk_id"]) == 16


def test_chunk_text_token_count(tokenizer: tiktoken.Encoding, long_text: str) -> None:
    """All chunks except possibly the last should be close to CHUNK_SIZE tokens.

    Note: decode/re-encode may shift token boundaries slightly, so we allow
    a small tolerance (analyst's original behavior).
    """
    chunks = chunk_text(long_text, "test", "doc1", tokenizer)
    for c in chunks[:-1]:
        tokens = tokenizer.encode(c["text"])
        assert abs(len(tokens) - CHUNK_SIZE) <= 30, (
            f"Token count {len(tokens)} too far from {CHUNK_SIZE}"
        )


def test_chunk_text_short(tokenizer: tiktoken.Encoding, short_text: str) -> None:
    chunks = chunk_text(short_text, "test", "doc1", tokenizer)
    assert len(chunks) == 1


def test_chunk_text_empty(tokenizer: tiktoken.Encoding) -> None:
    assert chunk_text("", "test", "doc1", tokenizer) == []
    assert chunk_text("   ", "test", "doc1", tokenizer) == []


def test_chunk_text_overlap(tokenizer: tiktoken.Encoding, long_text: str) -> None:
    """Consecutive chunks should share CHUNK_OVERLAP tokens."""
    chunks = chunk_text(long_text, "test", "doc1", tokenizer)
    if len(chunks) < 2:
        pytest.skip("Not enough chunks for overlap test")

    tokens_0 = tokenizer.encode(chunks[0]["text"])
    tokens_1 = tokenizer.encode(chunks[1]["text"])

    tail_of_first = tokens_0[-CHUNK_OVERLAP:]
    head_of_second = tokens_1[:CHUNK_OVERLAP]
    assert tail_of_first == head_of_second


# ---------------------------------------------------------------------------
# generate_chunk_id
# ---------------------------------------------------------------------------

def test_chunk_id_deterministic() -> None:
    id1 = generate_chunk_id("wiki", "Biology", 0)
    id2 = generate_chunk_id("wiki", "Biology", 0)
    assert id1 == id2


def test_chunk_id_uniqueness() -> None:
    id1 = generate_chunk_id("wiki", "Biology", 0)
    id2 = generate_chunk_id("wiki", "Chemistry", 0)
    assert id1 != id2


def test_chunk_id_different_index() -> None:
    id1 = generate_chunk_id("wiki", "Biology", 0)
    id2 = generate_chunk_id("wiki", "Biology", 1)
    assert id1 != id2


def test_chunk_id_length() -> None:
    cid = generate_chunk_id("pubmed", "12345", 42)
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


# ---------------------------------------------------------------------------
# classify_domain
# ---------------------------------------------------------------------------

def test_classify_domain_biology() -> None:
    assert classify_domain("DNA replication in eukaryotic cells", title="Gene Expression") == "biology"


def test_classify_domain_chemistry() -> None:
    assert classify_domain("Oxidation reactions of organic compounds", title="Chemical Bonding") == "chemistry"


def test_classify_domain_physics() -> None:
    assert classify_domain("Quantum mechanics and wave functions", title="Quantum Physics") == "physics"


def test_classify_domain_earth_science() -> None:
    assert classify_domain("Plate tectonics and earthquake zones", title="Geology of Earth") == "earth_science"


def test_classify_domain_materials_science() -> None:
    assert classify_domain("Graphene nanostructures for composite material applications", title="Nanotechnology") == "materials_science"


def test_classify_domain_unknown() -> None:
    result = classify_domain("The history of ancient Rome and its emperors", title="Roman History")
    assert result == "general_science"


def test_classify_domain_title_priority() -> None:
    """Title match should take priority over text body."""
    result = classify_domain(
        "This text mentions biology and cells many times biology biology",
        title="Quantum Physics Introduction",
    )
    assert result == "physics"


# ---------------------------------------------------------------------------
# load_domain_keywords
# ---------------------------------------------------------------------------

def test_load_domain_keywords() -> None:
    keywords = load_domain_keywords()
    assert "biology" in keywords
    assert "chemistry" in keywords
    assert "physics" in keywords
    assert "materials_science" in keywords
    assert "earth_science" in keywords
    assert len(keywords) == 5
    for domain, kw_list in keywords.items():
        assert len(kw_list) > 5, f"{domain} has too few keywords"


# ---------------------------------------------------------------------------
# write_chunks_jsonl
# ---------------------------------------------------------------------------

def test_write_chunks_jsonl() -> None:
    chunks = [
        {"chunk_id": "abc123", "text": "hello world", "source": "test", "source_id": "1", "domain": "biology"},
        {"chunk_id": "def456", "text": "foo bar", "source": "test", "source_id": "2", "domain": "physics"},
    ]
    buf = StringIO()
    count = write_chunks_jsonl(chunks, buf)
    assert count == 2

    buf.seek(0)
    lines = buf.readlines()
    assert len(lines) == 2

    parsed = json.loads(lines[0])
    assert parsed["chunk_id"] == "abc123"
    assert parsed["text"] == "hello world"
