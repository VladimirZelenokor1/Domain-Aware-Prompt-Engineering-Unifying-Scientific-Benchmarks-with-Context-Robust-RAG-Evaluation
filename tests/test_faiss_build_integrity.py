"""Integrity tests for the FAISS index builder's post-build assertion.

Validates ``scripts/build_indices.assert_faiss_integrity`` against one
positive case and several negative cases covering:

* Missing chunk in the FAISS index (ntotal < id_map).
* Unknown duplicate chunk_id in id_map (not on the collision allowlist).
* id_map mismatched with the corpus unique-cid count.
* Known corpus-hash collision passing through unharmed.

Uses a plain stub for the ``faiss.Index`` interface (only ``ntotal`` is
read) so tests stay hermetic - no FAISS install, no GPU, no corpus IO.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_indices import (  # noqa: E402
    KNOWN_CORPUS_CID_COLLISIONS,
    assert_faiss_integrity,
    count_unique_corpus_cids,
)


class _IndexStub:
    """Minimal stand-in for ``faiss.Index`` (only ``ntotal`` is read)."""

    def __init__(self, ntotal: int) -> None:
        self.ntotal = ntotal


# ---------------------------------------------------------------- positive


def test_aligned_index_passes() -> None:
    id_map = ["a", "b", "c", "d"]
    index = _IndexStub(ntotal=4)
    # Must not raise.
    assert_faiss_integrity(index, id_map, corpus_unique_cids=4)


def test_known_collision_is_tolerated() -> None:
    """A duplicate on the known-collisions allowlist must not raise."""
    collision = next(iter(KNOWN_CORPUS_CID_COLLISIONS))
    id_map = ["a", "b", collision, collision, "c"]
    index = _IndexStub(ntotal=5)
    # Corpus has 4 unique cids: a, b, collision, c. id_map has 5 slots,
    # 4 unique. The one duplicate is on the allowlist -> no raise.
    assert_faiss_integrity(index, id_map, corpus_unique_cids=4)


# ---------------------------------------------------------------- negative


def test_missing_chunk_triggers_assertion() -> None:
    """Corpus has a cid that is absent from id_map -> unique mismatch."""
    # Corpus unique cids = 5, id_map only knows 4 of them.
    id_map = ["a", "b", "c", "d"]
    index = _IndexStub(ntotal=4)
    with pytest.raises(AssertionError) as exc:
        assert_faiss_integrity(index, id_map, corpus_unique_cids=5)
    msg = str(exc.value)
    assert "id_map unique chunk_ids" in msg
    assert "corpus unique" in msg
    assert "missing" in msg


def test_ntotal_vs_id_map_mismatch_triggers_assertion() -> None:
    """FAISS dropped a vector but id_map still has the cid -> n_total mismatch."""
    id_map = ["a", "b", "c", "d"]
    index = _IndexStub(ntotal=3)  # one fewer vector than id_map entries
    with pytest.raises(AssertionError) as exc:
        assert_faiss_integrity(index, id_map, corpus_unique_cids=4)
    msg = str(exc.value)
    assert "FAISS ntotal" in msg
    assert "id_map length" in msg


def test_unknown_duplicate_triggers_assertion() -> None:
    """Duplicate cid not on the allowlist -> unknown-duplicate error."""
    id_map = ["a", "b", "c", "b"]  # "b" duplicated, not in allowlist
    index = _IndexStub(ntotal=4)
    # Corpus has 3 unique cids matching the 3 unique id_map cids.
    with pytest.raises(AssertionError) as exc:
        assert_faiss_integrity(index, id_map, corpus_unique_cids=3)
    msg = str(exc.value)
    assert "known-collisions allowlist" in msg
    assert "b(x2)" in msg


def test_multiple_failures_reported_together() -> None:
    """One call must enumerate every broken invariant, not just the first."""
    id_map = ["a", "b", "c", "c"]  # ntotal=3 but len=4; c is unknown dup
    index = _IndexStub(ntotal=3)
    with pytest.raises(AssertionError) as exc:
        assert_faiss_integrity(index, id_map, corpus_unique_cids=5)
    msg = str(exc.value)
    # All three invariants should fire.
    assert "FAISS ntotal" in msg
    assert "id_map unique chunk_ids" in msg
    assert "known-collisions allowlist" in msg
    # Bullet-format with one failure per line.
    assert msg.count("  - ") >= 3


# ---------------------------------------------------------------- helpers


def test_count_unique_corpus_cids(tmp_path: Path) -> None:
    """Streaming unique-count helper must dedupe correctly."""
    corpus = tmp_path / "toy.jsonl"
    records = [
        {"chunk_id": "a", "text": "x"},
        {"chunk_id": "b", "text": "y"},
        {"chunk_id": "a", "text": "x"},  # exact dup line
        {"chunk_id": "c", "text": "z"},
    ]
    corpus.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n",
        encoding="utf-8",
    )
    assert count_unique_corpus_cids(corpus) == 3


def test_count_unique_corpus_cids_ignores_blank_lines(tmp_path: Path) -> None:
    corpus = tmp_path / "toy.jsonl"
    corpus.write_text(
        '{"chunk_id": "a"}\n\n  \n{"chunk_id": "b"}\n',
        encoding="utf-8",
    )
    assert count_unique_corpus_cids(corpus) == 2
