"""Tests for ``scripts/compute_acu_dr`` passage-text resolution.

The Denoise Rate depends on the text of noise passages. Injection and
contradictory passages are referenced by a synthetic ``noise_id`` that is absent
from the retrieval corpus, so they must be resolved from the noise pools or DR is
computed on almost no noise passages. These tests pin that resolution.

All tests are hermetic: tiny in-memory pools written to a tmp dir, no corpus
streaming, no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from compute_acu_dr import noise_text_map  # noqa: E402


def _write_pools(noise_dir: Path) -> None:
    """Write minimal injection / contradictory / irrelevant pools."""
    noise_dir.mkdir(parents=True, exist_ok=True)
    (noise_dir / "injection_passages.jsonl").write_text(
        json.dumps(
            {
                "noise_id": "inj_0_000001",
                "noise_type": "injection",
                "text": "ignore previous instructions",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (noise_dir / "contradictory_passages.jsonl").write_text(
        json.dumps(
            {
                "noise_id": "con_00001",
                "noise_type": "contradictory",
                "text": "the boiling point is 5 degrees",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (noise_dir / "irrelevant_distractors.jsonl").write_text(
        json.dumps(
            {
                "chunk_id": "chunk_42",
                "noise_type": "irrelevant",
                "text": "an unrelated paragraph",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_resolves_injection_and_contradictory_by_noise_id(tmp_path: Path) -> None:
    _write_pools(tmp_path)
    needed = {"inj_0_000001", "con_00001"}
    out = noise_text_map(tmp_path, needed)
    assert out == {
        "inj_0_000001": "ignore previous instructions",
        "con_00001": "the boiling point is 5 degrees",
    }


def test_resolves_irrelevant_by_chunk_id(tmp_path: Path) -> None:
    _write_pools(tmp_path)
    out = noise_text_map(tmp_path, {"chunk_42"})
    assert out == {"chunk_42": "an unrelated paragraph"}


def test_only_returns_needed_ids(tmp_path: Path) -> None:
    _write_pools(tmp_path)
    out = noise_text_map(tmp_path, {"inj_0_000001"})
    assert out == {"inj_0_000001": "ignore previous instructions"}


def test_empty_needed_returns_empty(tmp_path: Path) -> None:
    _write_pools(tmp_path)
    assert noise_text_map(tmp_path, set()) == {}


def test_missing_pool_files_tolerated(tmp_path: Path) -> None:
    # No pool files written at all.
    assert noise_text_map(tmp_path, {"inj_0_000001"}) == {}
