"""Phase 6 noise-gate tests.

Requirements (approved 2026-04-21):

  (a) Level 0.0 MUST start successfully when contradictory review is
      absent or below threshold.
  (b) Level 0.2 MUST refuse to start in the same condition with
      SystemExit(2).

These tests are written BEFORE any Phase 6 production run so the gate's
policy is fixed in code, not in reviewer memory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from rag_gate import (  # noqa: E402
    RISKY_COMPOSITION_KEYS,
    assert_noise_gate,
    level_requires_contradictory,
)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def noise_config() -> dict:
    """Replica of the ``noise`` section of configs/noise.yaml."""
    return {
        "seed": 42,
        "top_k": 10,
        "levels": {
            "0.0": {"replaced": 0, "composition": {}},
            "0.2": {
                "replaced": 2,
                "composition": {"irrelevant": 1, "contradictory_or_injection": 1},
            },
            "0.4": {
                "replaced": 4,
                "composition": {
                    "irrelevant": 1,
                    "contradictory": 1,
                    "injection": 1,
                    "any": 1,
                },
            },
            "0.6": {
                "replaced": 6,
                "composition": {"irrelevant": 2, "contradictory": 2, "injection": 2},
            },
        },
        "pools": {
            "irrelevant": "corpus/noise/irrelevant_distractors.jsonl",
            "injection": "corpus/noise/injection_passages.jsonl",
            "contradictory": "corpus/noise/contradictory_passages.jsonl",
        },
        "contradictory": {
            "min_acceptance_rate": 0.80,
            "review_stats_path": "outputs/noise_review/contradictory_review_stats.json",
        },
    }


@pytest.fixture
def project_tree(tmp_path: Path) -> Path:
    """Create a fake project root with the always-required pools present."""
    for name in ("irrelevant_distractors", "injection_passages"):
        p = tmp_path / "corpus" / "noise" / f"{name}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}\n", encoding="utf-8")
    return tmp_path


def _write_contradictory_pool(root: Path) -> Path:
    path = root / "corpus" / "noise" / "contradictory_passages.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def _write_review_stats(root: Path, acceptance_rate: float) -> Path:
    path = root / "outputs" / "noise_review" / "contradictory_review_stats.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"acceptance_rate": acceptance_rate}), encoding="utf-8")
    return path


# =========================================================================
# level_requires_contradictory helper
# =========================================================================


def test_level_0_does_not_require_contradictory(noise_config: dict) -> None:
    assert level_requires_contradictory(0.0, noise_config) is False


def test_level_02_requires_contradictory_via_one_of_key(noise_config: dict) -> None:
    # "contradictory_or_injection" is a risky key.
    assert level_requires_contradictory(0.2, noise_config) is True


def test_level_04_requires_contradictory_direct(noise_config: dict) -> None:
    assert level_requires_contradictory(0.4, noise_config) is True


def test_level_06_requires_contradictory_direct(noise_config: dict) -> None:
    assert level_requires_contradictory(0.6, noise_config) is True


def test_risky_keys_match_assembler_semantics() -> None:
    """Sanity: RISKY_COMPOSITION_KEYS is exactly the set we document."""
    assert RISKY_COMPOSITION_KEYS == {
        "contradictory",
        "contradictory_or_injection",
        "any",
    }


# =========================================================================
# (a) level 0.0 passes without contradictory review
# =========================================================================


def test_level_0_passes_when_review_stats_absent(
    noise_config: dict,
    project_tree: Path,
) -> None:
    # No review stats file on disk - level 0.0 must still pass.
    assert_noise_gate(0.0, noise_config, project_root=project_tree)


def test_level_0_passes_when_acceptance_rate_below_threshold(
    noise_config: dict,
    project_tree: Path,
) -> None:
    # Contradictory pool present AND below-threshold stats present.
    # Level 0.0 still ignores contradictory entirely.
    _write_contradictory_pool(project_tree)
    _write_review_stats(project_tree, acceptance_rate=0.50)
    assert_noise_gate(0.0, noise_config, project_root=project_tree)


def test_level_0_fails_if_irrelevant_pool_missing(
    noise_config: dict,
    tmp_path: Path,
) -> None:
    # Injection present, irrelevant NOT present.
    (tmp_path / "corpus" / "noise" / "injection_passages.jsonl").parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    (tmp_path / "corpus" / "noise" / "injection_passages.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.0, noise_config, project_root=tmp_path)
    assert exc.value.code == 2


# =========================================================================
# (b) level 0.2 refuses when contradictory review is absent or below
# =========================================================================


def test_level_02_exits_2_when_review_stats_absent(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    # NO review stats file.
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.2, noise_config, project_root=project_tree)
    assert exc.value.code == 2


def test_level_02_exits_2_when_acceptance_rate_below_threshold(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    _write_review_stats(project_tree, acceptance_rate=0.50)
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.2, noise_config, project_root=project_tree)
    assert exc.value.code == 2


def test_level_02_exits_2_when_contradictory_pool_missing(
    noise_config: dict,
    project_tree: Path,
) -> None:
    # Stats file present with passing rate, but pool file absent.
    _write_review_stats(project_tree, acceptance_rate=0.95)
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.2, noise_config, project_root=project_tree)
    assert exc.value.code == 2


def test_level_02_passes_when_stats_at_threshold(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    _write_review_stats(project_tree, acceptance_rate=0.80)
    assert_noise_gate(0.2, noise_config, project_root=project_tree)


def test_level_02_passes_when_stats_above_threshold(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    _write_review_stats(project_tree, acceptance_rate=0.94)
    assert_noise_gate(0.2, noise_config, project_root=project_tree)


# =========================================================================
# error paths
# =========================================================================


def test_malformed_stats_json_exits_2(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    stats_path = (
        project_tree / "outputs" / "noise_review" / "contradictory_review_stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.2, noise_config, project_root=project_tree)
    assert exc.value.code == 2


def test_missing_acceptance_rate_field_exits_2(
    noise_config: dict,
    project_tree: Path,
) -> None:
    _write_contradictory_pool(project_tree)
    stats_path = (
        project_tree / "outputs" / "noise_review" / "contradictory_review_stats.json"
    )
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({"sampled": 50}), encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        assert_noise_gate(0.2, noise_config, project_root=project_tree)
    assert exc.value.code == 2


def test_custom_review_stats_path_override(
    noise_config: dict,
    project_tree: Path,
    tmp_path: Path,
) -> None:
    """Caller-supplied review_stats_path must take precedence."""
    _write_contradictory_pool(project_tree)
    custom = tmp_path / "alt_stats.json"
    custom.write_text(json.dumps({"acceptance_rate": 0.95}), encoding="utf-8")
    # Default location absent; custom location present and passing.
    assert_noise_gate(
        0.2,
        noise_config,
        project_root=project_tree,
        review_stats_path=custom,
    )
