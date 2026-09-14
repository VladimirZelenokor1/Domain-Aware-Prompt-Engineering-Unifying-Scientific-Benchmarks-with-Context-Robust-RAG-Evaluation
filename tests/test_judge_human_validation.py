"""Tests for ``scripts/judge_human_validation`` (RQ3 preliminary judge-human kappa).

Pure helpers for the single-annotator judge-vs-human rubric validation: blind
sampling, quadratic-weighted Cohen's kappa, and its bootstrap CI. Hermetic:
numpy only, no data files, no models.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from judge_human_validation import (  # noqa: E402
    bootstrap_kappa_ci,
    quadratic_weighted_kappa,
    sample_items,
)


# --- quadratic_weighted_kappa ------------------------------------------------


def test_qwk_perfect_agreement_is_one() -> None:
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]) == 1.0


def test_qwk_near_agreement_high() -> None:
    # off by one on a couple of items -> high but < 1
    k = quadratic_weighted_kappa([5, 4, 3, 2, 1, 0], [5, 4, 2, 2, 1, 0])
    assert 0.8 < k < 1.0


def test_qwk_constant_rater_is_zero() -> None:
    # one rater never varies -> expected == observed disagreement, kappa 0
    assert quadratic_weighted_kappa([3, 3, 3, 3], [0, 1, 2, 3]) == 0.0


def test_qwk_systematic_disagreement_negative() -> None:
    # inverse ratings -> strong negative agreement
    assert quadratic_weighted_kappa([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]) < 0.0


def test_qwk_symmetric() -> None:
    a, b = [0, 2, 4, 1, 3], [1, 2, 3, 1, 5]
    assert quadratic_weighted_kappa(a, b) == quadratic_weighted_kappa(b, a)


# --- bootstrap_kappa_ci ------------------------------------------------------


def test_bootstrap_ci_brackets_and_ordered() -> None:
    human = [5, 4, 3, 2, 1, 0, 5, 4, 3, 2]
    judge = [5, 4, 2, 2, 1, 1, 4, 4, 3, 1]
    lo, hi = bootstrap_kappa_ci(human, judge, n_boot=200, seed=42)
    assert lo <= hi
    assert -1.0 <= lo <= 1.0 and -1.0 <= hi <= 1.0


def test_bootstrap_ci_deterministic_with_seed() -> None:
    human = [5, 4, 3, 2, 1, 0]
    judge = [5, 4, 2, 2, 1, 0]
    assert bootstrap_kappa_ci(human, judge, n_boot=100, seed=7) == bootstrap_kappa_ci(
        human, judge, n_boot=100, seed=7
    )


# --- sample_items ------------------------------------------------------------


def test_sample_items_deterministic_and_capped() -> None:
    items = [{"id": i} for i in range(100)]
    s1 = sample_items(items, 10, seed=42)
    s2 = sample_items(items, 10, seed=42)
    assert s1 == s2  # deterministic
    assert len(s1) == 10
    assert {d["id"] for d in s1} <= {d["id"] for d in items}


def test_sample_items_n_larger_than_pool_returns_all_shuffled() -> None:
    items = [{"id": i} for i in range(5)]
    s = sample_items(items, 50, seed=1)
    assert len(s) == 5
    assert {d["id"] for d in s} == {0, 1, 2, 3, 4}
