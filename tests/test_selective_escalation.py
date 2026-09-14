"""Tests for ``scripts/analyze_selective_escalation`` (RQ3 escalation, role iii).

Pure helpers: point-biserial, the escalation rule, threshold tuning, and the
escalation gain (delta r_pb). Hermetic: numpy only, no judge data, no models.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_selective_escalation import (  # noqa: E402
    best_threshold,
    budget_curve,
    delta_rpb,
    escalate,
    point_biserial,
    split_indices,
    uncertainty_signal,
)


# --- point_biserial ----------------------------------------------------------


def test_point_biserial_perfect_separation() -> None:
    r = point_biserial([1.0, 2.0, 4.0, 5.0], [0, 0, 1, 1])
    assert r > 0.9


def test_point_biserial_constant_is_zero() -> None:
    assert point_biserial([3, 3, 3, 3], [0, 1, 0, 1]) == 0.0


# --- escalate ----------------------------------------------------------------


def test_escalate_replaces_low_confidence_with_judge_c() -> None:
    base = [2.0, 3.0, 4.0]
    c = [5.0, 5.0, 5.0]
    conf = [0.2, 0.9, 0.1]
    # threshold 0.5 -> items 0 and 2 (conf<0.5) escalate to c, item 1 keeps base
    assert escalate(base, c, conf, 0.5) == [5.0, 3.0, 5.0]


def test_escalate_skips_when_c_missing() -> None:
    base = [2.0, 3.0]
    c = [None, 5.0]
    conf = [0.1, 0.1]
    assert escalate(base, c, conf, 0.5) == [2.0, 5.0]


# --- delta_rpb ---------------------------------------------------------------


def test_delta_rpb_positive_when_c_improves_alignment() -> None:
    # base tracks gold poorly on the low-confidence items; c fixes them
    base = [3.0, 3.0, 3.0, 3.0]
    gold = [0, 0, 1, 1]
    c = [1.0, 1.0, 5.0, 5.0]  # c perfectly separates
    conf = [0.1, 0.1, 0.1, 0.1]  # all low -> all escalate
    d = delta_rpb(base, c, conf, gold, 0.5)
    assert d > 0.0


# --- best_threshold ----------------------------------------------------------


def test_best_threshold_picks_maximising_value() -> None:
    base = [3.0, 3.0, 3.0, 3.0]
    gold = [0, 0, 1, 1]
    c = [1.0, 1.0, 5.0, 5.0]
    conf = [0.1, 0.2, 0.3, 0.4]
    t, d = best_threshold(base, c, conf, gold, [0.0, 0.25, 0.5, 1.0])
    assert d >= 0.0
    assert t in {0.0, 0.25, 0.5, 1.0}


# --- split_indices -----------------------------------------------------------


def test_uncertainty_signal_confidence_and_disagreement() -> None:
    conf = [0.9, 0.5, None]
    rub_a = [4.0, 2.0, 5.0]
    rub_b = [4.0, 5.0, 1.0]
    # confidence: uncertainty = -conf; None -> inf (most uncertain)
    assert uncertainty_signal(conf, rub_a, rub_b, "confidence") == [-0.9, -0.5, float("inf")]
    # disagreement: |a-b|
    assert uncertainty_signal(conf, rub_a, rub_b, "disagreement") == [0.0, 3.0, 4.0]


def test_budget_curve_monotone_and_endpoints() -> None:
    base = [3.0, 3.0, 3.0, 3.0]
    gold = [0, 0, 1, 1]
    c = [1.0, 1.0, 5.0, 5.0]  # c separates gold perfectly
    unc = [0.4, 0.3, 0.2, 0.1]  # ranking; top fractions escalate first
    curve = budget_curve(base, c, unc, gold, [0.0, 0.5, 1.0])
    by_frac = {row["fraction"]: row for row in curve}
    assert by_frac[0.0]["n_escalated"] == 0
    assert by_frac[0.0]["delta_rpb"] == 0.0  # nothing escalated -> no change
    assert by_frac[1.0]["n_escalated"] == 4  # all escalated
    assert by_frac[1.0]["delta_rpb"] >= by_frac[0.0]["delta_rpb"]  # c helps


def test_split_indices_deterministic_disjoint_cover() -> None:
    tune, report = split_indices(10, frac=0.5, seed=42)
    assert sorted(tune + report) == list(range(10))  # cover, disjoint
    assert set(tune) & set(report) == set()
    assert split_indices(10, 0.5, 42) == (tune, report)  # deterministic
