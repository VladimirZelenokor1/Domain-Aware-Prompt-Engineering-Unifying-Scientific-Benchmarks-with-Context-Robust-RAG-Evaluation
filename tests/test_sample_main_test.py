"""Unit tests for ``scripts/sample_main_test.py``.

Covers three invariant surfaces:

* sampling math: quotas hit target within rounding, determinism under
  seed, acceptance invariant (every non-empty source cell represented);
* stopping criteria: smallest cell, non-empty cell count, domain loss;
* CLI: ``--check-only``, exit codes, report emission.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from sample_main_test import (  # noqa: E402
    MAX_DOMAIN_LOSS,
    MIN_CELL_ITEMS,
    MIN_NON_EMPTY_CELLS,
    check_stopping_criteria,
    cross_tabulate,
    effective_sample_quotas,
    sample_stratified,
    verify_every_cell_represented,
)


# =========================================================================
# Factories
# =========================================================================


def _make_record(
    idx: int,
    domain: str = "Biology",
    level: str = "L1",
    qtype: str = "mcq-4-choices",
) -> dict:
    """Minimal SciKnowEval-shaped record."""
    return {
        "question_id": f"ske-toy-{idx:05d}",
        "domain": domain,
        "type": qtype,
        "details": {"level": level, "task": "t", "subtask": "s", "source": "x"},
        "question": f"q {idx}",
        "answer": f"a {idx}",
    }


def _make_grid(
    domains: list[str],
    levels: list[str],
    types: list[str],
    per_cell: int,
    start_id: int = 0,
) -> list[dict]:
    """Dense (domain × level × type) grid with ``per_cell`` items each."""
    out: list[dict] = []
    i = start_id
    for d in domains:
        for lvl in levels:
            for t in types:
                for _ in range(per_cell):
                    out.append(_make_record(i, d, lvl, t))
                    i += 1
    return out


# =========================================================================
# quotas + sampling
# =========================================================================


def test_quotas_hit_target_when_all_large() -> None:
    """With every cell >> n_floor, effective ≈ target."""
    records = _make_grid(["Bio", "Chem"], ["L1", "L2"], ["mcq-4-choices"], per_cell=200)
    cells = cross_tabulate(records)
    quotas = effective_sample_quotas(cells, target=300, n_floor=3)
    assert abs(sum(quotas.values()) - 300) <= 4  # tolerate rounding


def test_quotas_take_all_small_triggers() -> None:
    """Cells <= n_floor are taken whole, contributing floor_sum."""
    records = (
        _make_grid(
            ["Bio"], ["L1"], ["mcq-4-choices"], per_cell=1
        )  # 1 small cell, 1 item
        + _make_grid(
            ["Bio"], ["L1"], ["open-ended-qa"], per_cell=2
        )  # 1 small cell, 2 items
        + _make_grid(["Bio"], ["L1"], ["true_or_false"], per_cell=500)  # 1 large cell
    )
    cells = cross_tabulate(records)
    quotas = effective_sample_quotas(cells, target=50, n_floor=3)
    # The two small cells (1 + 2 items) are taken whole.
    small_quota = sum(quotas[c] for c in cells if len(cells[c]) <= 3)
    assert small_quota == 3


def test_sample_size_matches_target_tolerance_4() -> None:
    records = _make_grid(
        ["Biology", "Chemistry", "Physics", "Material"],
        ["L1", "L2", "L3"],
        ["mcq-4-choices", "open-ended-qa"],
        per_cell=200,
    )
    out = sample_stratified(records, target=1500, n_floor=3, seed=42)
    assert abs(len(out) - 1500) <= 24  # 24 = 1 per cell of rounding drift


def test_sample_is_deterministic_under_seed() -> None:
    records = _make_grid(
        ["Bio", "Chem"],
        ["L1", "L2"],
        ["mcq-4-choices"],
        per_cell=50,
    )
    a = sample_stratified(records, target=100, n_floor=3, seed=7)
    b = sample_stratified(records, target=100, n_floor=3, seed=7)
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]


def test_sample_varies_with_seed() -> None:
    records = _make_grid(
        ["Bio", "Chem"],
        ["L1", "L2"],
        ["mcq-4-choices"],
        per_cell=50,
    )
    a = sample_stratified(records, target=100, n_floor=3, seed=1)
    b = sample_stratified(records, target=100, n_floor=3, seed=2)
    assert [r["question_id"] for r in a] != [r["question_id"] for r in b]


def test_every_non_empty_source_cell_represented() -> None:
    records = _make_grid(
        ["Biology", "Chemistry", "Physics", "Material"],
        ["L1", "L2", "L3"],
        ["mcq-4-choices", "open-ended-qa"],
        per_cell=100,
    )
    out = sample_stratified(records, target=300, n_floor=3, seed=42)
    ok, missing = verify_every_cell_represented(records, out)
    assert ok, f"missing cells: {missing}"


def test_sample_is_sorted_by_question_id() -> None:
    records = _make_grid(["Bio"], ["L1"], ["mcq-4-choices"], per_cell=20)
    out = sample_stratified(records, target=10, n_floor=3, seed=42)
    ids = [r["question_id"] for r in out]
    assert ids == sorted(ids)


# =========================================================================
# stopping criteria
# =========================================================================


_FULL_TYPES = [
    "mcq-4-choices",
    "mcq-2-choices",
    "open-ended-qa",
    "true_or_false",
    "relation_extraction",
    "filling",
]
_FULL_DOMAINS = ["Biology", "Chemistry", "Physics", "Material"]
_FULL_LEVELS = ["L1", "L2", "L3"]
# 4 x 3 x 6 = 72 cells, comfortably above MIN_NON_EMPTY_CELLS = 35.


def test_criteria_pass_when_all_fat() -> None:
    pre = _make_grid(_FULL_DOMAINS, _FULL_LEVELS, _FULL_TYPES, per_cell=60)
    post = _make_grid(_FULL_DOMAINS, _FULL_LEVELS, _FULL_TYPES, per_cell=40)
    assert check_stopping_criteria(cross_tabulate(pre), cross_tabulate(post)) == []


def test_criteria_trip_when_smallest_cell_below_threshold() -> None:
    pre = _make_grid(["Biology"], ["L1", "L2"], ["mcq-4-choices"], 100)
    # Second cell has only 5 items (below MIN_CELL_ITEMS = 30).
    post = _make_grid(["Biology"], ["L1"], ["mcq-4-choices"], 80) + _make_grid(
        ["Biology"], ["L2"], ["mcq-4-choices"], MIN_CELL_ITEMS - 25
    )
    violations = check_stopping_criteria(cross_tabulate(pre), cross_tabulate(post))
    assert any("smallest non-empty" in v for v in violations)


def test_criteria_trip_when_non_empty_cells_drop() -> None:
    # Pre has 40 dense cells; post keeps only 30.
    pre = _make_grid(
        ["Bio", "Chem", "Phys", "Mat"],
        ["L1", "L2"],
        [
            "mcq-4-choices",
            "open-ended-qa",
            "true_or_false",
            "relation_extraction",
            "filling",
        ],
        50,
    )
    post = _make_grid(
        ["Bio"],
        ["L1", "L2"],
        [
            "mcq-4-choices",
            "open-ended-qa",
            "true_or_false",
            "relation_extraction",
            "filling",
        ],
        50,
    )
    violations = check_stopping_criteria(cross_tabulate(pre), cross_tabulate(post))
    assert any("non-empty cells dropped" in v for v in violations)
    # Sanity: fewer non-empty cells than the threshold.
    assert sum(1 for v in cross_tabulate(post).values() if v) < MIN_NON_EMPTY_CELLS


def test_criteria_trip_when_domain_loses_more_than_half() -> None:
    pre = _make_grid(["Biology"], ["L1", "L2"], ["mcq-4-choices"], 100)
    post = _make_grid(["Biology"], ["L1"], ["mcq-4-choices"], 40)  # 40/200 = 80% loss
    violations = check_stopping_criteria(cross_tabulate(pre), cross_tabulate(post))
    assert any("lost" in v and "Biology" in v for v in violations)
    # Confirm the loss exceeds the threshold we're asserting against.
    assert 1 - 40 / 200 > MAX_DOMAIN_LOSS


def test_criteria_allow_exactly_50_percent_loss() -> None:
    """Threshold is 'more than 50%'; exactly 50% must not trip."""
    pre = _make_grid(["Biology"], ["L1", "L2"], ["mcq-4-choices"], 100)
    # L1 retained fully, L2 dropped -> 50% loss exactly, should be allowed.
    post = _make_grid(["Biology"], ["L1"], ["mcq-4-choices"], 100)
    violations = check_stopping_criteria(cross_tabulate(pre), cross_tabulate(post))
    # No domain-loss violation should be raised.
    assert not any("lost" in v for v in violations)


# =========================================================================
# CLI end-to-end
# =========================================================================


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


def _run_cli(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    """Invoke the script with an isolated I/O root.

    Grid is 4 domains x 3 levels x 6 types = 72 cells, all >= 30 items,
    so every stopping criterion passes by construction.
    """
    pre = _make_grid(_FULL_DOMAINS, _FULL_LEVELS, _FULL_TYPES, per_cell=60)
    post = _make_grid(_FULL_DOMAINS, _FULL_LEVELS, _FULL_TYPES, per_cell=40)
    pre_path = tmp_path / "main_test.json"
    post_path = tmp_path / "main_test_retrievable.json"
    out_path = tmp_path / "main_test_sampled.json"
    report_path = tmp_path / "stratification.json"
    _write_json(pre_path, pre)
    _write_json(post_path, post)

    script = _SCRIPTS_DIR / "sample_main_test.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prefilter",
            str(pre_path),
            "--input",
            str(post_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            *extra,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result


def test_cli_happy_path_exits_zero_and_writes_report(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--target", "300", "--n-floor", "3", "--seed", "42")
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "main_test_sampled.json").exists()
    report = json.loads((tmp_path / "stratification.json").read_text())
    assert report["seed"] == 42
    assert report["target"] == 300
    assert report["acceptance"]["every_source_cell_represented"] is True
    assert report["stopping_criteria"]["passed"] is True


def test_cli_check_only_prints_status_and_exits_zero(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "--check-only")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["proceed"] is True
    assert payload["violations"] == []


def test_cli_refuses_on_violation_without_force(tmp_path: Path) -> None:
    # Build a post set that trips the "domain loses > 50%" criterion.
    pre = _make_grid(
        ["Biology", "Chemistry"],
        ["L1", "L2"],
        ["mcq-4-choices"],
        per_cell=100,
    )
    post = _make_grid(["Biology"], ["L1", "L2"], ["mcq-4-choices"], per_cell=80)
    pre_path = tmp_path / "main_test.json"
    post_path = tmp_path / "main_test_retrievable.json"
    out_path = tmp_path / "main_test_sampled.json"
    report_path = tmp_path / "stratification.json"
    _write_json(pre_path, pre)
    _write_json(post_path, post)

    script = _SCRIPTS_DIR / "sample_main_test.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prefilter",
            str(pre_path),
            "--input",
            str(post_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--target",
            "100",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stderr
    assert not out_path.exists()


def test_cli_force_bypasses_criteria(tmp_path: Path) -> None:
    pre = _make_grid(
        ["Biology", "Chemistry"],
        ["L1", "L2"],
        ["mcq-4-choices"],
        per_cell=100,
    )
    post = _make_grid(["Biology"], ["L1", "L2"], ["mcq-4-choices"], per_cell=80)
    pre_path = tmp_path / "main_test.json"
    post_path = tmp_path / "main_test_retrievable.json"
    out_path = tmp_path / "main_test_sampled.json"
    report_path = tmp_path / "stratification.json"
    _write_json(pre_path, pre)
    _write_json(post_path, post)

    script = _SCRIPTS_DIR / "sample_main_test.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--prefilter",
            str(pre_path),
            "--input",
            str(post_path),
            "--output",
            str(out_path),
            "--report",
            str(report_path),
            "--target",
            "100",
            "--force",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.exists()
    # Report must still record violations even when force was used.
    report = json.loads(report_path.read_text())
    assert report["stopping_criteria"]["passed"] is False
    assert len(report["stopping_criteria"]["violations"]) >= 1
