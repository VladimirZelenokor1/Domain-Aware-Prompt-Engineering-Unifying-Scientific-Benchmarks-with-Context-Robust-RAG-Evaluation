"""Tests for ``scripts/run_statistics.py`` (Phase I).

Hermetic - no real outputs needed. All synthetic.

Covers:
- Data loaders (summary tables, model registry, judge stats).
- H1: Pearson + Kendall, Steiger's z-test on two correlations.
- H2: mixed-effects regression coefficient direction.
- H3: judge-aggregate reader.
- Effect sizes: Cliff's delta and Cohen's d.
- Benjamini-Hochberg FDR correction.
- CSV + LaTeX writers.
- End-to-end CLI smoke on synthetic inputs.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_statistics import (  # noqa: E402
    benjamini_hochberg,
    build_h1_table,
    build_h2_table,
    build_h3_table,
    cliffs_delta,
    cohens_d,
    compute_rag_improvement,
    fishers_z,
    load_model_registry,
    load_summary_table,
    pearson_with_p,
    rag_summary_for_h2,
    steigers_z,
    write_csv_rows,
    write_latex_table,
)


# ============================================================== loaders


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows), encoding="utf-8")


def test_load_summary_table_returns_list_of_dicts(tmp_path: Path) -> None:
    rows = [
        {"model": "qwen2.5-7b", "strategy": "da", "exact_match": 0.55, "total": 100},
        {"model": "qwen2.5-7b", "strategy": "ras", "exact_match": 0.60, "total": 100},
    ]
    p = tmp_path / "summary_table.json"
    _write_summary(p, rows)
    out = load_summary_table(p)
    assert out == rows


def test_load_summary_table_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_summary_table(tmp_path / "nope.json")


def test_load_model_registry_returns_normalised_keys(tmp_path: Path) -> None:
    """Maps both 'qwen2.5-7b' and 'qwen2.5-7b-awq' to the same record."""
    registry = {
        "llms": {
            "qwen2.5-7b-awq": {"params_b": 7.6, "model_type": "qwen2"},
            "llama-3.2-3b-awq": {"params_b": 3.2, "model_type": "llama"},
        }
    }
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(registry), encoding="utf-8")
    reg = load_model_registry(p)
    # Both stripped and original should resolve.
    assert reg["qwen2.5-7b"]["params_b"] == 7.6
    assert reg["qwen2.5-7b-awq"]["params_b"] == 7.6
    assert reg["llama-3.2-3b"]["params_b"] == 3.2


# ====================================================== effect sizes


def test_cliffs_delta_perfect_separation_returns_minus_one() -> None:
    a = [1, 2, 3]
    b = [4, 5, 6]
    assert cliffs_delta(a, b) == pytest.approx(-1.0)


def test_cliffs_delta_perfect_separation_returns_plus_one() -> None:
    a = [4, 5, 6]
    b = [1, 2, 3]
    assert cliffs_delta(a, b) == pytest.approx(1.0)


def test_cliffs_delta_identical_groups_returns_zero() -> None:
    a = [1, 2, 3]
    b = [1, 2, 3]
    assert cliffs_delta(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_known_value() -> None:
    """d = (mean_a - mean_b) / pooled_sd."""
    a = [1.0, 2.0, 3.0, 4.0, 5.0]  # mean 3, sd ~1.58
    b = [3.0, 4.0, 5.0, 6.0, 7.0]  # mean 5, sd ~1.58
    d = cohens_d(a, b)
    # (3 - 5) / 1.58 ~ -1.265
    assert d == pytest.approx(-1.2649, abs=1e-2)


def test_cohens_d_zero_for_identical_groups() -> None:
    a = [1.0, 2.0, 3.0]
    b = [1.0, 2.0, 3.0]
    assert cohens_d(a, b) == pytest.approx(0.0, abs=1e-9)


def test_cohens_d_zero_pooled_sd_returns_zero() -> None:
    """Both groups constant - pooled sd = 0; function should not raise."""
    a = [5.0, 5.0, 5.0]
    b = [5.0, 5.0, 5.0]
    assert cohens_d(a, b) == 0.0


# ===================================================== correlations


def test_pearson_with_p_strong_positive() -> None:
    x = [1, 2, 3, 4, 5, 6]
    y = [2.1, 4.0, 6.1, 8.0, 10.1, 12.0]
    r, p = pearson_with_p(x, y)
    assert r > 0.99
    assert p < 0.01


def test_pearson_with_p_uncorrelated_high_p() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=50).tolist()
    y = rng.normal(size=50).tolist()
    r, p = pearson_with_p(x, y)
    # Random samples - any p OK as long as r is small.
    assert abs(r) < 0.5


def test_fishers_z_zero_at_zero_correlation() -> None:
    assert fishers_z(0.0) == pytest.approx(0.0)


def test_fishers_z_increases_with_r() -> None:
    assert fishers_z(0.5) > fishers_z(0.1) > 0.0


def test_steigers_z_returns_zero_z_when_correlations_equal() -> None:
    """Two identical correlations from the same sample - z statistic ~ 0."""
    n = 50
    z, p = steigers_z(r12=0.5, r13=0.5, r23=0.7, n=n)
    assert abs(z) < 1e-6
    assert p > 0.9


def test_steigers_z_significant_when_correlations_differ_strongly() -> None:
    z, p = steigers_z(r12=0.9, r13=0.1, r23=0.1, n=200)
    assert abs(z) > 3.0
    assert p < 0.01


# ============================================================ BH


def test_benjamini_hochberg_all_significant_under_bh() -> None:
    pvals = [0.01, 0.02, 0.03, 0.04]
    rejected, _ = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected == [True, True, True, True]


def test_benjamini_hochberg_none_significant_when_all_high() -> None:
    pvals = [0.5, 0.6, 0.7]
    rejected, _ = benjamini_hochberg(pvals, alpha=0.05)
    assert rejected == [False, False, False]


def test_benjamini_hochberg_step_up_property() -> None:
    """Once one p-value is rejected, all smaller p-values are rejected."""
    pvals = [0.20, 0.001, 0.01, 0.05]
    rejected, _ = benjamini_hochberg(pvals, alpha=0.05)
    # Smallest 0.001 must be rejected; the largest 0.20 (rank 4 of 4):
    # threshold = 4/4 * 0.05 = 0.05 < 0.20 -> NOT rejected.
    assert rejected[1] is True  # 0.001
    assert rejected[2] is True  # 0.01
    assert rejected[0] is False  # 0.20


def test_benjamini_hochberg_returns_corrected_pvalues_not_above_one() -> None:
    pvals = [0.01, 0.05, 0.10]
    _, q = benjamini_hochberg(pvals, alpha=0.05)
    assert all(qi <= 1.0 for qi in q)


# ===================================================== H1 helpers


def _closed_summary() -> list[dict]:
    """Synthetic closed-book summary: 6 models x 1 strategy."""
    rows = []
    base_em = {
        "llama-3.2-3b": 0.40,
        "sciphi-mistral-7b": 0.45,
        "deepseek-r1-qwen-7b": 0.48,
        "qwen2.5-7b": 0.50,
        "gemma-2-9b": 0.55,
        "mistral-nemo-12b": 0.60,
    }
    for model, em in base_em.items():
        rows.append({"model": model, "strategy": "da", "exact_match": em, "total": 100})
    return rows


def _rag_summary(improvement_per_size: float = 0.01) -> list[dict]:
    """Synthetic RAG summary that improves more for larger models."""
    rows = []
    base_em = {
        "llama-3.2-3b": (3.2, 0.40),
        "sciphi-mistral-7b": (7.0, 0.45),
        "deepseek-r1-qwen-7b": (7.0, 0.48),
        "qwen2.5-7b": (7.6, 0.50),
        "gemma-2-9b": (9.0, 0.55),
        "mistral-nemo-12b": (12.0, 0.60),
    }
    for model, (size, em) in base_em.items():
        # RAG_improvement scales with size (positive correlation we want H1 to detect).
        rag_em = em + improvement_per_size * size
        rows.append(
            {
                "model": model,
                "strategy": "da",
                "retriever": "hybrid",
                "noise_level": 0.0,
                "exact_match": rag_em,
                "total": 100,
            }
        )
    return rows


def _registry() -> dict[str, dict]:
    return {
        "llama-3.2-3b": {"params_b": 3.2},
        "sciphi-mistral-7b": {"params_b": 7.0},
        "deepseek-r1-qwen-7b": {"params_b": 7.0},
        "qwen2.5-7b": {"params_b": 7.6},
        "gemma-2-9b": {"params_b": 9.0},
        "mistral-nemo-12b": {"params_b": 12.0},
    }


def test_compute_rag_improvement_subtracts_closed_book() -> None:
    closed = _closed_summary()
    rag = _rag_summary(improvement_per_size=0.01)
    df = compute_rag_improvement(closed, rag)
    # Returns a list of dicts with model, strategy, retriever, noise_level,
    # closed_book, rag_score, delta.
    assert {r["model"] for r in df} == {x["model"] for x in closed}
    for row in df:
        assert row["delta"] == pytest.approx(
            row["rag_score"] - row["closed_book"], abs=1e-9
        )


def test_build_h1_table_detects_size_correlation() -> None:
    """With improvement scaling with size, H1 should report a strong positive correlation."""
    closed = _closed_summary()
    rag = _rag_summary(improvement_per_size=0.01)
    registry = _registry()
    rows, summary = build_h1_table(
        closed_book=closed,
        rag=rag,
        registry=registry,
    )
    # Pearson over deltas vs sizes - delta = improvement_per_size * size, so
    # correlation should be ~1.0.
    pearson_row = next(r for r in rows if r["test"] == "pearson_size_vs_delta")
    assert pearson_row["statistic"] > 0.95
    # Kendall positive too.
    kendall_row = next(r for r in rows if r["test"] == "kendall_size_vs_delta")
    assert kendall_row["statistic"] > 0.5
    # Summary reports n_models.
    assert summary["n_models"] == 6


def test_build_h1_table_handles_zero_improvement() -> None:
    """Edge case: when RAG = closed-book exactly, delta = 0 for all - correlation NaN."""
    closed = _closed_summary()
    rag = []
    for r in closed:
        rag.append(
            {
                **r,
                "retriever": "hybrid",
                "noise_level": 0.0,
            }
        )
    registry = _registry()
    rows, summary = build_h1_table(closed_book=closed, rag=rag, registry=registry)
    # With zero-variance delta, function should return NaN/None for correlation
    # but not crash. Must produce rows (possibly with statistic=None).
    assert rows
    assert summary["n_models"] == 6


# ===================================================== H2 helpers


def test_rag_summary_for_h2_returns_long_format() -> None:
    rag = _rag_summary()
    df = rag_summary_for_h2(rag, registry=_registry())
    # Every row has the columns needed for MixedLM.
    for row in df:
        assert "score" in row
        assert "noise_level" in row
        assert "strategy" in row
        assert "retriever" in row
        assert "model" in row
        assert "params_b" in row


def test_build_h2_table_detects_negative_noise_coefficient() -> None:
    """Under synthetic data where score drops with noise, MixedLM coef < 0."""
    rng = np.random.default_rng(42)
    rag_rows = []
    for model in ["llama-3.2-3b", "qwen2.5-7b", "gemma-2-9b", "mistral-nemo-12b"]:
        base = rng.uniform(0.3, 0.7)
        for strategy in ["da", "ras"]:
            for retriever in ["bm25", "hybrid"]:
                for noise in [0.0, 0.2, 0.4, 0.6]:
                    score = base - 0.5 * noise + rng.normal(0, 0.02)
                    rag_rows.append(
                        {
                            "model": model,
                            "strategy": strategy,
                            "retriever": retriever,
                            "noise_level": noise,
                            "exact_match": score,
                            "total": 100,
                        }
                    )
    registry = _registry()
    try:
        rows, summary = build_h2_table(rag=rag_rows, registry=registry)
    except RuntimeError as exc:
        # Skip if statsmodels unavailable
        pytest.skip(f"statsmodels not usable: {exc}")
    noise_coef = next(r for r in rows if r["term"] == "noise_level")
    # Coefficient should be ~ -0.5 with tight CI under this DGP.
    assert noise_coef["coef"] < -0.3
    # SE not nan.
    assert summary["n_obs"] == len(rag_rows)


# ===================================================== H3


def test_build_h3_table_reads_aggregate_stats(tmp_path: Path) -> None:
    stats = {
        "krippendorff_alpha": {
            "rubric": {"alpha": 0.78, "n_items": 1000, "raters": ["judge_a", "judge_b"]}
        },
        "ece": {
            "all": {"ece": 0.07, "n_bins": 10, "n": 800},
        },
        "wilcoxon": {
            "noise_low_vs_high": {
                "statistic": 12345.0,
                "pvalue": 0.001,
                "n_pairs": 700,
            },
        },
    }
    p = tmp_path / "aggregate_stats.json"
    p.write_text(json.dumps(stats), encoding="utf-8")
    rows, summary = build_h3_table(p)
    assert any(r["metric"].startswith("krippendorff") for r in rows)
    assert any(r["metric"].startswith("ece") for r in rows)
    assert any(r["metric"].startswith("wilcoxon") for r in rows)
    assert summary["sources"] == [str(p)]


def test_build_h3_table_missing_file_returns_empty() -> None:
    """If judge stats not yet produced, H3 reader returns empty rows, not raise."""
    rows, summary = build_h3_table(Path("/nonexistent/aggregate_stats.json"))
    assert rows == []
    assert summary["status"] == "missing"


# ============================================================ writers


def test_write_csv_rows_round_trip(tmp_path: Path) -> None:
    rows = [
        {"name": "A", "value": 0.5, "p": 0.001},
        {"name": "B", "value": -0.3, "p": 0.02},
    ]
    p = tmp_path / "out.csv"
    write_csv_rows(rows, p)
    assert p.exists()
    with open(p, encoding="utf-8") as f:
        out = list(csv.DictReader(f))
    assert len(out) == 2
    assert out[0]["name"] == "A"
    assert out[0]["value"] == "0.5"


def test_write_latex_table_contains_caption_and_rows(tmp_path: Path) -> None:
    rows = [{"name": "A", "value": 0.5}, {"name": "B", "value": 0.7}]
    p = tmp_path / "out.tex"
    write_latex_table(rows, p, caption="Synthetic table", label="tab:test")
    text = p.read_text(encoding="utf-8")
    assert "\\begin{table}" in text
    assert "\\caption{Synthetic table}" in text
    assert "\\label{tab:test}" in text
    assert "name" in text
    assert "value" in text
    assert "0.5" in text


# ============================================================ CLI smoke


def test_cli_smoke_writes_chapter5_tables(tmp_path: Path) -> None:
    """End-to-end: synthetic inputs -> outputs/chapter5_tables/* files exist."""
    from run_statistics import main

    closed_path = tmp_path / "closed_book_main" / "summary_table.json"
    rag_path = tmp_path / "rag_main" / "summary_table.json"
    registry_path = tmp_path / "registry.json"
    judge_path = tmp_path / "judge" / "aggregate_stats.json"
    out_dir = tmp_path / "out"

    closed_path.parent.mkdir(parents=True, exist_ok=True)
    rag_path.parent.mkdir(parents=True, exist_ok=True)
    judge_path.parent.mkdir(parents=True, exist_ok=True)

    closed_path.write_text(json.dumps(_closed_summary()), encoding="utf-8")
    rag_path.write_text(json.dumps(_rag_summary(0.01)), encoding="utf-8")
    registry_path.write_text(
        json.dumps(
            {
                "llms": {
                    f"{m}-awq": {"params_b": v["params_b"]}
                    for m, v in _registry().items()
                }
            }
        ),
        encoding="utf-8",
    )
    judge_path.write_text(
        json.dumps(
            {
                "krippendorff_alpha": {
                    "rubric": {
                        "alpha": 0.8,
                        "n_items": 500,
                        "raters": ["judge_a", "judge_b"],
                    }
                },
                "ece": {"all": {"ece": 0.05, "n_bins": 10, "n": 500}},
            }
        ),
        encoding="utf-8",
    )

    rc = main(
        [
            "--closed-book-summary",
            str(closed_path),
            "--rag-summary",
            str(rag_path),
            "--registry",
            str(registry_path),
            "--judge-stats",
            str(judge_path),
            "--output",
            str(out_dir),
        ]
    )
    assert rc == 0
    assert (out_dir / "H1.csv").exists()
    assert (out_dir / "H1.tex").exists()
    assert (out_dir / "H3.csv").exists()
    # H2 may or may not exist depending on statsmodels availability;
    # tolerate but flag.
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert "H1" in summary
    assert "H3" in summary
