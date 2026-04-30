"""Phase I: hypothesis tests and effect sizes for chapter 5.

Reads per-cell metric summaries from Phase D (closed-book) and Phase F
(RAG main), plus the judge-aggregate output from Phase H, and emits CSV
and LaTeX tables that back the H1 / H2 / H3 hypothesis sections of the
thesis (see ``docs/blockers.md`` Section 7).

H1 (closed-book vs RAG)
    Pearson and Kendall correlations between model size and per-model
    RAG improvement, plus Steiger's z-test comparing the two
    correlations corr(size, closed_book) and corr(size, rag).

H2 (noise robustness)
    Mixed-effects regression
        score ~ noise_level + strategy + retriever + (1|model)
    The coefficient on ``noise_level`` quantifies sensitivity to noise.
    statsmodels is a required dep; if not installed, H2 is skipped with
    a logged warning.

H3 (judge reliability)
    Reads ``outputs/judge/aggregate_stats.json`` produced by
    ``scripts/judge_aggregate.py`` and reformats it as a table.

Effect sizes
    Cliff's delta and Cohen's d for each pair of models on a chosen
    score column.

Multiple testing
    Benjamini-Hochberg FDR control at alpha=0.05 over all reported
    p-values (H1, H2, H3 collated).

Outputs
    outputs/chapter5_tables/
        H1.csv, H1.tex
        H2.csv, H2.tex            (only if statsmodels available)
        H3.csv, H3.tex
        effect_sizes.csv, effect_sizes.tex
        p_values_corrected.csv
        summary.json              (machine-readable digest)

Usage
    python scripts/run_statistics.py
    python scripts/run_statistics.py --hypothesis H1 H3
    python scripts/run_statistics.py \\
        --closed-book-summary outputs/closed_book_main/summary_table.json \\
        --rag-summary         outputs/rag_main/summary_table.json \\
        --registry            models/MODEL_REGISTRY.json \\
        --judge-stats         outputs/judge/aggregate_stats.json \\
        --output              outputs/chapter5_tables
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as _scipy_stats

try:
    import statsmodels.api as _sm  # noqa: F401
    import statsmodels.formula.api as _smf

    _STATSMODELS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only on minimal envs
    _STATSMODELS_AVAILABLE = False
    _smf = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "chapter5_tables"
DEFAULT_CLOSED_BOOK_SUMMARY = (
    PROJECT_ROOT / "outputs" / "closed_book_main" / "summary_table.json"
)
DEFAULT_RAG_SUMMARY = PROJECT_ROOT / "outputs" / "rag_main" / "summary_table.json"
DEFAULT_REGISTRY = PROJECT_ROOT / "models" / "MODEL_REGISTRY.json"
DEFAULT_JUDGE_STATS = PROJECT_ROOT / "outputs" / "judge" / "aggregate_stats.json"

SCORE_COL = "exact_match"  # Primary cell-level score for H1/H2.

# =========================================================================
# Loaders
# =========================================================================


def load_summary_table(path: Path) -> list[dict]:
    """Load ``summary_table.json`` produced by ``compute_metrics.py``.

    Returns the JSON-decoded list of cell dicts. Each cell has at least
    ``model``, ``strategy``, ``exact_match``, and ``total``; RAG cells
    additionally carry ``retriever`` and ``noise_level``.
    """
    if not path.exists():
        raise FileNotFoundError(f"summary table not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"summary table at {path} is not a list")
    return data


def _strip_awq_suffix(name: str) -> str:
    return name[:-4] if name.endswith("-awq") else name


def load_model_registry(path: Path) -> dict[str, dict]:
    """Load ``MODEL_REGISTRY.json`` and expose both '<name>-awq' and
    '<name>' keys for convenience.

    The downstream code uses short config-style names (no ``-awq``
    suffix), but the registry stores the suffixed key. This loader
    normalises both forms so callers don't need to remember which.
    """
    if not path.exists():
        raise FileNotFoundError(f"model registry not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    llms = data.get("llms") or {}
    out: dict[str, dict] = {}
    for full_name, record in llms.items():
        out[full_name] = record
        out[_strip_awq_suffix(full_name)] = record
    return out


# =========================================================================
# Effect sizes
# =========================================================================


def cliffs_delta(group_a: list[float], group_b: list[float]) -> float:
    """Cliff's delta non-parametric effect size in [-1, 1].

    delta = (n_greater - n_less) / (n_a * n_b), where n_greater is the
    number of (a, b) pairs with a > b, and similarly for less.
    """
    if not group_a or not group_b:
        return 0.0
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    n_greater = int(np.sum(a[:, None] > b[None, :]))
    n_less = int(np.sum(a[:, None] < b[None, :]))
    total = a.size * b.size
    return (n_greater - n_less) / total


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d using pooled standard deviation.

    Returns 0.0 when both groups are constant (pooled sd = 0) so the
    function is total without raising.
    """
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)
    if a.size < 2 or b.size < 2:
        return 0.0
    mean_a, mean_b = float(a.mean()), float(b.mean())
    var_a = float(a.var(ddof=1))
    var_b = float(b.var(ddof=1))
    pooled = math.sqrt(
        ((a.size - 1) * var_a + (b.size - 1) * var_b) / (a.size + b.size - 2)
    )
    if pooled == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled


# =========================================================================
# Correlations and Steiger's z
# =========================================================================


def pearson_with_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Pearson r and two-sided p-value via scipy.

    Returns (nan, nan) if input has fewer than 3 points or zero
    variance, so callers can detect degenerate data.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 3 or y_arr.size < 3:
        return float("nan"), float("nan")
    if x_arr.std(ddof=1) == 0 or y_arr.std(ddof=1) == 0:
        return float("nan"), float("nan")
    res = _scipy_stats.pearsonr(x_arr, y_arr)
    return float(res.statistic), float(res.pvalue)


def kendall_tau_with_p(x: list[float], y: list[float]) -> tuple[float, float]:
    """Kendall's tau and two-sided p-value."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.size < 3 or y_arr.size < 3:
        return float("nan"), float("nan")
    res = _scipy_stats.kendalltau(x_arr, y_arr)
    return float(res.statistic), float(res.pvalue)


def fishers_z(r: float) -> float:
    """Fisher z-transform of a Pearson correlation."""
    r = max(min(r, 0.999_999), -0.999_999)
    return 0.5 * math.log((1 + r) / (1 - r))


def steigers_z(r12: float, r13: float, r23: float, n: int) -> tuple[float, float]:
    """Steiger's z for comparing two dependent correlations.

    Tests H0: corr(X, Y1) == corr(X, Y2) given a sample of size n where
    Y1 and Y2 are correlated as r23. Implements Steiger (1980),
    equation 12, simplified two-sided p-value.

    Args:
        r12: corr(X, Y1).
        r13: corr(X, Y2).
        r23: corr(Y1, Y2).
        n: sample size.

    Returns:
        (z, p_two_sided).
    """
    if n < 4:
        return float("nan"), float("nan")
    z1 = fishers_z(r12)
    z2 = fishers_z(r13)
    rm2 = (r12 * r12 + r13 * r13) / 2.0
    f = (1.0 - r23) / (2.0 * (1.0 - rm2)) if rm2 < 1.0 else 0.0
    h = (1.0 - f * rm2) / (1.0 - rm2) if rm2 < 1.0 else 1.0
    z = (z1 - z2) * math.sqrt((n - 3) / (2.0 * (1.0 - r23) * h))
    p = 2.0 * (1.0 - _scipy_stats.norm.cdf(abs(z)))
    return float(z), float(p)


# =========================================================================
# Multiple testing - Benjamini-Hochberg FDR
# =========================================================================


def benjamini_hochberg(
    pvals: list[float], alpha: float = 0.05
) -> tuple[list[bool], list[float]]:
    """Benjamini-Hochberg step-up FDR control.

    Args:
        pvals: List of raw two-sided p-values.
        alpha: Target false discovery rate.

    Returns:
        (rejected, q_values) - rejected[i] is True iff pvals[i] survives
        BH at level alpha; q_values[i] is the BH-adjusted p-value
        (capped at 1.0).
    """
    p_arr = np.asarray(pvals, dtype=float)
    n = p_arr.size
    if n == 0:
        return [], []
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    # Adjusted p-values: q_(i) = min_{k>=i} (p_(k) * n / k), capped at 1.
    q_sorted = np.empty(n, dtype=float)
    running_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = ranked[i] * n / rank
        running_min = min(running_min, q)
        q_sorted[i] = running_min
    q_sorted = np.minimum(q_sorted, 1.0)
    rejected_sorted = q_sorted <= alpha
    # Restore original order.
    q_out = np.empty(n, dtype=float)
    rej_out = np.empty(n, dtype=bool)
    for new_idx, original_idx in enumerate(order):
        q_out[original_idx] = q_sorted[new_idx]
        rej_out[original_idx] = rejected_sorted[new_idx]
    return rej_out.tolist(), q_out.tolist()


# =========================================================================
# H1: closed-book vs RAG correlation analysis
# =========================================================================


def compute_rag_improvement(closed_book: list[dict], rag: list[dict]) -> list[dict]:
    """Per-(model, strategy) RAG improvement = mean RAG score - closed-book.

    Aggregates RAG cells across all retrievers and noise levels per
    (model, strategy) by averaging ``exact_match``. Closed-book cells
    are matched 1:1 by (model, strategy).
    """
    closed_index: dict[tuple[str, str], float] = {}
    for r in closed_book:
        key = (r["model"], r["strategy"])
        closed_index[key] = float(r.get(SCORE_COL, 0.0))

    rag_groups: dict[tuple[str, str], list[float]] = {}
    for r in rag:
        key = (r["model"], r["strategy"])
        rag_groups.setdefault(key, []).append(float(r.get(SCORE_COL, 0.0)))

    out = []
    for key, scores in rag_groups.items():
        rag_mean = sum(scores) / len(scores) if scores else 0.0
        closed_score = closed_index.get(key)
        if closed_score is None:
            continue
        out.append(
            {
                "model": key[0],
                "strategy": key[1],
                "closed_book": closed_score,
                "rag_score": rag_mean,
                "delta": rag_mean - closed_score,
                "rag_cells": len(scores),
            }
        )
    return out


def _per_model_means(rows: list[dict], registry: dict[str, dict]) -> list[dict]:
    """Average over strategies to produce one (model, params_b, mean_delta)
    record per model. Models missing from the registry are skipped."""
    by_model: dict[str, list[dict]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    out = []
    for model, group in by_model.items():
        reg = registry.get(model) or registry.get(_strip_awq_suffix(model))
        if not reg or "params_b" not in reg:
            logger.warning("model %r missing params_b in registry; skipping", model)
            continue
        out.append(
            {
                "model": model,
                "params_b": float(reg["params_b"]),
                "closed_book_mean": float(np.mean([g["closed_book"] for g in group])),
                "rag_mean": float(np.mean([g["rag_score"] for g in group])),
                "delta_mean": float(np.mean([g["delta"] for g in group])),
            }
        )
    out.sort(key=lambda r: r["params_b"])
    return out


def build_h1_table(
    *,
    closed_book: list[dict],
    rag: list[dict],
    registry: dict[str, dict],
) -> tuple[list[dict], dict]:
    """H1: model-size effect on RAG improvement.

    Returns ``(rows, summary)`` where ``rows`` is a list of
    test-result dicts suitable for CSV / LaTeX, and ``summary`` is a
    machine-readable digest used by the orchestrator.
    """
    deltas = compute_rag_improvement(closed_book, rag)
    per_model = _per_model_means(deltas, registry)
    if not per_model:
        return [], {"n_models": 0, "status": "no_models"}

    sizes = [r["params_b"] for r in per_model]
    delta = [r["delta_mean"] for r in per_model]
    closed = [r["closed_book_mean"] for r in per_model]
    rag_mean = [r["rag_mean"] for r in per_model]

    r_pearson, p_pearson = pearson_with_p(sizes, delta)
    r_kendall, p_kendall = kendall_tau_with_p(sizes, delta)
    r_size_closed, _ = pearson_with_p(sizes, closed)
    r_size_rag, _ = pearson_with_p(sizes, rag_mean)
    r_closed_rag, _ = pearson_with_p(closed, rag_mean)
    z, p_steiger = steigers_z(
        r12=r_size_rag,
        r13=r_size_closed,
        r23=r_closed_rag,
        n=len(per_model),
    )

    rows = [
        {
            "test": "pearson_size_vs_delta",
            "description": "Pearson r between model size and RAG improvement",
            "statistic": r_pearson,
            "pvalue": p_pearson,
            "n": len(per_model),
        },
        {
            "test": "kendall_size_vs_delta",
            "description": "Kendall's tau (rank) between model size and RAG improvement",
            "statistic": r_kendall,
            "pvalue": p_kendall,
            "n": len(per_model),
        },
        {
            "test": "pearson_size_vs_closed_book",
            "description": "Pearson r between model size and closed-book score",
            "statistic": r_size_closed,
            "pvalue": float("nan"),
            "n": len(per_model),
        },
        {
            "test": "pearson_size_vs_rag",
            "description": "Pearson r between model size and RAG score",
            "statistic": r_size_rag,
            "pvalue": float("nan"),
            "n": len(per_model),
        },
        {
            "test": "steigers_z_size_corr_change",
            "description": (
                "Steiger's z comparing corr(size, RAG) vs corr(size, closed-book)"
            ),
            "statistic": z,
            "pvalue": p_steiger,
            "n": len(per_model),
        },
    ]
    summary = {
        "n_models": len(per_model),
        "per_model": per_model,
        "pearson_r": r_pearson,
        "pearson_p": p_pearson,
        "kendall_tau": r_kendall,
        "kendall_p": p_kendall,
        "steigers_z": z,
        "steigers_p": p_steiger,
    }
    return rows, summary


# =========================================================================
# H2: mixed-effects regression
# =========================================================================


def rag_summary_for_h2(rag: list[dict], registry: dict[str, dict]) -> list[dict]:
    """Long-format records ready for ``MixedLM.from_formula``.

    Each row: (model, strategy, retriever, noise_level, params_b, score).
    """
    out = []
    for r in rag:
        reg = registry.get(r["model"]) or registry.get(_strip_awq_suffix(r["model"]))
        params_b = float(reg["params_b"]) if reg and "params_b" in reg else float("nan")
        out.append(
            {
                "model": r["model"],
                "strategy": r["strategy"],
                "retriever": r["retriever"],
                "noise_level": float(r["noise_level"]),
                "params_b": params_b,
                "score": float(r.get(SCORE_COL, 0.0)),
            }
        )
    return out


def build_h2_table(
    *,
    rag: list[dict],
    registry: dict[str, dict],
) -> tuple[list[dict], dict]:
    """H2: noise robustness via mixed-effects regression.

    Returns the per-term coefficient table plus a summary digest.
    """
    if not _STATSMODELS_AVAILABLE:
        raise RuntimeError(
            "statsmodels not installed; install with `pip install statsmodels` "
            "to compute H2 mixed-effects regression."
        )
    long = rag_summary_for_h2(rag, registry)
    if not long:
        return [], {"n_obs": 0, "status": "no_data"}

    import pandas as pd  # local to avoid hard dep at module import

    df = pd.DataFrame(long)
    formula = "score ~ noise_level + C(strategy) + C(retriever)"
    model = _smf.mixedlm(formula, df, groups=df["model"])
    try:
        result = model.fit(method="lbfgs", reml=True)
    except Exception as exc:  # numerical issues - report and exit gracefully
        logger.warning("MixedLM fit failed: %s", exc)
        return [], {"n_obs": len(df), "status": "fit_failed", "error": str(exc)}

    rows: list[dict] = []
    for term, coef in result.params.items():
        rows.append(
            {
                "term": term,
                "coef": float(coef),
                "std_err": float(result.bse.get(term, float("nan"))),
                "z": float(result.tvalues.get(term, float("nan"))),
                "pvalue": float(result.pvalues.get(term, float("nan"))),
            }
        )
    summary = {
        "n_obs": int(len(df)),
        "n_groups": int(df["model"].nunique()),
        "formula": formula,
        "log_likelihood": float(result.llf),
        "aic": float(getattr(result, "aic", float("nan"))),
        "converged": bool(getattr(result, "converged", True)),
    }
    return rows, summary


# =========================================================================
# H3: judge-aggregate reader
# =========================================================================


def build_h3_table(stats_path: Path) -> tuple[list[dict], dict]:
    """H3: read judge_aggregate output and reformat as a flat table.

    The schema produced by ``judge_aggregate.py`` may evolve; this
    reader is intentionally permissive - it only extracts the fields it
    knows and silently ignores unknown sub-trees.
    """
    if not stats_path.exists():
        return [], {"sources": [], "status": "missing"}
    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return [], {
            "sources": [str(stats_path)],
            "status": "decode_error",
            "error": str(exc),
        }

    rows: list[dict] = []
    krip = data.get("krippendorff_alpha") or {}
    for scope, payload in krip.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "metric": f"krippendorff_alpha[{scope}]",
                "value": float(payload.get("alpha", float("nan"))),
                "n": int(payload.get("n_items", 0)),
                "details": (f"raters={payload.get('raters', 'unknown')}"),
            }
        )

    ece = data.get("ece") or {}
    for scope, payload in ece.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "metric": f"ece[{scope}]",
                "value": float(payload.get("ece", float("nan"))),
                "n": int(payload.get("n", 0)),
                "details": f"n_bins={payload.get('n_bins', 'unknown')}",
            }
        )

    wilcoxon = data.get("wilcoxon") or {}
    for scope, payload in wilcoxon.items():
        if not isinstance(payload, dict):
            continue
        rows.append(
            {
                "metric": f"wilcoxon[{scope}]",
                "value": float(payload.get("statistic", float("nan"))),
                "n": int(payload.get("n_pairs", 0)),
                "details": f"pvalue={payload.get('pvalue', float('nan'))}",
            }
        )

    return rows, {"sources": [str(stats_path)], "status": "ok"}


# =========================================================================
# Effect sizes table
# =========================================================================


def build_effect_sizes_table(
    closed_book: list[dict],
    rag: list[dict],
) -> list[dict]:
    """Per-pair Cliff's delta and Cohen's d on cell-level scores.

    Compares each pair of MODELS using the cell-level score column;
    for closed-book and RAG separately.
    """
    rows: list[dict] = []

    def _pairs(rows_in: list[dict], label: str) -> None:
        by_model: dict[str, list[float]] = {}
        for r in rows_in:
            by_model.setdefault(r["model"], []).append(float(r.get(SCORE_COL, 0.0)))
        models = sorted(by_model.keys())
        for a, b in combinations(models, 2):
            rows.append(
                {
                    "scope": label,
                    "model_a": a,
                    "model_b": b,
                    "n_a": len(by_model[a]),
                    "n_b": len(by_model[b]),
                    "cliffs_delta": cliffs_delta(by_model[a], by_model[b]),
                    "cohens_d": cohens_d(by_model[a], by_model[b]),
                }
            )

    if closed_book:
        _pairs(closed_book, "closed_book")
    if rag:
        _pairs(rag, "rag")
    return rows


# =========================================================================
# Writers
# =========================================================================


def _format_value(value: Any) -> str:
    """Display formatter for LaTeX tables. Trims floats to 4dp."""
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        return f"{value:.4f}" if abs(value) < 1000 else f"{value:.4g}"
    return str(value)


def _csv_value(value: Any) -> str:
    """CSV-side stringifier - keep raw float repr for round-trip with pandas."""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return repr(value) if not value.is_integer() else f"{value:.1f}"
    return str(value)


def write_csv_rows(rows: list[dict], path: Path) -> None:
    """Write ``rows`` as CSV with the union of all keys preserved.

    Floats are written via ``repr`` so a downstream ``pd.read_csv`` /
    ``csv.DictReader`` sees the same numeric value back. Only the LaTeX
    writer applies display formatting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in fields})


def _latex_escape(s: str) -> str:
    replacements = [
        ("\\", "\\textbackslash{}"),
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("$", "\\$"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    out = s
    for src, tgt in replacements:
        out = out.replace(src, tgt)
    return out


def write_latex_table(
    rows: list[dict],
    path: Path,
    *,
    caption: str,
    label: str,
) -> None:
    """Render ``rows`` as a LaTeX tabular environment with a caption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            f"% empty table: {caption}\n",
            encoding="utf-8",
        )
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fields.append(key)
    col_spec = "l" * len(fields)
    lines: list[str] = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{_latex_escape(caption)}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(_latex_escape(f) for f in fields) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        cells = [_latex_escape(_format_value(row.get(f, ""))) for f in fields]
        lines.append(" & ".join(cells) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================================================================
# Orchestration / CLI
# =========================================================================


def _collect_pvalues(
    h1_rows: list[dict],
    h2_rows: list[dict],
    h3_rows: list[dict],
) -> list[dict]:
    """Flatten p-values from all H tables for FDR correction."""
    out = []
    for row in h1_rows:
        p = row.get("pvalue")
        if isinstance(p, float) and not math.isnan(p):
            out.append({"hypothesis": "H1", "test": row["test"], "pvalue": p})
    for row in h2_rows:
        p = row.get("pvalue")
        if isinstance(p, float) and not math.isnan(p):
            out.append({"hypothesis": "H2", "test": row["term"], "pvalue": p})
    for row in h3_rows:
        # H3 metrics typically don't carry an explicit p-value column.
        details = row.get("details", "")
        if isinstance(details, str) and details.startswith("pvalue="):
            try:
                p = float(details.split("=", 1)[1])
                out.append({"hypothesis": "H3", "test": row["metric"], "pvalue": p})
            except ValueError:
                pass
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase I: hypothesis tests and effect sizes for chapter 5",
    )
    parser.add_argument(
        "--hypothesis",
        nargs="+",
        choices=["H1", "H2", "H3"],
        default=["H1", "H2", "H3"],
    )
    parser.add_argument(
        "--closed-book-summary",
        type=Path,
        default=DEFAULT_CLOSED_BOOK_SUMMARY,
    )
    parser.add_argument("--rag-summary", type=Path, default=DEFAULT_RAG_SUMMARY)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--judge-stats", type=Path, default=DEFAULT_JUDGE_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--alpha", type=float, default=0.05, help="BH FDR target")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    closed_book: list[dict] = []
    rag: list[dict] = []
    registry: dict[str, dict] = {}

    if args.closed_book_summary.exists():
        closed_book = load_summary_table(args.closed_book_summary)
    else:
        logger.warning("closed-book summary missing: %s", args.closed_book_summary)
    if args.rag_summary.exists():
        rag = load_summary_table(args.rag_summary)
    else:
        logger.warning("rag summary missing: %s", args.rag_summary)
    if args.registry.exists():
        registry = load_model_registry(args.registry)
    else:
        logger.warning("model registry missing: %s", args.registry)

    summary: dict[str, Any] = {}

    h1_rows: list[dict] = []
    if "H1" in args.hypothesis and closed_book and rag and registry:
        h1_rows, h1_summary = build_h1_table(
            closed_book=closed_book, rag=rag, registry=registry
        )
        write_csv_rows(h1_rows, out_dir / "H1.csv")
        write_latex_table(
            h1_rows,
            out_dir / "H1.tex",
            caption="H1: model size vs RAG improvement",
            label="tab:h1",
        )
        summary["H1"] = h1_summary

    h2_rows: list[dict] = []
    if "H2" in args.hypothesis and rag and registry:
        if not _STATSMODELS_AVAILABLE:
            logger.warning(
                "statsmodels not installed - H2 mixed-effects regression skipped"
            )
            summary["H2"] = {"status": "statsmodels_missing"}
        else:
            h2_rows, h2_summary = build_h2_table(rag=rag, registry=registry)
            write_csv_rows(h2_rows, out_dir / "H2.csv")
            write_latex_table(
                h2_rows,
                out_dir / "H2.tex",
                caption="H2: noise robustness mixed-effects regression",
                label="tab:h2",
            )
            summary["H2"] = h2_summary

    h3_rows: list[dict] = []
    if "H3" in args.hypothesis:
        h3_rows, h3_summary = build_h3_table(args.judge_stats)
        write_csv_rows(h3_rows, out_dir / "H3.csv")
        write_latex_table(
            h3_rows,
            out_dir / "H3.tex",
            caption="H3: judge inter-rater reliability and calibration",
            label="tab:h3",
        )
        summary["H3"] = h3_summary

    effect_rows = build_effect_sizes_table(closed_book=closed_book, rag=rag)
    if effect_rows:
        write_csv_rows(effect_rows, out_dir / "effect_sizes.csv")
        write_latex_table(
            effect_rows,
            out_dir / "effect_sizes.tex",
            caption="Pairwise effect sizes (Cliff's delta, Cohen's d)",
            label="tab:effect_sizes",
        )
        summary["effect_sizes"] = {"n_pairs": len(effect_rows)}

    pvals = _collect_pvalues(h1_rows, h2_rows, h3_rows)
    if pvals:
        rejected, q = benjamini_hochberg([r["pvalue"] for r in pvals], alpha=args.alpha)
        for row, ok, qi in zip(pvals, rejected, q):
            row["bh_q"] = qi
            row["rejected_at_alpha"] = ok
        write_csv_rows(pvals, out_dir / "p_values_corrected.csv")
        summary["bh_fdr"] = {
            "alpha": args.alpha,
            "n_tests": len(pvals),
            "n_rejected": int(sum(rejected)),
        }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Phase I tables written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
