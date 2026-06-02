"""H2 (RQ2): closed-book competence vs RAG robustness under retrieval noise.

Thesis H2: domain-aware quality (rubric) under RAG relates to closed-book
competence and degrades as retrieval noise rises (0% -> 60%). Two levels of
analysis:

(1) Question-level mixed-effects model (inferential):
        rubric ~ closed_book_correct + noise_level + C(strategy) + (1 | model)
    closed_book_correct is the predictor of interest; noise_level and strategy
    are fixed effects; model is a random intercept.

(2) Model-level descriptive correlation (n = 6, no inferential claim):
    per-model robustness slope = OLS slope of mean rubric on noise_level
    (negative = quality drops with noise); Spearman between mean closed-book
    rubric and robustness slope; flag "competent but context-fragile" models
    (above-median closed-book rubric AND below-median robustness slope).

Rubric per item = mean over judge_a/judge_b on the judged RAG subset.
closed_book_correct = exact-match correctness of the same (model, strategy,
question) in the closed-book condition. Read-only.

Usage:
    python scripts/analyze_h2_competence_robustness.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

RAG_CELL_RE = re.compile(r"^(bm25|dense|hybrid)_noise([0-9.]+)_(da|ras|ctl|sc)$")


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def closed_book_correct(root: Path) -> dict[tuple, float]:
    """(model, strategy, question_id) -> closed-book exact-match correctness."""
    from compute_metrics import compute_exact_match, get_predicted_answer  # noqa: PLC0415

    out: dict[tuple, float] = {}
    for f in glob.glob(str(root / "closed_book_main_test" / "*" / "*.jsonl")):
        if f.endswith("_metrics.json"):
            continue
        model, strategy = Path(f).parent.name, Path(f).stem
        for r in _read_jsonl(f):
            qtype = r.get("question_type", "")
            out[(model, strategy, r.get("question_id"))] = (
                1.0
                if compute_exact_match(
                    get_predicted_answer(r), r.get("gold_answer", ""), qtype
                )
                else 0.0
            )
    return out


_JUDGE_FIELDS = ("rubric", "faithfulness", "citation_precision", "citation_recall")


def rag_rubric_rows(root: Path) -> list[dict]:
    """One row per judged RAG item with mean-over-judges rubric/faithfulness/citation."""
    acc: dict[tuple, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(str(root / "judge" / jid / "rag_main" / "*" / "*.jsonl")):
            m = RAG_CELL_RE.match(Path(f).stem)
            if not m:
                continue
            retr, noise, strat = m.group(1), float(m.group(2)), m.group(3)
            model = Path(f).parent.name
            for r in _read_jsonl(f):
                key = (model, strat, retr, noise, r.get("question_id"))
                for fld in _JUDGE_FIELDS:
                    if r.get(fld) is not None:
                        acc[key][fld].append(float(r[fld]))
    rows = []
    for (model, strat, retr, noise, qid), vals in acc.items():
        row = {
            "model": model,
            "strategy": strat,
            "retriever": retr,
            "noise_level": noise,
            "question_id": qid,
        }
        for fld in _JUDGE_FIELDS:
            row[fld] = float(np.mean(vals[fld])) if vals[fld] else float("nan")
        rows.append(row)
    return rows


def closed_book_rubric(root: Path) -> dict[str, float]:
    """Mean closed-book judge rubric per model (competence measure)."""
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(
            str(root / "judge" / jid / "closed_book_main_test" / "*" / "*.jsonl")
        ):
            model = Path(f).parent.name
            for r in _read_jsonl(f):
                acc[model].append(float(r.get("rubric", 0)))
    return {m: float(np.mean(v)) for m, v in acc.items()}


def main() -> None:
    """Run the H2 mixed model and model-level robustness correlation."""
    parser = argparse.ArgumentParser(description="H2 competence vs robustness")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    root = args.outputs_root

    cb = closed_book_correct(root)
    rows = rag_rubric_rows(root)
    for r in rows:
        r["closed_book_correct"] = cb.get((r["model"], r["strategy"], r["question_id"]))
    rows = [r for r in rows if r["closed_book_correct"] is not None]
    print(f"RAG judged items joined to closed-book: {len(rows)}")

    # ---- (1) question-level mixed-effects model --------------------------------
    try:
        import pandas as pd  # noqa: PLC0415
        import statsmodels.formula.api as smf  # noqa: PLC0415

        df = pd.DataFrame(rows)
        model = smf.mixedlm(
            "rubric ~ closed_book_correct + noise_level + C(strategy)",
            df,
            groups=df["model"],
        )
        res = model.fit(method="lbfgs", reml=True)
        print(
            "\n[H2.1] mixed model: rubric ~ closed_book_correct + noise_level + C(strategy) + (1|model)"
        )
        for term in ("closed_book_correct", "noise_level"):
            if term in res.params.index:
                print(
                    f"  {term:24s} coef={res.params[term]:+.4f}  p={res.pvalues[term]:.4g}"
                )
        gv = float(res.cov_re.iloc[0, 0]) if res.cov_re.size else float("nan")
        if gv < 1e-6:
            print(
                f"  (random-effect variance singular: {gv:.2e} - model differences "
                "absorbed by closed_book_correct; see OLS below for clean estimates)"
            )

        # OLS with model as a fixed effect - well-defined when the random
        # intercept is singular; the trustworthy estimate for the report.
        ols = smf.ols(
            "rubric ~ closed_book_correct + noise_level + C(strategy) + C(model)", df
        ).fit()
        print("\n[H2.1b] OLS (model as fixed effect) - robustness check:")
        for term in ("closed_book_correct", "noise_level"):
            print(
                f"  {term:24s} coef={ols.params[term]:+.4f}  p={ols.pvalues[term]:.4g}"
            )
        print(f"  R^2 = {ols.rsquared:.3f}  n = {int(ols.nobs)}")
    except ImportError:
        print("\n[H2.1] statsmodels/pandas not installed - skip mixed model")

    # ---- (2) model-level: thesis robustness slope (eq 12) + Spearman -----------
    from scipy.stats import spearmanr  # noqa: PLC0415

    cb_rubric = closed_book_rubric(root)

    def mean_hybrid(model: str, noise: float) -> float:
        vals = [
            r["rubric"]
            for r in rows
            if r["model"] == model
            and r["retriever"] == "hybrid"
            and r["noise_level"] == noise
        ]
        return float(np.mean(vals)) if vals else float("nan")

    print(
        "\n[H2.2] thesis robustness slope (eq 12: normalised 0%->60% drop on hybrid)"
        " vs closed-book rubric"
    )
    names, comp, drop = [], [], []
    for m in sorted({r["model"] for r in rows}):
        r0, r60 = mean_hybrid(m, 0.0), mean_hybrid(m, 0.6)
        delta = (r0 - r60) / r0 if r0 else float("nan")  # eq 12; >0 = degradation
        c = cb_rubric.get(m, float("nan"))
        names.append(m)
        comp.append(c)
        drop.append(delta)
        print(
            f"  {m:24s} closed_book_rubric={c:.3f}  hybrid r0={r0:.3f} r60={r60:.3f}"
            f"  norm_drop={delta:+.3f}"
        )

    rho, p = spearmanr(comp, drop)
    # model-level bootstrap CI (n=6 - tiny; thesis-specified, reported with caveat)
    rng = np.random.default_rng(42)
    comp_a, drop_a = np.array(comp), np.array(drop)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, len(names), len(names))
        if len(set(comp_a[idx])) > 1 and len(set(drop_a[idx])) > 1:
            boot.append(spearmanr(comp_a[idx], drop_a[idx]).statistic)
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (float("nan"), float("nan"))
    print(
        f"\n  Spearman(closed-book rubric, normalised drop) n={len(names)}: "
        f"rho={rho:.3f} p={p:.3g}  95% bootstrap CI [{lo:.3f}, {hi:.3f}] (n=6, descriptive)"
    )
    med_c, med_d = float(np.median(comp)), float(np.median(drop))
    fragile = [
        names[i]
        for i in range(len(names))
        if comp[i] > med_c and drop[i] > med_d  # high competence, big drop
    ]
    print(
        f"  'competent but context-fragile' (high closed-book + big drop): {fragile or 'none'}"
    )

    # ---- (3) closed-book competence vs RAG citation/faithfulness (RQ2) ---------
    print("\n[H2.3] closed-book rubric vs mean RAG grounding metrics (descriptive)")
    for fld in ("faithfulness", "citation_precision", "citation_recall"):
        xs, ys = [], []
        for m in names:
            mv = [r[fld] for r in rows if r["model"] == m and not np.isnan(r[fld])]
            if mv and not np.isnan(cb_rubric.get(m, float("nan"))):
                xs.append(cb_rubric[m])
                ys.append(float(np.mean(mv)))
        if len(xs) > 2:
            rr, pp = spearmanr(xs, ys)
            print(
                f"  Spearman(closed-book rubric, mean {fld:18s}) = {rr:+.3f} p={pp:.3g}"
            )


if __name__ == "__main__":
    main()
