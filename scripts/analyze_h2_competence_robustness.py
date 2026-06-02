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


def rag_rubric_rows(root: Path) -> list[dict]:
    """One row per judged RAG item: model, strategy, noise_level, rubric (mean a/b)."""
    acc: dict[tuple, list[int]] = collections.defaultdict(list)
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(str(root / "judge" / jid / "rag_main" / "*" / "*.jsonl")):
            m = RAG_CELL_RE.match(Path(f).stem)
            if not m:
                continue
            retr, noise, strat = m.group(1), float(m.group(2)), m.group(3)
            model = Path(f).parent.name
            for r in _read_jsonl(f):
                key = (model, strat, retr, noise, r.get("question_id"))
                acc[key].append(int(r.get("rubric", 0)))
    rows = []
    for (model, strat, retr, noise, qid), rubrics in acc.items():
        rows.append(
            {
                "model": model,
                "strategy": strat,
                "retriever": retr,
                "noise_level": noise,
                "question_id": qid,
                "rubric": float(np.mean(rubrics)),
            }
        )
    return rows


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

    # ---- (2) model-level robustness slope + Spearman ---------------------------
    from scipy.stats import spearmanr  # noqa: PLC0415

    by_model = collections.defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    print(
        "\n[H2.2] per-model robustness slope (rubric vs noise) and closed-book rubric"
    )
    slopes, cb_means, names = [], [], []
    for m, mrows in sorted(by_model.items()):
        noise = np.array([r["noise_level"] for r in mrows])
        rub = np.array([r["rubric"] for r in mrows])
        slope = (
            float(np.polyfit(noise, rub, 1)[0])
            if len(set(noise.tolist())) > 1
            else float("nan")
        )
        cb_mean = float(
            np.mean([r["rubric"] for r in mrows if r["noise_level"] == 0.0])
        )
        slopes.append(slope)
        cb_means.append(cb_mean)
        names.append(m)
        print(f"  {m:24s} noise0_rubric={cb_mean:.3f}  robustness_slope={slope:+.4f}")

    rho, p = spearmanr(cb_means, slopes)
    print(
        f"\n  Spearman(noise0 rubric, robustness slope) over n={len(names)} models: rho={rho:.3f} p={p:.3g}"
    )
    med_cb, med_slope = float(np.median(cb_means)), float(np.median(slopes))
    fragile = [
        names[i]
        for i in range(len(names))
        if cb_means[i] > med_cb and slopes[i] < med_slope
    ]
    print(
        f"  'competent but context-fragile' (high noise0 rubric, low slope): {fragile or 'none'}"
    )


if __name__ == "__main__":
    main()
