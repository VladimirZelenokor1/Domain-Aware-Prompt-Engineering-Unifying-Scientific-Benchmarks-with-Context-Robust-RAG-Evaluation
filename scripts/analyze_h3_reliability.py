"""H3 (RQ3): judge reliability - per-judge ECE, alpha with bootstrap CI, Cliff's delta.

Completes the thesis H3 protocol (Section 3.11) beyond judge_aggregate:

(a) Stability - paired Wilcoxon on the perturbation audit (class 1 surface,
    class 2 semantic) with Cliff's delta effect size per class.
(b) Inter-judge consistency - Krippendorff's alpha (ordinal) between judge_a
    and judge_b over every jointly rated answer, with a 10,000-resample
    bootstrap 95% CI; evaluated against the pre-registered 0.40-0.80 band.
(c) Calibration - Expected Calibration Error per judge (B=10 bins) on the
    MCQ/short-answer subset; H3c holds if ECE > 0.05 for every judge.

Read-only. Usage:
    python scripts/analyze_h3_reliability.py
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

UNAMBIGUOUS = {"mcq-4-choices", "mcq-2-choices", "true_or_false", "filling"}
QID_IDX = re.compile(r"-(\d+)$")


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def compute_ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error with equally spaced bins."""
    if len(conf) == 0:
        return 0.0
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (conf >= edges[i]) & (
            conf < edges[i + 1] if i < n_bins - 1 else conf <= edges[i + 1]
        )
        if mask.sum():
            ece += (mask.sum() / len(conf)) * abs(
                conf[mask].mean() - correct[mask].mean()
            )
    return float(ece)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta effect size between samples a and b."""
    if not a or not b:
        return 0.0
    x, y = np.asarray(a), np.asarray(b)
    gt = int(np.sum(x[:, None] > y[None, :]))
    lt = int(np.sum(x[:, None] < y[None, :]))
    return (gt - lt) / (x.size * y.size)


def source_correct_mcq(root: Path) -> dict[str, float]:
    """cell+qid -> MCQ/short-answer correctness (1/0); only unambiguous types."""
    from compute_metrics import compute_exact_match, get_predicted_answer  # noqa: PLC0415

    out: dict[str, float] = {}
    for sub in ("closed_book_main_test", "rag_main", "qasper_main"):
        d = root / sub
        if not d.exists():
            continue
        for f in glob.glob(str(d / "*" / "*.jsonl")):
            if f.endswith("_metrics.json"):
                continue
            cell = "/".join(Path(f).relative_to(d.parent).parts)
            for r in _read_jsonl(f):
                if r.get("question_type") in UNAMBIGUOUS:
                    key = f"{cell}::{r.get('question_id')}"
                    out[key] = (
                        1.0
                        if compute_exact_match(
                            get_predicted_answer(r),
                            r.get("gold_answer", ""),
                            r.get("question_type", ""),
                        )
                        else 0.0
                    )
    return out


def judge_items(judge_root: Path, jid: str) -> dict[str, dict]:
    """cell+qid -> {rubric, self_confidence} for one judge (main tracks)."""
    out: dict[str, dict] = {}
    base = judge_root / jid
    for sub in ("closed_book_main_test", "rag_main", "qasper_main"):
        for f in glob.glob(str(base / sub / "*" / "*.jsonl")):
            cell = "/".join(Path(f).relative_to(base).parts)
            for r in _read_jsonl(f):
                out[f"{cell}::{r.get('question_id')}"] = {
                    "rubric": int(r.get("rubric", 0)),
                    "conf": r.get("self_confidence"),
                }
    return out


def perturb_rubric(judge_root: Path, split: str) -> dict[tuple, float]:
    """(model, strategy, idx) -> mean-over-judges rubric for a perturbation split."""
    acc: dict[tuple, list[int]] = collections.defaultdict(list)
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(str(judge_root / jid / split / "*" / "*.jsonl")):
            model, stem = Path(f).parent.name, Path(f).stem
            for r in _read_jsonl(f):
                m = QID_IDX.search(r.get("question_id", ""))
                if m:
                    acc[(model, stem, m.group(1))].append(int(r.get("rubric", 0)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def main() -> None:
    """Run H3 (a) stability, (b) inter-judge alpha + CI, (c) per-judge ECE."""
    import krippendorff  # noqa: PLC0415
    from scipy.stats import wilcoxon  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="H3 judge reliability")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    jr = args.outputs_root / "judge"

    a, b = judge_items(jr, "judge_a"), judge_items(jr, "judge_b")

    # (b) alpha + bootstrap CI
    items = sorted(set(a) & set(b))
    ra = np.array([a[i]["rubric"] for i in items], float)
    rb = np.array([b[i]["rubric"] for i in items], float)
    alpha = krippendorff.alpha(
        reliability_data=[ra, rb], level_of_measurement="ordinal"
    )
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(10000):
        idx = rng.integers(0, len(items), len(items))
        boot.append(
            krippendorff.alpha(
                reliability_data=[ra[idx], rb[idx]], level_of_measurement="ordinal"
            )
        )
    lo, hi = np.percentile(boot, [2.5, 97.5])
    band = "in 0.40-0.80 band" if 0.40 <= alpha <= 0.80 else "OUTSIDE 0.40-0.80"
    print(
        f"[H3b] Krippendorff alpha(a,b) = {alpha:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
        f"n={len(items)}  ({band}; < 0.85: {alpha < 0.85})"
    )

    # (c) per-judge ECE on the MCQ/short-answer subset
    correct = source_correct_mcq(args.outputs_root)
    print("\n[H3c] per-judge ECE (MCQ/short-answer subset, B=10):")
    all_above = True
    for jid, jmap in (("judge_a", a), ("judge_b", b)):
        conf, corr = [], []
        for key, v in jmap.items():
            if v["conf"] is not None and key in correct:
                conf.append(float(v["conf"]))
                corr.append(correct[key])
        ece = compute_ece(np.array(conf), np.array(corr))
        all_above = all_above and ece > 0.05
        print(f"  {jid}: ECE = {ece:.4f}  (n={len(conf)})  > 0.05: {ece > 0.05}")
    print(f"  H3c (ECE > 0.05 for every judge): {all_above}")

    # (a) perturbation stability with Cliff's delta
    base = perturb_rubric(jr, "perturb_base")
    if base:
        print("\n[H3a] perturbation stability (paired Wilcoxon + Cliff's delta):")
        for split in ("perturb_class1", "perturb_class2"):
            pert = perturb_rubric(jr, split)
            keys = sorted(set(base) & set(pert))
            if not keys:
                continue
            bv = [base[k] for k in keys]
            pv = [pert[k] for k in keys]
            stat, p = wilcoxon(bv, pv)
            d = cliffs_delta(bv, pv)
            cls = "surface(class1)" if "class1" in split else "semantic(class2)"
            print(
                f"  {cls}: n={len(keys)} base={np.mean(bv):.3f} pert={np.mean(pv):.3f} "
                f"Wilcoxon p={p:.4g} Cliff's delta={d:+.3f}"
            )
    else:
        print(
            "\n[H3a] no perturbation judge outputs found - run the perturbation audit first"
        )


if __name__ == "__main__":
    main()
