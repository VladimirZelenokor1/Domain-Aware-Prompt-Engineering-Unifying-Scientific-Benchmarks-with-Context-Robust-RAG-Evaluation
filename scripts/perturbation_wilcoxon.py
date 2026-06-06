"""H3 perturbation Wilcoxon: base vs class1 (surface) and base vs class2 (semantic).

Reads judge rubric scores for the three aligned perturbation splits (produced
by generate_perturbations.py + run_inference --split-file + run_judge) and runs
a paired Wilcoxon signed-rank test of base-vs-perturbed rubric per class.

Expectation:
    class1 (surface typos/whitespace) -> NOT significant (judges/models robust)
    class2 (semantic negation)         -> significant     (sensitive to meaning)

Pairs are matched on (model, strategy, question-index); the index is the
numeric suffix of the question_id, shared across the aligned splits.

Usage:
    python scripts/perturbation_wilcoxon.py --judge-root outputs/judge
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

QID_IDX = re.compile(r"-(\d+)$")
SPLITS = ("perturb_base", "perturb_class1", "perturb_class2")

# All perturbed splits compared against perturb_base, with a human label.
# class1/class2 = question-level (surface typos / rule-based negation);
# class2pad/class2trunc = draft 3.6.4 answer-level semantic-degradation.
_COMPARISONS = {
    "perturb_class1": "surface (class1, question typos/whitespace)",
    "perturb_class2": "semantic-negation (class2, question negation)",
    "perturb_class2pad": "semantic-padding (class2, draft 3.6.4 answer padding)",
    "perturb_class2trunc": "semantic-truncation (class2, draft 3.6.4 answer truncation)",
}


def _rubrics(judge_root: Path, split: str) -> dict[tuple[str, str, str], list[int]]:
    """Map (model, strategy, index) -> [rubric per judge] for one split."""
    out: dict[tuple[str, str, str], list[int]] = {}
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(str(judge_root / jid / split / "*" / "*.jsonl")):
            parts = Path(f).parts
            model, stem = parts[-2], Path(f).stem
            for line in open(f, encoding="utf-8"):
                if not line.strip():
                    continue
                rec = json.loads(line)
                m = QID_IDX.search(rec.get("question_id", ""))
                if not m:
                    continue
                out.setdefault((model, stem, m.group(1)), []).append(
                    int(rec.get("rubric", 0))
                )
    return out


def _mean_by_pair(d: dict[tuple, list[int]]) -> dict[tuple, float]:
    return {k: sum(v) / len(v) for k, v in d.items()}


def main() -> None:
    """Run the paired Wilcoxon for class1 and class2 against base."""
    from judge_aggregate import compute_wilcoxon  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="H3 perturbation Wilcoxon")
    parser.add_argument(
        "--judge-root", type=Path, default=PROJECT_ROOT / "outputs" / "judge"
    )
    args = parser.parse_args()

    base = _mean_by_pair(_rubrics(args.judge_root, "perturb_base"))
    if not base:
        print("No base perturbation judge outputs found - run the pipeline first.")
        return

    for split, cls in _COMPARISONS.items():
        pert = _mean_by_pair(_rubrics(args.judge_root, split))
        keys = sorted(set(base) & set(pert))
        if not keys:
            print(f"{split}: no paired records (skip)")
            continue
        b = np.array([base[k] for k in keys], float)
        p = np.array([pert[k] for k in keys], float)
        res = compute_wilcoxon(b, p)
        verdict = "significant" if res["significant"] else "NOT significant"
        print(
            f"{cls}: n={len(keys)} pairs  mean rubric base={b.mean():.3f} "
            f"-> perturbed={p.mean():.3f}  delta={p.mean() - b.mean():+.3f}  "
            f"Wilcoxon p={res['p_value']:.4g}  -> {verdict}"
        )
    print(
        "expected: class1 NOT significant (robust to surface noise); "
        "class2 / class2pad / class2trunc significant (sensitive to meaning change)"
    )


if __name__ == "__main__":
    main()
