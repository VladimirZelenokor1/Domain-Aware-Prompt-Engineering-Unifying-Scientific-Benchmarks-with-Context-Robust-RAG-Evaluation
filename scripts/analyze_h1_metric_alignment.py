"""H1 (RQ1): domain-aware rubric vs lexical metrics, alignment with gold.

Thesis H1 has two parts, both on the closed-book condition:

(a) Metric-gold alignment. On the unambiguous-gold subset (MCQ, true/false,
    fill-in), the point-biserial correlation between the LLM-judge rubric score
    and binary correctness is expected to be HIGHER than for each lexical metric
    (ROUGE-L, BLEU-4). Tested with the Williams/Steiger test for two dependent
    correlations that share the correctness variable, Bonferroni-corrected for
    the comparisons.

(b) Strategy-ranking divergence. On the open-ended subset, the ranking of
    prompting strategies by mean rubric differs from the ranking by each lexical
    metric (Kendall's tau < 1), with the largest divergence expected for RAS.

Rubric per item = mean over judge_a/judge_b (judged subset, N per cell).
Correctness = compute_exact_match (question-type aware). Lexical metrics are
computed on the model's answer text vs the gold answer text. Only the judged
subset has rubric scores, so all H1 statistics are computed on that subset;
the per-type Ns are printed for transparency.

Read-only. Usage:
    python scripts/analyze_h1_metric_alignment.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

# question_type groups (SciKnowEval)
UNAMBIGUOUS = {"mcq-4-choices", "mcq-2-choices", "true_or_false", "filling"}
OPEN_ENDED = {"open-ended-qa", "relation_extraction"}
STRATEGIES = ["da", "ras", "ctl", "sc"]


def _read_jsonl(path: str) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def _answer_text(rec: dict) -> str:
    """Raw answer text for lexical metrics (not the normalized letter)."""
    parsed = rec.get("parsed") or {}
    return (
        parsed.get("answer")
        or rec.get("raw_response")
        or parsed.get("answer_normalized")
        or ""
    )


def load_closed_book(root: Path) -> dict[tuple, dict]:
    """Map (model, strategy, question_id) -> record fields for closed-book."""
    from compute_metrics import compute_exact_match, get_predicted_answer  # noqa: PLC0415

    out: dict[tuple, dict] = {}
    for f in glob.glob(str(root / "closed_book_main_test" / "*" / "*.jsonl")):
        if f.endswith("_metrics.json"):
            continue
        model = Path(f).parent.name
        strategy = Path(f).stem
        for r in _read_jsonl(f):
            qid = r.get("question_id")
            qtype = r.get("question_type", "")
            gold = r.get("gold_answer", "")
            out[(model, strategy, qid)] = {
                "qtype": qtype,
                "correct": 1.0
                if compute_exact_match(get_predicted_answer(r), gold, qtype)
                else 0.0,
                "answer_text": _answer_text(r),
                "gold": gold,
            }
    return out


def load_rubric(root: Path) -> dict[tuple, float]:
    """Map (model, strategy, question_id) -> mean rubric over judges (closed-book)."""
    acc: dict[tuple, list[int]] = collections.defaultdict(list)
    for jid in ("judge_a", "judge_b"):
        for f in glob.glob(
            str(root / "judge" / jid / "closed_book_main_test" / "*" / "*.jsonl")
        ):
            model = Path(f).parent.name
            strategy = Path(f).stem
            for r in _read_jsonl(f):
                acc[(model, strategy, r.get("question_id"))].append(
                    int(r.get("rubric", 0))
                )
    return {k: float(np.mean(v)) for k, v in acc.items()}


def _rouge_l(pred: str, gold: str) -> float:
    from compute_metrics import compute_rouge_l  # noqa: PLC0415

    return compute_rouge_l(pred, gold)


def _bleu(pred: str, gold: str) -> float:
    if not pred or not gold:
        return 0.0
    from sacrebleu.metrics import BLEU  # noqa: PLC0415

    return min(
        BLEU(effective_order=True).sentence_score(pred, [gold]).score / 100.0, 1.0
    )


def point_biserial(metric: np.ndarray, correct: np.ndarray) -> float:
    from scipy.stats import pointbiserialr  # noqa: PLC0415

    if len(set(correct.tolist())) < 2:  # need both classes
        return float("nan")
    return float(pointbiserialr(correct, metric).statistic)


def williams_test(r_jk: float, r_jh: float, r_kh: float, n: int) -> tuple[float, float]:
    """Williams' test: is r_jk (rubric,correct) > r_jh (lexical,correct)?

    j = correctness (shared), k = rubric, h = lexical metric; r_kh = rubric vs
    lexical. Returns (t statistic, two-sided p value).
    """
    from scipy.stats import t as t_dist  # noqa: PLC0415

    det = 1 - r_jk**2 - r_jh**2 - r_kh**2 + 2 * r_jk * r_jh * r_kh
    num = (r_jk - r_jh) * np.sqrt((n - 1) * (1 + r_kh))
    den = np.sqrt(
        2 * ((n - 1) / (n - 3)) * det + ((r_jk + r_jh) ** 2 / 4) * (1 - r_kh) ** 3
    )
    if den == 0:
        return float("nan"), float("nan")
    t = num / den
    p = 2 * (1 - t_dist.cdf(abs(t), df=n - 3))
    return float(t), float(p)


def main() -> None:
    """Run H1 (a) metric-gold alignment and (b) strategy-ranking divergence."""
    parser = argparse.ArgumentParser(description="H1 metric-gold alignment")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    root = args.outputs_root

    cb = load_closed_book(root)
    rubric = load_rubric(root)
    keys = [k for k in rubric if k in cb]  # judged subset with closed-book data
    print(f"judged closed-book items: {len(keys)}")

    # assemble per-item arrays on the unambiguous-gold subset
    rows = []
    for k in keys:
        c = cb[k]
        rows.append(
            {
                "qtype": c["qtype"],
                "rubric": rubric[k],
                "correct": c["correct"],
                "rouge": _rouge_l(c["answer_text"], c["gold"]),
                "bleu": _bleu(c["answer_text"], c["gold"]),
                # exact-match lexical metric = string equality of answer text vs
                # gold text (distinct from the type-aware correctness label)
                "em": 1.0
                if c["answer_text"].strip().lower() == str(c["gold"]).strip().lower()
                else 0.0,
                "strategy": k[1],
            }
        )

    print("\n[H1a] point-biserial r(metric, correctness) on unambiguous-gold subset")
    sub = [r for r in rows if r["qtype"] in UNAMBIGUOUS]
    print(f"  n = {len(sub)}  types = {sorted({r['qtype'] for r in sub})}")
    correct = np.array([r["correct"] for r in sub])
    rub = np.array([r["rubric"] for r in sub])
    rouge = np.array([r["rouge"] for r in sub])
    bleu = np.array([r["bleu"] for r in sub])

    em = np.array([r["em"] for r in sub])

    r_rub = point_biserial(rub, correct)
    print(f"  r_pb(rubric,  correct) = {r_rub:.3f}")
    n_cmp = 3  # rubric vs {rouge, bleu, exact_match} (thesis: Bonferroni /3)
    for name, arr in (("rouge_l", rouge), ("bleu_4", bleu), ("exact_match", em)):
        r_lex = point_biserial(arr, correct)
        r_kh = float(np.corrcoef(rub, arr)[0, 1])
        t, p = williams_test(r_rub, r_lex, r_kh, len(sub))
        bonf = args.alpha / n_cmp
        verdict = "rubric HIGHER (sig)" if (p < bonf and r_rub > r_lex) else "n.s."
        print(
            f"  r_pb({name:7s}, correct) = {r_lex:.3f}  | Williams t={t:.2f} "
            f"p={p:.4g} (Bonferroni a={bonf:.3f}) -> {verdict}"
        )

    print("\n[H1b] strategy ranking divergence on open-ended subset (Kendall tau)")
    from scipy.stats import kendalltau  # noqa: PLC0415

    opn = [r for r in rows if r["qtype"] in OPEN_ENDED]
    print(f"  n = {len(opn)}  types = {sorted({r['qtype'] for r in opn})}")

    def rank_by(metric: str) -> list[str]:
        means = {
            s: np.mean([r[metric] for r in opn if r["strategy"] == s])
            for s in STRATEGIES
            if any(r["strategy"] == s for r in opn)
        }
        return [s for s, _ in sorted(means.items(), key=lambda kv: -kv[1])]

    ranks = {m: rank_by(m) for m in ("rubric", "rouge", "bleu", "em")}
    for m, rk in ranks.items():
        print(f"  {m:6s} ranking: {rk}")
    # all pairwise Kendall tau between metric-induced rankings
    pos = {m: {s: i for i, s in enumerate(rk)} for m, rk in ranks.items()}
    metrics = [m for m in ranks if len(ranks[m]) == len(ranks["rubric"]) > 1]
    print("  pairwise Kendall tau:")
    for i, a in enumerate(metrics):
        for b in metrics[i + 1 :]:
            common = ranks["rubric"]
            tau = kendalltau(
                [pos[a][s] for s in common], [pos[b][s] for s in common]
            ).statistic
            print(f"    {a:6s} vs {b:6s}: tau = {tau:.3f}")


if __name__ == "__main__":
    main()
