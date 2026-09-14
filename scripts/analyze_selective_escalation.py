"""Selective escalation of low-confidence judgements to judge_c (RQ3, role iii).

Implements the Jung-et-al.-style selective-escalation check: the composite
open-weight judgement (mean of judge_a, judge_b rubric) is escalated to the
higher-capacity judge_c on the lowest-confidence items, under a confidence
threshold. We measure whether this improves how well the composite tracks the
SciKnowEval gold correctness, via the point-biserial r_pb(score, correctness)
on the unambiguous-gold subset.

Design (documented for the thesis, closes audit A-9):
- composite base S0 = mean(rubric_a, rubric_b) per item.
- confidence = mean(self_confidence_a, self_confidence_b); lower => escalate.
- escalated S1 = judge_c rubric on items with confidence < threshold (and where
  judge_c scored the item), else S0.
- gold y = binary exact-match correctness of the same item, restricted to
  unambiguous-gold types (mcq-2/4, true_or_false, filling).
- The confidence threshold is tuned on one half of the common pool (judge_a +
  judge_b + judge_c all scored) and the gain delta r_pb is reported on the held
  -out other half - so threshold tuning and reporting use disjoint items.

Usage:
    python scripts/analyze_selective_escalation.py
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

logger = logging.getLogger(__name__)

UNAMBIGUOUS = {"mcq-4-choices", "mcq-2-choices", "true_or_false", "filling"}
JUDGE_ROOT = PROJECT_ROOT / "outputs" / "judge"
INFERENCE_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "selective_escalation.json"


# =========================================================================
# Pure helpers
# =========================================================================


def point_biserial(continuous: list[float], binary: list[int]) -> float:
    """Point-biserial correlation (Pearson of a continuous score with a 0/1 label).

    Returns:
        Correlation in [-1, 1]; 0.0 if either series is constant or n < 2.
    """
    c = np.asarray(continuous, dtype=float)
    b = np.asarray(binary, dtype=float)
    if len(c) < 2 or c.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(c, b)[0, 1])


def escalate(
    base: list[float],
    c_scores: list[float | None],
    confidences: list[float | None],
    threshold: float,
) -> list[float]:
    """Replace base scores with judge_c where confidence is below ``threshold``.

    An item escalates only if judge_c scored it and its confidence is known and
    strictly below the threshold; otherwise the base composite is kept.
    """
    out = []
    for b, c, conf in zip(base, c_scores, confidences):
        if c is not None and conf is not None and conf < threshold:
            out.append(c)
        else:
            out.append(b)
    return out


def delta_rpb(
    base: list[float],
    c_scores: list[float | None],
    confidences: list[float | None],
    gold: list[int],
    threshold: float,
) -> float:
    """Gain in point-biserial alignment from escalating at ``threshold``."""
    escalated = escalate(base, c_scores, confidences, threshold)
    return point_biserial(escalated, gold) - point_biserial(base, gold)


def best_threshold(
    base: list[float],
    c_scores: list[float | None],
    confidences: list[float | None],
    gold: list[int],
    thresholds: list[float],
) -> tuple[float, float]:
    """Pick the threshold maximising delta r_pb (tuning step).

    Returns:
        (best_threshold, best_delta_rpb).
    """
    best_t, best_d = thresholds[0], float("-inf")
    for t in thresholds:
        d = delta_rpb(base, c_scores, confidences, gold, t)
        if d > best_d:
            best_d, best_t = d, t
    return best_t, best_d


def split_indices(n: int, frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Deterministically split range(n) into (tune, report) by ``frac``."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    cut = int(n * frac)
    tune = sorted(idx[:cut])
    report = sorted(idx[cut:])
    return tune, report


def uncertainty_signal(
    confidences: list[float | None],
    rub_a: list[float],
    rub_b: list[float],
    signal: str,
) -> list[float]:
    """Per-item uncertainty used to rank escalation (higher = escalate first).

    Args:
        confidences: mean judge_a/b self-confidence per item (None allowed).
        rub_a: judge_a rubric per item.
        rub_b: judge_b rubric per item.
        signal: "confidence" (uncertainty = -confidence) or "disagreement"
            (uncertainty = |rubric_a - rubric_b|).

    Returns:
        Uncertainty scores; missing confidence ranks as maximally uncertain.
    """
    if signal == "disagreement":
        return [abs(a - b) for a, b in zip(rub_a, rub_b)]
    return [(-c if c is not None else float("inf")) for c in confidences]


def budget_curve(
    base: list[float],
    c_scores: list[float | None],
    uncertainty: list[float],
    gold: list[int],
    fractions: list[float],
) -> list[dict]:
    """Delta r_pb when escalating only the most-uncertain ``frac`` of items.

    Models a fixed API budget: rank items by uncertainty (descending) and
    escalate the top ``frac`` to judge_c, leaving the rest on the base composite.

    Returns:
        One row per fraction: {fraction, n_escalated, escalated_rpb, delta_rpb}.
    """
    n = len(base)
    order = sorted(range(n), key=lambda i: uncertainty[i], reverse=True)
    base_rpb = point_biserial(base, gold)
    rows = []
    for frac in fractions:
        k = int(round(frac * n))
        esc_set = set(order[:k])
        scored = [
            c_scores[i] if (i in esc_set and c_scores[i] is not None) else base[i]
            for i in range(n)
        ]
        rpb = point_biserial(scored, gold)
        rows.append(
            {
                "fraction": round(frac, 2),
                "n_escalated": k,
                "escalated_rpb": round(rpb, 4),
                "delta_rpb": round(rpb - base_rpb, 4),
            }
        )
    return rows


# =========================================================================
# IO: assemble the common pool (judge_a + judge_b + judge_c, with gold)
# =========================================================================


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _judge_map(judge_root: Path, jid: str) -> dict[str, dict]:
    """item_id -> {rubric, self_confidence} for one judge across all cells."""
    out: dict[str, dict] = {}
    for f in glob.glob(str(judge_root / jid / "**" / "*.jsonl"), recursive=True):
        fp = Path(f)
        rel = fp.relative_to(judge_root / jid)  # track/model/cell.jsonl
        item_prefix = f"{rel.parts[0]}/{rel.parts[1]}/{fp.stem}"
        for r in _read_jsonl(fp):
            out[f"{item_prefix}/{r.get('question_id')}"] = {
                "rubric": r.get("rubric"),
                "self_confidence": r.get("self_confidence"),
            }
    return out


def _gold_map(inference_root: Path, judge_root: Path) -> dict[str, int]:
    """item_id -> binary correctness for unambiguous-type items (mirrors judge_a paths)."""
    from compute_metrics import compute_exact_match, get_predicted_answer  # noqa: PLC0415

    out: dict[str, int] = {}
    for f in glob.glob(str(judge_root / "judge_a" / "**" / "*.jsonl"), recursive=True):
        rel = Path(f).relative_to(judge_root / "judge_a")
        inf_path = inference_root / rel
        if not inf_path.exists():
            continue
        item_prefix = f"{rel.parts[0]}/{rel.parts[1]}/{Path(f).stem}"
        for r in _read_jsonl(inf_path):
            qtype = r.get("question_type") or r.get("type") or ""
            if qtype not in UNAMBIGUOUS:
                continue
            ok = compute_exact_match(
                get_predicted_answer(r), r.get("gold_answer", ""), qtype
            )
            out[f"{item_prefix}/{r.get('question_id')}"] = 1 if ok else 0
    return out


def assemble_pool(judge_root: Path, inference_root: Path) -> list[dict]:
    """Common pool: items scored by judge_a, judge_b AND judge_c with gold correctness."""
    ja = _judge_map(judge_root, "judge_a")
    jb = _judge_map(judge_root, "judge_b")
    jc = _judge_map(judge_root, "judge_c")
    gold = _gold_map(inference_root, judge_root)
    pool = []
    for item_id in sorted(set(ja) & set(jb) & set(jc) & set(gold)):
        ra, rb = ja[item_id]["rubric"], jb[item_id]["rubric"]
        if ra is None or rb is None or jc[item_id]["rubric"] is None:
            continue
        ca = ja[item_id]["self_confidence"]
        cb = jb[item_id]["self_confidence"]
        conf = (
            float(np.mean([x for x in (ca, cb) if x is not None]))
            if any(x is not None for x in (ca, cb))
            else None
        )
        pool.append(
            {
                "item_id": item_id,
                "base": (float(ra) + float(rb)) / 2.0,
                "rub_a": float(ra),
                "rub_b": float(rb),
                "c": float(jc[item_id]["rubric"]),
                "conf": conf,
                "gold": gold[item_id],
            }
        )
    return pool


# =========================================================================
# CLI
# =========================================================================


def main(argv: list[str] | None = None) -> int:
    """Tune the escalation threshold on half the pool and report delta r_pb on the rest."""
    parser = argparse.ArgumentParser(description="Selective escalation to judge_c (RQ3)")
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT)
    parser.add_argument("--inference-root", type=Path, default=INFERENCE_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signal",
        choices=["confidence", "disagreement"],
        default="confidence",
        help="Routing signal: judge_a/b self-confidence, or |rubric_a - rubric_b|",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    pool = assemble_pool(args.judge_root, args.inference_root)
    if len(pool) < 10:
        logger.error("Common pool too small (%d) - has judge_c been run?", len(pool))
        return 1
    logger.info("Common pool (a&b&c, unambiguous gold): %d items", len(pool))

    tune_idx, report_idx = split_indices(len(pool), 0.5, args.seed)
    thresholds = [round(t, 2) for t in np.arange(0.0, 1.01, 0.05)]

    def cols(idx: list[int]) -> dict[str, list]:
        return {k: [pool[i][k] for i in idx] for k in ("base", "c", "conf", "rub_a", "rub_b", "gold")}

    t = cols(tune_idx)
    best_t, tune_delta = best_threshold(t["base"], t["c"], t["conf"], t["gold"], thresholds)

    r = cols(report_idx)
    base_rpb = point_biserial(r["base"], r["gold"])
    esc = escalate(r["base"], r["c"], r["conf"], best_t)
    esc_rpb = point_biserial(esc, r["gold"])
    n_esc = sum(
        1 for c, conf in zip(r["c"], r["conf"]) if c is not None and conf is not None and conf < best_t
    )

    # Fixed-budget curve on the report half: escalate only the most-uncertain K%.
    unc = uncertainty_signal(r["conf"], r["rub_a"], r["rub_b"], args.signal)
    curve = budget_curve(r["base"], r["c"], unc, r["gold"], [0.1, 0.2, 0.3, 0.5, 1.0])

    result = {
        "n_pool": len(pool),
        "n_tune": len(tune_idx),
        "n_report": len(report_idx),
        "routing_signal": args.signal,
        "tuned_threshold": {
            "best_threshold": best_t,
            "tune_delta_rpb": round(tune_delta, 4),
            "report_escalated_rpb": round(esc_rpb, 4),
            "report_delta_rpb": round(esc_rpb - base_rpb, 4),
            "report_escalated_fraction": round(n_esc / len(report_idx), 4),
        },
        "report_base_rpb": round(base_rpb, 4),
        "budget_curve": curve,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, args.out.open("w", encoding="utf-8"), indent=2)

    print("\n" + "=" * 64)
    print("SELECTIVE ESCALATION TO judge_c (RQ3, role iii)")
    print("=" * 64)
    print(f"common pool n={result['n_pool']} (tune {result['n_tune']} / report {result['n_report']})")
    print(f"routing signal: {args.signal}")
    print(f"report base r_pb: {base_rpb:.4f}")
    print("\nfixed-budget curve (escalate most-uncertain K%, held-out half):")
    print(f"  {'budget':>7} {'n_esc':>6} {'esc_rpb':>9} {'delta_rpb':>10}")
    for row in curve:
        print(
            f"  {row['fraction']*100:>6.0f}% {row['n_escalated']:>6} "
            f"{row['escalated_rpb']:>9.4f} {row['delta_rpb']:>+10.4f}"
        )
    print(f"\nfree-optimum threshold (conf<{best_t}): escalates "
          f"{result['tuned_threshold']['report_escalated_fraction']*100:.0f}%, "
          f"delta r_pb {result['tuned_threshold']['report_delta_rpb']:+.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
