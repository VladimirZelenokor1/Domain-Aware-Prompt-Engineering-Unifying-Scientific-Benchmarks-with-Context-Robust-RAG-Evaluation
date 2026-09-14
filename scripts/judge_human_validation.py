"""Preliminary single-annotator judge-vs-human rubric validation (RQ3).

The LLM judges are otherwise validated only against automatic signals
(SciKnowEval gold, inter-judge Krippendorff alpha, perturbation tests). This
tool adds a *preliminary, single-annotator* human check: it exports a blind,
shuffled sample of judged answers for a human to re-score on the 0-5 rubric,
then computes judge-vs-human agreement (quadratic-weighted Cohen's kappa with a
bootstrap CI, plus Spearman). This executes the original kappa definition
(judge vs human) that the thesis otherwise lists as future work.

Honest framing for the writeup: a single annotator is weaker than two or more
independent raters; report it as *preliminary* validation, with n and a
confidence interval, not as established expert agreement.

Workflow:
    # 1. export a blind sample (no judge scores shown), shuffled
    python scripts/judge_human_validation.py extract --n 80 --seed 42
    # 2. a human fills the empty `human_rubric` column (0-5) in the CSV
    # 3. compute agreement
    python scripts/judge_human_validation.py score
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import random
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
JUDGE_ROOT = PROJECT_ROOT / "outputs" / "judge"
INFERENCE_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_SAMPLE = PROJECT_ROOT / "outputs" / "judge_human_sample.csv"
DEFAULT_KEY = PROJECT_ROOT / "outputs" / "judge_human_key.json"
DEFAULT_AGREEMENT = PROJECT_ROOT / "outputs" / "judge_human_agreement.json"
MIN_RATING, MAX_RATING = 0, 5
CSV_FIELDS = ["item_id", "track", "model", "cell", "question", "answer", "gold", "human_rubric"]


# =========================================================================
# Pure stats helpers
# =========================================================================


def quadratic_weighted_kappa(
    y1: list[int], y2: list[int], min_rating: int = MIN_RATING, max_rating: int = MAX_RATING
) -> float:
    """Quadratic-weighted Cohen's kappa for ordinal ratings.

    Args:
        y1: First rater's integer ratings.
        y2: Second rater's integer ratings (same length as ``y1``).
        min_rating: Lowest possible rating.
        max_rating: Highest possible rating.

    Returns:
        Quadratic-weighted kappa in [-1, 1]; 0.0 if a rater is constant
        (expected disagreement is zero).
    """
    a = np.asarray(y1, dtype=int)
    b = np.asarray(y2, dtype=int)
    n = len(a)
    k = max_rating - min_rating + 1
    observed = np.zeros((k, k), dtype=float)
    for x, y in zip(a, b):
        observed[x - min_rating, y - min_rating] += 1
    weights = np.zeros((k, k), dtype=float)
    for i in range(k):
        for j in range(k):
            weights[i, j] = ((i - j) ** 2) / ((k - 1) ** 2)
    hist1 = observed.sum(axis=1)
    hist2 = observed.sum(axis=0)
    expected = np.outer(hist1, hist2) / n
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return 0.0
    return 1.0 - float((weights * observed).sum()) / denom


def bootstrap_kappa_ci(
    y1: list[int],
    y2: list[int],
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
    min_rating: int = MIN_RATING,
    max_rating: int = MAX_RATING,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the quadratic-weighted kappa.

    Args:
        y1: First rater's ratings.
        y2: Second rater's ratings.
        n_boot: Bootstrap resamples.
        seed: RNG seed (deterministic).
        ci: Central interval mass (e.g. 0.95).
        min_rating: Lowest possible rating.
        max_rating: Highest possible rating.

    Returns:
        Tuple (lower, upper) percentile bounds.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(y1, dtype=int)
    b = np.asarray(y2, dtype=int)
    n = len(a)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        stats.append(quadratic_weighted_kappa(a[idx], b[idx], min_rating, max_rating))
    lo = float(np.percentile(stats, (1 - ci) / 2 * 100))
    hi = float(np.percentile(stats, (1 + ci) / 2 * 100))
    return lo, hi


def sample_items(items: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Deterministically shuffle ``items`` and take the first ``n``."""
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled[:n]


# =========================================================================
# Extraction (build the blind annotation sample)
# =========================================================================


def _judge_relative(path: Path, judge_root: Path, judge_id: str) -> Path:
    """track/model/cell.jsonl relative path of a judge cell file."""
    return path.relative_to(judge_root / judge_id)


def _answer_text(rec: dict) -> str:
    parsed = rec.get("parsed") or {}
    if rec.get("strategy") == "sc" and rec.get("sc_result"):
        return rec["sc_result"].get("final_answer") or parsed.get("answer") or ""
    return parsed.get("answer") or rec.get("raw_response") or ""


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def collect_judged_items(
    judge_root: Path, inference_root: Path, primary: str = "judge_a", secondary: str = "judge_b"
) -> list[dict]:
    """Join primary-judge rubric records to inference answers (+ secondary rubric).

    Walks every primary-judge cell file, mirrors its path into the inference
    tree for question/answer/gold, and into the secondary-judge tree for that
    judge's rubric.

    Returns:
        List of item dicts (item_id, track, model, cell, question, answer, gold,
        judge_a, judge_b).
    """
    items: list[dict] = []
    pri_files = sorted(glob.glob(str(judge_root / primary / "**" / "*.jsonl"), recursive=True))
    for pf in pri_files:
        pf_path = Path(pf)
        rel = _judge_relative(pf_path, judge_root, primary)
        inf_path = inference_root / rel
        if not inf_path.exists():
            logger.warning("No inference file for %s (skipping)", rel)
            continue
        inf_map = {r.get("question_id"): r for r in _load_jsonl(inf_path)}
        sec_path = judge_root / secondary / rel
        sec_map = (
            {r.get("question_id"): r.get("rubric") for r in _load_jsonl(sec_path)}
            if sec_path.exists()
            else {}
        )
        track = rel.parts[0]
        model = rel.parts[1]
        cell = pf_path.stem
        for jrec in _load_jsonl(pf_path):
            qid = jrec.get("question_id")
            inf = inf_map.get(qid)
            if inf is None:
                continue
            items.append(
                {
                    "item_id": f"{track}/{model}/{cell}/{qid}",
                    "track": track,
                    "model": model,
                    "cell": cell,
                    "question": inf.get("question", ""),
                    "answer": _answer_text(inf),
                    "gold": inf.get("gold_answer", ""),
                    "judge_a": jrec.get("rubric"),
                    "judge_b": sec_map.get(qid),
                }
            )
    return items


def write_sample(items: list[dict], sample_path: Path, key_path: Path) -> None:
    """Write the blind annotation CSV (no judge scores) and the hidden key JSON."""
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    with sample_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for it in items:
            writer.writerow(
                {
                    "item_id": it["item_id"],
                    "track": it["track"],
                    "model": it["model"],
                    "cell": it["cell"],
                    "question": it["question"],
                    "answer": it["answer"],
                    "gold": it["gold"],
                    "human_rubric": "",  # annotator fills 0-5
                }
            )
    key = {
        it["item_id"]: {"judge_a": it["judge_a"], "judge_b": it["judge_b"]}
        for it in items
    }
    with key_path.open("w", encoding="utf-8") as fh:
        json.dump(key, fh, indent=2)


# =========================================================================
# Scoring
# =========================================================================


def _spearman(y1: list[float], y2: list[float]) -> float:
    from scipy.stats import spearmanr  # noqa: PLC0415

    rho, _ = spearmanr(y1, y2)
    return float(rho)


def score_agreement(
    sample_path: Path, key_path: Path, n_boot: int, seed: int
) -> dict:
    """Compute judge-vs-human agreement from the filled CSV and the hidden key."""
    key = json.load(key_path.open(encoding="utf-8"))
    human: list[int] = []
    ja: list[int] = []
    jb: list[int] = []
    n_blank = 0
    with sample_path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            val = (row.get("human_rubric") or "").strip()
            if val == "":
                n_blank += 1
                continue
            scores = key.get(row["item_id"], {})
            if scores.get("judge_a") is None:
                continue
            h = int(round(float(val)))
            human.append(h)
            ja.append(int(round(float(scores["judge_a"]))))
            jb.append(scores.get("judge_b"))

    paired_a = [(h, a) for h, a in zip(human, ja)]
    paired_b = [(h, b) for h, b in zip(human, jb) if b is not None]
    h_a = [h for h, _ in paired_a]
    a_only = [a for _, a in paired_a]
    h_b = [h for h, _ in paired_b]
    b_only = [int(round(float(b))) for _, b in paired_b]
    # per-item mean judge rubric (mean of a,b where b exists, else a), aligned to human
    mean_judge = [(a + b) / 2 if b is not None else a for a, b in zip(ja, jb)]

    result: dict = {
        "annotator": "single (preliminary)",
        "n_scored": len(human),
        "n_blank": n_blank,
        "judge_a": {
            "n": len(h_a),
            "quadratic_weighted_kappa": round(quadratic_weighted_kappa(h_a, a_only), 4),
            "kappa_ci95": [round(x, 4) for x in bootstrap_kappa_ci(h_a, a_only, n_boot, seed)],
            "spearman": round(_spearman(h_a, a_only), 4) if len(h_a) > 2 else None,
        },
    }
    if h_b:
        result["judge_b"] = {
            "n": len(h_b),
            "quadratic_weighted_kappa": round(quadratic_weighted_kappa(h_b, b_only), 4),
            "kappa_ci95": [round(x, 4) for x in bootstrap_kappa_ci(h_b, b_only, n_boot, seed)],
            "spearman": round(_spearman(h_b, b_only), 4) if len(h_b) > 2 else None,
        }
    if len(human) > 2:
        result["mean_judge_spearman"] = round(_spearman(human, mean_judge), 4)
    return result


# =========================================================================
# CLI
# =========================================================================


def main(argv: list[str] | None = None) -> int:
    """CLI: ``extract`` a blind sample, then ``score`` judge-vs-human agreement."""
    parser = argparse.ArgumentParser(description="Preliminary judge-vs-human rubric validation")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    sub = parser.add_subparsers(dest="mode", required=True)

    pe = sub.add_parser("extract", help="Export a blind, shuffled annotation sample")
    pe.add_argument("--judge-root", type=Path, default=JUDGE_ROOT)
    pe.add_argument("--inference-root", type=Path, default=INFERENCE_ROOT)
    pe.add_argument("--n", type=int, default=80, help="Sample size to annotate")
    pe.add_argument("--seed", type=int, default=42)
    pe.add_argument("--out", type=Path, default=DEFAULT_SAMPLE)
    pe.add_argument("--key-out", type=Path, default=DEFAULT_KEY)

    ps = sub.add_parser("score", help="Compute kappa/CI from the filled CSV")
    ps.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    ps.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ps.add_argument("--n-boot", type=int, default=1000)
    ps.add_argument("--seed", type=int, default=42)
    ps.add_argument("--out", type=Path, default=DEFAULT_AGREEMENT)

    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    if args.mode == "extract":
        items = collect_judged_items(args.judge_root, args.inference_root)
        if not items:
            logger.error("No judged items found under %s", args.judge_root)
            return 1
        chosen = sample_items(items, args.n, args.seed)
        write_sample(chosen, args.out, args.key_out)
        logger.info(
            "Wrote %d blind items to %s (judge scores hidden in %s). "
            "Fill the human_rubric column (0-5), then run `score`.",
            len(chosen), args.out, args.key_out,
        )
        return 0

    result = score_agreement(args.sample, args.key, args.n_boot, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(result, args.out.open("w", encoding="utf-8"), indent=2)
    logger.info("Wrote %s", args.out)

    print("\n" + "=" * 60)
    print("JUDGE vs HUMAN (preliminary, single annotator)")
    print("=" * 60)
    print(f"n scored: {result['n_scored']}  (blank: {result['n_blank']})")
    for jid in ("judge_a", "judge_b"):
        if jid in result:
            r = result[jid]
            ci = r["kappa_ci95"]
            print(
                f"  {jid}: QWK = {r['quadratic_weighted_kappa']} "
                f"(95% CI [{ci[0]}, {ci[1]}], n={r['n']}); Spearman = {r['spearman']}"
            )
    if "mean_judge_spearman" in result:
        print(f"  mean judge: Spearman = {result['mean_judge_spearman']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
