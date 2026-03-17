"""Compute automated metrics for closed-book experiment results.

Computes Exact Match, ROUGE-L, BLEU-4, and parse rate across all
model x strategy cells, with breakdowns by domain, level, and question type.

Usage:
    python scripts/compute_metrics.py
    python scripts/compute_metrics.py --base-dir outputs/closed_book
    python scripts/compute_metrics.py --models qwen2.5-7b --strategies da,ras
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent

# Question types where EM compares letter labels only
MC_TYPES = {"mcq-4-choices", "mcq-2-choices"}
TF_TYPE = "true_or_false"
# Types where ROUGE-L / BLEU are meaningful (anything non-MC)
TEXT_TYPES = {"open-ended-qa", "filling", "relation_extraction", "true_or_false"}


# =========================================================================
# Normalization helpers
# =========================================================================

def normalize_mc_answer(answer: str | None) -> str:
    """Normalize MC answer to uppercase letter (A-D)."""
    if not answer:
        return ""
    ans = answer.strip().upper()
    if len(ans) == 1 and ans in "ABCD":
        return ans
    # Handle "A) ...", "A. ...", "(A) ..."
    for char in ans:
        if char in "ABCD":
            return char
        if char.isalpha() and char not in "ABCD":
            break
    return ans


def normalize_tf_answer(answer: str | None) -> str:
    """Normalize True/False answer to 'yes' or 'no'."""
    if not answer:
        return ""
    ans = answer.strip().lower()
    if ans in ("yes", "true", "correct", "right"):
        return "yes"
    if ans in ("no", "false", "incorrect", "wrong"):
        return "no"
    return ans


def normalize_text_answer(answer: str | None) -> str:
    """Normalize open-ended answer: lowercase, strip whitespace."""
    if not answer:
        return ""
    return answer.strip().lower()


def get_predicted_answer(record: dict) -> str:
    """Extract the predicted answer from a record (handles SC and non-SC)."""
    strategy = record.get("strategy", "")
    if strategy == "sc" and record.get("sc_result"):
        return record["sc_result"].get("final_answer_normalized", "") or ""
    return record.get("parsed", {}).get("answer_normalized", "") or ""


def get_parse_success(record: dict) -> bool:
    """Check if parsing was successful (handles SC and non-SC)."""
    strategy = record.get("strategy", "")
    if strategy == "sc" and record.get("sc_result"):
        return record["sc_result"].get("sc_success", False)
    return record.get("parsed", {}).get("parse_success", False)


# =========================================================================
# Exact Match
# =========================================================================

def compute_exact_match(predicted: str, gold: str, question_type: str) -> bool:
    """Compute exact match for a single question.

    Args:
        predicted: Normalized predicted answer.
        gold: Gold answer string.
        question_type: Type of question (determines normalization).

    Returns:
        True if exact match, False otherwise.
    """
    if not predicted:
        return False

    if question_type in MC_TYPES:
        pred_norm = normalize_mc_answer(predicted)
        gold_norm = normalize_mc_answer(gold)
        return pred_norm == gold_norm

    if question_type == TF_TYPE:
        pred_norm = normalize_tf_answer(predicted)
        gold_norm = normalize_tf_answer(gold)
        return pred_norm == gold_norm

    # Open-ended, filling, relation_extraction
    pred_norm = normalize_text_answer(predicted)
    gold_norm = normalize_text_answer(gold)
    return pred_norm == gold_norm


# =========================================================================
# ROUGE-L
# =========================================================================

_ROUGE_SCORER = None


def _get_rouge_scorer() -> rouge_scorer.RougeScorer:
    global _ROUGE_SCORER  # noqa: PLW0603
    if _ROUGE_SCORER is None:
        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return _ROUGE_SCORER


def compute_rouge_l(predicted: str, gold: str) -> float:
    """Compute ROUGE-L F1 score for a single text pair.

    Returns:
        ROUGE-L F1 score in [0, 1].
    """
    if not predicted or not gold:
        return 0.0
    scorer = _get_rouge_scorer()
    scores = scorer.score(gold, predicted)
    return scores["rougeL"].fmeasure


# =========================================================================
# BLEU-4
# =========================================================================

def compute_bleu_corpus(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU-4.

    Args:
        predictions: List of predicted texts.
        references: List of reference texts (one per prediction).

    Returns:
        BLEU-4 score in [0, 1] (sacrebleu returns 0-100, we divide by 100).
    """
    if not predictions or not references:
        return 0.0
    bleu = BLEU(effective_order=True)
    # sacrebleu expects references as list of lists
    refs_wrapped = [[r] for r in references]
    result = bleu.corpus_score(predictions, list(zip(*refs_wrapped)))
    return min(result.score / 100.0, 1.0)


# =========================================================================
# Per-cell metrics computation
# =========================================================================

def compute_cell_metrics(records: list[dict]) -> dict[str, Any]:
    """Compute all metrics for a single cell (model x strategy).

    Args:
        records: List of JSONL records for one cell.

    Returns:
        Dict with overall metrics and breakdowns.
    """
    if not records:
        return {}

    model = records[0].get("model", "unknown")
    strategy = records[0].get("strategy", "unknown")
    total = len(records)

    # Accumulators for breakdowns
    groups: dict[str, dict[str, list]] = {
        "overall": {"_all": []},
        "by_domain": defaultdict(list),
        "by_level": defaultdict(list),
        "by_type": defaultdict(list),
    }

    for rec in records:
        predicted = get_predicted_answer(rec)
        gold = rec.get("gold_answer", "")
        qtype = rec.get("question_type", "")
        domain = rec.get("domain", "unknown")
        level = rec.get("details", {}).get("level", "unknown")
        parse_ok = get_parse_success(rec)

        entry = {
            "predicted": predicted,
            "gold": gold,
            "question_type": qtype,
            "parse_success": parse_ok,
            "em": compute_exact_match(predicted, gold, qtype),
        }

        # ROUGE-L only for text types
        if qtype in TEXT_TYPES:
            entry["rouge_l"] = compute_rouge_l(predicted, gold)
        else:
            entry["rouge_l"] = None

        groups["overall"]["_all"].append(entry)
        groups["by_domain"][domain].append(entry)
        groups["by_level"][level].append(entry)
        groups["by_type"][qtype].append(entry)

    def _aggregate(entries: list[dict]) -> dict[str, Any]:
        n = len(entries)
        if n == 0:
            return {"count": 0}

        em_count = sum(1 for e in entries if e["em"])
        parse_count = sum(1 for e in entries if e["parse_success"])

        result: dict[str, Any] = {
            "count": n,
            "exact_match": round(em_count / n, 4),
            "parse_rate": round(parse_count / n, 4),
        }

        # ROUGE-L: only for text-type entries
        rouge_vals = [e["rouge_l"] for e in entries if e["rouge_l"] is not None]
        if rouge_vals:
            result["rouge_l_f1"] = round(sum(rouge_vals) / len(rouge_vals), 4)
            result["rouge_l_count"] = len(rouge_vals)

        # BLEU-4: only for text-type entries
        text_preds = [e["predicted"] for e in entries if e["question_type"] in TEXT_TYPES and e["predicted"]]
        text_golds = [e["gold"] for e in entries if e["question_type"] in TEXT_TYPES and e["predicted"]]
        if text_preds:
            result["bleu_4"] = round(compute_bleu_corpus(text_preds, text_golds), 4)
            result["bleu_4_count"] = len(text_preds)

        return result

    parse_ok_total = sum(1 for e in groups["overall"]["_all"] if e["parse_success"])

    metrics: dict[str, Any] = {
        "model": model,
        "strategy": strategy,
        "total_questions": total,
        "parse_rate": round(parse_ok_total / total, 4),
        "overall": _aggregate(groups["overall"]["_all"]),
        "by_domain": {k: _aggregate(v) for k, v in sorted(groups["by_domain"].items())},
        "by_level": {k: _aggregate(v) for k, v in sorted(groups["by_level"].items())},
        "by_type": {k: _aggregate(v) for k, v in sorted(groups["by_type"].items())},
    }

    return metrics


# =========================================================================
# I/O helpers
# =========================================================================

def load_cell(path: Path) -> list[dict]:
    """Load JSONL records from a cell file."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def discover_cells(base_dir: Path) -> list[tuple[str, str, Path]]:
    """Discover all model/strategy cells in base_dir.

    Returns:
        List of (model_name, strategy, path) tuples.
    """
    cells = []
    if not base_dir.exists():
        return cells
    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            strategy = jsonl.stem
            if strategy.endswith("_metrics"):
                continue
            cells.append((model_dir.name, strategy, jsonl))
    return cells


# =========================================================================
# Summary table
# =========================================================================

def build_summary_table(all_metrics: list[dict]) -> list[dict]:
    """Build a flat summary table from all cell metrics."""
    rows = []
    for m in all_metrics:
        row: dict[str, Any] = {
            "model": m["model"],
            "strategy": m["strategy"],
            "total": m["total_questions"],
            "parse_rate": m["parse_rate"],
            "exact_match": m["overall"]["exact_match"],
        }
        if "rouge_l_f1" in m["overall"]:
            row["rouge_l_f1"] = m["overall"]["rouge_l_f1"]
        if "bleu_4" in m["overall"]:
            row["bleu_4"] = m["overall"]["bleu_4"]
        rows.append(row)
    return rows


def print_summary(summary: list[dict]) -> None:
    """Pretty-print summary table to console."""
    print("\n" + "=" * 85)
    print("CLOSED-BOOK RESULTS (dev.json)")
    print("ROUGE-L and BLEU-4 computed on non-MC questions only (open-ended, T/F, fill, relext)")
    print("=" * 85)
    header = f"{'Model':<22} {'Strategy':<10} {'Parse%':>8} {'EM':>8} {'ROUGE-L*':>10} {'BLEU-4*':>9}"
    print(header)
    print("-" * 85)

    for row in summary:
        rouge = f"{row['rouge_l_f1']:.4f}" if "rouge_l_f1" in row else "     n/a"
        bleu = f"{row['bleu_4']:.4f}" if "bleu_4" in row else "    n/a"
        print(
            f"{row['model']:<22} {row['strategy']:<10} "
            f"{row['parse_rate'] * 100:>7.1f}% "
            f"{row['exact_match']:>8.4f} "
            f"{rouge:>10} "
            f"{bleu:>9}"
        )

    print("-" * 85)

    # Best EM
    best_em = max(summary, key=lambda r: r["exact_match"])
    print(f"Best EM:      {best_em['model']} + {best_em['strategy']} ({best_em['exact_match']:.4f})")

    # Best ROUGE-L
    rouge_rows = [r for r in summary if "rouge_l_f1" in r]
    if rouge_rows:
        best_rouge = max(rouge_rows, key=lambda r: r["rouge_l_f1"])
        print(f"Best ROUGE-L: {best_rouge['model']} + {best_rouge['strategy']} ({best_rouge['rouge_l_f1']:.4f})")

    print("=" * 80)


def print_analysis(all_metrics: list[dict], summary: list[dict]) -> None:
    """Print quick analysis observations."""
    print("\nQUICK ANALYSIS")
    print("-" * 60)

    # 1. Best model overall (avg EM across strategies)
    model_ems: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        model_ems[row["model"]].append(row["exact_match"])
    model_avg = {m: sum(v) / len(v) for m, v in model_ems.items()}
    best_model = max(model_avg, key=model_avg.get)  # type: ignore[arg-type]
    print(f"1. Best model (avg EM): {best_model} ({model_avg[best_model]:.4f})")

    # 2. Best strategy overall (avg EM across models)
    strat_ems: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        strat_ems[row["strategy"]].append(row["exact_match"])
    strat_avg = {s: sum(v) / len(v) for s, v in strat_ems.items()}
    best_strat = max(strat_avg, key=strat_avg.get)  # type: ignore[arg-type]
    print(f"2. Best strategy (avg EM): {best_strat} ({strat_avg[best_strat]:.4f})")

    # 3. RAS vs DA comparison
    da_avg = strat_avg.get("da", 0)
    ras_avg = strat_avg.get("ras", 0)
    delta = ras_avg - da_avg
    direction = "improves" if delta > 0 else "degrades"
    print(f"3. RAS vs DA: RAS {direction} by {abs(delta):.4f} (DA={da_avg:.4f}, RAS={ras_avg:.4f})")

    # 4. Model x strategy interaction
    print("4. Best strategy per model:")
    for model in sorted(model_ems.keys()):
        model_rows = [r for r in summary if r["model"] == model]
        best = max(model_rows, key=lambda r: r["exact_match"])
        print(f"   {model:<22} -> {best['strategy']} (EM={best['exact_match']:.4f})")

    # 5. Llama-3B vs 7B models
    llama_avg = model_avg.get("llama-3.2-3b", 0)
    others = {m: v for m, v in model_avg.items() if m != "llama-3.2-3b"}
    if others:
        avg_7b = sum(others.values()) / len(others)
        gap = avg_7b - llama_avg
        print(f"5. Llama-3B vs 7B avg: gap={gap:.4f} (3B={llama_avg:.4f}, 7B-avg={avg_7b:.4f})")

    # 6. Parse rate issues
    low_parse = [r for r in summary if r["parse_rate"] < 0.95]
    if low_parse:
        print("6. Low parse rate (<95%):")
        for r in low_parse:
            print(f"   {r['model']}/{r['strategy']}: {r['parse_rate']*100:.1f}%")
    else:
        print("6. All cells have parse rate >= 95%")

    print()


# =========================================================================
# Main
# =========================================================================

def main() -> None:
    """Run metrics computation across all cells."""
    parser = argparse.ArgumentParser(description="Compute closed-book metrics")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "closed_book",
        help="Base directory with model subdirectories",
    )
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model filter")
    parser.add_argument("--strategies", type=str, default=None, help="Comma-separated strategy filter")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    model_filter = set(args.models.split(",")) if args.models else None
    strat_filter = set(args.strategies.split(",")) if args.strategies else None

    cells = discover_cells(args.base_dir)
    if model_filter:
        cells = [(m, s, p) for m, s, p in cells if m in model_filter]
    if strat_filter:
        cells = [(m, s, p) for m, s, p in cells if s in strat_filter]

    if not cells:
        logger.error("No cells found in %s", args.base_dir)
        sys.exit(1)

    logger.info("Found %d cells to process", len(cells))

    all_metrics = []
    for model_name, strategy, path in cells:
        logger.info("Computing metrics for %s/%s ...", model_name, strategy)
        records = load_cell(path)
        metrics = compute_cell_metrics(records)

        # Save per-cell metrics
        metrics_path = path.parent / f"{strategy}_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        logger.info("Saved %s", metrics_path)

        all_metrics.append(metrics)

    # Build and save summary table
    summary = build_summary_table(all_metrics)
    summary_path = args.base_dir / "summary_table.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary to %s", summary_path)

    # Print results
    print_summary(summary)
    print_analysis(all_metrics, summary)


if __name__ == "__main__":
    main()
