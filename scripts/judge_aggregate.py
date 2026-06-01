"""Judge output aggregation - inter-rater reliability and calibration metrics.

Aggregates judge JSONL outputs from outputs/judge/{judge_id}/ into:
    - Krippendorff's alpha (inter-rater reliability)
    - Expected Calibration Error (ECE) on MCQ subset
    - Wilcoxon signed-rank test for perturbation analysis
    - Per-domain and per-judge breakdowns

Usage:
    python scripts/judge_aggregate.py \\
        --judge-dir outputs/judge \\
        --source-dirs outputs/closed_book_main outputs/rag_main \\
        --config configs/judge.yaml
    # Output: outputs/judge/aggregate_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    import krippendorff as _krippendorff_lib

    _KRIPPENDORFF_AVAILABLE = True
except ImportError:
    _KRIPPENDORFF_AVAILABLE = False
    _krippendorff_lib = None  # type: ignore[assignment]

from scipy.stats import wilcoxon as _scipy_wilcoxon

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent


# =========================================================================
# I/O helpers
# =========================================================================


def read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to the .jsonl file.

    Returns:
        List of parsed records. Empty list if file is empty or missing.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _cell_key(record: dict) -> str:
    """Build the cell-level identifier for a record.

    The inter-rater unit is a single judged answer = (cell, question). The
    same ``question_id`` is judged in every matrix cell (model x strategy x
    retriever x noise), so joining on ``question_id`` alone collapses 216
    cells into one (last-write-wins). ``_cell`` is the source-relative path
    (``<source_root>/<model>/<file>``) attached at load time; combined with
    ``question_id`` it uniquely identifies the answer.

    Args:
        record: A judge or source record carrying ``_cell`` and ``question_id``.

    Returns:
        Composite key ``"<cell>::<question_id>"``.
    """
    return f"{record.get('_cell', '')}::{record.get('question_id', '')}"


def _load_all_judge_records(judge_dir: Path) -> list[dict]:
    """Load every JSONL record from all judge sub-directories.

    Each record is tagged with ``_cell`` (the source-relative path, with the
    leading ``judge_id`` component stripped) so judge and source records can
    be joined per answer rather than per ``question_id``.

    Args:
        judge_dir: Root directory containing {judge_id}/{...}/*.jsonl files.

    Returns:
        Flat list of all judge records across all judges.
    """
    records: list[dict] = []
    for jsonl_path in sorted(judge_dir.rglob("*.jsonl")):
        if jsonl_path.name == "aggregate_stats.json":
            continue
        # judge_dir/<judge_id>/<source_root>/<model>/<file> -> "<source_root>/<model>/<file>"
        rel_parts = jsonl_path.relative_to(judge_dir).parts[1:]
        cell = "/".join(rel_parts)
        for rec in read_jsonl(jsonl_path):
            rec["_cell"] = cell
            records.append(rec)
    logger.debug("Loaded %d total judge records from %s", len(records), judge_dir)
    return records


def _load_all_source_records(source_dirs: list[Path]) -> list[dict]:
    """Load every JSONL record from all source directories.

    Each record is tagged with ``_cell`` (path relative to the parent of the
    source dir, i.e. ``<source_root>/<model>/<file>``) so it matches the
    ``_cell`` attached to judge records.

    Args:
        source_dirs: List of root directories containing source inference outputs.

    Returns:
        Flat list of all source records.
    """
    records: list[dict] = []
    for source_dir in source_dirs:
        for jsonl_path in sorted(source_dir.rglob("*.jsonl")):
            cell = "/".join(jsonl_path.relative_to(source_dir.parent).parts)
            for rec in read_jsonl(jsonl_path):
                rec["_cell"] = cell
                records.append(rec)
    logger.debug(
        "Loaded %d source records from %s dirs", len(records), len(source_dirs)
    )
    return records


# =========================================================================
# Krippendorff alpha
# =========================================================================


def compute_krippendorff_alpha(
    ratings: dict[str, dict[str, int]],
    level: str = "ordinal",
) -> float:
    """Compute Krippendorff's alpha from rater -> {item_id: rating} dicts.

    Uses the krippendorff library when available; falls back to percent
    agreement scaled to [-1, 1] otherwise.

    Args:
        ratings: Dict of {rater_id: {question_id: rubric_score}}.
        level: Measurement level ("ordinal", "interval", "nominal").

    Returns:
        Alpha value (-1 to 1, 1 = perfect agreement).

    Raises:
        RuntimeError: If krippendorff is not installed and fallback cannot run.
    """
    rater_ids = sorted(ratings.keys())
    all_items = sorted({qid for r in ratings.values() for qid in r})

    if not rater_ids or not all_items:
        return 0.0

    # Build reliability matrix: rows = raters, cols = items; missing = NaN
    matrix = np.full((len(rater_ids), len(all_items)), np.nan)
    for row_idx, rater_id in enumerate(rater_ids):
        for col_idx, item_id in enumerate(all_items):
            if item_id in ratings[rater_id]:
                matrix[row_idx, col_idx] = ratings[rater_id][item_id]

    if _KRIPPENDORFF_AVAILABLE:
        try:
            alpha: float = float(
                _krippendorff_lib.alpha(
                    reliability_data=matrix,
                    level_of_measurement=level,
                )
            )
            return alpha
        except Exception as exc:
            logger.warning(
                "krippendorff.alpha failed (%s); falling back to percent agreement", exc
            )

    # Fallback: simple percent agreement scaled to [-1, 1]
    logger.warning(
        "krippendorff not installed - using percent-agreement fallback. "
        "Install with: pip install krippendorff"
    )
    agreements = 0
    comparisons = 0
    for col_idx in range(len(all_items)):
        col_vals = matrix[:, col_idx]
        observed = col_vals[~np.isnan(col_vals)]
        if len(observed) < 2:
            continue
        for i in range(len(observed)):
            for j in range(i + 1, len(observed)):
                comparisons += 1
                if observed[i] == observed[j]:
                    agreements += 1
    if comparisons == 0:
        return 0.0
    pct = agreements / comparisons
    return float(2 * pct - 1)


# =========================================================================
# ECE
# =========================================================================


def compute_ece(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error.

    Bins predictions by confidence, then computes the weighted average of
    |avg_confidence - avg_accuracy| per bin.

    Args:
        confidences: Array of self_confidence values (0-1).
        correctness: Binary array (1 if model answer correct, 0 otherwise).
        n_bins: Number of equal-width bins covering [0, 1].

    Returns:
        ECE value (0 = perfectly calibrated).
    """
    if len(confidences) == 0:
        return 0.0

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)

    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        bin_size = int(mask.sum())
        if bin_size == 0:
            continue
        avg_conf = float(confidences[mask].mean())
        avg_acc = float(correctness[mask].mean())
        ece += (bin_size / n) * abs(avg_conf - avg_acc)

    return float(ece)


# =========================================================================
# Wilcoxon signed-rank test
# =========================================================================


def compute_wilcoxon(
    scores_before: np.ndarray,
    scores_after: np.ndarray,
) -> dict[str, Any]:
    """Wilcoxon signed-rank test for paired perturbation analysis.

    Args:
        scores_before: Scores in the baseline condition.
        scores_after: Scores in the perturbed condition.

    Returns:
        Dict with keys:
            - statistic (float): Wilcoxon test statistic.
            - p_value (float): Two-sided p-value.
            - significant (bool): True if p_value < 0.05.
    """
    if np.array_equal(scores_before, scores_after):
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    stat, p_val = _scipy_wilcoxon(scores_before, scores_after)
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "significant": bool(p_val < 0.05),
    }


# =========================================================================
# Per-domain breakdown
# =========================================================================


def aggregate_per_domain(
    judge_records: list[dict],
    source_records: list[dict],
) -> dict[str, dict[str, Any]]:
    """Per-domain breakdown of rubric, faithfulness, and coverage.

    Joins judge records to source records on question_id to obtain domain.

    Args:
        judge_records: List of judge output dicts (with rubric, faithfulness, coverage).
        source_records: List of source inference dicts (with question_id, domain).

    Returns:
        Mapping of {domain: {avg_rubric, avg_faithfulness, avg_coverage, count}}.
    """
    domain_map: dict[str, str] = {
        rec["question_id"]: rec["domain"]
        for rec in source_records
        if "question_id" in rec and "domain" in rec
    }

    buckets: dict[str, dict[str, list[float]]] = {}
    for rec in judge_records:
        qid = rec.get("question_id", "")
        domain = domain_map.get(qid)
        if domain is None:
            continue
        if domain not in buckets:
            buckets[domain] = {"rubric": [], "faithfulness": [], "coverage": []}
        buckets[domain]["rubric"].append(float(rec.get("rubric", 0)))
        faith = rec.get("faithfulness")
        if faith is not None:
            buckets[domain]["faithfulness"].append(float(faith))
        cov = rec.get("coverage")
        if cov is not None:
            buckets[domain]["coverage"].append(float(cov))

    result: dict[str, dict[str, Any]] = {}
    for domain, vals in buckets.items():
        result[domain] = {
            "avg_rubric": float(np.mean(vals["rubric"])) if vals["rubric"] else 0.0,
            "avg_faithfulness": float(np.mean(vals["faithfulness"]))
            if vals["faithfulness"]
            else None,
            "avg_coverage": float(np.mean(vals["coverage"]))
            if vals["coverage"]
            else None,
            "count": len(vals["rubric"]),
        }

    return result


# =========================================================================
# Per-judge stats
# =========================================================================


def _aggregate_per_judge(judge_records: list[dict]) -> dict[str, dict[str, Any]]:
    """Compute per-judge average metrics.

    Args:
        judge_records: All judge records across all judges.

    Returns:
        Mapping of {judge_id: {avg_rubric, avg_faithfulness, avg_coverage, avg_self_confidence, count}}.
    """
    buckets: dict[str, dict[str, list[float]]] = {}
    for rec in judge_records:
        jid = rec.get("judge_id", "unknown")
        if jid not in buckets:
            buckets[jid] = {
                "rubric": [],
                "faithfulness": [],
                "coverage": [],
                "self_confidence": [],
            }
        buckets[jid]["rubric"].append(float(rec.get("rubric", 0)))
        faith = rec.get("faithfulness")
        if faith is not None:
            buckets[jid]["faithfulness"].append(float(faith))
        cov = rec.get("coverage")
        if cov is not None:
            buckets[jid]["coverage"].append(float(cov))
        conf = rec.get("self_confidence")
        if conf is not None:
            buckets[jid]["self_confidence"].append(float(conf))

    result: dict[str, dict[str, Any]] = {}
    for jid, vals in buckets.items():
        result[jid] = {
            "avg_rubric": float(np.mean(vals["rubric"])) if vals["rubric"] else 0.0,
            "avg_faithfulness": float(np.mean(vals["faithfulness"]))
            if vals["faithfulness"]
            else None,
            "avg_coverage": float(np.mean(vals["coverage"]))
            if vals["coverage"]
            else None,
            "avg_self_confidence": float(np.mean(vals["self_confidence"]))
            if vals["self_confidence"]
            else None,
            "count": len(vals["rubric"]),
        }
    return result


# =========================================================================
# ECE on MCQ subset
# =========================================================================


def _compute_ece_mcq(
    judge_records: list[dict],
    source_records: list[dict],
) -> tuple[float, int]:
    """Compute ECE restricted to MCQ questions.

    Pairs ``self_confidence`` from each judge record with the binary
    correctness of the matching source answer. The join is per answer
    (``_cell`` + ``question_id``), not per ``question_id``, so the
    confidence/correctness pair refers to the exact answer that was judged.
    Correctness reuses ``compute_metrics`` extraction/normalisation
    (``get_predicted_answer`` + ``compute_exact_match``) rather than a naive
    string compare, so MCQ letters are normalised consistently with the
    accuracy tables.

    Args:
        judge_records: All judge output records (tagged with ``_cell``).
        source_records: All source inference records (tagged with ``_cell``).

    Returns:
        Tuple of (ECE value, number of MCQ pairs); (0.0, 0) if none found.
    """
    from compute_metrics import (  # noqa: PLC0415
        compute_exact_match,
        get_predicted_answer,
    )

    source_map: dict[str, dict] = {
        _cell_key(rec): rec for rec in source_records if "question_id" in rec
    }

    confidences: list[float] = []
    correctness: list[float] = []

    for rec in judge_records:
        src = source_map.get(_cell_key(rec))
        if src is None:
            continue
        qtype = src.get("question_type", "")
        if "mcq" not in qtype:
            continue
        conf = rec.get("self_confidence")
        if conf is None:
            continue
        predicted = get_predicted_answer(src)
        gold = src.get("gold_answer", "")
        correct = 1.0 if compute_exact_match(predicted, gold, qtype) else 0.0
        confidences.append(float(conf))
        correctness.append(correct)

    if not confidences:
        return 0.0, 0

    ece = compute_ece(np.array(confidences), np.array(correctness))
    return ece, len(confidences)


# =========================================================================
# Wilcoxon for perturbation classes
# =========================================================================


def _compute_perturbation_wilcoxon(
    judge_records: list[dict],
    source_records: list[dict],
    class_key: str,
) -> dict[str, Any]:
    """Compute Wilcoxon for a perturbation class (noise_level low vs high).

    Splits source records into two groups by noise_level threshold (0.5)
    and compares rubric scores between the two groups using matched question_ids.

    Args:
        judge_records: Judge output records with rubric scores.
        source_records: Source inference records with noise_level field.
        class_key: Identifier for this perturbation class (for logging).

    Returns:
        Dict with statistic, p_value, significant; or zeros if insufficient data.
    """
    source_map: dict[str, dict] = {
        rec["question_id"]: rec for rec in source_records if "question_id" in rec
    }
    judge_map: dict[str, float] = {
        rec["question_id"]: float(rec.get("rubric", 0))
        for rec in judge_records
        if "question_id" in rec
    }

    low_noise: list[float] = []
    high_noise: list[float] = []

    for qid, src in source_map.items():
        noise_level = src.get("noise_level")
        if noise_level is None:
            continue
        rubric = judge_map.get(qid)
        if rubric is None:
            continue
        if float(noise_level) < 0.5:
            low_noise.append(rubric)
        else:
            high_noise.append(rubric)

    min_len = min(len(low_noise), len(high_noise))
    if min_len < 2:
        logger.debug(
            "Insufficient paired samples for Wilcoxon %s (low=%d, high=%d)",
            class_key,
            len(low_noise),
            len(high_noise),
        )
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    scores_before = np.array(low_noise[:min_len])
    scores_after = np.array(high_noise[:min_len])
    return compute_wilcoxon(scores_before, scores_after)


# =========================================================================
# Main aggregation
# =========================================================================


def aggregate_judge_outputs(
    judge_dir: Path,
    source_dirs: list[Path],
    config: dict,
) -> dict[str, Any]:
    """Main aggregation function over all judge outputs.

    Reads all judge JSONL files from judge_dir and all source JSONL files
    from source_dirs, then computes inter-rater reliability, calibration,
    and domain/judge breakdowns.

    Args:
        judge_dir: Root directory containing judge outputs per judge_id.
        source_dirs: List of source inference output directories.
        config: Optional config dict (currently unused; reserved for future params).

    Returns:
        Dict with keys:
            - krippendorff_ab: Alpha between judge_a and judge_b.
            - krippendorff_abc: Alpha between judge_a, judge_b, judge_c (calibration subset).
            - ece_mcq: ECE on MCQ subset.
            - wilcoxon_class1: Wilcoxon result for noise perturbation class 1.
            - wilcoxon_class2: Wilcoxon result for noise perturbation class 2.
            - per_domain: Domain breakdown dict.
            - per_judge: Per-judge metrics dict.
    """
    all_judge_records = _load_all_judge_records(judge_dir)
    all_source_records = _load_all_source_records(source_dirs)

    logger.info(
        "Aggregating %d judge records against %d source records",
        len(all_judge_records),
        len(all_source_records),
    )

    # Group judge records by judge_id -> {cell::question_id: rubric}. The
    # composite key keeps every judged answer distinct (see _cell_key);
    # joining on question_id alone would collapse all matrix cells.
    judge_ratings: dict[str, dict[str, int]] = {}
    for rec in all_judge_records:
        jid = rec.get("judge_id", "unknown")
        item = _cell_key(rec)
        rubric = int(rec.get("rubric", 0))
        if jid not in judge_ratings:
            judge_ratings[jid] = {}
        judge_ratings[jid][item] = rubric

    # Krippendorff alpha between judge_a and judge_b
    ratings_ab: dict[str, dict[str, int]] = {
        jid: ratings
        for jid, ratings in judge_ratings.items()
        if jid in {"judge_a", "judge_b"}
    }
    krippendorff_ab: float = (
        compute_krippendorff_alpha(ratings_ab) if len(ratings_ab) >= 2 else 0.0
    )

    # Krippendorff alpha among judge_a, judge_b, judge_c (all three)
    ratings_abc: dict[str, dict[str, int]] = {
        jid: ratings
        for jid, ratings in judge_ratings.items()
        if jid in {"judge_a", "judge_b", "judge_c"}
    }
    krippendorff_abc: float = (
        compute_krippendorff_alpha(ratings_abc) if len(ratings_abc) >= 2 else 0.0
    )

    # Number of answers jointly rated by judge_a and judge_b (the alpha unit).
    items_a = set(judge_ratings.get("judge_a", {}))
    items_b = set(judge_ratings.get("judge_b", {}))
    n_ab_items = len(items_a & items_b)

    # ECE on MCQ subset
    ece_mcq, ece_mcq_n = _compute_ece_mcq(all_judge_records, all_source_records)

    # Wilcoxon for two perturbation classes
    wilcoxon_class1 = _compute_perturbation_wilcoxon(
        all_judge_records, all_source_records, class_key="class1"
    )
    wilcoxon_class2 = _compute_perturbation_wilcoxon(
        all_judge_records, all_source_records, class_key="class2"
    )

    # Per-domain breakdown
    per_domain = aggregate_per_domain(all_judge_records, all_source_records)

    # Per-judge stats
    per_judge = _aggregate_per_judge(all_judge_records)

    return {
        "krippendorff_ab": krippendorff_ab,
        "krippendorff_abc": krippendorff_abc,
        "n_ab_items": n_ab_items,
        "ece_mcq": ece_mcq,
        "ece_mcq_n": ece_mcq_n,
        "wilcoxon_class1": wilcoxon_class1,
        "wilcoxon_class2": wilcoxon_class2,
        "per_domain": per_domain,
        "per_judge": per_judge,
    }


# =========================================================================
# CLI
# =========================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate judge outputs into inter-rater reliability and calibration metrics."
    )
    parser.add_argument(
        "--judge-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "judge",
        help="Root directory of judge outputs (default: outputs/judge).",
    )
    parser.add_argument(
        "--source-dirs",
        type=Path,
        nargs="+",
        default=[
            PROJECT_ROOT / "outputs" / "closed_book_main",
            PROJECT_ROOT / "outputs" / "rag_main",
        ],
        help="Source inference output directories.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "judge.yaml",
        help="Path to judge config YAML.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: {judge_dir}/aggregate_stats.json).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the judge aggregation CLI."""
    args = _parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config: dict = {}
    if args.config.exists():
        with args.config.open() as fh:
            config = yaml.safe_load(fh) or {}
        logger.debug("Loaded config from %s", args.config)
    else:
        logger.warning("Config not found at %s; using defaults.", args.config)

    stats = aggregate_judge_outputs(
        judge_dir=args.judge_dir,
        source_dirs=args.source_dirs,
        config=config,
    )

    output_path = args.output or (args.judge_dir / "aggregate_stats.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(stats, fh, indent=2)

    logger.info("Aggregate stats written to %s", output_path)


if __name__ == "__main__":
    main()
