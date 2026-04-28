"""Stratified sampling of QASPER questions for Track B experiments.

Downloads QASPER dataset, flattens questions, classifies answer types,
and produces a stratified sample compatible with run_rag_inference.

Usage:
    python scripts/sample_qasper.py --target 600 --seed 42
    python scripts/sample_qasper.py --target 600 --seed 42 --from-flat data/qasper/all_questions.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "qasper" / "sample.json"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "qasper_sampled_stratification.json"


# =========================================================================
# Answer type classification
# =========================================================================


def classify_answer_type(answer: dict) -> str:
    """Classify a single QASPER answer annotation into an answer type.

    Priority: unanswerable > yes_no > extractive > free_form > unknown.

    Args:
        answer: QASPER answer annotation dict with keys
            unanswerable, yes_no, extractive_spans, free_form_answer.

    Returns:
        One of: 'unanswerable', 'yes_no', 'extractive', 'free_form', 'unknown'.
    """
    if answer.get("unanswerable"):
        return "unanswerable"
    if answer.get("yes_no") is not None:
        return "yes_no"
    if answer.get("extractive_spans"):
        return "extractive"
    free_form = answer.get("free_form_answer", "")
    if free_form and free_form.strip():
        return "free_form"
    return "unknown"


# =========================================================================
# Question flattening
# =========================================================================


def _extract_answer_text(answer: dict, answer_type: str) -> str:
    """Extract answer text from annotation based on type."""
    if answer_type == "unanswerable":
        return ""
    if answer_type == "yes_no":
        return "Yes" if answer.get("yes_no") else "No"
    if answer_type == "extractive":
        spans = answer.get("extractive_spans", [])
        return " ".join(spans) if spans else ""
    if answer_type == "free_form":
        return answer.get("free_form_answer", "")
    return ""


def flatten_questions(
    dataset: dict[str, list[dict]],
) -> list[dict]:
    """Flatten QASPER dataset into a list of question records.

    Args:
        dataset: Dict of split_name -> list of paper dicts (QASPER format).

    Returns:
        List of flat question dicts with keys: paper_id, question_id,
        question, answer_type, answer_text, split.
    """
    flat: list[dict] = []

    for split_name, papers in dataset.items():
        for paper in papers:
            paper_id = paper["id"]
            qas = paper["qas"]
            questions = qas["question"]
            question_ids = qas["question_id"]
            answers_list = qas["answers"]["answer"]

            for q_text, q_id, answer_annotations in zip(
                questions, question_ids, answers_list
            ):
                # Use first annotator's answer
                if not answer_annotations:
                    continue
                answer = answer_annotations[0]
                answer_type = classify_answer_type(answer)
                answer_text = _extract_answer_text(answer, answer_type)

                flat.append(
                    {
                        "paper_id": paper_id,
                        "question_id": q_id,
                        "question": q_text,
                        "answer_type": answer_type,
                        "answer_text": answer_text,
                        "split": split_name,
                    }
                )

    return flat


# =========================================================================
# Stratified sampling
# =========================================================================


def sample_qasper(
    questions: list[dict],
    target: int,
    seed: int = 42,
) -> list[dict]:
    """Stratified sample by answer_type, output in SciKnowEval format.

    Args:
        questions: Flat question list from flatten_questions.
        target: Desired sample size.
        seed: Random seed for reproducibility.

    Returns:
        List of sampled records in SciKnowEval-compatible format.
    """
    if target >= len(questions):
        sampled_flat = list(questions)
    else:
        # Group by answer_type
        strata: dict[str, list[int]] = defaultdict(list)
        for i, q in enumerate(questions):
            strata[q["answer_type"]].append(i)

        rng = random.Random(seed)
        total = len(questions)
        kept: list[int] = []

        for atype in sorted(strata.keys()):
            indices = strata[atype]
            # Proportional allocation, at least 1 per stratum
            quota = max(1, round(len(indices) / total * target))
            quota = min(quota, len(indices))
            kept.extend(rng.sample(indices, quota))

        # If over target, trim; if under, pad from remaining
        if len(kept) > target:
            rng.shuffle(kept)
            kept = kept[:target]
        elif len(kept) < target:
            used = set(kept)
            remaining = [i for i in range(len(questions)) if i not in used]
            rng.shuffle(remaining)
            kept.extend(remaining[: target - len(kept)])

        kept.sort()
        sampled_flat = [questions[i] for i in kept]

    # Convert to SciKnowEval-compatible format
    output: list[dict] = []
    for q in sampled_flat:
        output.append(
            {
                "question": q["question"],
                "answer": q["answer_text"],
                "type": q["answer_type"],
                "domain": "CS",
                "details": {
                    "paper_id": q["paper_id"],
                    "question_id": q["question_id"],
                    "answer_type": q["answer_type"],
                    "source": "qasper",
                },
                "answerKey": "",
                "choices": {"text": [], "label": []},
            }
        )

    return output


# =========================================================================
# Reporting
# =========================================================================


def build_stratification_report(
    source: list[dict],
    sample: list[dict],
    target: int,
    seed: int,
) -> dict:
    """Build a stratification report for the QASPER sample.

    Args:
        source: Full flattened question list.
        sample: Sampled records (SciKnowEval format).
        target: Requested target size.
        seed: Random seed used.

    Returns:
        Report dict with source/sample breakdown by answer_type.
    """
    source_types: dict[str, int] = defaultdict(int)
    for q in source:
        source_types[q["answer_type"]] += 1

    sample_types: dict[str, int] = defaultdict(int)
    for q in sample:
        sample_types[q.get("type", q.get("answer_type", "unknown"))] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "target": target,
        "source_total": len(source),
        "sample_total": len(sample),
        "source_by_answer_type": dict(sorted(source_types.items())),
        "sample_by_answer_type": dict(sorted(sample_types.items())),
    }


# =========================================================================
# CLI
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Stratified sampling of QASPER questions for Track B",
    )
    parser.add_argument("--target", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--from-flat",
        type=Path,
        default=None,
        help="Pre-flattened questions JSON (skip HF download)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.from_flat is not None:
        logger.info("Loading pre-flattened questions from %s", args.from_flat)
        with open(args.from_flat, encoding="utf-8") as f:
            flat = json.load(f)
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets library not installed. Run: pip install datasets")
            return 1

        logger.info("Loading QASPER dataset from HuggingFace...")
        ds = load_dataset("allenai/qasper")
        flat = flatten_questions(ds)

        # Save flattened questions for future reuse
        flat_path = args.output.parent / "all_questions.json"
        flat_path.parent.mkdir(parents=True, exist_ok=True)
        with open(flat_path, "w", encoding="utf-8") as f:
            json.dump(flat, f, indent=2, ensure_ascii=False)
        logger.info("Flattened %d questions saved to %s", len(flat), flat_path)

    sample = sample_qasper(flat, target=args.target, seed=args.seed)
    logger.info("Sampled %d / %d (target=%d)", len(sample), len(flat), args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    logger.info("Sample written: %s", args.output)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = build_stratification_report(flat, sample, args.target, args.seed)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Report written: %s", args.report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
