"""Batch validation for prompt builder and response parser pipeline.

Loads questions from SciKnowEval dev set, builds prompts, generates
mock responses, parses them, and reports parse rate statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prompt_builder import build_prompt
from response_parser import parse_response

logger = logging.getLogger(__name__)

# Mock response templates for each strategy
MOCK_TEMPLATES: dict[str, dict[str, str]] = {
    "da": {
        "mc": "ANSWER: {letter}\nJUSTIFICATION: This is the correct answer based on scientific evidence.",
        "tf": "ANSWER: No\nJUSTIFICATION: The statement is incorrect based on safety protocols.",
        "open": "ANSWER: The procedure involves standard laboratory techniques.\nJUSTIFICATION: This follows established methodology.",
    },
    "ras": {
        "mc": "ANSWER: {letter}\nKEY REASONING: The analysis of the options points to this answer.\nUNCERTAINTY: Minor uncertainty about edge cases.\nCONFIDENCE: high",
        "tf": "ANSWER: No\nKEY REASONING: Safety standards require specific procedures.\nUNCERTAINTY: None.\nCONFIDENCE: high",
        "open": "ANSWER: The experimental design follows standard protocols.\nKEY REASONING: Based on established literature.\nUNCERTAINTY: Sample size effects unknown.\nCONFIDENCE: medium",
    },
    "ctl": {
        "mc": "ANSWER: {letter}\nRATIONALE: Step-by-step analysis of the options leads to this conclusion.",
        "tf": "ANSWER: No\nRATIONALE: Careful consideration of safety requirements shows this is false.",
        "open": "ANSWER: The method requires careful preparation.\nRATIONALE: Standard procedures dictate these steps.",
    },
    "sc": {
        "mc": "ANSWER: {letter}\nJUSTIFICATION: This is the correct answer.",
        "tf": "ANSWER: No\nJUSTIFICATION: The statement is incorrect.",
        "open": "ANSWER: Standard laboratory procedure.\nJUSTIFICATION: Follows established methods.",
    },
}

BAD_MOCKS = [
    "I think the answer might be something related to chemistry but I'm not entirely sure about this one.",
    "",
]


def _get_question_category(record: dict) -> str:
    """Determine question category for mock selection."""
    qtype = record.get("type", "")
    if qtype in ("mcq-4-choices", "mcq-2-choices"):
        return "mc"
    elif qtype == "true_or_false":
        return "tf"
    else:
        return "open"


def _generate_mock(record: dict, strategy: str) -> str:
    """Generate a mock response for a given record and strategy."""
    category = _get_question_category(record)
    template = MOCK_TEMPLATES[strategy][category]

    if category == "mc":
        answer_key = record.get("answerKey", "A")
        if not answer_key:
            labels = record.get("choices", {}).get("label", ["A"])
            answer_key = labels[0] if labels else "A"
        return template.format(letter=answer_key)

    return template


def validate(
    data_path: Path,
    strategy: str,
    limit: int | None = None,
) -> dict:
    """Run validation on dev set with mock responses.

    Args:
        data_path: Path to dev.json.
        strategy: Prompt strategy to validate.
        limit: Max number of records to process.

    Returns:
        Summary dict with parse rate statistics.
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    if limit:
        # Take a representative sample
        random.seed(42)
        data = random.sample(data, min(limit, len(data)))

    good_total = 0
    good_parsed = 0
    bad_total = 0
    bad_correctly_failed = 0

    for record in data:
        question = record["question"]
        choices = record.get("choices")
        qtype = record.get("type", "open-ended-qa")

        # Verify prompt builds without error
        prompt = build_prompt(
            question=question,
            strategy=strategy,
            choices=choices,
        )

        # Check no unsubstituted placeholders
        assert "{question}" not in prompt
        assert "{options_block}" not in prompt
        assert "{passages_block}" not in prompt

        # Generate and parse good mock
        mock = _generate_mock(record, strategy)
        result = parse_response(
            raw_response=mock,
            strategy=strategy,
            question_type=qtype,
            choices=choices if choices and choices.get("text") else None,
        )
        good_total += 1
        if result.parse_success:
            good_parsed += 1

    # Test bad mocks
    sample_record = data[0] if data else None
    if sample_record:
        qtype = sample_record.get("type", "open-ended-qa")
        choices = sample_record.get("choices")

        for bad_mock in BAD_MOCKS:
            result = parse_response(
                raw_response=bad_mock,
                strategy=strategy,
                question_type=qtype,
                choices=choices if choices and choices.get("text") else None,
            )
            bad_total += 1
            if not result.parse_success or result.refusal:
                bad_correctly_failed += 1

    return {
        "strategy": strategy,
        "good_total": good_total,
        "good_parsed": good_parsed,
        "bad_total": bad_total,
        "bad_correctly_failed": bad_correctly_failed,
        "parse_rate": good_parsed / good_total * 100 if good_total else 0,
        "false_positive_rate": (
            (bad_total - bad_correctly_failed) / bad_total * 100
            if bad_total
            else 0
        ),
    }


def main() -> None:
    """CLI entry point for prompt validator."""
    parser = argparse.ArgumentParser(
        description="Validate prompt builder + parser pipeline on mock responses"
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to SciKnowEval dev.json",
    )
    parser.add_argument(
        "--strategy",
        default=None,
        choices=["da", "ras", "ctl", "sc"],
        help="Strategy to validate (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process (default: all)",
    )

    args = parser.parse_args()
    data_path = Path(args.data)

    if not data_path.exists():
        sys.stderr.write(f"Error: {data_path} not found\n")
        sys.exit(1)

    strategies = [args.strategy] if args.strategy else ["da", "ras", "ctl", "sc"]

    print("\nMOCK VALIDATION RESULTS:")
    print(
        f"  {'Strategy':<10} {'Good mocks':<12} {'Parsed':<8} "
        f"{'Bad mocks':<11} {'Correct fail':<14} {'Parse rate':<12} "
        f"{'FP rate'}"
    )
    print("  " + "-" * 85)

    for strat in strategies:
        result = validate(data_path, strat, args.limit)
        print(
            f"  {result['strategy']:<10} "
            f"{result['good_total']:<12} "
            f"{result['good_parsed']:<8} "
            f"{result['bad_total']:<11} "
            f"{result['bad_correctly_failed']:<14} "
            f"{result['parse_rate']:<12.1f} "
            f"{result['false_positive_rate']:.1f}%"
        )

    print()


if __name__ == "__main__":
    main()
