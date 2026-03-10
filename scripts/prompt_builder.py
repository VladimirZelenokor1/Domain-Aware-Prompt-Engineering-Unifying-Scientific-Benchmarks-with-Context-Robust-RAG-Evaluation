"""Prompt builder for SciKnowEval experiments.

Loads prompt templates and assembles complete prompts for LLM inference
across DA, RAS, CTL, and SC strategies in closed-book and RAG modes.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
VALID_STRATEGIES = {"da", "ras", "ctl", "sc"}


def load_template(strategy: str, prompts_dir: Path = PROMPTS_DIR) -> str:
    """Load a prompt template file by strategy name.

    Args:
        strategy: One of 'da', 'ras', 'ctl', 'sc'.
        prompts_dir: Path to the prompts directory.

    Returns:
        Template string with {question} and {options_block} placeholders.

    Raises:
        ValueError: If strategy is not recognized.
        FileNotFoundError: If template file does not exist.
    """
    strategy = strategy.lower()
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Must be one of: {VALID_STRATEGIES}"
        )

    # SC uses the DA template
    filename = "da.txt" if strategy == "sc" else f"{strategy}.txt"
    template_path = prompts_dir / filename

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    return template_path.read_text(encoding="utf-8")


def format_options(choices: dict[str, list] | None) -> str:
    """Format MC choices into an options block string.

    Args:
        choices: Dict with 'text' (list[str]) and 'label' (list[str]),
                 or None for open-ended questions.

    Returns:
        Formatted string like "A) text\\nB) text\\n..." or empty string
        if choices are empty or None.
    """
    if choices is None:
        return ""

    texts = choices.get("text", [])
    labels = choices.get("label", [])

    if not texts:
        return ""

    # Generate labels dynamically if fewer labels than texts
    if len(labels) < len(texts):
        labels = [chr(65 + i) for i in range(len(texts))]

    lines = [f"{label}) {text}" for label, text in zip(labels, texts)]
    return "\n".join(lines)


def format_rag_context(
    passages: list[str], prompts_dir: Path = PROMPTS_DIR
) -> str:
    """Format RAG passages into context block.

    Args:
        passages: List of passage texts (up to top_k=10).
        prompts_dir: Path to the prompts directory.

    Returns:
        Formatted RAG context block with numbered passages,
        or empty string if passages list is empty.
    """
    if not passages:
        return ""

    template_path = prompts_dir / "rag_context.txt"
    if not template_path.exists():
        raise FileNotFoundError(f"RAG template not found: {template_path}")

    rag_template = template_path.read_text(encoding="utf-8")

    numbered = [f"[{i + 1}] {text}" for i, text in enumerate(passages)]
    passages_block = "\n".join(numbered)

    return rag_template.replace("{passages_block}", passages_block)


def build_prompt(
    question: str,
    strategy: str,
    choices: dict[str, list] | None = None,
    passages: list[str] | None = None,
    prompts_dir: Path = PROMPTS_DIR,
) -> str:
    """Build a complete prompt for a given question and strategy.

    Args:
        question: The question text.
        strategy: One of 'da', 'ras', 'ctl', 'sc'.
        choices: MC choices dict, or None for open-ended.
        passages: RAG passages list, or None for closed-book.
        prompts_dir: Path to the prompts directory.

    Returns:
        Complete prompt string ready for LLM inference.
    """
    template = load_template(strategy, prompts_dir)
    options_block = format_options(choices)

    prompt = template.replace("{question}", question)
    prompt = prompt.replace("{options_block}", options_block)

    # Prepend RAG context if passages provided
    if passages:
        rag_block = format_rag_context(passages, prompts_dir)
        prompt = rag_block + "\n\n" + prompt

    return prompt


def main() -> None:
    """CLI entry point for prompt builder."""
    parser = argparse.ArgumentParser(
        description="Build prompts for SciKnowEval experiments"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(VALID_STRATEGIES),
        help="Prompt strategy to use",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question text",
    )
    parser.add_argument(
        "--choices",
        default=None,
        help='JSON string of choices dict, e.g. \'{"text":["A","B"],"label":["A","B"]}\'',
    )
    parser.add_argument(
        "--passages",
        default=None,
        help='JSON string of passages list, e.g. \'["passage 1","passage 2"]\'',
    )

    args = parser.parse_args()

    choices = json.loads(args.choices) if args.choices else None
    passages = json.loads(args.passages) if args.passages else None

    prompt = build_prompt(
        question=args.question,
        strategy=args.strategy,
        choices=choices,
        passages=passages,
    )

    sys.stdout.write(prompt + "\n")


if __name__ == "__main__":
    main()
