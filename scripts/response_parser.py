"""Response parser for SciKnowEval LLM outputs.

6-step pipeline: strip think blocks -> strip preamble -> strip markdown
-> extract fields -> normalize answer -> extract citations -> validate.

Designed for >95% parse rate across 6 different LLMs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STRATEGIES = {"da", "ras", "ctl", "sc"}

STRATEGY_FIELDS: dict[str, list[str]] = {
    "da": ["ANSWER", "JUSTIFICATION"],
    "ras": ["ANSWER", "KEY REASONING", "UNCERTAINTY", "CONFIDENCE"],
    "ctl": ["ANSWER", "RATIONALE"],
    "sc": ["ANSWER", "JUSTIFICATION"],
}

MC_TYPES = {"mcq-4-choices", "mcq-2-choices"}
TF_TYPE = "true_or_false"
OPEN_TYPES = {"open-ended-qa", "relation_extraction", "filling"}

ALL_FIELD_NAMES = [
    "ANSWER",
    "JUSTIFICATION",
    "KEY REASONING",
    "UNCERTAINTY",
    "CONFIDENCE",
    "RATIONALE",
]

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Step 1: Think block removal (DeepSeek-R1)
THINK_PATTERN = re.compile(
    r"<\|?think\|?>.*?<\|?/?think\|?>", re.DOTALL
)

# Step 2: Preamble - find first field marker
_FIELD_NAMES_RE = "|".join(re.escape(f) for f in ALL_FIELD_NAMES)
FIRST_FIELD_PATTERN = re.compile(
    rf"(?:^|\n)\s*(?:\*{{1,3}})?(?:{_FIELD_NAMES_RE})(?:\*{{1,3}})?\s*:",
    re.IGNORECASE,
)

# Step 3: Markdown cleanup
MARKDOWN_BOLD_PATTERN = re.compile(
    rf"\*{{1,3}}({_FIELD_NAMES_RE})\*{{1,3}}", re.IGNORECASE
)
MARKDOWN_CODE_PATTERN = re.compile(
    rf"`({_FIELD_NAMES_RE})`", re.IGNORECASE
)
MARKDOWN_HEADER_PATTERN = re.compile(r"^#{1,4}\s*", re.MULTILINE)
MARKDOWN_CODEBLOCK_PATTERN = re.compile(r"```\w*\n?|```", re.MULTILINE)

# Step 4: Field extraction - built per strategy (see _build_field_pattern)

# Step 5a: MC answer normalization
MC_LETTER_DIRECT = re.compile(r"^\s*\(?([A-Fa-f])\)?\s*$")
MC_LETTER_WITH_TEXT = re.compile(
    r"^\s*\(?([A-Fa-f])\)?[\)\.\:\s]+\S", re.IGNORECASE
)
MC_ANSWER_IS = re.compile(
    r"(?:answer|option|choice)\s+(?:is\s+)?[\(\[]?([A-Fa-f])[\)\]]?\b",
    re.IGNORECASE,
)
MC_TRAILING_LETTER = re.compile(r"\b([A-Fa-f])\s*\.?\s*$")

# Step 5b: True/false normalization
TF_PATTERN = re.compile(
    r"\b(yes|no|true|false|correct|incorrect|right|wrong)\b",
    re.IGNORECASE,
)
TF_MAP = {
    "yes": "Yes",
    "true": "Yes",
    "correct": "Yes",
    "right": "Yes",
    "no": "No",
    "false": "No",
    "incorrect": "No",
    "wrong": "No",
}

# Step 6: Citation extraction
CITATION_BRACKET = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
CITATION_PAREN = re.compile(r"\((\d+)\)")
CITATION_PASSAGE = re.compile(r"[Pp]assage\s+(\d+)")
CITATION_SOURCE = re.compile(r"[Ss]ource\s+(\d+)")

# Refusal detection
REFUSAL_PATTERN = re.compile(
    r"(?:I\s+cannot|I'm\s+unable|I\s+don't\s+have\s+(?:enough|sufficient)"
    r"|as an AI|I\s+apologize|I\s+can't\s+(?:answer|provide)"
    r"|I\s+am\s+not\s+able|I\s+do\s+not\s+have)",
    re.IGNORECASE,
)

# Chinese field labels (Qwen fallback)
CHINESE_ANSWER = re.compile(r"(?:答案|回答)\s*[:：]\s*(.+?)(?:\n|$)")
CHINESE_REASONING = re.compile(r"(?:理由|解释|说明|推理)\s*[:：]\s*(.+?)(?=(?:答案|回答|理由|解释|说明|推理|置信度|信心|不确定)\s*[:：]|$)", re.DOTALL)
CHINESE_CONFIDENCE = re.compile(r"(?:置信度|信心)\s*[:：]\s*(.+?)(?:\n|$)")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ParsedResponse:
    """Result of parsing a single model response."""

    answer: str | None = None
    answer_normalized: str | None = None
    justification: str | None = None
    rationale: str | None = None
    key_reasoning: str | None = None
    uncertainty: str | None = None
    confidence: str | None = None
    citations: list[int] = field(default_factory=list)
    parse_success: bool = False
    parse_errors: list[str] = field(default_factory=list)
    raw_response: str = ""
    refusal: bool = False
    think_block_removed: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class SCResult:
    """Result of SC aggregation over multiple samples."""

    final_answer: str = ""
    final_answer_normalized: str = ""
    vote_counts: dict[str, int] = field(default_factory=dict)
    total_samples: int = 0
    valid_samples: int = 0
    sc_success: bool = False
    agreement_ratio: float = 0.0
    tie_broken: bool = False
    per_sample: list[ParsedResponse] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["per_sample"] = [s.to_dict() for s in self.per_sample]
        return d


# ---------------------------------------------------------------------------
# Pipeline step 1: Strip think blocks
# ---------------------------------------------------------------------------


def _strip_think_blocks(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks (DeepSeek-R1).

    Returns:
        Tuple of (cleaned text, whether blocks were removed).
    """
    cleaned = THINK_PATTERN.sub("", text)
    removed = cleaned != text
    return cleaned.strip(), removed


# ---------------------------------------------------------------------------
# Pipeline step 2: Strip preamble
# ---------------------------------------------------------------------------


def _strip_preamble(text: str) -> str:
    """Remove text before first recognized field label."""
    match = FIRST_FIELD_PATTERN.search(text)
    if match:
        return text[match.start():].lstrip("\n")
    return text


# ---------------------------------------------------------------------------
# Pipeline step 3: Strip markdown
# ---------------------------------------------------------------------------


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting from field labels."""
    text = MARKDOWN_CODEBLOCK_PATTERN.sub("", text)
    text = MARKDOWN_BOLD_PATTERN.sub(r"\1", text)
    text = MARKDOWN_CODE_PATTERN.sub(r"\1", text)
    text = MARKDOWN_HEADER_PATTERN.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Pipeline step 4: Extract fields
# ---------------------------------------------------------------------------


def _build_field_pattern(fields: list[str]) -> re.Pattern:
    """Build regex to extract field_name: value pairs."""
    escaped = [re.escape(f) for f in fields]
    joined = "|".join(escaped)
    return re.compile(
        rf"({joined})\s*:\s*(.*?)(?=(?:{joined})\s*:|$)",
        re.DOTALL | re.IGNORECASE,
    )


def _extract_fields(text: str, strategy: str) -> dict[str, str]:
    """Extract field_name -> field_value pairs for a strategy."""
    fields = STRATEGY_FIELDS.get(strategy, STRATEGY_FIELDS["da"])
    pattern = _build_field_pattern(fields)

    result: dict[str, str] = {}
    for match in pattern.finditer(text):
        field_name = match.group(1).strip().upper()
        field_value = match.group(2).strip()
        result[field_name] = field_value

    return result


def _extract_fields_chinese(text: str) -> dict[str, str]:
    """Fallback: extract fields using Chinese labels."""
    result: dict[str, str] = {}

    m = CHINESE_ANSWER.search(text)
    if m:
        result["ANSWER"] = m.group(1).strip()

    m = CHINESE_REASONING.search(text)
    if m:
        result["JUSTIFICATION"] = m.group(1).strip()

    m = CHINESE_CONFIDENCE.search(text)
    if m:
        result["CONFIDENCE"] = m.group(1).strip()

    return result


# ---------------------------------------------------------------------------
# Pipeline step 5: Normalize answer
# ---------------------------------------------------------------------------


def _normalize_mc(raw: str, choices: dict[str, list]) -> str | None:
    """Normalize MC answer to a single uppercase letter.

    Tries multiple patterns, then falls back to fuzzy matching.
    """
    answer = raw.strip()
    if not answer:
        return None

    texts = choices.get("text", [])
    labels = choices.get("label", [])
    if len(labels) < len(texts):
        labels = [chr(65 + i) for i in range(len(texts))]

    valid_labels = {lbl.upper() for lbl in labels}

    # 1. Direct letter match: "B" or "(B)" or "B)"
    m = MC_LETTER_DIRECT.match(answer)
    if m and m.group(1).upper() in valid_labels:
        return m.group(1).upper()

    # 2. Letter + text: "B) Water is..."
    m = MC_LETTER_WITH_TEXT.match(answer)
    if m and m.group(1).upper() in valid_labels:
        return m.group(1).upper()

    # 3. "The answer is B" pattern
    m = MC_ANSWER_IS.search(answer)
    if m and m.group(1).upper() in valid_labels:
        return m.group(1).upper()

    # 4. Trailing letter at end of answer
    m = MC_TRAILING_LETTER.search(answer)
    if m and m.group(1).upper() in valid_labels:
        return m.group(1).upper()

    # 5. Fuzzy match against choice texts
    answer_lower = answer.lower()
    best_ratio = 0.0
    best_label = None
    for lbl, txt in zip(labels, texts):
        ratio = difflib.SequenceMatcher(
            None, answer_lower, txt.lower()
        ).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_label = lbl.upper()

    if best_ratio >= 0.8 and best_label is not None:
        return best_label

    # 6. Check if answer contains exactly one valid letter
    found_letters = [
        c.upper() for c in answer if c.upper() in valid_labels
    ]
    unique_found = set(found_letters)
    if len(unique_found) == 1:
        return unique_found.pop()

    return None


def _normalize_tf(raw: str) -> str | None:
    """Normalize true/false answer to 'Yes' or 'No'."""
    answer = raw.strip().lower()
    if not answer:
        return None

    # Direct match
    if answer in TF_MAP:
        return TF_MAP[answer]

    # Starts with yes/no
    for prefix in ("yes", "no"):
        if answer.startswith(prefix):
            return TF_MAP[prefix]

    # Regex search for first match
    m = TF_PATTERN.search(answer)
    if m:
        return TF_MAP[m.group(1).lower()]

    return None


def _normalize_open(raw: str) -> str:
    """Normalize open-ended answer (minimal processing)."""
    answer = raw.strip()
    # Remove trailing "END" marker (seen in relation_extraction)
    if answer.endswith("END"):
        answer = answer[:-3].strip()
    return answer


def _normalize_answer(
    raw: str,
    question_type: str,
    choices: dict[str, list] | None = None,
) -> str | None:
    """Normalize answer based on question type.

    Returns:
        Normalized answer string, or None if normalization fails.
    """
    if not raw or not raw.strip():
        return None

    if question_type in MC_TYPES and choices:
        return _normalize_mc(raw, choices)
    elif question_type == TF_TYPE:
        return _normalize_tf(raw)
    else:
        return _normalize_open(raw)


# ---------------------------------------------------------------------------
# Pipeline step 6: Extract citations
# ---------------------------------------------------------------------------


def _extract_citations(text: str, max_passage: int = 10) -> list[int]:
    """Extract passage citation numbers from text.

    Tries multiple citation formats and deduplicates.
    """
    citations: set[int] = set()

    # [1], [2, 3], [1, 2, 3]
    for m in CITATION_BRACKET.finditer(text):
        for num_str in m.group(1).split(","):
            num_str = num_str.strip()
            if num_str.isdigit():
                citations.add(int(num_str))

    # (1), (2)
    for m in CITATION_PAREN.finditer(text):
        citations.add(int(m.group(1)))

    # Passage 1, passage 2
    for m in CITATION_PASSAGE.finditer(text):
        citations.add(int(m.group(1)))

    # Source 1, source 2
    for m in CITATION_SOURCE.finditer(text):
        citations.add(int(m.group(1)))

    # Filter to valid range
    return sorted(c for c in citations if 1 <= c <= max_passage)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detect_refusal(text: str) -> bool:
    """Detect if response is a model refusal."""
    return bool(REFUSAL_PATTERN.search(text))


def _detect_language(text: str) -> str:
    """Simple language detection based on CJK character ratio."""
    if not text:
        return "en"
    non_ws = re.sub(r"\s", "", text)
    if not non_ws:
        return "en"
    cjk_count = sum(1 for c in non_ws if "\u4e00" <= c <= "\u9fff")
    ratio = cjk_count / len(non_ws)
    return "zh" if ratio > 0.3 else "en"


def _fallback_first_line(text: str) -> str | None:
    """Last-resort: extract the first non-empty line as answer."""
    for line in text.strip().split("\n"):
        line = line.strip()
        if line:
            return line
    return None


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------


def parse_response(
    raw_response: str,
    strategy: str,
    question_type: str,
    choices: dict[str, list] | None = None,
    mode: str = "closed_book",
) -> ParsedResponse:
    """Parse a single LLM response through the 6-step pipeline.

    Args:
        raw_response: Raw model output string.
        strategy: One of 'da', 'ras', 'ctl', 'sc'.
        question_type: One of the SciKnowEval question types.
        choices: MC choices dict for answer validation.
        mode: 'closed_book' or 'rag'.

    Returns:
        ParsedResponse with extracted and normalized fields.
    """
    result = ParsedResponse(raw_response=raw_response)
    errors: list[str] = []

    # Early exit: empty response
    if not raw_response or not raw_response.strip():
        result.parse_errors = ["empty_response"]
        return result

    text = raw_response

    # Step 1: Strip think blocks
    text, think_removed = _strip_think_blocks(text)
    result.think_block_removed = think_removed

    # Check refusal before field extraction
    if _detect_refusal(text):
        result.refusal = True

    # Step 2: Strip preamble
    text = _strip_preamble(text)

    # Step 3: Strip markdown
    text = _strip_markdown(text)

    # Step 4: Extract fields
    fields = _extract_fields(text, strategy)

    # Fallback: try Chinese labels if no ANSWER found
    if "ANSWER" not in fields:
        lang = _detect_language(raw_response)
        if lang == "zh":
            fields = _extract_fields_chinese(text)

    # Step 5: Normalize answer
    raw_answer = fields.get("ANSWER")
    if raw_answer:
        result.answer = raw_answer
        normalized = _normalize_answer(raw_answer, question_type, choices)
        result.answer_normalized = normalized
        if normalized is None and question_type in MC_TYPES:
            errors.append("mc_normalization_failed")
    elif not result.refusal:
        # Last-resort: try first line extraction
        fallback = _fallback_first_line(text)
        if fallback:
            result.answer = fallback
            normalized = _normalize_answer(fallback, question_type, choices)
            result.answer_normalized = normalized
            errors.append("answer_field_missing_used_fallback")
        else:
            errors.append("answer_field_missing")

    # Populate other fields
    result.justification = fields.get("JUSTIFICATION")
    result.key_reasoning = fields.get("KEY REASONING")
    result.uncertainty = fields.get("UNCERTAINTY")
    result.confidence = (
        fields.get("CONFIDENCE", "").strip().lower() or None
    )
    result.rationale = fields.get("RATIONALE")

    # Step 6: Extract citations (RAG only)
    if mode == "rag":
        # Search across all extracted text
        all_text = " ".join(v for v in fields.values() if v)
        result.citations = _extract_citations(all_text)

    # Determine parse_success
    if result.refusal:
        result.parse_success = True
        result.answer = result.answer or "REFUSED"
        result.answer_normalized = result.answer_normalized or "REFUSED"
    elif result.answer_normalized is not None:
        result.parse_success = True
    elif result.answer is not None and question_type in OPEN_TYPES:
        # For open-ended, having any answer text is success
        result.answer_normalized = result.answer
        result.parse_success = True
    else:
        result.parse_success = False

    result.parse_errors = errors
    return result


# ---------------------------------------------------------------------------
# SC aggregation
# ---------------------------------------------------------------------------


def _normalize_for_sc_vote(
    answer: str | None,
    question_type: str,
    choices: dict[str, list] | None = None,
) -> str | None:
    """Aggressive normalization for SC vote comparison."""
    if answer is None:
        return None

    if question_type in MC_TYPES and choices:
        return _normalize_mc(answer, choices)
    elif question_type == TF_TYPE:
        return _normalize_tf(answer)
    else:
        # Open-ended: strip, lowercase, remove trailing punctuation
        normalized = answer.strip().lower()
        normalized = re.sub(r"[.!?,;:]+$", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized or None


def aggregate_sc(
    samples: list[ParsedResponse],
    question_type: str,
    choices: dict[str, list] | None = None,
    min_valid: int = 3,
) -> SCResult:
    """Aggregate SC samples via majority vote.

    Args:
        samples: List of ParsedResponse from N SC runs.
        question_type: Question type for normalization dispatch.
        choices: MC choices for MC questions.
        min_valid: Minimum valid parses required for quorum.

    Returns:
        SCResult with final answer and vote metadata.
    """
    result = SCResult(
        total_samples=len(samples),
        per_sample=samples,
    )

    # Filter to successfully parsed, non-refusal samples
    valid = [
        s for s in samples if s.parse_success and not s.refusal
    ]
    result.valid_samples = len(valid)

    if not valid:
        result.sc_success = False
        return result

    # Normalize answers for voting
    votes: list[tuple[str, int]] = []  # (normalized_answer, sample_index)
    for i, s in enumerate(valid):
        vote_answer = s.answer_normalized or s.answer
        normalized = _normalize_for_sc_vote(vote_answer, question_type, choices)
        if normalized is not None:
            votes.append((normalized, i))

    if not votes:
        result.sc_success = False
        return result

    # Count votes
    vote_counter = Counter(v[0] for v in votes)
    result.vote_counts = dict(vote_counter.most_common())

    # Check quorum
    if len(votes) < min_valid:
        result.sc_success = False
        # Still provide best answer from available votes
        winner = vote_counter.most_common(1)[0][0]
        result.final_answer_normalized = winner
        result.final_answer = winner
        result.agreement_ratio = vote_counter[winner] / len(votes)
        return result

    # Determine winner
    most_common = vote_counter.most_common()
    top_count = most_common[0][1]
    tied = [ans for ans, cnt in most_common if cnt == top_count]

    if len(tied) == 1:
        winner = tied[0]
        result.tie_broken = False
    else:
        # Tie-break: first occurrence among tied answers
        for norm_ans, _ in votes:
            if norm_ans in tied:
                winner = norm_ans
                break
        else:
            winner = tied[0]
        result.tie_broken = True

    result.final_answer_normalized = winner
    result.final_answer = winner
    result.sc_success = True
    result.agreement_ratio = vote_counter[winner] / len(votes)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for response parser."""
    parser = argparse.ArgumentParser(
        description="Parse LLM responses for SciKnowEval experiments"
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(VALID_STRATEGIES),
        help="Prompt strategy used",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Raw response text (use \\n for newlines)",
    )
    parser.add_argument(
        "--question-type",
        default="mcq-4-choices",
        help="Question type (default: mcq-4-choices)",
    )
    parser.add_argument(
        "--choices",
        default=None,
        help="JSON string of choices dict",
    )
    parser.add_argument(
        "--mode",
        default="closed_book",
        choices=["closed_book", "rag"],
        help="Inference mode",
    )

    args = parser.parse_args()

    raw_text = args.input.replace("\\n", "\n")
    choices = json.loads(args.choices) if args.choices else None

    result = parse_response(
        raw_response=raw_text,
        strategy=args.strategy,
        question_type=args.question_type,
        choices=choices,
        mode=args.mode,
    )

    sys.stdout.write(json.dumps(result.to_dict(), indent=2) + "\n")


if __name__ == "__main__":
    main()
