"""Tests for ``scripts/canary_contamination`` (thesis section 3.5.1 Layer 2).

Canary-completion contamination probe: truncate a benchmark question to its
opening clause, ask each model to reconstruct the full question and its
reference answer, and measure how often the gold answer is regenerated. A
higher regeneration rate on one dataset than another (cross-dataset Fisher)
flags differential memorisation.

Hermetic: pure helpers + a stub engine, no vLLM, no GPU, no network.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

# ROUGE-L tests need rouge_score (a declared dep, installed on the server/CI;
# may be absent in a lean local env). Pure helpers below always run.
requires_rouge = pytest.mark.skipif(
    importlib.util.find_spec("rouge_score") is None,
    reason="rouge_score not installed",
)

from canary_contamination import (  # noqa: E402
    build_canary_prompt,
    extract_completion_parts,
    fishers_2x2,
    is_regenerated,
    run_model_canary,
    truncate_question,
)


class _StubCompletion:
    def __init__(self, text: str) -> None:
        self.text = text


class _StubRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_StubCompletion(text)]


class _StubEngine:
    """Engine-agnostic stub: returns a queued completion per prompt in order."""

    def __init__(self, completions: list[str]) -> None:
        self._completions = completions

    def generate(self, prompts: list[str], sampling_params: object) -> list:  # noqa: ARG002
        return [_StubRequestOutput(self._completions[i]) for i in range(len(prompts))]


# --- truncate_question -------------------------------------------------------


def test_truncate_question_takes_first_clause() -> None:
    q = "What is the role of mitochondria, and how does it relate to ATP?"
    # first clause = up to the comma
    assert truncate_question(q) == "What is the role of mitochondria"


def test_truncate_question_collapses_whitespace() -> None:
    assert truncate_question("  What   is  X?  ") == "What is X"


def test_truncate_question_falls_back_to_ratio_when_clause_too_short() -> None:
    # "Q:" leading clause is < 3 words -> ratio prefix of the whole token list
    q = "Define the second law of thermodynamics precisely please"
    out = truncate_question(q, keep_ratio=0.5)
    assert out == "Define the second law"  # 8 tokens -> first 4


def test_truncate_question_empty() -> None:
    assert truncate_question("") == ""


# --- build_canary_prompt -----------------------------------------------------


def test_build_canary_prompt_inserts_prefix() -> None:
    template = 'Opening:\n"{prefix}"\nGo.'
    assert build_canary_prompt(template, "What is X") == 'Opening:\n"What is X"\nGo.'


# --- extract_completion_parts ------------------------------------------------


def test_extract_completion_parts_parses_question_and_answer() -> None:
    text = "QUESTION: What is the boiling point of water?\nANSWER: 100 degrees Celsius"
    q, a = extract_completion_parts(text)
    assert q == "What is the boiling point of water?"
    assert a == "100 degrees Celsius"


def test_extract_completion_parts_fallback_uses_whole_text_as_answer() -> None:
    text = "100 degrees Celsius"  # no labels
    q, a = extract_completion_parts(text)
    assert q == ""
    assert a == "100 degrees Celsius"


# --- is_regenerated ----------------------------------------------------------


def test_is_regenerated_exact_match_open_ended() -> None:
    assert is_regenerated("photosynthesis", "Photosynthesis", "open-ended-qa") is True


@requires_rouge
def test_is_regenerated_high_rouge_counts_as_regenerated() -> None:
    gold = "the mitochondria is the powerhouse of the cell"
    out = "the mitochondria is the powerhouse of the cell indeed"
    assert is_regenerated(out, gold, "open-ended-qa", rouge_threshold=0.75) is True


@requires_rouge
def test_is_regenerated_low_overlap_is_false() -> None:
    assert (
        is_regenerated("a completely unrelated sentence", "quantum entanglement", "open-ended-qa")
        is False
    )


def test_is_regenerated_empty_is_false() -> None:
    assert is_regenerated("", "anything", "open-ended-qa") is False
    assert is_regenerated("anything", "", "open-ended-qa") is False


# --- fishers_2x2 -------------------------------------------------------------


def test_fishers_2x2_significant_difference() -> None:
    # 40/50 vs 5/50 regenerated -> strongly significant
    odds, p = fishers_2x2(40, 50, 5, 50)
    assert p < 0.05
    assert odds > 1.0


def test_fishers_2x2_no_difference() -> None:
    odds, p = fishers_2x2(10, 50, 10, 50)
    assert p > 0.05


# --- run_model_canary (engine-agnostic) --------------------------------------


@requires_rouge
def test_run_model_canary_scores_with_stub_engine() -> None:
    records = [
        {"question": "What is X, exactly?", "gold": "alpha", "qtype": "open-ended-qa", "dataset": "qasper"},
        {"question": "What is Y, exactly?", "gold": "beta", "qtype": "open-ended-qa", "dataset": "qasper"},
    ]
    # first completion regenerates gold, second does not
    engine = _StubEngine(
        [
            "QUESTION: What is X, exactly?\nANSWER: alpha",
            "QUESTION: What is Y, exactly?\nANSWER: something else entirely",
        ]
    )
    template = '"{prefix}"'
    out = run_model_canary(engine, object(), records, template, rouge_threshold=0.75)
    assert [r["regenerated"] for r in out] == [True, False]
    # the question-verbatim reconstruction of the first item is exact -> ~1.0
    assert out[0]["q_verbatim"] > 0.9
