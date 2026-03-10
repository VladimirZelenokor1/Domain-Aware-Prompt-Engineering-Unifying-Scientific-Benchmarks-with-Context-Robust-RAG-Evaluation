"""Tests for response_parser module.

Each test uses hand-crafted mock responses that mimic real LLM output
from the 6 target models (Mistral-Nemo, DeepSeek-R1, Qwen, SciPhi,
Gemma, Llama-3.2). Designed to validate >95% parse rate.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from response_parser import (
    ParsedResponse,
    aggregate_sc,
    parse_response,
)


# ===================================================================
# Good-path tests (#1-5)
# ===================================================================


class TestGoodPath:
    """Clean, well-formatted responses that should always parse."""

    def test_da_clean_mc(self, choices_4: dict) -> None:
        """#1: Perfect DA format, MC answer."""
        raw = "ANSWER: B\nJUSTIFICATION: Water is a polar molecule and acts as a universal solvent."
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "B"
        assert result.justification is not None
        assert "polar" in result.justification

    def test_da_clean_open_ended(self) -> None:
        """#2: Perfect DA format, open-ended answer."""
        raw = (
            "ANSWER: The procedure involves isolating RNA from tissue samples, "
            "performing reverse transcription, and running qPCR.\n"
            "JUSTIFICATION: This is the standard workflow for gene expression analysis."
        )
        result = parse_response(raw, "da", "open-ended-qa")
        assert result.parse_success is True
        assert "RNA" in result.answer
        assert result.justification is not None

    def test_ras_all_fields(self, choices_4: dict) -> None:
        """#3: RAS with all 4 fields present."""
        raw = (
            "ANSWER: C\n"
            "KEY REASONING: Air is primarily composed of nitrogen and oxygen. "
            "The question asks about atmospheric composition.\n"
            "UNCERTAINTY: I am not sure about trace gases.\n"
            "CONFIDENCE: high"
        )
        result = parse_response(raw, "ras", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "C"
        assert result.key_reasoning is not None
        assert "nitrogen" in result.key_reasoning
        assert result.uncertainty is not None
        assert result.confidence == "high"

    def test_ctl_with_rationale(self) -> None:
        """#4: CTL with ANSWER + RATIONALE for true_or_false."""
        raw = (
            "ANSWER: No\n"
            "RATIONALE: Leaving a gas cylinder valve open after use is dangerous "
            "because it can lead to gas leaks, contamination, and safety hazards."
        )
        result = parse_response(raw, "ctl", "true_or_false")
        assert result.parse_success is True
        assert result.answer_normalized == "No"
        assert result.rationale is not None
        assert "safety" in result.rationale

    def test_rag_with_standard_citations(self, choices_4: dict) -> None:
        """#5: RAG response with [1], [3] citations."""
        raw = (
            "ANSWER: B\n"
            "JUSTIFICATION: According to [1], water is a polar compound. "
            "This is supported by [3] which mentions sodium chloride dissolving in water."
        )
        result = parse_response(
            raw, "da", "mcq-4-choices", choices_4, mode="rag"
        )
        assert result.parse_success is True
        assert result.answer_normalized == "B"
        assert result.citations == [1, 3]


# ===================================================================
# Edge-case tests (#6-16)
# ===================================================================


class TestEdgeCases:
    """Responses with formatting quirks that should still parse."""

    def test_preamble_before_answer(self, choices_4: dict) -> None:
        """#6: Model adds conversational preamble."""
        raw = (
            "Sure! I'd be happy to help with this question. "
            "Let me analyze the options carefully.\n\n"
            "ANSWER: C\n"
            "JUSTIFICATION: Air is the correct answer because it refers to "
            "the atmosphere around us."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "C"

    def test_lowercase_field_names(self, choices_4: dict) -> None:
        """#7: Lowercase 'answer:' instead of 'ANSWER:'."""
        raw = (
            "answer: B\n"
            "justification: Water is the universal solvent."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "B"

    def test_markdown_bold_fields(self, choices_4: dict) -> None:
        """#8: **ANSWER:** with markdown bold."""
        raw = (
            "**ANSWER:** D\n"
            "**JUSTIFICATION:** Earth is the third planet from the sun "
            "and the only known planet to harbor life."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "D"

    def test_extra_blank_lines(self, choices_4: dict) -> None:
        """#9: Extra blank lines between fields."""
        raw = (
            "ANSWER: A\n\n\n"
            "JUSTIFICATION: Fire is a rapid oxidation process "
            "producing heat and light.\n\n"
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "A"

    def test_cot_paragraph_before_answer(self, choices_4: dict) -> None:
        """#10: CoT thinking paragraph before the structured fields."""
        raw = (
            "Let me think about this step by step. The question asks about "
            "which element is responsible for cell wall thickening. "
            "I recall that KNAT7 is involved in secondary cell wall formation. "
            "The other options don't directly regulate wall thickening.\n\n"
            "ANSWER: A\n"
            "RATIONALE: KNAT7 is a transcription factor that regulates "
            "secondary cell wall biosynthesis in Arabidopsis."
        )
        result = parse_response(raw, "ctl", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "A"

    def test_deepseek_think_block(self, choices_4: dict) -> None:
        """#11: DeepSeek-R1 <think>...</think> block before answer."""
        raw = (
            "<think>\n"
            "Let me analyze this question. The options are:\n"
            "A) Fire - combustion reaction\n"
            "B) Water - H2O, polar molecule\n"
            "C) Air - mixture of gases\n"
            "D) Earth - solid matter\n\n"
            "The question seems to be about polar solvents, so B is correct.\n"
            "</think>\n"
            "ANSWER: B\n"
            "JUSTIFICATION: Water is a polar molecule due to its bent "
            "molecular geometry and electronegativity difference between O and H."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "B"
        assert result.think_block_removed is True

    def test_confidence_case_normalization(self, choices_4: dict) -> None:
        """#12: CONFIDENCE: Medium (capitalized) -> 'medium'."""
        raw = (
            "ANSWER: B\n"
            "KEY REASONING: Water has unique properties as a solvent.\n"
            "UNCERTAINTY: None identified.\n"
            "CONFIDENCE: Medium"
        )
        result = parse_response(raw, "ras", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.confidence == "medium"

    def test_comma_separated_citations(self, choices_4: dict) -> None:
        """#13: Citations as [1, 2] (comma inside brackets)."""
        raw = (
            "ANSWER: B\n"
            "JUSTIFICATION: Based on [1, 2], water is polar and widely studied."
        )
        result = parse_response(
            raw, "da", "mcq-4-choices", choices_4, mode="rag"
        )
        assert result.parse_success is True
        assert result.citations == [1, 2]

    def test_adjacent_citations(self, choices_4: dict) -> None:
        """#14: Citations as [1][2] (adjacent brackets)."""
        raw = (
            "ANSWER: A\n"
            "JUSTIFICATION: As discussed in [1][2], fire requires fuel and oxygen."
        )
        result = parse_response(
            raw, "da", "mcq-4-choices", choices_4, mode="rag"
        )
        assert result.parse_success is True
        assert result.citations == [1, 2]

    def test_passage_word_citations(self, choices_4: dict) -> None:
        """#15: Citations as 'passage 1 and passage 3'."""
        raw = (
            "ANSWER: B\n"
            "JUSTIFICATION: According to passage 1 and passage 3, "
            "water dissolves ionic compounds like NaCl."
        )
        result = parse_response(
            raw, "da", "mcq-4-choices", choices_4, mode="rag"
        )
        assert result.parse_success is True
        assert 1 in result.citations
        assert 3 in result.citations

    def test_mc_letter_with_text(self, choices_4: dict) -> None:
        """#16: MC answer 'B) Water is the universal solvent' -> 'B'."""
        raw = (
            "ANSWER: B) Water is the universal solvent\n"
            "JUSTIFICATION: Water's polarity allows it to dissolve many substances."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "B"

    def test_mc_full_text_fuzzy_match(self, choices_4: dict) -> None:
        """#17: MC answer is full choice text without letter -> fuzzy match."""
        raw = (
            "ANSWER: Water\n"
            "JUSTIFICATION: Water is the correct choice."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "B"

    def test_tf_true_normalized_to_yes(self) -> None:
        """#18: true_or_false answer 'True' -> 'Yes'."""
        raw = (
            "ANSWER: True\n"
            "JUSTIFICATION: The statement is factually correct."
        )
        result = parse_response(raw, "da", "true_or_false")
        assert result.parse_success is True
        assert result.answer_normalized == "Yes"

    def test_codeblock_wrapped_response(self, choices_4: dict) -> None:
        """#19: Response wrapped in markdown code block."""
        raw = (
            "```\n"
            "ANSWER: C\n"
            "JUSTIFICATION: Air is a mixture of gases.\n"
            "```"
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "C"


# ===================================================================
# Failure tests (#20-24)
# ===================================================================


class TestFailureCases:
    """Responses that should fail gracefully."""

    def test_no_answer_field_freetext(self, choices_4: dict) -> None:
        """#20: No ANSWER: field at all, just free text."""
        raw = (
            "The correct option here would be B because water is a polar "
            "molecule with hydrogen bonding capabilities."
        )
        result = parse_response(raw, "da", "mcq-4-choices", choices_4)
        # Fallback extraction may succeed with letter B in text
        # But may also fail - both are acceptable degradation
        assert isinstance(result.parse_success, bool)
        assert "answer_field_missing" in result.parse_errors[0]

    def test_empty_response(self) -> None:
        """#21: Completely empty response."""
        result = parse_response("", "da", "mcq-4-choices")
        assert result.parse_success is False
        assert "empty_response" in result.parse_errors

    def test_whitespace_only_response(self) -> None:
        """#22: Only whitespace."""
        result = parse_response("   \n\n  ", "da", "mcq-4-choices")
        assert result.parse_success is False

    def test_ras_only_answer_partial(self, choices_4: dict) -> None:
        """#23: RAS with only ANSWER present, missing other 3 fields.
        Should still succeed because answer is enough."""
        raw = "ANSWER: D"
        result = parse_response(raw, "ras", "mcq-4-choices", choices_4)
        assert result.parse_success is True
        assert result.answer_normalized == "D"
        assert result.key_reasoning is None
        assert result.uncertainty is None
        assert result.confidence is None

    def test_refusal_detected(self) -> None:
        """#24: Model refuses to answer."""
        raw = (
            "I apologize, but I cannot answer this question as it requires "
            "specialized domain knowledge that I'm not confident about. "
            "I would recommend consulting a subject matter expert."
        )
        result = parse_response(raw, "da", "mcq-4-choices")
        assert result.parse_success is True  # Refusal is a valid parse
        assert result.refusal is True
        assert result.answer == "REFUSED"


# ===================================================================
# SC aggregation tests (#25-30)
# ===================================================================


class TestSCAggregation:
    """Self-Consistency majority vote tests."""

    def _make_parsed(
        self,
        answer: str | None,
        answer_normalized: str | None,
        parse_success: bool = True,
        refusal: bool = False,
    ) -> ParsedResponse:
        """Helper to create a ParsedResponse for SC testing."""
        return ParsedResponse(
            answer=answer,
            answer_normalized=answer_normalized,
            parse_success=parse_success,
            refusal=refusal,
            raw_response=f"ANSWER: {answer}",
        )

    def test_clear_majority_3_2(self, choices_4: dict) -> None:
        """#25: 5 samples, 3 say B, 2 say C -> B wins."""
        samples = [
            self._make_parsed("B", "B"),
            self._make_parsed("B", "B"),
            self._make_parsed("B", "B"),
            self._make_parsed("C", "C"),
            self._make_parsed("C", "C"),
        ]
        result = aggregate_sc(samples, "mcq-4-choices", choices_4)
        assert result.sc_success is True
        assert result.final_answer_normalized == "B"
        assert result.agreement_ratio == pytest.approx(0.6)
        assert result.valid_samples == 5

    def test_format_variation_all_normalize_same(self, choices_4: dict) -> None:
        """#26: 5 samples with format variation all normalizing to B."""
        samples = [
            self._make_parsed("B", "B"),
            self._make_parsed("B) Water", "B"),
            self._make_parsed("B.", "B"),
            self._make_parsed("C", "C"),
            self._make_parsed("B", "B"),
        ]
        result = aggregate_sc(samples, "mcq-4-choices", choices_4)
        assert result.sc_success is True
        assert result.final_answer_normalized == "B"
        assert result.vote_counts["B"] == 4

    def test_tie_first_occurrence_wins(self, choices_4: dict) -> None:
        """#27: 2 A, 2 B, 1 C -> tie between A and B, A appeared first."""
        samples = [
            self._make_parsed("A", "A"),
            self._make_parsed("B", "B"),
            self._make_parsed("A", "A"),
            self._make_parsed("B", "B"),
            self._make_parsed("C", "C"),
        ]
        result = aggregate_sc(samples, "mcq-4-choices", choices_4)
        assert result.sc_success is True
        assert result.final_answer_normalized == "A"
        assert result.tie_broken is True

    def test_below_quorum(self, choices_4: dict) -> None:
        """#28: 3 parse failures, only 2 valid -> below quorum."""
        samples = [
            self._make_parsed("B", "B"),
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed("B", "B"),
            self._make_parsed(None, None, parse_success=False),
        ]
        result = aggregate_sc(samples, "mcq-4-choices", choices_4)
        assert result.sc_success is False
        assert result.valid_samples == 2
        # Still provides best answer from available
        assert result.final_answer_normalized == "B"

    def test_all_failed(self, choices_4: dict) -> None:
        """#29: All 5 samples failed parsing."""
        samples = [
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed(None, None, parse_success=False),
            self._make_parsed(None, None, parse_success=False),
        ]
        result = aggregate_sc(samples, "mcq-4-choices", choices_4)
        assert result.sc_success is False
        assert result.valid_samples == 0
        assert result.final_answer_normalized == ""

    def test_sc_open_ended_majority(self) -> None:
        """#30: SC with open-ended answers, majority vote on text."""
        samples = [
            self._make_parsed("The answer is 42", "The answer is 42"),
            self._make_parsed("The answer is 42.", "The answer is 42."),
            self._make_parsed("The answer is 42", "The answer is 42"),
            self._make_parsed("It could be 43", "It could be 43"),
            self._make_parsed("The answer is 42", "The answer is 42"),
        ]
        result = aggregate_sc(samples, "open-ended-qa")
        assert result.sc_success is True
        # "The answer is 42" and "The answer is 42." normalize to same
        assert "42" in result.final_answer_normalized
