"""Tests for prompt_builder module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure scripts/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from prompt_builder import (
    build_prompt,
    format_options,
    load_template,
)


class TestLoadTemplate:
    """Tests for load_template function."""

    def test_load_da_template(self) -> None:
        template = load_template("da")
        assert "{question}" in template
        assert "{options_block}" in template
        assert "ANSWER:" in template
        assert "JUSTIFICATION:" in template

    def test_load_ras_template(self) -> None:
        template = load_template("ras")
        assert "KEY REASONING:" in template
        assert "UNCERTAINTY:" in template
        assert "CONFIDENCE:" in template

    def test_load_ctl_template(self) -> None:
        template = load_template("ctl")
        assert "RATIONALE:" in template
        assert "step by step" in template

    def test_sc_loads_da_template(self) -> None:
        sc_template = load_template("sc")
        da_template = load_template("da")
        assert sc_template == da_template

    def test_invalid_strategy_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown strategy"):
            load_template("invalid")


class TestFormatOptions:
    """Tests for format_options function."""

    def test_mcq4_formats_all_options(self, sample_mcq4_record: dict) -> None:
        result = format_options(sample_mcq4_record["choices"])
        assert "A)" in result
        assert "B)" in result
        assert "C)" in result
        assert "D)" in result
        assert "KNOTTED ARABIDOPSIS THALIANA7 (KNAT7)" in result

    def test_mcq2_formats_two_options(self, sample_mcq2_record: dict) -> None:
        result = format_options(sample_mcq2_record["choices"])
        assert "A)" in result
        assert "B)" in result
        assert "C)" not in result

    def test_open_ended_returns_empty(self, sample_open_ended_record: dict) -> None:
        result = format_options(sample_open_ended_record["choices"])
        assert result == ""

    def test_none_choices_returns_empty(self) -> None:
        result = format_options(None)
        assert result == ""

    def test_dynamic_labels_for_five_choices(self) -> None:
        """When labels < texts, generate A-E dynamically."""
        choices = {
            "text": ["opt1", "opt2", "opt3", "opt4", "opt5"],
            "label": ["A", "B", "C", "D"],  # Only 4 labels for 5 texts
        }
        result = format_options(choices)
        assert "E) opt5" in result


class TestBuildPrompt:
    """Tests for build_prompt function."""

    def test_da_mcq4_complete_prompt(self, sample_mcq4_record: dict) -> None:
        prompt = build_prompt(
            question=sample_mcq4_record["question"],
            strategy="da",
            choices=sample_mcq4_record["choices"],
        )
        # Question is present
        assert "cell expansion" in prompt
        # Options are present
        assert "A)" in prompt
        assert "D)" in prompt
        # No unsubstituted placeholders
        assert "{question}" not in prompt
        assert "{options_block}" not in prompt

    def test_ras_open_ended_no_options(self, sample_open_ended_record: dict) -> None:
        prompt = build_prompt(
            question=sample_open_ended_record["question"],
            strategy="ras",
            choices=sample_open_ended_record["choices"],
        )
        assert "biomarkers" in prompt
        assert "KEY REASONING:" in prompt
        assert "A)" not in prompt

    def test_ctl_true_false_no_options(self, sample_tf_record: dict) -> None:
        prompt = build_prompt(
            question=sample_tf_record["question"],
            strategy="ctl",
            choices=sample_tf_record["choices"],
        )
        assert "gas cylinder" in prompt
        assert "RATIONALE:" in prompt
        assert "A)" not in prompt

    def test_rag_prepends_passages(
        self, sample_mcq4_record: dict, sample_passages: list[str]
    ) -> None:
        prompt = build_prompt(
            question=sample_mcq4_record["question"],
            strategy="da",
            choices=sample_mcq4_record["choices"],
            passages=sample_passages,
        )
        # RAG context appears before the question
        rag_pos = prompt.index("[1]")
        question_pos = prompt.index("cell expansion")
        assert rag_pos < question_pos
        # All passages numbered
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt
        assert "Cite relevant passages" in prompt

    def test_rag_ten_passages(self, sample_mcq4_record: dict) -> None:
        passages = [f"Passage number {i+1} content." for i in range(10)]
        prompt = build_prompt(
            question=sample_mcq4_record["question"],
            strategy="da",
            choices=sample_mcq4_record["choices"],
            passages=passages,
        )
        assert "[10]" in prompt
        assert "Passage number 10 content." in prompt

    def test_latex_preserved_in_question(self) -> None:
        question = r"Calculate $\Delta G$ for the isothermal expansion of $2.25 \mathrm{~mol}$."
        prompt = build_prompt(question=question, strategy="da")
        assert r"$\Delta G$" in prompt
        assert r"$2.25 \mathrm{~mol}$" in prompt

    def test_filling_no_options(self, sample_filling_record: dict) -> None:
        prompt = build_prompt(
            question=sample_filling_record["question"],
            strategy="da",
            choices=sample_filling_record["choices"],
        )
        assert "NH4 + NO2" in prompt
        assert "A)" not in prompt

    def test_no_unsubstituted_placeholders_any_strategy(
        self, sample_mcq4_record: dict
    ) -> None:
        for strategy in ("da", "ras", "ctl", "sc"):
            prompt = build_prompt(
                question=sample_mcq4_record["question"],
                strategy=strategy,
                choices=sample_mcq4_record["choices"],
            )
            assert "{" not in prompt, f"Unsubstituted placeholder in {strategy}"
