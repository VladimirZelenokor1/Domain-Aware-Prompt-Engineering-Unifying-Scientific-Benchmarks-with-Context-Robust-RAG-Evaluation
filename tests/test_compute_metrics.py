"""Tests for scripts/compute_metrics.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from compute_metrics import (  # noqa: E402
    build_summary_table,
    compute_bleu_corpus,
    compute_cell_metrics,
    compute_exact_match,
    compute_rouge_l,
    discover_cells,
    get_parse_success,
    get_predicted_answer,
    load_cell,
    normalize_mc_answer,
    normalize_text_answer,
    normalize_tf_answer,
)


# =========================================================================
# Fixtures
# =========================================================================

def _make_record(
    gold: str,
    predicted: str,
    qtype: str = "mcq-4-choices",
    domain: str = "Biology",
    level: str = "L1",
    parse_success: bool = True,
    strategy: str = "da",
    model: str = "test-model",
) -> dict:
    """Create a minimal inference record for testing."""
    rec = {
        "question_id": "test-001",
        "question": "Test question?",
        "question_type": qtype,
        "domain": domain,
        "details": {"level": level, "task": "test", "subtask": "test", "source": "test"},
        "gold_answer": gold,
        "choices": None,
        "model": model,
        "strategy": strategy,
        "prompt": "test prompt",
        "raw_response": predicted,
        "parsed": {
            "answer": predicted,
            "answer_normalized": predicted,
            "parse_success": parse_success,
            "parse_errors": [] if parse_success else ["parse_failed"],
            "raw_response": predicted,
        },
        "sc_result": None,
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "timestamp": "2026-03-17T00:00:00+00:00",
    }
    return rec


def _make_sc_record(
    gold: str,
    predicted: str,
    qtype: str = "mcq-4-choices",
    sc_success: bool = True,
    **kwargs,
) -> dict:
    """Create an SC strategy record."""
    rec = _make_record(gold, predicted, qtype=qtype, strategy="sc", **kwargs)
    rec["sc_result"] = {
        "final_answer": predicted,
        "final_answer_normalized": predicted,
        "vote_counts": {predicted: 3},
        "total_samples": 5,
        "valid_samples": 5 if sc_success else 1,
        "sc_success": sc_success,
        "agreement_ratio": 0.6,
        "tie_broken": False,
    }
    return rec


# =========================================================================
# MC normalization tests
# =========================================================================

class TestNormalizeMC:
    def test_single_letter(self):
        assert normalize_mc_answer("B") == "B"

    def test_lowercase(self):
        assert normalize_mc_answer("b") == "B"

    def test_letter_with_paren(self):
        assert normalize_mc_answer("A) 1.02") == "A"

    def test_empty(self):
        assert normalize_mc_answer("") == ""

    def test_none(self):
        assert normalize_mc_answer(None) == ""

    def test_whitespace(self):
        assert normalize_mc_answer("  C  ") == "C"


class TestNormalizeTF:
    def test_yes(self):
        assert normalize_tf_answer("Yes") == "yes"

    def test_true(self):
        assert normalize_tf_answer("True") == "yes"

    def test_no(self):
        assert normalize_tf_answer("No") == "no"

    def test_false(self):
        assert normalize_tf_answer("false") == "no"

    def test_empty(self):
        assert normalize_tf_answer("") == ""


class TestNormalizeText:
    def test_basic(self):
        assert normalize_text_answer("Hello World") == "hello world"

    def test_strip(self):
        assert normalize_text_answer("  foo  ") == "foo"

    def test_none(self):
        assert normalize_text_answer(None) == ""


# =========================================================================
# Exact Match tests
# =========================================================================

class TestExactMatch:
    def test_mc_correct(self):
        assert compute_exact_match("B", "B", "mcq-4-choices") is True

    def test_mc_case_insensitive(self):
        assert compute_exact_match("b", "B", "mcq-4-choices") is True

    def test_mc_wrong(self):
        assert compute_exact_match("A", "B", "mcq-4-choices") is False

    def test_mc_empty_predicted(self):
        assert compute_exact_match("", "B", "mcq-4-choices") is False

    def test_tf_correct(self):
        assert compute_exact_match("Yes", "Yes", "true_or_false") is True

    def test_tf_true_maps_to_yes(self):
        assert compute_exact_match("True", "Yes", "true_or_false") is True

    def test_tf_wrong(self):
        assert compute_exact_match("Yes", "No", "true_or_false") is False

    def test_open_correct(self):
        assert compute_exact_match("Water", "water", "open-ended-qa") is True

    def test_open_wrong(self):
        assert compute_exact_match("ice", "water", "open-ended-qa") is False

    def test_mc2_correct(self):
        assert compute_exact_match("A", "A", "mcq-2-choices") is True


# =========================================================================
# ROUGE-L tests
# =========================================================================

class TestRougeL:
    def test_identical_strings(self):
        score = compute_rouge_l("the cat sat on the mat", "the cat sat on the mat")
        assert 0.99 <= score <= 1.0

    def test_different_strings(self):
        score = compute_rouge_l("the dog ran in the park", "the cat sat on the mat")
        assert 0.0 <= score <= 1.0

    def test_empty_predicted(self):
        assert compute_rouge_l("", "some text") == 0.0

    def test_empty_gold(self):
        assert compute_rouge_l("some text", "") == 0.0

    def test_partial_overlap(self):
        score = compute_rouge_l("the cat sat", "the cat sat on the mat")
        assert 0.3 < score < 1.0


# =========================================================================
# BLEU-4 tests
# =========================================================================

class TestBleu4:
    def test_identical(self):
        score = compute_bleu_corpus(
            ["the cat sat on the mat"],
            ["the cat sat on the mat"],
        )
        assert 0.0 <= score <= 1.01  # sacrebleu can slightly exceed 1.0 due to float precision

    def test_empty_lists(self):
        assert compute_bleu_corpus([], []) == 0.0

    def test_returns_float(self):
        score = compute_bleu_corpus(
            ["hello world foo bar baz"],
            ["hello world foo bar qux"],
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# =========================================================================
# Record helper tests
# =========================================================================

class TestRecordHelpers:
    def test_get_predicted_da(self):
        rec = _make_record("B", "B", strategy="da")
        assert get_predicted_answer(rec) == "B"

    def test_get_predicted_sc(self):
        rec = _make_sc_record("B", "B")
        assert get_predicted_answer(rec) == "B"

    def test_get_parse_success_da(self):
        rec = _make_record("B", "B", parse_success=True)
        assert get_parse_success(rec) is True

    def test_get_parse_success_da_fail(self):
        rec = _make_record("B", "", parse_success=False)
        assert get_parse_success(rec) is False

    def test_get_parse_success_sc(self):
        rec = _make_sc_record("B", "B", sc_success=True)
        assert get_parse_success(rec) is True

    def test_get_parse_success_sc_fail(self):
        rec = _make_sc_record("B", "B", sc_success=False)
        assert get_parse_success(rec) is False


# =========================================================================
# Cell metrics tests
# =========================================================================

class TestCellMetrics:
    def test_basic_mc_cell(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices", domain="Biology", level="L1"),
            _make_record("B", "B", qtype="mcq-4-choices", domain="Biology", level="L1"),
            _make_record("C", "A", qtype="mcq-4-choices", domain="Chemistry", level="L2"),
            _make_record("D", "D", qtype="mcq-4-choices", domain="Chemistry", level="L2"),
        ]
        m = compute_cell_metrics(records)
        assert m["total_questions"] == 4
        assert m["overall"]["exact_match"] == 0.75
        assert m["overall"]["count"] == 4

    def test_parse_failure_counts_as_wrong(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices"),
            _make_record("B", "", qtype="mcq-4-choices", parse_success=False),
        ]
        m = compute_cell_metrics(records)
        assert m["overall"]["exact_match"] == 0.5
        assert m["parse_rate"] == 0.5

    def test_domain_breakdown(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices", domain="Biology"),
            _make_record("B", "C", qtype="mcq-4-choices", domain="Physics"),
        ]
        m = compute_cell_metrics(records)
        assert "Biology" in m["by_domain"]
        assert "Physics" in m["by_domain"]
        assert m["by_domain"]["Biology"]["exact_match"] == 1.0
        assert m["by_domain"]["Physics"]["exact_match"] == 0.0

    def test_level_breakdown(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices", level="L1"),
            _make_record("B", "B", qtype="mcq-4-choices", level="L3"),
        ]
        m = compute_cell_metrics(records)
        assert "L1" in m["by_level"]
        assert "L3" in m["by_level"]

    def test_type_breakdown(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices"),
            _make_record("water", "water", qtype="open-ended-qa"),
        ]
        m = compute_cell_metrics(records)
        assert "mcq-4-choices" in m["by_type"]
        assert "open-ended-qa" in m["by_type"]

    def test_rouge_only_for_text_types(self):
        records = [
            _make_record("A", "A", qtype="mcq-4-choices"),
        ]
        m = compute_cell_metrics(records)
        # MC type should not have rouge_l_f1
        assert "rouge_l_f1" not in m["by_type"]["mcq-4-choices"]

    def test_rouge_present_for_open_ended(self):
        records = [
            _make_record("the cat sat", "the cat sat on mat", qtype="open-ended-qa"),
        ]
        m = compute_cell_metrics(records)
        assert "rouge_l_f1" in m["by_type"]["open-ended-qa"]
        assert 0.0 < m["by_type"]["open-ended-qa"]["rouge_l_f1"] <= 1.0

    def test_empty_records(self):
        assert compute_cell_metrics([]) == {}

    def test_sc_records(self):
        records = [
            _make_sc_record("A", "A", qtype="mcq-4-choices"),
            _make_sc_record("B", "C", qtype="mcq-4-choices"),
        ]
        m = compute_cell_metrics(records)
        assert m["overall"]["exact_match"] == 0.5


# =========================================================================
# Summary table tests
# =========================================================================

class TestSummaryTable:
    def test_schema(self):
        metrics = [
            compute_cell_metrics([
                _make_record("A", "A", qtype="mcq-4-choices", model="m1"),
            ]),
        ]
        table = build_summary_table(metrics)
        assert len(table) == 1
        row = table[0]
        assert "model" in row
        assert "strategy" in row
        assert "exact_match" in row
        assert "parse_rate" in row


# =========================================================================
# I/O tests
# =========================================================================

class TestIO:
    def test_load_cell(self, tmp_path):
        p = tmp_path / "test.jsonl"
        records = [
            {"question_id": "q1", "gold_answer": "A"},
            {"question_id": "q2", "gold_answer": "B"},
        ]
        with open(p, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        loaded = load_cell(p)
        assert len(loaded) == 2
        assert loaded[0]["question_id"] == "q1"

    def test_discover_cells(self, tmp_path):
        model_dir = tmp_path / "model1"
        model_dir.mkdir()
        (model_dir / "da.jsonl").write_text('{"a":1}\n')
        (model_dir / "ras.jsonl").write_text('{"a":1}\n')
        cells = discover_cells(tmp_path)
        assert len(cells) == 2
        assert cells[0][0] == "model1"  # model name
        assert cells[0][1] == "da"  # strategy

    def test_discover_skips_metrics_files(self, tmp_path):
        model_dir = tmp_path / "model1"
        model_dir.mkdir()
        (model_dir / "da.jsonl").write_text('{"a":1}\n')
        (model_dir / "da_metrics.jsonl").write_text('{"a":1}\n')
        cells = discover_cells(tmp_path)
        assert len(cells) == 1
        assert cells[0][1] == "da"
