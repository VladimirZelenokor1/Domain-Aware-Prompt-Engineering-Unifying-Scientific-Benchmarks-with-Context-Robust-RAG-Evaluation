"""Tests for judge aggregation (scripts/judge_aggregate.py).

All tests are self-contained: no GPU, no real model files required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

krippendorff = pytest.importorskip("krippendorff")

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from judge_aggregate import (  # noqa: E402
    _cell_key,
    _compute_ece_mcq,
    aggregate_judge_outputs,
    aggregate_per_domain,
    compute_ece,
    compute_krippendorff_alpha,
    compute_wilcoxon,
)


# =========================================================================
# Krippendorff alpha
# =========================================================================


class TestKrippendorffAlpha:
    """Tests for compute_krippendorff_alpha."""

    def test_krippendorff_alpha_perfect_agreement(self) -> None:
        """Two raters with identical ordinal ratings -> alpha ~ 1.0."""
        ratings: dict[str, dict[str, int]] = {
            "rater_a": {f"q{i}": i % 5 for i in range(20)},
            "rater_b": {f"q{i}": i % 5 for i in range(20)},
        }
        alpha = compute_krippendorff_alpha(ratings, level="ordinal")
        assert alpha == pytest.approx(1.0, abs=1e-6)

    def test_krippendorff_alpha_random_agreement(self) -> None:
        """Raters with completely random (but seeded) disagreement -> alpha near 0.

        Exact value varies with data; we only check it is not strongly positive.
        With random disagreeing data, alpha is typically <= 0.2.
        """
        rng = np.random.default_rng(seed=0)
        ratings = {
            "rater_a": {f"q{i}": int(rng.integers(0, 5)) for i in range(50)},
            "rater_b": {f"q{i}": int(rng.integers(0, 5)) for i in range(50)},
        }
        alpha = compute_krippendorff_alpha(ratings, level="ordinal")
        assert alpha < 0.3

    def test_krippendorff_alpha_returns_float(self) -> None:
        """Return type is a Python float."""
        ratings = {
            "rater_a": {"q1": 3, "q2": 4, "q3": 2},
            "rater_b": {"q1": 3, "q2": 4, "q3": 2},
        }
        result = compute_krippendorff_alpha(ratings, level="ordinal")
        assert isinstance(result, float)

    def test_krippendorff_alpha_missing_items_handled(self) -> None:
        """Raters may not have rated all items (missing -> NaN internally)."""
        ratings = {
            "rater_a": {"q1": 4, "q2": 3, "q3": 5},
            "rater_b": {"q1": 4, "q3": 5},  # q2 missing
        }
        alpha = compute_krippendorff_alpha(ratings, level="ordinal")
        assert -1.0 <= alpha <= 1.0


# =========================================================================
# ECE
# =========================================================================


class TestComputeECE:
    """Tests for compute_ece."""

    def test_ece_perfect_calibration(self) -> None:
        """Confidence equals accuracy per bin -> ECE ~ 0.0."""
        n = 1000
        rng = np.random.default_rng(seed=42)
        confidences = rng.uniform(0.0, 1.0, size=n)
        # correctness = 1 with probability equal to confidence
        correctness = (rng.uniform(size=n) < confidences).astype(float)
        ece = compute_ece(confidences, correctness, n_bins=10)
        # Expect small ECE for well-calibrated data (tolerance for randomness)
        assert ece < 0.1

    def test_ece_overconfident(self) -> None:
        """Constant high confidence (0.9) but low accuracy (0.2) -> ECE > 0.5."""
        n = 200
        confidences = np.full(n, 0.9)
        correctness = np.where(np.arange(n) < int(n * 0.2), 1.0, 0.0)
        ece = compute_ece(confidences, correctness, n_bins=10)
        assert ece > 0.5

    def test_ece_zero_items(self) -> None:
        """Empty arrays -> ECE = 0.0."""
        ece = compute_ece(np.array([]), np.array([]), n_bins=10)
        assert ece == pytest.approx(0.0)

    def test_ece_returns_float(self) -> None:
        """Return type is a Python float."""
        ece = compute_ece(np.array([0.8, 0.6, 0.4]), np.array([1.0, 1.0, 0.0]))
        assert isinstance(ece, float)

    def test_ece_value_in_unit_range(self) -> None:
        """ECE is always in [0, 1]."""
        rng = np.random.default_rng(seed=7)
        confidences = rng.uniform(0.0, 1.0, size=100)
        correctness = rng.integers(0, 2, size=100).astype(float)
        ece = compute_ece(confidences, correctness)
        assert 0.0 <= ece <= 1.0


# =========================================================================
# Wilcoxon
# =========================================================================


class TestComputeWilcoxon:
    """Tests for compute_wilcoxon."""

    def test_wilcoxon_significant_difference(self) -> None:
        """Scores before much higher than scores after -> p < 0.05."""
        rng = np.random.default_rng(seed=0)
        scores_before = rng.uniform(3.5, 5.0, size=50)
        scores_after = rng.uniform(0.0, 1.5, size=50)
        result = compute_wilcoxon(scores_before, scores_after)
        assert result["p_value"] < 0.05
        assert result["significant"] is True

    def test_wilcoxon_no_difference(self) -> None:
        """Identical scores -> p > 0.05 (cannot reject null)."""
        scores = np.array([3.0, 4.0, 2.0, 5.0, 3.5, 4.5, 1.0, 2.5] * 5)
        result = compute_wilcoxon(scores.copy(), scores.copy())
        assert result["p_value"] > 0.05
        assert result["significant"] is False

    def test_wilcoxon_result_has_required_keys(self) -> None:
        """Result dict contains statistic, p_value, significant."""
        scores_a = np.arange(1.0, 21.0)
        scores_b = np.arange(1.0, 21.0) + 2.0
        result = compute_wilcoxon(scores_a, scores_b)
        assert "statistic" in result
        assert "p_value" in result
        assert "significant" in result

    def test_wilcoxon_significant_flag_is_bool(self) -> None:
        """significant field is a Python bool."""
        scores_a = np.ones(20)
        scores_b = np.ones(20) * 2
        result = compute_wilcoxon(scores_a, scores_b)
        assert isinstance(result["significant"], bool)


# =========================================================================
# ECE on MCQ subset (composite-key join + normalized correctness)
# =========================================================================


class TestComputeECEMcq:
    """Tests for _compute_ece_mcq (join + correctness on the MCQ subset)."""

    def _judge(self, cell: str, qid: str, conf: float) -> dict:
        return {"_cell": cell, "question_id": qid, "self_confidence": conf}

    def _source(self, cell: str, qid: str, pred: str, gold: str) -> dict:
        return {
            "_cell": cell,
            "question_id": qid,
            "question_type": "mcq-4-choices",
            "gold_answer": gold,
            "parsed": {"answer_normalized": pred, "parse_success": True},
        }

    def test_ece_mcq_joins_per_cell_not_per_qid(self) -> None:
        """Same qid in two cells must stay distinct (one pair each, not collapsed)."""
        judge = [
            self._judge("rag_main/m/bm25_noise0.0_da.jsonl", "q1", 0.9),
            self._judge("rag_main/m/hybrid_noise0.6_da.jsonl", "q1", 0.9),
        ]
        source = [
            self._source(
                "rag_main/m/bm25_noise0.0_da.jsonl", "q1", "A", "A"
            ),  # correct
            self._source(
                "rag_main/m/hybrid_noise0.6_da.jsonl", "q1", "B", "A"
            ),  # wrong
        ]
        _ece, n = _compute_ece_mcq(judge, source)
        assert n == 2  # both cells counted, not collapsed to one qid

    def test_ece_mcq_overconfident_high(self) -> None:
        """High confidence but wrong MCQ answers -> large ECE."""
        judge = [self._judge("c/m/f.jsonl", f"q{i}", 0.9) for i in range(20)]
        source = [self._source("c/m/f.jsonl", f"q{i}", "B", "A") for i in range(20)]
        ece, n = _compute_ece_mcq(judge, source)
        assert n == 20
        assert ece > 0.5

    def test_ece_mcq_normalizes_letter_labels(self) -> None:
        """'A)' vs gold 'A' counts as correct via MC normalization."""
        judge = [self._judge("c/m/f.jsonl", "q1", 0.8)]
        source = [self._source("c/m/f.jsonl", "q1", "A) because ...", "A")]
        ece, n = _compute_ece_mcq(judge, source)
        assert n == 1
        # confidence 0.8 vs accuracy 1.0 -> ECE = 0.2
        assert ece == pytest.approx(0.2, abs=1e-6)

    def test_ece_mcq_skips_non_mcq(self) -> None:
        """Non-MCQ question types are excluded -> no pairs."""
        judge = [self._judge("c/m/f.jsonl", "q1", 0.8)]
        source = [
            {
                "_cell": "c/m/f.jsonl",
                "question_id": "q1",
                "question_type": "open-ended-qa",
                "gold_answer": "x",
                "parsed": {"answer_normalized": "x", "parse_success": True},
            }
        ]
        ece, n = _compute_ece_mcq(judge, source)
        assert n == 0
        assert ece == pytest.approx(0.0)

    def test_cell_key_combines_cell_and_qid(self) -> None:
        """Composite key is '<cell>::<question_id>'."""
        rec = {"_cell": "rag_main/m/f.jsonl", "question_id": "q9"}
        assert _cell_key(rec) == "rag_main/m/f.jsonl::q9"


# =========================================================================
# Aggregate per domain
# =========================================================================


class TestAggregatePerDomain:
    """Tests for aggregate_per_domain."""

    def _make_judge_record(
        self,
        question_id: str,
        domain: str,
        rubric: int = 3,
        faithfulness: float = 0.7,
        coverage: float = 0.6,
    ) -> dict:
        return {
            "question_id": question_id,
            "judge_id": "judge_a",
            "rubric": rubric,
            "rationale": "ok",
            "faithfulness": faithfulness,
            "citation_precision": 0.5,
            "citation_recall": 0.5,
            "coverage": coverage,
            "self_confidence": 0.8,
        }

    def _make_source_record(self, question_id: str, domain: str) -> dict:
        return {
            "question_id": question_id,
            "domain": domain,
            "question": "q?",
            "gold_answer": "ans",
        }

    def test_per_domain_breakdown_keys(self) -> None:
        """Each domain entry has avg_rubric, avg_faithfulness, avg_coverage, count."""
        judge_records = [
            self._make_judge_record("q1", "Biology"),
            self._make_judge_record("q2", "Biology"),
            self._make_judge_record("q3", "Chemistry"),
        ]
        source_records = [
            self._make_source_record("q1", "Biology"),
            self._make_source_record("q2", "Biology"),
            self._make_source_record("q3", "Chemistry"),
        ]
        result = aggregate_per_domain(judge_records, source_records)
        assert "Biology" in result
        assert "Chemistry" in result
        for domain_data in result.values():
            assert "avg_rubric" in domain_data
            assert "avg_faithfulness" in domain_data
            assert "avg_coverage" in domain_data
            assert "count" in domain_data

    def test_per_domain_count_correct(self) -> None:
        """Count matches number of records per domain."""
        judge_records = [
            self._make_judge_record("q1", "Biology"),
            self._make_judge_record("q2", "Biology"),
            self._make_judge_record("q3", "Chemistry"),
        ]
        source_records = [
            self._make_source_record("q1", "Biology"),
            self._make_source_record("q2", "Biology"),
            self._make_source_record("q3", "Chemistry"),
        ]
        result = aggregate_per_domain(judge_records, source_records)
        assert result["Biology"]["count"] == 2
        assert result["Chemistry"]["count"] == 1

    def test_per_domain_avg_rubric_correct(self) -> None:
        """avg_rubric is the mean of rubric scores for that domain."""
        judge_records = [
            self._make_judge_record("q1", "Biology", rubric=4),
            self._make_judge_record("q2", "Biology", rubric=2),
        ]
        source_records = [
            self._make_source_record("q1", "Biology"),
            self._make_source_record("q2", "Biology"),
        ]
        result = aggregate_per_domain(judge_records, source_records)
        assert result["Biology"]["avg_rubric"] == pytest.approx(3.0)


# =========================================================================
# Full aggregation pipeline
# =========================================================================


class TestAggregateJudgeOutputs:
    """Tests for aggregate_judge_outputs (end-to-end with mocked filesystem)."""

    def _write_jsonl(self, path: Path, records: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as fh:
            for rec in records:
                fh.write(json.dumps(rec) + "\n")

    def _make_judge_record(
        self,
        question_id: str,
        judge_id: str = "judge_a",
        rubric: int = 3,
        self_confidence: float = 0.7,
        domain: str = "Biology",
    ) -> dict:
        return {
            "question_id": question_id,
            "judge_id": judge_id,
            "rubric": rubric,
            "rationale": "ok",
            "faithfulness": 0.6,
            "citation_precision": 0.5,
            "citation_recall": 0.5,
            "coverage": 0.7,
            "self_confidence": self_confidence,
        }

    def _make_source_record(
        self,
        question_id: str,
        domain: str = "Biology",
        question_type: str = "open-ended-qa",
        parsed_answer: str = "A",
        gold_answer: str = "A",
    ) -> dict:
        return {
            "question_id": question_id,
            "domain": domain,
            "question": "What?",
            "gold_answer": gold_answer,
            "question_type": question_type,
            "parsed": {"answer": parsed_answer, "parse_success": True},
        }

    @pytest.fixture()
    def mock_judge_dir(self, tmp_path: Path) -> Path:
        """Create mock judge outputs: judge_a and judge_b rating the same 10 items."""
        judge_dir = tmp_path / "judge"
        items = [f"q{i}" for i in range(10)]

        # judge_a and judge_b have identical rubric ratings (perfect agreement)
        for judge_id in ["judge_a", "judge_b"]:
            records = [
                self._make_judge_record(qid, judge_id=judge_id, rubric=4)
                for qid in items
            ]
            self._write_jsonl(judge_dir / judge_id / "model_a" / "da.jsonl", records)
        # judge_c rates calibration subset (first 5)
        calib_records = [
            self._make_judge_record(f"q{i}", judge_id="judge_c", rubric=4)
            for i in range(5)
        ]
        self._write_jsonl(judge_dir / "judge_c" / "model_a" / "da.jsonl", calib_records)
        return judge_dir

    @pytest.fixture()
    def mock_source_dir(self, tmp_path: Path) -> Path:
        """Create mock source output with matching question_ids."""
        source_dir = tmp_path / "outputs" / "closed_book_main"
        records = [
            self._make_source_record(f"q{i}", domain="Biology") for i in range(10)
        ]
        self._write_jsonl(source_dir / "model_a" / "da.jsonl", records)
        return source_dir

    def test_aggregate_output_has_all_keys(
        self,
        mock_judge_dir: Path,
        mock_source_dir: Path,
    ) -> None:
        """Output dict contains all required top-level keys."""
        result = aggregate_judge_outputs(
            judge_dir=mock_judge_dir,
            source_dirs=[mock_source_dir],
            config={},
        )
        required_keys = {
            "krippendorff_ab",
            "krippendorff_abc",
            "ece_mcq",
            "wilcoxon_class1",
            "wilcoxon_class2",
            "per_domain",
            "per_judge",
        }
        assert required_keys.issubset(result.keys())

    def test_per_domain_breakdown_present(
        self,
        mock_judge_dir: Path,
        mock_source_dir: Path,
    ) -> None:
        """per_domain contains at least one domain with required sub-keys."""
        result = aggregate_judge_outputs(
            judge_dir=mock_judge_dir,
            source_dirs=[mock_source_dir],
            config={},
        )
        per_domain = result["per_domain"]
        assert isinstance(per_domain, dict)
        # At least one domain must be present
        assert len(per_domain) >= 1
        for domain_data in per_domain.values():
            assert "avg_rubric" in domain_data
            assert "count" in domain_data

    def test_krippendorff_ab_perfect_agreement(
        self,
        mock_judge_dir: Path,
        mock_source_dir: Path,
    ) -> None:
        """judge_a and judge_b have identical ratings -> krippendorff_ab ~ 1.0."""
        result = aggregate_judge_outputs(
            judge_dir=mock_judge_dir,
            source_dirs=[mock_source_dir],
            config={},
        )
        assert result["krippendorff_ab"] == pytest.approx(1.0, abs=1e-5)

    def test_per_judge_stats_present(
        self,
        mock_judge_dir: Path,
        mock_source_dir: Path,
    ) -> None:
        """per_judge contains entries for each judge_id with avg metrics."""
        result = aggregate_judge_outputs(
            judge_dir=mock_judge_dir,
            source_dirs=[mock_source_dir],
            config={},
        )
        per_judge = result["per_judge"]
        assert isinstance(per_judge, dict)
        assert "judge_a" in per_judge
        for judge_stats in per_judge.values():
            assert "avg_rubric" in judge_stats
            assert "avg_faithfulness" in judge_stats
