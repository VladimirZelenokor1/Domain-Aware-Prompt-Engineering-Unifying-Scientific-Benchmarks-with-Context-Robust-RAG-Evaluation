"""Tests for LLM-judge scorer (scripts/run_judge.py).

All tests use MockJudgeLLM and MockNLI - no GPU, no real models required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_judge import (  # noqa: E402
    APIJudge,
    _judge_relative_path,
    build_claims_prompt,
    build_coverage_prompt,
    compute_citation_metrics,
    compute_faithfulness,
    iter_output_files,
    load_judge_prompts,
    parse_claims_response,
    parse_coverage_response,
    parse_rubric_response,
    read_jsonl,
    save_judge_output,
    score_record,
    score_records,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# =========================================================================
# Mock helpers
# =========================================================================


class MockCompletionOutput:
    """Mimics vllm.outputs.CompletionOutput."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.token_ids = tuple(range(len(text.split())))


class MockRequestOutput:
    """Mimics vllm.outputs.RequestOutput."""

    def __init__(self, outputs: list[MockCompletionOutput]) -> None:
        self.outputs = outputs
        self.prompt_token_ids = list(range(50))


class MockJudgeLLM:
    """Mock judge that returns canned rubric/claims/coverage JSON.

    Inspects prompts to determine which template was used and returns
    the appropriate JSON response.
    """

    def __init__(
        self,
        rubric_response: str | None = None,
        claims_response: str | None = None,
        coverage_response: str | None = None,
    ) -> None:
        self._rubric = rubric_response or json.dumps(
            {"rubric": 4, "rationale": "Good answer", "self_confidence": 0.85}
        )
        self._claims = claims_response or json.dumps(
            ["The sky is blue", "Water is wet"]
        )
        self._coverage = coverage_response or json.dumps(
            {
                "key_points_total": 5,
                "key_points_covered": 4,
                "missing_points": ["point5"],
            }
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any = None,
        use_tqdm: bool = True,
    ) -> list[MockRequestOutput]:
        """Return canned responses based on prompt content."""
        results: list[MockRequestOutput] = []
        for prompt in prompts:
            if "Rate the model answer" in prompt or "rubric" in prompt.lower():
                text = self._rubric
            elif "atomic factual claims" in prompt.lower():
                text = self._claims
            elif "key points" in prompt.lower() or "coverage" in prompt.lower():
                text = self._coverage
            else:
                text = self._rubric  # default fallback
            results.append(MockRequestOutput([MockCompletionOutput(text=text)]))
        return results


class MockNLI:
    """Mock CrossEncoder returning canned NLI logits.

    Args:
        entailment_prob: Probability to assign to entailment label.
            Used to generate logits such that softmax gives this probability.
    """

    def __init__(self, entailment_prob: float = 0.9) -> None:
        self._entailment_prob = entailment_prob

    def predict(
        self,
        pairs: list[list[str]],
        apply_softmax: bool = False,
    ) -> np.ndarray:
        """Return NLI scores for each pair.

        Label order: [contradiction=0, entailment=1, neutral=2].
        """
        n = len(pairs)
        if apply_softmax:
            # Return softmax probabilities directly
            remainder = (1.0 - self._entailment_prob) / 2.0
            return np.array(
                [[remainder, self._entailment_prob, remainder]] * n,
                dtype=np.float32,
            )
        # Return raw logits that produce ~entailment_prob after softmax
        # For simplicity, use log-scale values
        logit_e = np.log(self._entailment_prob + 1e-10)
        logit_other = np.log((1.0 - self._entailment_prob) / 2.0 + 1e-10)
        return np.array(
            [[logit_other, logit_e, logit_other]] * n,
            dtype=np.float32,
        )


class MockSamplingParams:
    """Minimal SamplingParams stand-in for tests."""

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 42,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture()
def templates() -> dict[str, str]:
    """Load real judge prompt templates from configs/."""
    prompts_dir = PROJECT_ROOT / "configs" / "judge_prompts"
    return load_judge_prompts(prompts_dir)


@pytest.fixture()
def judge_llm() -> MockJudgeLLM:
    """Default mock judge LLM."""
    return MockJudgeLLM()


@pytest.fixture()
def nli_model() -> MockNLI:
    """Default mock NLI model with high entailment."""
    return MockNLI(entailment_prob=0.9)


@pytest.fixture()
def sampling_params() -> MockSamplingParams:
    """Default sampling params for judge."""
    return MockSamplingParams(temperature=0.0, max_tokens=1024, seed=42)


@pytest.fixture()
def rag_record() -> dict:
    """Sample RAG inference output record with passages."""
    return {
        "question_id": "ske-dev-00001",
        "question": "What is the function of mitochondria?",
        "question_type": "open-ended-qa",
        "domain": "Biology",
        "gold_answer": "Mitochondria produce ATP through cellular respiration.",
        "model": "qwen2.5-7b",
        "strategy": "da",
        "raw_response": "Mitochondria are the powerhouse of the cell.",
        "parsed": {
            "answer": "Mitochondria are the powerhouse of the cell.",
            "answer_normalized": "mitochondria are the powerhouse of the cell.",
            "parse_success": True,
        },
        "passages_used": [
            {
                "chunk_id": "chunk_0001",
                "text": "Mitochondria generate ATP.",
                "noise_type": "real",
            },
            {
                "chunk_id": "chunk_0002",
                "text": "Chloroplasts perform photosynthesis.",
                "noise_type": "corpus_random",
            },
            {
                "chunk_id": "chunk_0003",
                "text": "ATP is the energy currency of cells.",
                "noise_type": "real",
            },
        ],
        "retriever": "hybrid",
        "noise_level": 0.4,
    }


@pytest.fixture()
def closed_book_record() -> dict:
    """Sample closed-book inference output record (no passages)."""
    return {
        "question_id": "ske-dev-00002",
        "question": "Is water a polar molecule?",
        "question_type": "mcq-2-choices",
        "domain": "Chemistry",
        "gold_answer": "A",
        "model": "qwen2.5-7b",
        "strategy": "da",
        "raw_response": "ANSWER: A\nJUSTIFICATION: Water is polar.",
        "parsed": {
            "answer": "A",
            "answer_normalized": "A",
            "parse_success": True,
        },
    }


# =========================================================================
# Prompt template loading
# =========================================================================


class TestLoadPrompts:
    """Tests for loading judge prompt templates."""

    def test_load_judge_prompts_returns_three_templates(self) -> None:
        prompts_dir = PROJECT_ROOT / "configs" / "judge_prompts"
        templates = load_judge_prompts(prompts_dir)
        assert "rubric" in templates
        assert "claims" in templates
        assert "coverage" in templates

    def test_load_judge_prompts_templates_have_placeholders(
        self,
        templates: dict[str, str],
    ) -> None:
        assert "{question}" in templates["rubric"]
        assert "{answer}" in templates["claims"]
        assert "{gold_answer}" in templates["coverage"]


# =========================================================================
# Rubric parsing
# =========================================================================


class TestParseRubricResponse:
    """Tests for rubric response parsing."""

    def test_parse_rubric_response_valid(self) -> None:
        raw = '{"rubric": 4, "rationale": "good", "self_confidence": 0.8}'
        result = parse_rubric_response(raw)
        assert result["rubric"] == 4
        assert result["rationale"] == "good"
        assert result["self_confidence"] == pytest.approx(0.8)

    def test_parse_rubric_response_malformed(self) -> None:
        raw = "This is not valid JSON at all"
        result = parse_rubric_response(raw)
        assert result["rubric"] == 0
        assert result["self_confidence"] == pytest.approx(0.0)
        assert "rationale" in result

    def test_rubric_in_valid_range(self) -> None:
        for score in range(6):
            raw = json.dumps(
                {"rubric": score, "rationale": "ok", "self_confidence": 0.5}
            )
            result = parse_rubric_response(raw)
            assert 0 <= result["rubric"] <= 5

    def test_self_confidence_in_unit_range(self) -> None:
        for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
            raw = json.dumps({"rubric": 3, "rationale": "ok", "self_confidence": conf})
            result = parse_rubric_response(raw)
            assert 0.0 <= result["self_confidence"] <= 1.0

    def test_parse_rubric_clamps_out_of_range(self) -> None:
        raw = json.dumps({"rubric": 10, "rationale": "ok", "self_confidence": 1.5})
        result = parse_rubric_response(raw)
        assert result["rubric"] <= 5
        assert result["self_confidence"] <= 1.0

    def test_parse_rubric_json_with_markdown_fence(self) -> None:
        raw = '```json\n{"rubric": 3, "rationale": "ok", "self_confidence": 0.7}\n```'
        result = parse_rubric_response(raw)
        assert result["rubric"] == 3


# =========================================================================
# Claims decomposition
# =========================================================================


class TestClaimsDecomposition:
    """Tests for claims prompt building and parsing."""

    def test_decompose_claims_returns_list(self) -> None:
        raw = '["claim one", "claim two"]'
        claims = parse_claims_response(raw)
        assert isinstance(claims, list)
        assert len(claims) == 2
        assert claims[0] == "claim one"

    def test_decompose_claims_malformed_fallback(self) -> None:
        raw = "Not a JSON array"
        claims = parse_claims_response(raw)
        assert isinstance(claims, list)
        assert len(claims) >= 1

    def test_build_claims_prompt_has_answer(self, templates: dict[str, str]) -> None:
        prompt = build_claims_prompt(templates["claims"], "Water is H2O.")
        assert "Water is H2O." in prompt

    def test_decompose_claims_markdown_fence(self) -> None:
        raw = '```json\n["a", "b", "c"]\n```'
        claims = parse_claims_response(raw)
        assert len(claims) == 3


# =========================================================================
# Faithfulness (NLI-based)
# =========================================================================


class TestFaithfulness:
    """Tests for NLI-based faithfulness scoring."""

    def test_faithfulness_all_entailed(self) -> None:
        nli = MockNLI(entailment_prob=0.9)
        claims = ["Claim A", "Claim B", "Claim C"]
        passages = [
            {"chunk_id": "c1", "text": "Supporting passage.", "noise_type": "real"},
        ]
        score = compute_faithfulness(claims, passages, nli, threshold=0.5)
        assert score == pytest.approx(1.0)

    def test_faithfulness_none_entailed(self) -> None:
        nli = MockNLI(entailment_prob=0.1)
        claims = ["Claim A", "Claim B"]
        passages = [
            {"chunk_id": "c1", "text": "Unrelated passage.", "noise_type": "real"},
        ]
        score = compute_faithfulness(claims, passages, nli, threshold=0.5)
        assert score == pytest.approx(0.0)

    def test_faithfulness_partial(self) -> None:
        """2 out of 4 claims entailed -> 0.5."""

        class PartialNLI:
            """NLI that entails first 2 claims but not the last 2."""

            def __init__(self) -> None:
                self._call_count = 0

            def predict(
                self,
                pairs: list[list[str]],
                apply_softmax: bool = False,
            ) -> np.ndarray:
                results = []
                for pair in pairs:
                    claim = pair[1]
                    if "entailed" in claim:
                        # High entailment
                        results.append([0.05, 0.9, 0.05])
                    else:
                        # Low entailment
                        results.append([0.7, 0.1, 0.2])
                return np.array(results, dtype=np.float32)

        nli = PartialNLI()
        claims = ["entailed A", "entailed B", "not supported C", "not supported D"]
        passages = [
            {"chunk_id": "c1", "text": "Some passage.", "noise_type": "real"},
        ]
        score = compute_faithfulness(claims, passages, nli, threshold=0.5)
        assert score == pytest.approx(0.5)

    def test_faithfulness_empty_claims(self) -> None:
        nli = MockNLI(entailment_prob=0.9)
        passages = [{"chunk_id": "c1", "text": "Passage.", "noise_type": "real"}]
        score = compute_faithfulness([], passages, nli, threshold=0.5)
        assert score == pytest.approx(0.0)

    def test_faithfulness_in_unit_range(self, nli_model: MockNLI) -> None:
        claims = ["Claim A", "Claim B"]
        passages = [{"chunk_id": "c1", "text": "Passage.", "noise_type": "real"}]
        score = compute_faithfulness(claims, passages, nli_model, threshold=0.5)
        assert 0.0 <= score <= 1.0


# =========================================================================
# Citation precision / recall
# =========================================================================


class TestCitationMetrics:
    """Tests for citation precision and recall."""

    def test_citation_precision_recall(self) -> None:
        """Check precision and recall math with a mix of relevant/irrelevant."""

        class SelectiveNLI:
            """NLI that marks passages containing 'supports' as entailing."""

            def predict(
                self,
                pairs: list[list[str]],
                apply_softmax: bool = False,
            ) -> np.ndarray:
                results = []
                for pair in pairs:
                    passage_text = pair[0]
                    if "supports" in passage_text:
                        results.append([0.05, 0.9, 0.05])
                    else:
                        results.append([0.7, 0.1, 0.2])
                return np.array(results, dtype=np.float32)

        nli = SelectiveNLI()
        claims = ["Some claim"]
        passages = [
            {"chunk_id": "c1", "text": "supports the claim 1", "noise_type": "real"},
            {
                "chunk_id": "c2",
                "text": "random unrelated noise",
                "noise_type": "corpus_random",
            },
            {"chunk_id": "c3", "text": "supports the claim 2", "noise_type": "real"},
        ]
        metrics = compute_citation_metrics(claims, passages, nli, threshold=0.5)
        # 2 of 3 passages are relevant -> precision = 2/3
        assert metrics["citation_precision"] == pytest.approx(2.0 / 3.0, abs=0.01)
        # 2 real passages, both relevant -> recall = 2/2 = 1.0
        assert metrics["citation_recall"] == pytest.approx(1.0)

    def test_citation_no_passages(self) -> None:
        nli = MockNLI(entailment_prob=0.9)
        claims = ["Some claim"]
        metrics = compute_citation_metrics(claims, [], nli, threshold=0.5)
        assert metrics["citation_precision"] == pytest.approx(0.0)
        assert metrics["citation_recall"] == pytest.approx(0.0)


# =========================================================================
# Coverage parsing
# =========================================================================


class TestCoverage:
    """Tests for coverage response parsing."""

    def test_coverage_parsing(self) -> None:
        raw = '{"key_points_total": 10, "key_points_covered": 7, "missing_points": ["a", "b", "c"]}'
        score = parse_coverage_response(raw)
        assert score == pytest.approx(0.7)

    def test_coverage_parsing_zero_total(self) -> None:
        raw = '{"key_points_total": 0, "key_points_covered": 0, "missing_points": []}'
        score = parse_coverage_response(raw)
        assert score == pytest.approx(0.0)

    def test_coverage_parsing_malformed(self) -> None:
        raw = "not json"
        score = parse_coverage_response(raw)
        assert score == pytest.approx(0.0)

    def test_build_coverage_prompt(self, templates: dict[str, str]) -> None:
        prompt = build_coverage_prompt(
            templates["coverage"], "gold answer", "model answer"
        )
        assert "gold answer" in prompt
        assert "model answer" in prompt


# =========================================================================
# Full pipeline (score_record)
# =========================================================================


class TestAPIJudge:
    """Tests for the proprietary-API judge (judge_c), no real network."""

    def test_provider_detected_from_model_name(self) -> None:
        assert APIJudge("claude-sonnet-4-6").provider == "anthropic"
        assert APIJudge("gpt-4o").provider == "openai"

    def test_generate_returns_vllm_shaped_outputs(self) -> None:
        judge = APIJudge("gpt-4o", cost_cap_usd=100.0)
        judge._complete = lambda prompt: ('{"rubric": 3}', 100, 20)  # type: ignore[method-assign]
        outs = judge.generate(["p1", "p2"])
        assert len(outs) == 2
        assert outs[0].outputs[0].text == '{"rubric": 3}'
        assert judge.spent_usd > 0.0

    def test_cost_cap_raises_before_overspending(self) -> None:
        judge = APIJudge("gpt-4o", cost_cap_usd=0.0)
        judge._complete = lambda prompt: ("x", 10, 10)  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="cost cap"):
            judge.generate(["p"])


class TestSaveJudgeOutput:
    """Saving must tolerate stray surrogate code points in judge text."""

    def test_source_root_disambiguates_colliding_names(self, tmp_path: Path) -> None:
        # rag_main and qasper_main share model/filename; outputs must not collide
        rec_rag = {"question_id": "rag1", "rubric": 1}
        rec_qasper = {"question_id": "qasper1", "rubric": 5}
        out_rag = save_judge_output(
            "judge_a",
            Path("outputs/rag_main/qwen2.5-7b/bm25_noise0.0_da.jsonl"),
            [rec_rag],
            tmp_path,
        )
        out_qasper = save_judge_output(
            "judge_a",
            Path("outputs/qasper_main/qwen2.5-7b/bm25_noise0.0_da.jsonl"),
            [rec_qasper],
            tmp_path,
        )
        assert out_rag != out_qasper
        assert out_rag.parent.name == "qwen2.5-7b"
        assert out_rag.parent.parent.name == "rag_main"
        assert out_qasper.parent.parent.name == "qasper_main"
        assert _judge_relative_path(
            Path("outputs/qasper_main/qwen2.5-7b/bm25_noise0.0_da.jsonl")
        ) == Path("qasper_main/qwen2.5-7b/bm25_noise0.0_da.jsonl")

    def test_handles_surrogate_in_rationale(self, tmp_path: Path) -> None:
        rec = {
            "question_id": "x",
            "rubric": 3,
            "rationale": "bad\udc0cchar",  # lone surrogate that breaks utf-8
        }
        out = save_judge_output(
            "judge_a", Path("outputs/rag_main/qwen2.5-7b/da.jsonl"), [rec], tmp_path
        )
        rows = [json.loads(line) for line in open(out, encoding="utf-8")]
        assert len(rows) == 1
        assert rows[0]["rubric"] == 3


class TestScoreRecordsBatched:
    """Tests for the batched score_records path used by the pipeline."""

    def test_batched_matches_per_record_keys_and_count(
        self,
        rag_record: dict,
        closed_book_record: dict,
        judge_llm: MockJudgeLLM,
        nli_model: MockNLI,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        sampling_set = {
            "rubric": sampling_params,
            "claims": sampling_params,
            "coverage": sampling_params,
        }
        records = [rag_record, closed_book_record]
        results = score_records(
            records=records,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_set=sampling_set,
        )
        assert len(results) == 2
        for r in results:
            assert 0 <= r["rubric"] <= 5
            assert 0.0 <= r["coverage"] <= 1.0
        # RAG record gets faithfulness; closed-book stays None
        assert results[0]["faithfulness"] is not None
        assert results[1]["faithfulness"] is None

    def test_handles_null_answer_and_gold(
        self,
        judge_llm: MockJudgeLLM,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        # Empty-output records have parsed.answer=None; gold/question may be
        # null too. Must not crash str.replace().
        rec = {
            "question_id": "x",
            "question": None,
            "gold_answer": None,
            "raw_response": None,
            "parsed": {"answer": None},
        }
        sampling_set = {
            "rubric": sampling_params,
            "claims": sampling_params,
            "coverage": sampling_params,
        }
        results = score_records(
            records=[rec],
            judge_llm=judge_llm,
            nli_model=None,
            templates=templates,
            sampling_set=sampling_set,
        )
        assert len(results) == 1
        assert 0 <= results[0]["rubric"] <= 5


class TestScoreRecord:
    """Tests for the full score_record pipeline."""

    def test_score_record_returns_all_metrics(
        self,
        rag_record: dict,
        judge_llm: MockJudgeLLM,
        nli_model: MockNLI,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        result = score_record(
            record=rag_record,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_params=sampling_params,
        )
        expected_keys = {
            "rubric",
            "faithfulness",
            "citation_precision",
            "citation_recall",
            "coverage",
            "self_confidence",
        }
        assert expected_keys.issubset(result.keys())

    def test_score_record_rubric_in_range(
        self,
        rag_record: dict,
        judge_llm: MockJudgeLLM,
        nli_model: MockNLI,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        result = score_record(
            record=rag_record,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_params=sampling_params,
        )
        assert 0 <= result["rubric"] <= 5

    def test_score_record_faithfulness_in_range(
        self,
        rag_record: dict,
        judge_llm: MockJudgeLLM,
        nli_model: MockNLI,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        result = score_record(
            record=rag_record,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_params=sampling_params,
        )
        assert 0.0 <= result["faithfulness"] <= 1.0

    def test_score_record_self_confidence_in_range(
        self,
        rag_record: dict,
        judge_llm: MockJudgeLLM,
        nli_model: MockNLI,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        result = score_record(
            record=rag_record,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_params=sampling_params,
        )
        assert 0.0 <= result["self_confidence"] <= 1.0

    def test_score_record_closed_book_no_passages(
        self,
        closed_book_record: dict,
        judge_llm: MockJudgeLLM,
        templates: dict[str, str],
        sampling_params: MockSamplingParams,
    ) -> None:
        result = score_record(
            record=closed_book_record,
            judge_llm=judge_llm,
            nli_model=None,
            templates=templates,
            sampling_params=sampling_params,
        )
        assert result["faithfulness"] is None
        assert result["citation_precision"] is None
        assert result["citation_recall"] is None
        # rubric and coverage should still be present
        assert 0 <= result["rubric"] <= 5
        assert 0.0 <= result["coverage"] <= 1.0
        assert 0.0 <= result["self_confidence"] <= 1.0


# =========================================================================
# I/O utilities
# =========================================================================


class TestIO:
    """Tests for JSONL I/O and file discovery."""

    def test_read_jsonl(self, tmp_path: Path) -> None:
        p = tmp_path / "test.jsonl"
        records = [{"a": 1}, {"b": 2}]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n")
        loaded = read_jsonl(p)
        assert len(loaded) == 2
        assert loaded[0]["a"] == 1

    def test_read_jsonl_empty(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        loaded = read_jsonl(p)
        assert loaded == []

    def test_save_judge_output(self, tmp_path: Path) -> None:
        scored = [
            {"question_id": "q1", "rubric": 4},
            {"question_id": "q2", "rubric": 3},
        ]
        out = save_judge_output(
            judge_id="judge_a",
            source_path=Path("outputs/closed_book_main/qwen2.5-7b/da.jsonl"),
            scored_records=scored,
            output_dir=tmp_path,
        )
        assert out.exists()
        assert "judge_a" in str(out)
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_iter_output_files(self, tmp_path: Path) -> None:
        # Create nested JSONL files
        d = tmp_path / "model_a"
        d.mkdir()
        (d / "da.jsonl").write_text('{"a": 1}\n')
        (d / "ras.jsonl").write_text('{"b": 2}\n')
        files = iter_output_files([tmp_path])
        assert len(files) >= 2
        assert all(f.suffix == ".jsonl" for f in files)
