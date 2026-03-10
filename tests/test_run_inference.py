"""Tests for run_inference module.

All tests use MockLLM - no GPU required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure scripts/ is importable
_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_inference import (  # noqa: E402
    MockLLM,
    _MockSamplingParams,
    build_prompts_batch,
    count_existing_records,
    create_sampling_params,
    get_choices_or_none,
    get_gold_answer,
    get_model_config,
    get_output_path,
    load_config,
    load_dataset,
    process_outputs,
    run_cell,
    write_checkpoint,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "closed_book.yaml"
DEV_DATA = PROJECT_ROOT / "data" / "sciknoweval" / "dev.json"


# =========================================================================
# Config tests
# =========================================================================


class TestConfig:
    """Tests for config loading."""

    def test_load_config_valid(self) -> None:
        config = load_config(CONFIG_PATH)
        assert "models" in config
        assert "strategies" in config
        assert "inference" in config
        assert "output" in config
        assert "data" in config
        assert config["inference"]["seed"] == 42
        assert config["inference"]["max_tokens"] == 1024
        assert len(config["models"]) == 6

    def test_load_config_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_get_model_config_unknown_raises(self) -> None:
        config = load_config(CONFIG_PATH)
        with pytest.raises(KeyError, match="not-a-model"):
            get_model_config(config, "not-a-model")


# =========================================================================
# Data loading tests
# =========================================================================


class TestDataLoading:
    """Tests for dataset loading and field extraction."""

    def test_load_dataset_adds_question_ids(self) -> None:
        data = load_dataset(DEV_DATA, split_name="dev")
        assert len(data) == 2839
        assert data[0]["question_id"] == "ske-dev-00000"
        assert data[100]["question_id"] == "ske-dev-00100"
        # All records have question_id
        assert all("question_id" in r for r in data)

    def test_get_gold_answer_mcq(self, sample_mcq4_record: dict) -> None:
        assert get_gold_answer(sample_mcq4_record) == "A"

    def test_get_gold_answer_open(self, sample_open_ended_record: dict) -> None:
        gold = get_gold_answer(sample_open_ended_record)
        assert gold == "1. Install R packages.\n2. Acquire datasets.\n3. Analyze."

    def test_get_gold_answer_tf(self, sample_tf_record: dict) -> None:
        assert get_gold_answer(sample_tf_record) == "No"

    def test_get_choices_or_none_mcq(self, sample_mcq4_record: dict) -> None:
        choices = get_choices_or_none(sample_mcq4_record)
        assert choices is not None
        assert len(choices["text"]) == 4

    def test_get_choices_or_none_open(self, sample_open_ended_record: dict) -> None:
        choices = get_choices_or_none(sample_open_ended_record)
        assert choices is None


# =========================================================================
# Prompt building tests
# =========================================================================


class TestPromptBuilding:
    """Tests for batch prompt construction."""

    def test_batch_length_matches(
        self,
        sample_mcq4_record: dict,
        sample_open_ended_record: dict,
        sample_tf_record: dict,
    ) -> None:
        records = [sample_mcq4_record, sample_open_ended_record, sample_tf_record]
        prompts = build_prompts_batch(records, "da")
        assert len(prompts) == 3

    def test_no_unsubstituted_placeholders(
        self,
        sample_mcq4_record: dict,
    ) -> None:
        prompts = build_prompts_batch([sample_mcq4_record], "ras")
        assert "{question}" not in prompts[0]
        assert "{options_block}" not in prompts[0]


# =========================================================================
# SamplingParams tests
# =========================================================================


class TestSamplingParams:
    """Tests for sampling parameter creation."""

    def test_greedy_params_da(self) -> None:
        config = load_config(CONFIG_PATH)
        params = create_sampling_params("da", config)
        assert params.temperature == 0.0
        assert params.n == 1
        assert params.seed == 42
        assert params.max_tokens == 1024

    def test_sc_params_n5(self) -> None:
        config = load_config(CONFIG_PATH)
        params = create_sampling_params("sc", config)
        assert params.temperature == 0.7
        assert params.n == 5
        assert params.seed == 42
        assert params.max_tokens == 1024


# =========================================================================
# MockLLM tests
# =========================================================================


class TestMockLLM:
    """Tests for the mock inference engine."""

    def test_generates_correct_count(self, sample_mcq4_record: dict) -> None:
        records = [sample_mcq4_record] * 3
        mock = MockLLM(records=records)
        params = _MockSamplingParams(n=1)
        outputs = mock.generate(["p1", "p2", "p3"], params)
        assert len(outputs) == 3
        assert len(outputs[0].outputs) == 1

    def test_sc_has_five_outputs(self, sample_mcq4_record: dict) -> None:
        mock = MockLLM(records=[sample_mcq4_record])
        params = _MockSamplingParams(n=5)
        outputs = mock.generate(["prompt"], params)
        assert len(outputs) == 1
        assert len(outputs[0].outputs) == 5

    def test_mock_responses_parseable(self, sample_mcq4_record: dict) -> None:
        from response_parser import parse_response

        mock = MockLLM(records=[sample_mcq4_record])
        params = _MockSamplingParams(n=1)
        params._strategy = "da"
        outputs = mock.generate(["prompt"], params)
        raw = outputs[0].outputs[0].text

        result = parse_response(
            raw_response=raw,
            strategy="da",
            question_type="mcq-4-choices",
            choices=sample_mcq4_record["choices"],
        )
        assert result.parse_success
        assert result.answer_normalized == "A"


# =========================================================================
# Process outputs tests
# =========================================================================


class TestProcessOutputs:
    """Tests for response processing pipeline."""

    def test_da_record_schema(self, sample_mcq4_record: dict) -> None:
        sample_mcq4_record["question_id"] = "ske-dev-00000"
        records = [sample_mcq4_record]
        prompts = build_prompts_batch(records, "da")

        mock = MockLLM(records=records)
        params = _MockSamplingParams(n=1)
        params._strategy = "da"
        outputs = mock.generate(prompts, params)

        result = process_outputs(records, prompts, outputs, "da", "test-model")
        assert len(result) == 1
        rec = result[0]

        # Required fields
        assert rec["question_id"] == "ske-dev-00000"
        assert rec["question_type"] == "mcq-4-choices"
        assert rec["model"] == "test-model"
        assert rec["strategy"] == "da"
        assert rec["sc_result"] is None
        assert rec["parsed"]["parse_success"] is True
        assert isinstance(rec["prompt_tokens"], int)
        assert isinstance(rec["completion_tokens"], int)
        assert "timestamp" in rec

    def test_sc_aggregation(self, sample_mcq4_record: dict) -> None:
        sample_mcq4_record["question_id"] = "ske-dev-00000"
        records = [sample_mcq4_record]
        prompts = build_prompts_batch(records, "sc")

        mock = MockLLM(records=records)
        params = _MockSamplingParams(n=5)
        params._strategy = "sc"
        outputs = mock.generate(prompts, params)

        result = process_outputs(records, prompts, outputs, "sc", "test-model")
        assert len(result) == 1
        rec = result[0]

        assert rec["sc_result"] is not None
        sc = rec["sc_result"]
        assert sc["total_samples"] == 5
        assert sc["valid_samples"] == 5
        assert sc["sc_success"] is True
        assert len(sc["per_sample"]) == 5


# =========================================================================
# Checkpoint tests
# =========================================================================


class TestCheckpointing:
    """Tests for checkpoint write/resume."""

    def test_count_existing_empty(self, tmp_path: Path) -> None:
        assert count_existing_records(tmp_path / "missing.jsonl") == 0

    def test_count_existing_valid_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write(json.dumps({"b": 2}) + "\n")
            f.write(json.dumps({"c": 3}) + "\n")
            f.write("NOT VALID JSON\n")  # corrupt line
        assert count_existing_records(path) == 3

    def test_write_checkpoint_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "output" / "model" / "da.jsonl"
        records_a = [{"id": 1}, {"id": 2}]
        records_b = [{"id": 3}, {"id": 4}]

        write_checkpoint(records_a, path)
        write_checkpoint(records_b, path)

        count = count_existing_records(path)
        assert count == 4

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 4
        assert json.loads(lines[0])["id"] == 1
        assert json.loads(lines[3])["id"] == 4


# =========================================================================
# Output path tests
# =========================================================================


class TestOutputPath:
    """Tests for output path construction."""

    def test_path_format(self) -> None:
        path = get_output_path(Path("outputs/closed_book"), "llama-3.2-3b", "da")
        assert path == Path("outputs/closed_book/llama-3.2-3b/da.jsonl")

    def test_path_sc(self) -> None:
        path = get_output_path(Path("out"), "model-x", "sc")
        assert path == Path("out/model-x/sc.jsonl")


# =========================================================================
# End-to-end tests
# =========================================================================


class TestRunCell:
    """End-to-end tests with MockLLM."""

    def test_mock_da_end_to_end(self) -> None:
        config = load_config(CONFIG_PATH)
        # Redirect output to temp dir
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["base_dir"] = tmpdir
            summary = run_cell(
                model_name="MOCK",
                strategy="da",
                config=config,
                split="dev",
                limit=5,
                mock=True,
            )

        assert summary["status"] == "complete"
        assert summary["processed"] == 5
        assert summary["parse_rate"] == 100.0

    def test_mock_sc_end_to_end(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["base_dir"] = tmpdir
            summary = run_cell(
                model_name="MOCK",
                strategy="sc",
                config=config,
                split="dev",
                limit=3,
                mock=True,
            )

        assert summary["status"] == "complete"
        assert summary["processed"] == 3
        assert summary["parse_rate"] == 100.0

    def test_resume_skips_processed(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["base_dir"] = tmpdir

            # First run: 5 records
            run_cell(
                model_name="MOCK",
                strategy="da",
                config=config,
                split="dev",
                limit=5,
                mock=True,
            )

            # Second run with same limit: should skip all
            summary = run_cell(
                model_name="MOCK",
                strategy="da",
                config=config,
                split="dev",
                limit=5,
                mock=True,
            )
            assert summary["status"] == "skipped"
            assert summary["processed"] == 0
            assert summary["skipped"] == 5

    def test_resume_continues(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["base_dir"] = tmpdir

            # First run: 3 records
            run_cell(
                model_name="MOCK",
                strategy="ras",
                config=config,
                split="dev",
                limit=3,
                mock=True,
            )

            # Second run: limit 8 - should process 5 more
            summary = run_cell(
                model_name="MOCK",
                strategy="ras",
                config=config,
                split="dev",
                limit=8,
                mock=True,
            )
            assert summary["status"] == "complete"
            assert summary["skipped"] == 3
            assert summary["processed"] == 5
            assert summary["total"] == 8

    def test_jsonl_format_valid(self) -> None:
        config = load_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as tmpdir:
            config["output"]["base_dir"] = tmpdir
            run_cell(
                model_name="MOCK",
                strategy="da",
                config=config,
                split="dev",
                limit=10,
                mock=True,
            )

            output_path = get_output_path(Path(tmpdir), "MOCK", "da")
            with open(output_path) as f:
                lines = f.readlines()
            assert len(lines) == 10

            for i, line in enumerate(lines):
                rec = json.loads(line)  # Should not raise
                assert rec["question_id"] == f"ske-dev-{i:05d}"
                assert rec["strategy"] == "da"
                assert rec["model"] == "MOCK"
                assert "parsed" in rec
                assert "prompt" in rec
