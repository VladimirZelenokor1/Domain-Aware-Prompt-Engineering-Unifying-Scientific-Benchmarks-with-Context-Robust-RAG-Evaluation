# Phase F/H Blockers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock Phases D/F/H on the server by updating configs for 6 models and writing run_rag_inference.py, run_rag_experiment.py, run_judge.py, and judge_aggregate.py with full test coverage.

**Architecture:** RAG scripts extend the closed-book pipeline (run_inference.py / run_experiment.py) by injecting retrieval + noise assembly before prompt building. Judge scripts load a judge LLM (vLLM or API) + NLI model (DeBERTa), score each output record on 6 metrics, and aggregate inter-rater reliability.

**Tech Stack:** Python 3.10+, vLLM, sentence-transformers (CrossEncoder for NLI), PyYAML, pytest. On server: A100 40GB MIG. Locally: RTX 4060 8GB + MockLLM for tests.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `configs/closed_book.yaml` | Add 6 models, point main_test to sampled |
| Modify | `configs/rag.yaml` | Sync 6 models, fill stub |
| Modify | `models/MODEL_REGISTRY.json` | Unexclude deepseek, add gemma/mistral-nemo |
| Create | `scripts/run_rag_inference.py` | Single RAG cell: retrieve -> noise -> prompt -> infer -> save |
| Create | `scripts/run_rag_experiment.py` | 4D matrix orchestrator over run_rag_inference |
| Create | `tests/test_run_rag_inference.py` | Tests for RAG inference |
| Create | `tests/test_run_rag_experiment.py` | Tests for RAG experiment matrix |
| Create | `configs/judge_prompts/rubric_template.txt` | 6-point rubric scoring prompt |
| Create | `configs/judge_prompts/claims_template.txt` | Atomic claims decomposition prompt |
| Create | `configs/judge_prompts/coverage_template.txt` | Key-point coverage prompt |
| Create | `scripts/run_judge.py` | Judge scorer: rubric + NLI faithfulness + citations + coverage |
| Create | `scripts/judge_aggregate.py` | Krippendorff alpha, ECE, Wilcoxon, per-domain breakdown |
| Create | `tests/test_run_judge.py` | Tests for judge scorer |
| Create | `tests/test_judge_aggregate.py` | Tests for aggregation |

---

## Task 1: Update configs for 6 models on A100

**Files:**
- Modify: `configs/closed_book.yaml`
- Modify: `configs/rag.yaml`
- Modify: `models/MODEL_REGISTRY.json`

- [ ] **Step 1: Update closed_book.yaml**

```yaml
experiment: closed_book

data:
  dev: data/sciknoweval/dev.json
  main_test: data/sciknoweval/main_test_sampled.json

models:
  llama-3.2-3b:
    path: models/llama-3.2-3b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  qwen2.5-7b:
    path: models/qwen2.5-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  sciphi-mistral-7b:
    path: models/sciphi-mistral-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  deepseek-r1-qwen-7b:
    path: models/deepseek-r1-qwen-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  gemma-2-9b:
    path: models/gemma-2-9b-awq
    quantization: awq_marlin
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  mistral-nemo-12b:
    path: models/mistral-nemo-12b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16

strategies:
  - da
  - ras
  - ctl
  - sc

inference:
  seed: 42
  max_tokens: 1024
  temperature_greedy: 0.0
  temperature_sc: 0.7
  sc_samples: 5
  sc_min_valid: 3
  checkpoint_every: 500

output:
  base_dir: outputs/closed_book
```

- [ ] **Step 2: Update rag.yaml - sync 6 models, fill stub**

```yaml
experiment: rag

data:
  track_a:
    dev: data/sciknoweval/dev.json
    main_test: data/sciknoweval/main_test_sampled.json
  track_b:
    sample: data/qasper/sample.json

models:
  llama-3.2-3b:
    path: models/llama-3.2-3b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  qwen2.5-7b:
    path: models/qwen2.5-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  sciphi-mistral-7b:
    path: models/sciphi-mistral-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  deepseek-r1-qwen-7b:
    path: models/deepseek-r1-qwen-7b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  gemma-2-9b:
    path: models/gemma-2-9b-awq
    quantization: awq_marlin
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16
  mistral-nemo-12b:
    path: models/mistral-nemo-12b-awq
    quantization: awq
    gpu_memory_utilization: 0.85
    max_model_len: 4096
    enforce_eager: true
    dtype: float16

strategies:
  - da
  - ras
  - ctl
  - sc

retrievers:
  - bm25
  - dense
  - hybrid

noise:
  config: configs/noise.yaml
  levels: [0.0, 0.2, 0.4, 0.6]

inference:
  seed: 42
  max_tokens: 1024
  temperature_greedy: 0.0
  temperature_sc: 0.7
  sc_samples: 5
  sc_min_valid: 3
  top_k_passages: 10
  checkpoint_every: 500

output:
  base_dir: outputs/rag_main
```

- [ ] **Step 3: Update MODEL_REGISTRY.json**

Add entries for gemma-2-9b-awq and mistral-nemo-12b-awq under `llms`. Remove `"excluded": "hardware_constraint"` from deepseek-r1-qwen-7b-awq. Add `"note": "Re-enabled for A100 40GB server experiments"`.

- [ ] **Step 4: Run existing tests to verify no regression**

Run: `pytest tests/test_run_inference.py -x -q`
Expected: all pass (the config test expects `len(config["models"]) == 4` - this will need updating to 6 in the test).

- [ ] **Step 5: Fix test_run_inference.py model count assertion**

In `tests/test_run_inference.py:58`, change:
```python
assert len(config["models"]) == 4
```
to:
```python
assert len(config["models"]) == 6
```

- [ ] **Step 6: Run tests again**

Run: `pytest tests/test_run_inference.py -x -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add configs/closed_book.yaml configs/rag.yaml models/MODEL_REGISTRY.json tests/test_run_inference.py
git commit -m "feat: enable all 6 models for A100 40GB, point main_test to sampled"
```

---

## Task 2: RAG inference runner - tests

**Files:**
- Create: `tests/test_run_rag_inference.py`

This task writes all RAG inference tests FIRST (TDD). The tests use MockLLM + MockRetriever - no GPU needed.

- [ ] **Step 1: Write test file with MockRetriever and core tests**

```python
"""Tests for run_rag_inference module.

All tests use MockLLM + MockRetriever - no GPU required.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


# =========================================================================
# Mock retriever
# =========================================================================


class MockRetriever:
    """Deterministic mock for Retriever - returns canned passages."""

    def __init__(self, top_k: int = 10) -> None:
        self._top_k = top_k
        self.call_count = 0

    def _make_passages(self, query: str, top_k: int) -> list[dict]:
        self.call_count += 1
        return [
            {
                "chunk_id": f"mock_chunk_{i:03d}",
                "text": f"Mock passage {i} for query: {query[:30]}",
                "score": 1.0 - i * 0.05,
                "rank": i + 1,
            }
            for i in range(top_k)
        ]

    def retrieve_bm25(self, query: str, top_k: int = 10) -> list[dict]:
        return self._make_passages(query, top_k)

    def retrieve_dense(self, query: str, top_k: int = 10) -> list[dict]:
        return self._make_passages(query, top_k)

    def retrieve_hybrid(self, query: str, top_k: int = 10) -> list[dict]:
        return self._make_passages(query, top_k)


# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def rag_config() -> dict:
    with open(RAG_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    return tmp_path / "rag_out"


@pytest.fixture
def noise_pools(tmp_path: Path) -> dict:
    """Create minimal noise pool files and return paths dict."""
    pool_dir = tmp_path / "corpus" / "noise"
    pool_dir.mkdir(parents=True)
    for name, prefix in [
        ("irrelevant_distractors", "irr"),
        ("injection_passages", "inj"),
        ("contradictory_passages", "con"),
    ]:
        path = pool_dir / f"{name}.jsonl"
        lines = []
        for i in range(20):
            rec = {"noise_id": f"{prefix}_{i:03d}", "text": f"{prefix} passage {i}"}
            if prefix == "irr":
                rec["chunk_id"] = rec["noise_id"]
                rec["domain"] = ["biology", "chemistry", "physics"][i % 3]
            lines.append(json.dumps(rec))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "irrelevant": str(pool_dir / "irrelevant_distractors.jsonl"),
        "injection": str(pool_dir / "injection_passages.jsonl"),
        "contradictory": str(pool_dir / "contradictory_passages.jsonl"),
    }


# =========================================================================
# Config loading
# =========================================================================


class TestRagConfig:
    def test_load_rag_config_has_required_keys(self, rag_config: dict) -> None:
        assert "models" in rag_config
        assert "retrievers" in rag_config
        assert "noise" in rag_config
        assert "strategies" in rag_config
        assert len(rag_config["models"]) == 6

    def test_rag_config_has_noise_levels(self, rag_config: dict) -> None:
        assert rag_config["noise"]["levels"] == [0.0, 0.2, 0.4, 0.6]

    def test_rag_config_has_three_retrievers(self, rag_config: dict) -> None:
        assert set(rag_config["retrievers"]) == {"bm25", "dense", "hybrid"}


# =========================================================================
# Output path format
# =========================================================================


class TestRagOutputPath:
    def test_rag_output_path_format(self) -> None:
        from run_rag_inference import get_rag_output_path

        path = get_rag_output_path(
            Path("outputs/rag_main"), "qwen2.5-7b", "hybrid", 0.4, "ras"
        )
        assert path == Path("outputs/rag_main/qwen2.5-7b/hybrid_noise0.4_ras.jsonl")

    def test_rag_output_path_noise_zero(self) -> None:
        from run_rag_inference import get_rag_output_path

        path = get_rag_output_path(
            Path("out"), "llama-3.2-3b", "bm25", 0.0, "da"
        )
        assert path == Path("out/llama-3.2-3b/bm25_noise0.0_da.jsonl")


# =========================================================================
# RAG cell end-to-end (mock)
# =========================================================================


class TestRunRagCell:
    def test_mock_rag_da_produces_correct_records(
        self, rag_config: dict, mock_retriever: MockRetriever,
        tmp_output: Path, noise_pools: dict,
    ) -> None:
        from run_rag_inference import run_rag_cell

        rag_config["output"]["base_dir"] = str(tmp_output)

        summary = run_rag_cell(
            model_name="llama-3.2-3b",
            strategy="da",
            retriever_mode="bm25",
            noise_level=0.0,
            config=rag_config,
            split="dev",
            limit=5,
            mock=True,
            retriever=mock_retriever,
            noise_pool_paths=noise_pools,
        )

        assert summary["status"] == "complete"
        assert summary["processed"] == 5

        # Verify output JSONL
        from run_rag_inference import get_rag_output_path
        out_path = get_rag_output_path(tmp_output, "MOCK", "bm25", 0.0, "da")
        with open(out_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        assert len(records) == 5

        # Each record must have RAG-specific fields
        for rec in records:
            assert "passages_used" in rec
            assert isinstance(rec["passages_used"], list)
            assert len(rec["passages_used"]) > 0
            assert "retriever" in rec
            assert rec["retriever"] == "bm25"
            assert "noise_level" in rec
            assert rec["noise_level"] == 0.0

    def test_mock_rag_sc_produces_sc_result(
        self, rag_config: dict, mock_retriever: MockRetriever,
        tmp_output: Path, noise_pools: dict,
    ) -> None:
        from run_rag_inference import run_rag_cell

        rag_config["output"]["base_dir"] = str(tmp_output)

        summary = run_rag_cell(
            model_name="llama-3.2-3b",
            strategy="sc",
            retriever_mode="hybrid",
            noise_level=0.0,
            config=rag_config,
            split="dev",
            limit=3,
            mock=True,
            retriever=mock_retriever,
            noise_pool_paths=noise_pools,
        )

        assert summary["status"] == "complete"
        assert summary["processed"] == 3

    def test_resume_skips_processed_records(
        self, rag_config: dict, mock_retriever: MockRetriever,
        tmp_output: Path, noise_pools: dict,
    ) -> None:
        from run_rag_inference import run_rag_cell

        rag_config["output"]["base_dir"] = str(tmp_output)
        kwargs = dict(
            model_name="llama-3.2-3b",
            strategy="da",
            retriever_mode="bm25",
            noise_level=0.0,
            config=rag_config,
            split="dev",
            limit=5,
            mock=True,
            retriever=mock_retriever,
            noise_pool_paths=noise_pools,
        )

        run_rag_cell(**kwargs)
        summary = run_rag_cell(**kwargs)
        assert summary["status"] == "skipped"
        assert summary["processed"] == 0

    def test_citations_extracted_in_rag_mode(
        self, rag_config: dict, mock_retriever: MockRetriever,
        tmp_output: Path, noise_pools: dict,
    ) -> None:
        from run_rag_inference import run_rag_cell

        rag_config["output"]["base_dir"] = str(tmp_output)

        run_rag_cell(
            model_name="llama-3.2-3b",
            strategy="da",
            retriever_mode="hybrid",
            noise_level=0.0,
            config=rag_config,
            split="dev",
            limit=3,
            mock=True,
            retriever=mock_retriever,
            noise_pool_paths=noise_pools,
        )

        from run_rag_inference import get_rag_output_path
        out_path = get_rag_output_path(tmp_output, "MOCK", "hybrid", 0.0, "da")
        with open(out_path) as f:
            records = [json.loads(line) for line in f if line.strip()]

        # parsed should have citations key (may be empty for mock responses)
        for rec in records:
            assert "citations" in rec["parsed"]
```

- [ ] **Step 2: Run tests to confirm they fail**

Run: `pytest tests/test_run_rag_inference.py -x -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_rag_inference'`

- [ ] **Step 3: Commit test file**

```bash
git add tests/test_run_rag_inference.py
git commit -m "test: add RAG inference tests (red phase)"
```

---

## Task 3: RAG inference runner - implementation

**Files:**
- Create: `scripts/run_rag_inference.py`

- [ ] **Step 1: Write run_rag_inference.py**

Key design decisions:
- Imports `load_dataset`, `load_config`, `create_engine`, `release_engine`, `create_sampling_params`, `count_existing_records`, `write_checkpoint`, `MockLLM`, `_MockSamplingParams`, `truncate_long_prompts`, `get_gold_answer`, `get_choices_or_none` from `run_inference`.
- Imports `Retriever` from `retriever` (or uses injected mock).
- Imports `load_pools`, `assemble_noisy_context` from `noise_assembler`.
- Imports `build_prompt` from `prompt_builder` (already supports `passages` param).
- Imports `parse_response` from `response_parser` (with `mode="rag"`).
- Imports `assert_noise_gate` from `rag_gate`.
- New function `get_rag_output_path(base_dir, model, retriever, noise_level, strategy)` returns `{base_dir}/{model}/{retriever}_noise{noise_level}_{strategy}.jsonl`.
- New function `run_rag_cell(...)` - main loop: for each question, retrieve -> assemble noise -> build prompt (with passages) -> infer -> parse (mode=rag) -> save record with `passages_used`, `retriever`, `noise_level`.
- Resumable via `count_existing_records`.
- At startup: `assert_noise_gate(noise_level, noise_config)` for levels > 0.0.

The full implementation code is ~350 lines. Core function signature:

```python
def run_rag_cell(
    model_name: str,
    strategy: str,
    retriever_mode: str,
    noise_level: float,
    config: dict,
    split: str = "dev",
    limit: int | None = None,
    mock: bool = False,
    engine: Any | None = None,
    retriever: Any | None = None,
    noise_pool_paths: dict[str, str] | None = None,
) -> dict[str, Any]:
```

Output record extends closed-book record with:
```python
{
    # ... all closed-book fields ...
    "passages_used": [{"chunk_id": "...", "noise_type": "real|irrelevant|..."}],
    "retriever": "bm25|dense|hybrid",
    "noise_level": 0.0,
}
```

`parse_response` is called with `mode="rag"` to extract citations.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_run_rag_inference.py -x -v`
Expected: PASS

- [ ] **Step 3: Run linter**

Run: `ruff check scripts/run_rag_inference.py`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add scripts/run_rag_inference.py
git commit -m "feat: RAG inference runner for Phase F"
```

---

## Task 4: RAG experiment orchestrator - tests

**Files:**
- Create: `tests/test_run_rag_experiment.py`

- [ ] **Step 1: Write test file**

```python
"""Tests for run_rag_experiment module."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_CONFIG_PATH = PROJECT_ROOT / "configs" / "rag.yaml"


@pytest.fixture
def rag_config() -> dict:
    with open(RAG_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


class TestBuildRagMatrix:
    def test_full_matrix_size(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        matrix = build_rag_matrix(rag_config, mode="full")
        # 6 models x 4 strategies x 3 retrievers x 4 noise levels = 288
        assert len(matrix) == 288

    def test_fractional_matrix_size(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        matrix = build_rag_matrix(rag_config, mode="fractional")
        # hybrid x 4 noise x 6 models x 4 strats = 96
        # + bm25,dense x noise=0.0 x 6 x 4 = 48
        # = 144
        assert len(matrix) == 144

    def test_fractional_matrix_no_duplicates(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        matrix = build_rag_matrix(rag_config, mode="fractional")
        assert len(matrix) == len(set(matrix))

    def test_full_matrix_tuple_structure(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        matrix = build_rag_matrix(rag_config, mode="full")
        for cell in matrix:
            assert len(cell) == 4  # (model, strategy, retriever, noise_level)

    def test_unknown_mode_raises(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        with pytest.raises(ValueError, match="Unknown matrix mode"):
            build_rag_matrix(rag_config, mode="bogus")

    def test_outer_loop_is_model(self, rag_config: dict) -> None:
        from run_rag_experiment import build_rag_matrix

        matrix = build_rag_matrix(rag_config, mode="full")
        # First model's cells should come before second model's cells
        models = [cell[0] for cell in matrix]
        first_model = models[0]
        # All cells for first model are contiguous
        first_model_end = next(
            i for i, m in enumerate(models) if m != first_model
        )
        assert all(m == first_model for m in models[:first_model_end])
```

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_run_rag_experiment.py -x -q`
Expected: FAIL

- [ ] **Step 3: Commit**

```bash
git add tests/test_run_rag_experiment.py
git commit -m "test: add RAG experiment orchestrator tests (red phase)"
```

---

## Task 5: RAG experiment orchestrator - implementation

**Files:**
- Create: `scripts/run_rag_experiment.py`

- [ ] **Step 1: Write run_rag_experiment.py**

Key design:
- `build_rag_matrix(config, mode)` returns `list[tuple[str, str, str, float]]` - (model, strategy, retriever, noise_level).
- `mode="full"`: 6x4x3x4 = 288 cells.
- `mode="fractional"`: hybrid x all noise (96) + bm25/dense x noise=0.0 (48) = 144 cells. Deduplicated.
- Outer loop = model (minimizes LLM loads). Retriever created once outside model loop.
- `run_rag_experiment(config, ...)` iterates matrix, calls `run_rag_cell` per cell.
- Resumable: checks `count_existing_records(output_path) >= dataset_size` per cell.
- After each model, `release_engine(llm); torch.cuda.empty_cache()`.
- CLI: `--config`, `--matrix full|fractional`, `--split`, `--models`, `--strategies`, `--retrievers`, `--noise-levels`, `--limit`, `--mock`, `--dry-run`.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_run_rag_experiment.py -x -v`
Expected: PASS

- [ ] **Step 3: Run linter**

Run: `ruff check scripts/run_rag_experiment.py`
Expected: no errors

- [ ] **Step 4: Commit**

```bash
git add scripts/run_rag_experiment.py
git commit -m "feat: RAG experiment orchestrator for Phase F"
```

---

## Task 6: Judge prompt templates

**Files:**
- Create: `configs/judge_prompts/rubric_template.txt`
- Create: `configs/judge_prompts/claims_template.txt`
- Create: `configs/judge_prompts/coverage_template.txt`

- [ ] **Step 1: Create rubric_template.txt**

```text
You are an expert scientific evaluator. Rate the quality of the model's answer compared to the gold answer.

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{model_answer}

{passages_block}

Rate the model answer on a scale of 0-5:
- 0: Completely wrong, irrelevant, or empty
- 1: Mostly wrong with minor correct elements
- 2: Partially correct but major errors or omissions
- 3: Roughly correct but with notable errors or missing details
- 4: Mostly correct with only minor issues
- 5: Fully correct and complete

Respond with ONLY a JSON object:
{{"rubric": <0-5>, "rationale": "<brief explanation>", "self_confidence": <0.0-1.0>}}
```

- [ ] **Step 2: Create claims_template.txt**

```text
Decompose the following answer into a list of atomic factual claims. Each claim should be a single verifiable statement.

ANSWER:
{answer}

Respond with ONLY a JSON array of strings, e.g.:
["claim 1", "claim 2", "claim 3"]
```

- [ ] **Step 3: Create coverage_template.txt**

```text
Compare the model answer against the gold answer and identify key points.

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{model_answer}

List each key point from the gold answer and whether the model answer covers it.

Respond with ONLY a JSON object:
{{"key_points_total": <int>, "key_points_covered": <int>, "missing_points": ["point1", ...]}}
```

- [ ] **Step 4: Commit**

```bash
mkdir -p configs/judge_prompts
git add configs/judge_prompts/rubric_template.txt configs/judge_prompts/claims_template.txt configs/judge_prompts/coverage_template.txt
git commit -m "feat: judge prompt templates for Phase H"
```

---

## Task 7: Judge scorer - tests

**Files:**
- Create: `tests/test_run_judge.py`

- [ ] **Step 1: Write test file**

Tests use MockJudgeLLM + MockNLI. Key tests:
- `test_score_record_returns_all_metrics` - verify all 6 metric keys present
- `test_rubric_in_valid_range` - rubric score 0-5
- `test_faithfulness_in_unit_range` - faithfulness 0.0-1.0
- `test_citation_precision_recall_range` - both 0.0-1.0
- `test_coverage_in_unit_range` - coverage 0.0-1.0
- `test_self_confidence_in_unit_range` - 0.0-1.0
- `test_parse_rubric_response_valid_json` - parse LLM rubric output
- `test_parse_rubric_response_malformed_falls_back` - graceful degradation
- `test_decompose_claims_returns_list` - claims decomposition
- `test_faithfulness_all_entailed` - 100% entailment -> faithfulness=1.0
- `test_faithfulness_none_entailed` - 0% entailment -> faithfulness=0.0

- [ ] **Step 2: Run tests to confirm failure**

Run: `pytest tests/test_run_judge.py -x -q`
Expected: FAIL

- [ ] **Step 3: Commit**

```bash
git add tests/test_run_judge.py
git commit -m "test: add judge scorer tests (red phase)"
```

---

## Task 8: Judge scorer - implementation

**Files:**
- Create: `scripts/run_judge.py`

- [ ] **Step 1: Write run_judge.py**

Key design:
- `score_record(llm, nli_model, record, scoring_config, prompt_templates)` returns dict with 6 metrics.
- Rubric: build prompt from template, call LLM, parse JSON response, extract `rubric` (0-5) and `self_confidence` (0-1).
- Claims decomposition: build prompt from template, call LLM, parse JSON list.
- Faithfulness: for each claim, run NLI against each passage, `faithfulness = fraction(claims entailed by ANY passage)`. NLI uses `nli-deberta-v3-large` with label order [contradiction=0, entailment=1, neutral=2].
- Citation precision/recall: NLI-based. `relevant_passage = max(NLI(passage, claim).entailment for claim in claims) >= 0.5`. Precision = relevant/cited, recall = relevant/gold-relevant.
- Coverage: LLM-based via coverage template.
- CLI: `--judge judge_a|judge_b|judge_c`, `--target outputs/...`, `--calibration-subset N`, `--cost-cap-usd N`.
- For judge_a/judge_b: load via vLLM. For judge_c: API client (Anthropic or OpenAI).
- Resumable: per-output-file, skip records already judged.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_run_judge.py -x -v`
Expected: PASS

- [ ] **Step 3: Run linter**

Run: `ruff check scripts/run_judge.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/run_judge.py
git commit -m "feat: LLM-judge scorer for Phase H"
```

---

## Task 9: Judge aggregation - tests

**Files:**
- Create: `tests/test_judge_aggregate.py`

- [ ] **Step 1: Write test file**

Key tests:
- `test_krippendorff_alpha_perfect_agreement` - two identical raters -> alpha ~ 1.0
- `test_krippendorff_alpha_random_agreement` - random raters -> alpha ~ 0.0
- `test_ece_perfect_calibration` - confidence matches accuracy -> ECE ~ 0.0
- `test_ece_overconfident` - high confidence, low accuracy -> high ECE
- `test_aggregate_output_has_all_keys` - output dict has required fields
- `test_per_domain_breakdown_present` - per-domain metrics in output

- [ ] **Step 2: Run tests, confirm failure**

- [ ] **Step 3: Commit**

```bash
git add tests/test_judge_aggregate.py
git commit -m "test: add judge aggregation tests (red phase)"
```

---

## Task 10: Judge aggregation - implementation

**Files:**
- Create: `scripts/judge_aggregate.py`

- [ ] **Step 1: Write judge_aggregate.py**

Key functions:
- `compute_krippendorff_alpha(ratings_matrix, level="ordinal")` - wraps `krippendorff.alpha`.
- `compute_ece(confidences, correctness, n_bins=10)` - Expected Calibration Error.
- `compute_wilcoxon_perturbation(scores_before, scores_after)` - Wilcoxon signed-rank test.
- `aggregate_judge_outputs(judge_dirs, config)` - main function:
  - Load all judge JSONL files
  - Compute Krippendorff alpha (A,B) on full set, (A,B,C) on calibration subset
  - Compute ECE on MCQ subset
  - Per-domain breakdown of rubric, faithfulness
  - Output: `outputs/judge/aggregate_stats.json`
- CLI: `--judge-dir outputs/judge`, `--config configs/judge.yaml`

Dependencies: `krippendorff`, `scipy.stats` (for Wilcoxon).

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_judge_aggregate.py -x -v`
Expected: PASS

- [ ] **Step 3: Run linter**

Run: `ruff check scripts/judge_aggregate.py`

- [ ] **Step 4: Commit**

```bash
git add scripts/judge_aggregate.py
git commit -m "feat: judge aggregation with Krippendorff alpha and ECE"
```

---

## Commit Summary

| # | Message | Files |
|---|---------|-------|
| 1 | `feat: enable all 6 models for A100 40GB, point main_test to sampled` | configs/*, models/MODEL_REGISTRY.json, tests/test_run_inference.py |
| 2 | `test: add RAG inference tests (red phase)` | tests/test_run_rag_inference.py |
| 3 | `feat: RAG inference runner for Phase F` | scripts/run_rag_inference.py |
| 4 | `test: add RAG experiment orchestrator tests (red phase)` | tests/test_run_rag_experiment.py |
| 5 | `feat: RAG experiment orchestrator for Phase F` | scripts/run_rag_experiment.py |
| 6 | `feat: judge prompt templates for Phase H` | configs/judge_prompts/*.txt |
| 7 | `test: add judge scorer tests (red phase)` | tests/test_run_judge.py |
| 8 | `feat: LLM-judge scorer for Phase H` | scripts/run_judge.py |
| 9 | `test: add judge aggregation tests (red phase)` | tests/test_judge_aggregate.py |
| 10 | `feat: judge aggregation with Krippendorff alpha and ECE` | scripts/judge_aggregate.py |
