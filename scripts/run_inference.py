"""Closed-book inference runner for a single model x strategy cell.

Loads SciKnowEval data, builds prompts, runs vLLM batched inference,
parses responses, and writes JSONL output with checkpointing.

Usage:
    python scripts/run_inference.py --model llama-3.2-3b --strategy da
    python scripts/run_inference.py --model MOCK --strategy da --limit 10
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# ---------------------------------------------------------------------------
# Path setup (consistent with existing scripts)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from prompt_builder import build_prompt  # noqa: E402
from response_parser import (  # noqa: E402
    ParsedResponse,
    SCResult,
    aggregate_sc,
    parse_response,
)

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "closed_book.yaml"
VALID_STRATEGIES = {"da", "ras", "ctl", "sc"}
MC_TYPES = {"mcq-4-choices", "mcq-2-choices"}

# Mock templates (reused from prompt_validator.py pattern)
_MOCK_TEMPLATES: dict[str, dict[str, str]] = {
    "da": {
        "mc": "ANSWER: {letter}\nJUSTIFICATION: Based on scientific evidence this is correct.",
        "tf": "ANSWER: No\nJUSTIFICATION: The statement contradicts safety protocols.",
        "open": "ANSWER: The procedure follows standard laboratory techniques.\n"
                "JUSTIFICATION: This follows established methodology in the field.",
    },
    "ras": {
        "mc": "ANSWER: {letter}\nKEY REASONING: Analysis of options points to this answer.\n"
              "UNCERTAINTY: Minor uncertainty about edge cases.\nCONFIDENCE: high",
        "tf": "ANSWER: No\nKEY REASONING: Safety standards require specific procedures.\n"
              "UNCERTAINTY: None.\nCONFIDENCE: high",
        "open": "ANSWER: The experimental design follows standard protocols.\n"
                "KEY REASONING: Based on established literature.\n"
                "UNCERTAINTY: Sample size effects unknown.\nCONFIDENCE: medium",
    },
    "ctl": {
        "mc": "ANSWER: {letter}\nRATIONALE: Step-by-step analysis leads to this conclusion.",
        "tf": "ANSWER: No\nRATIONALE: Careful consideration of safety requirements shows this.",
        "open": "ANSWER: The method requires careful preparation.\n"
                "RATIONALE: Standard procedures dictate these steps.",
    },
    "sc": {
        "mc": "ANSWER: {letter}\nJUSTIFICATION: This is the correct answer.",
        "tf": "ANSWER: No\nJUSTIFICATION: The statement is incorrect.",
        "open": "ANSWER: Standard laboratory procedure.\n"
                "JUSTIFICATION: Follows established methods.",
    },
}


# =========================================================================
# Configuration
# =========================================================================

def load_config(config_path: Path) -> dict:
    """Load and validate closed_book.yaml configuration.

    Args:
        config_path: Path to YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file does not exist.
        KeyError: If required keys are missing.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for key in ("models", "strategies", "inference", "output", "data"):
        if key not in config:
            raise KeyError(f"Missing required config key: {key}")
    return config


def get_model_config(config: dict, model_name: str) -> dict:
    """Extract model-specific config.

    Args:
        config: Full config dict.
        model_name: Key from config['models'].

    Returns:
        Model config dict with keys: path, quantization, gpu_memory_utilization.

    Raises:
        KeyError: If model not found in config.
    """
    if model_name not in config["models"]:
        raise KeyError(
            f"Model '{model_name}' not in config. "
            f"Available: {list(config['models'].keys())}"
        )
    return config["models"][model_name]


# =========================================================================
# Data loading
# =========================================================================

def load_dataset(data_path: Path, split_name: str | None = None) -> list[dict]:
    """Load SciKnowEval JSON dataset and assign question_id to each record.

    Args:
        data_path: Path to JSON file.
        split_name: Split identifier for question_id (e.g. 'dev', 'main_test').
                    Auto-detected from filename if None.

    Returns:
        List of record dicts, each with added 'question_id' field.

    Raises:
        FileNotFoundError: If data file does not exist.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if split_name is None:
        stem = data_path.stem
        split_name = stem.replace("_", "-")

    with open(data_path, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    for idx, record in enumerate(data):
        record["question_id"] = f"ske-{split_name}-{idx:05d}"

    logger.info("Loaded %d records from %s (split=%s)", len(data), data_path.name, split_name)
    return data


def get_gold_answer(record: dict) -> str:
    """Extract gold answer from a SciKnowEval record.

    For MCQ types: returns answerKey (e.g. 'A', 'B').
    For everything else: returns answer field.

    Args:
        record: SciKnowEval question record.

    Returns:
        Gold answer string.
    """
    qtype = record.get("type", "")
    if qtype in MC_TYPES:
        return record.get("answerKey", "")
    return record.get("answer", "")


def get_choices_or_none(record: dict) -> dict[str, list] | None:
    """Return choices dict if non-empty, else None.

    Args:
        record: SciKnowEval question record.

    Returns:
        Choices dict or None.
    """
    choices = record.get("choices")
    if choices and choices.get("text"):
        return choices
    return None


# =========================================================================
# Prompt building
# =========================================================================

def build_prompts_batch(records: list[dict], strategy: str) -> list[str]:
    """Build prompts for a batch of records.

    Args:
        records: List of SciKnowEval question records.
        strategy: Prompt strategy (da, ras, ctl, sc).

    Returns:
        List of prompt strings aligned with records.
    """
    prompts: list[str] = []
    for record in records:
        prompt = build_prompt(
            question=record["question"],
            strategy=strategy,
            choices=get_choices_or_none(record),
        )
        prompts.append(prompt)
    return prompts


# =========================================================================
# vLLM engine management
# =========================================================================

def create_engine(model_cfg: dict, seed: int = 42) -> LLM:
    """Create vLLM LLM instance.

    Args:
        model_cfg: Dict with path, quantization, gpu_memory_utilization.
        seed: Global random seed.

    Returns:
        vllm.LLM instance.
    """
    from vllm import LLM as _LLM

    model_path = str(PROJECT_ROOT / model_cfg["path"])
    logger.info(
        "Loading model from %s (quantization=%s, gpu_mem=%.2f)",
        model_path,
        model_cfg.get("quantization"),
        model_cfg.get("gpu_memory_utilization", 0.9),
    )
    return _LLM(
        model=model_path,
        quantization=model_cfg.get("quantization"),
        gpu_memory_utilization=model_cfg.get("gpu_memory_utilization", 0.9),
        seed=seed,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        enforce_eager=model_cfg.get("enforce_eager", False),
        dtype="auto",
    )


def create_sampling_params(strategy: str, config: dict) -> SamplingParams:
    """Create SamplingParams based on strategy.

    Args:
        strategy: One of da, ras, ctl, sc.
        config: Full experiment config dict.

    Returns:
        vllm.SamplingParams instance.
    """
    from vllm import SamplingParams as _SamplingParams

    inf = config["inference"]
    if strategy == "sc":
        return _SamplingParams(
            temperature=inf["temperature_sc"],
            max_tokens=inf["max_tokens"],
            seed=inf["seed"],
            n=inf["sc_samples"],
        )
    return _SamplingParams(
        temperature=inf["temperature_greedy"],
        max_tokens=inf["max_tokens"],
        seed=inf["seed"],
        n=1,
    )


def release_engine(llm: Any) -> None:
    """Delete LLM and free GPU memory.

    Args:
        llm: vLLM LLM instance to release.
    """
    import torch

    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Engine released, GPU cache cleared")


# =========================================================================
# Mock engine (for testing without GPU)
# =========================================================================

class MockCompletionOutput:
    """Mimics vllm.outputs.CompletionOutput."""

    def __init__(self, text: str, token_ids: tuple[int, ...] | None = None) -> None:
        self.text = text
        self.token_ids = token_ids or tuple(range(len(text.split())))


class MockRequestOutput:
    """Mimics vllm.outputs.RequestOutput."""

    def __init__(
        self,
        outputs: list[MockCompletionOutput],
        prompt_token_ids: list[int] | None = None,
    ) -> None:
        self.outputs = outputs
        self.prompt_token_ids = prompt_token_ids or list(range(50))


def _get_question_category(record: dict) -> str:
    """Determine question category for mock selection."""
    qtype = record.get("type", "")
    if qtype in MC_TYPES:
        return "mc"
    if qtype == "true_or_false":
        return "tf"
    return "open"


def _generate_mock_text(record: dict, strategy: str) -> str:
    """Generate a single mock response text for a record."""
    category = _get_question_category(record)
    template = _MOCK_TEMPLATES[strategy][category]
    if category == "mc":
        answer_key = record.get("answerKey", "A") or "A"
        return template.format(letter=answer_key)
    return template


class MockLLM:
    """Deterministic mock for vLLM LLM, for testing without GPU.

    Generates formulaic responses that match each strategy's expected format.
    """

    def __init__(self, records: list[dict] | None = None) -> None:
        self._records = records or []

    def set_records(self, records: list[dict]) -> None:
        """Set the dataset records for context-aware mock generation."""
        self._records = records

    def generate(
        self,
        prompts: list[str] | str,
        sampling_params: Any = None,
        *,
        use_tqdm: bool = True,
    ) -> list[MockRequestOutput]:
        """Generate mock responses.

        Args:
            prompts: List of prompt strings.
            sampling_params: SamplingParams-like object (uses .n attribute).
            use_tqdm: Ignored.

        Returns:
            List of MockRequestOutput aligned with prompts.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        n_samples = getattr(sampling_params, "n", 1) if sampling_params else 1
        strategy = getattr(sampling_params, "_strategy", "da")

        results: list[MockRequestOutput] = []
        for i, _prompt in enumerate(prompts):
            record = self._records[i] if i < len(self._records) else {}
            completions: list[MockCompletionOutput] = []
            for _ in range(n_samples):
                text = _generate_mock_text(record, strategy)
                completions.append(MockCompletionOutput(text=text))
            results.append(MockRequestOutput(outputs=completions))
        return results


class _MockSamplingParams:
    """Minimal SamplingParams stand-in for MockLLM."""

    def __init__(
        self,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        seed: int | None = 42,
        n: int = 1,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.n = n
        self._strategy = "da"


# =========================================================================
# Inference execution
# =========================================================================

def run_batch(
    llm: Any,
    prompts: list[str],
    sampling_params: Any,
) -> list[Any]:
    """Run batched inference via vLLM (or MockLLM).

    Args:
        llm: vLLM LLM or MockLLM instance.
        prompts: List of prompt strings.
        sampling_params: vLLM SamplingParams or _MockSamplingParams.

    Returns:
        List of RequestOutput (or MockRequestOutput) aligned with prompts.
    """
    return llm.generate(prompts, sampling_params, use_tqdm=True)


# =========================================================================
# Response processing
# =========================================================================

def process_outputs(
    records: list[dict],
    prompts: list[str],
    outputs: list[Any],
    strategy: str,
    model_name: str,
) -> list[dict]:
    """Process vLLM outputs into serializable record dicts.

    Args:
        records: Source dataset records (aligned with outputs).
        prompts: Prompt strings (aligned with outputs).
        outputs: vLLM RequestOutput objects (aligned with records).
        strategy: Prompt strategy used.
        model_name: Short model name for the record.

    Returns:
        List of output record dicts ready for JSONL serialization.
    """
    results: list[dict] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for i, (record, prompt_text, output) in enumerate(zip(records, prompts, outputs)):
        qtype = record.get("type", "open-ended-qa")
        choices = get_choices_or_none(record)

        # Token counts
        prompt_tokens = (
            len(output.prompt_token_ids)
            if output.prompt_token_ids is not None
            else 0
        )

        if strategy == "sc":
            # SC: parse each of the n samples, then aggregate
            samples: list[ParsedResponse] = []
            total_completion_tokens = 0
            raw_texts: list[str] = []
            for comp in output.outputs:
                raw_text = comp.text
                raw_texts.append(raw_text)
                total_completion_tokens += len(comp.token_ids) if comp.token_ids else 0
                parsed = parse_response(
                    raw_response=raw_text,
                    strategy=strategy,
                    question_type=qtype,
                    choices=choices,
                    mode="closed_book",
                )
                samples.append(parsed)

            sc_result: SCResult = aggregate_sc(
                samples=samples,
                question_type=qtype,
                choices=choices,
            )

            # Top-level raw_response/parsed: use first sample
            top_raw = raw_texts[0] if raw_texts else ""
            top_parsed = samples[0].to_dict() if samples else {}

            result_record = {
                "question_id": record["question_id"],
                "question": record["question"],
                "question_type": qtype,
                "domain": record.get("domain", ""),
                "details": record.get("details", {}),
                "gold_answer": get_gold_answer(record),
                "choices": choices,
                "model": model_name,
                "strategy": strategy,
                "prompt": prompt_text,
                "raw_response": top_raw,
                "parsed": top_parsed,
                "sc_result": sc_result.to_dict(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "timestamp": now,
            }
        else:
            # DA/RAS/CTL: single sample
            comp = output.outputs[0]
            raw_text = comp.text
            completion_tokens = len(comp.token_ids) if comp.token_ids else 0

            parsed = parse_response(
                raw_response=raw_text,
                strategy=strategy,
                question_type=qtype,
                choices=choices,
                mode="closed_book",
            )

            result_record = {
                "question_id": record["question_id"],
                "question": record["question"],
                "question_type": qtype,
                "domain": record.get("domain", ""),
                "details": record.get("details", {}),
                "gold_answer": get_gold_answer(record),
                "choices": choices,
                "model": model_name,
                "strategy": strategy,
                "prompt": prompt_text,
                "raw_response": raw_text,
                "parsed": parsed.to_dict(),
                "sc_result": None,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "timestamp": now,
            }

        results.append(result_record)

    return results


# =========================================================================
# Checkpointing and output
# =========================================================================

def get_output_path(base_dir: Path, model_name: str, strategy: str) -> Path:
    """Return output JSONL path: {base_dir}/{model_name}/{strategy}.jsonl.

    Args:
        base_dir: Base output directory.
        model_name: Short model name.
        strategy: Prompt strategy.

    Returns:
        Path to JSONL output file.
    """
    return base_dir / model_name / f"{strategy}.jsonl"


def count_existing_records(output_path: Path) -> int:
    """Count valid JSONL lines in existing output file for resume.

    Args:
        output_path: Path to JSONL file.

    Returns:
        Number of valid JSON lines. 0 if file does not exist.
    """
    if not output_path.exists():
        return 0

    count = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt JSONL line at position %d", count)
    return count


def write_checkpoint(records: list[dict], output_path: Path) -> None:
    """Append records to JSONL file.

    Creates parent directories if needed. Opens in append mode and flushes.

    Args:
        records: List of record dicts to write.
        output_path: Path to JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
    logger.info("Checkpoint: wrote %d records to %s", len(records), output_path)


# =========================================================================
# Main pipeline
# =========================================================================

def run_cell(
    model_name: str,
    strategy: str,
    config: dict,
    split: str = "dev",
    limit: int | None = None,
    mock: bool = False,
    engine: Any | None = None,
) -> dict[str, Any]:
    """Run inference for a single model x strategy cell.

    Args:
        model_name: Key from config['models'] or 'MOCK'.
        strategy: One of 'da', 'ras', 'ctl', 'sc'.
        config: Loaded closed_book.yaml config.
        split: Dataset split ('dev' or 'main_test').
        limit: Max records to process (None = all).
        mock: Use MockLLM instead of real vLLM.
        engine: Pre-loaded vLLM engine (None = create new).

    Returns:
        Summary dict with total, processed, skipped, parse_rate, duration_s.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(f"Invalid strategy: {strategy}. Must be one of {VALID_STRATEGIES}")

    start_time = time.time()
    inf_config = config["inference"]

    # Load dataset
    data_path = PROJECT_ROOT / config["data"][split]
    dataset = load_dataset(data_path, split_name=split.replace("_", "-"))

    if limit is not None:
        dataset = dataset[:limit]

    # Check for resume
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    effective_model = model_name if not mock else "MOCK"
    output_path = get_output_path(output_dir, effective_model, strategy)
    existing = count_existing_records(output_path)

    if existing >= len(dataset):
        logger.info(
            "Cell %s/%s already complete (%d/%d records). Skipping.",
            effective_model, strategy, existing, len(dataset),
        )
        return {
            "model": effective_model,
            "strategy": strategy,
            "total": len(dataset),
            "processed": 0,
            "skipped": existing,
            "parse_rate": 0.0,
            "duration_s": 0.0,
            "status": "skipped",
        }

    if existing > 0:
        logger.info(
            "Resuming %s/%s from record %d/%d",
            effective_model, strategy, existing, len(dataset),
        )
        dataset = dataset[existing:]

    # Create engine
    owns_engine = False
    if engine is None:
        if mock:
            engine = MockLLM(records=dataset)
        else:
            model_cfg = get_model_config(config, model_name)
            engine = create_engine(model_cfg, seed=inf_config["seed"])
        owns_engine = True

    # Create sampling params
    if mock:
        params = _MockSamplingParams(
            temperature=inf_config["temperature_sc"] if strategy == "sc" else inf_config["temperature_greedy"],
            max_tokens=inf_config["max_tokens"],
            seed=inf_config["seed"],
            n=inf_config["sc_samples"] if strategy == "sc" else 1,
        )
        params._strategy = strategy
    else:
        params = create_sampling_params(strategy, config)

    # Process in checkpoint-sized chunks
    chunk_size = inf_config.get("checkpoint_every", 500)
    total_processed = 0
    total_parse_success = 0

    for chunk_start in range(0, len(dataset), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(dataset))
        chunk_records = dataset[chunk_start:chunk_end]

        # Update MockLLM records for this chunk
        if mock and isinstance(engine, MockLLM):
            engine.set_records(chunk_records)

        # Build prompts
        prompts = build_prompts_batch(chunk_records, strategy)

        # Run inference
        logger.info(
            "Generating %s/%s chunk [%d-%d] (%d prompts, n=%s)...",
            effective_model, strategy,
            existing + chunk_start,
            existing + chunk_end,
            len(prompts),
            getattr(params, "n", 1),
        )
        outputs = run_batch(engine, prompts, params)

        # Process outputs
        result_records = process_outputs(
            records=chunk_records,
            prompts=prompts,
            outputs=outputs,
            strategy=strategy,
            model_name=effective_model,
        )

        # Count parse successes
        for rec in result_records:
            if strategy == "sc":
                if rec.get("sc_result", {}).get("sc_success", False):
                    total_parse_success += 1
            else:
                if rec.get("parsed", {}).get("parse_success", False):
                    total_parse_success += 1

        # Write checkpoint
        write_checkpoint(result_records, output_path)
        total_processed += len(result_records)

        logger.info(
            "Progress: %d/%d processed (%.1f%% parse rate so far)",
            existing + total_processed,
            existing + len(dataset),
            total_parse_success / total_processed * 100 if total_processed else 0,
        )

    # Release engine if we created it
    if owns_engine and not mock:
        release_engine(engine)

    duration = time.time() - start_time
    parse_rate = total_parse_success / total_processed * 100 if total_processed else 0.0

    logger.info(
        "Cell %s/%s complete: %d records, %.1f%% parse rate, %.1fs",
        effective_model, strategy, total_processed, parse_rate, duration,
    )

    return {
        "model": effective_model,
        "strategy": strategy,
        "total": existing + total_processed,
        "processed": total_processed,
        "skipped": existing,
        "parse_rate": parse_rate,
        "duration_s": round(duration, 1),
        "status": "complete",
    }


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run closed-book inference for a single model x strategy cell",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model name (key from config) or 'MOCK' for testing",
    )
    parser.add_argument(
        "--strategy",
        required=True,
        choices=sorted(VALID_STRATEGIES),
        help="Prompt strategy",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "main_test"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to process (default: all)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_config(args.config)
    is_mock = args.model.upper() == "MOCK"

    summary = run_cell(
        model_name=args.model,
        strategy=args.strategy,
        config=config,
        split=args.split,
        limit=args.limit,
        mock=is_mock,
    )

    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
