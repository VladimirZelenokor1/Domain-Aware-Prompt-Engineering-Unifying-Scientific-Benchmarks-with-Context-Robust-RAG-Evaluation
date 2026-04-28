"""RAG inference runner for a single model x strategy x retriever x noise cell.

Extends the closed-book pipeline with retrieval, noise injection, and
citation extraction. Reuses as much as possible from run_inference.py.

Usage:
    python scripts/run_rag_inference.py \
        --config configs/rag.yaml \
        --model qwen2.5-7b \
        --strategy ras \
        --retriever hybrid \
        --noise-level 0.4 \
        --split dev

    python scripts/run_rag_inference.py \
        --model MOCK --strategy da --retriever bm25 --noise-level 0.0 --limit 10
"""

from __future__ import annotations

# Ensure CC is set early for triton kernel compilation in vLLM spawned
# subprocesses (WSL2 uses 'spawn' multiprocessing which loses PATH context).
import os as _os
import shutil as _shutil

if "CC" not in _os.environ:
    _cc = _shutil.which("gcc") or _shutil.which("cc") or "/usr/bin/gcc"
    _os.environ["CC"] = _cc

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Path setup (consistent with existing scripts)
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from noise_assembler import assemble_noisy_context, load_pools  # noqa: E402
from prompt_builder import build_prompt  # noqa: E402
from rag_gate import assert_noise_gate  # noqa: E402
from response_parser import (  # noqa: E402
    parse_response,
)
from run_inference import (  # noqa: E402
    MockLLM,
    _MockSamplingParams,
    count_existing_records,
    create_engine,
    create_sampling_params,
    get_choices_or_none,
    get_model_config,
    load_config,
    load_dataset,
    process_outputs,
    release_engine,
    run_batch,
    truncate_long_prompts,
    write_checkpoint,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "rag.yaml"
VALID_STRATEGIES = {"da", "ras", "ctl", "sc"}
VALID_RETRIEVERS = {"bm25", "dense", "hybrid"}


# =========================================================================
# Helpers
# =========================================================================


def resolve_split_path(config: dict, split: str) -> Path:
    """Resolve split name to data path from RAG config.

    Args:
        config: Loaded rag.yaml config dict.
        split: One of 'dev', 'main_test', 'track_b'.

    Returns:
        Absolute path to the dataset JSON file.

    Raises:
        ValueError: If split name is unknown.
    """
    data = config["data"]
    if split == "dev":
        return PROJECT_ROOT / data["track_a"]["dev"]
    elif split == "main_test":
        return PROJECT_ROOT / data["track_a"]["main_test"]
    elif split == "track_b":
        return PROJECT_ROOT / data["track_b"]["sample"]
    else:
        raise ValueError(f"Unknown split: {split}")


def get_rag_output_path(
    base_dir: Path,
    model: str,
    retriever: str,
    noise_level: float,
    strategy: str,
) -> Path:
    """Return output JSONL path for a RAG cell.

    Format: {base_dir}/{model}/{retriever}_noise{noise_level}_{strategy}.jsonl

    Args:
        base_dir: Base output directory.
        model: Short model name.
        retriever: Retriever mode (bm25, dense, hybrid).
        noise_level: Noise level float (0.0, 0.2, 0.4, 0.6).
        strategy: Prompt strategy.

    Returns:
        Path to JSONL output file.
    """
    return base_dir / model / f"{retriever}_noise{noise_level}_{strategy}.jsonl"


def _load_noise_config(config: dict) -> dict:
    """Load the noise config referenced in rag.yaml.

    Args:
        config: Loaded rag.yaml config dict.

    Returns:
        Parsed noise config dict (the 'noise' section of noise.yaml).
    """
    noise_yaml_path = PROJECT_ROOT / config["noise"]["config"]
    with open(noise_yaml_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return raw["noise"]


def _retrieve_passages(
    retriever: Any,
    retriever_mode: str,
    query: str,
    top_k: int,
) -> list[dict]:
    """Dispatch retrieval to the correct method.

    Args:
        retriever: Retriever instance (real or mock).
        retriever_mode: One of 'bm25', 'dense', 'hybrid'.
        query: Search query string.
        top_k: Number of passages to retrieve.

    Returns:
        List of passage dicts with chunk_id, text, score, rank.
    """
    method = getattr(retriever, f"retrieve_{retriever_mode}")
    return method(query, top_k=top_k)


# =========================================================================
# RAG output processing
# =========================================================================


def process_rag_outputs(
    records: list[dict],
    prompts: list[str],
    outputs: list[Any],
    strategy: str,
    model_name: str,
    passages_per_record: list[list[dict]],
    retriever_mode: str,
    noise_level: float,
) -> list[dict]:
    """Process vLLM outputs into RAG-specific record dicts.

    Calls process_outputs internally for closed-book fields, then adds
    RAG-specific fields: passages_used, retriever, noise_level, and
    re-parses with mode='rag' for citation extraction.

    Args:
        records: Source dataset records (aligned with outputs).
        prompts: Prompt strings (aligned with outputs).
        outputs: vLLM RequestOutput objects (aligned with records).
        strategy: Prompt strategy used.
        model_name: Short model name for the record.
        passages_per_record: List of assembled passage lists per record.
        retriever_mode: Retriever mode used.
        noise_level: Noise level used.

    Returns:
        List of output record dicts with RAG fields.
    """
    # Get base records from closed-book processing
    base_records = process_outputs(records, prompts, outputs, strategy, model_name)

    for i, rec in enumerate(base_records):
        # Build passages_used metadata
        assembled_passages = (
            passages_per_record[i] if i < len(passages_per_record) else []
        )
        rec["passages_used"] = [
            {
                "chunk_id": p.get("chunk_id") or p.get("noise_id", ""),
                "noise_type": p.get("noise_type", "real"),
            }
            for p in assembled_passages
        ]
        rec["retriever"] = retriever_mode
        rec["noise_level"] = noise_level

        # Re-parse with mode="rag" to extract citations
        qtype = rec.get("question_type", "open-ended-qa")
        choices = rec.get("choices")

        if strategy == "sc":
            # For SC, re-parse each sample with rag mode for citations
            if rec.get("sc_result") and rec["sc_result"].get("per_sample"):
                for sample in rec["sc_result"]["per_sample"]:
                    rag_parsed = parse_response(
                        raw_response=sample.get("raw_response", ""),
                        strategy=strategy,
                        question_type=qtype,
                        choices=choices,
                        mode="rag",
                    )
                    sample["citations"] = rag_parsed.citations
            # Also add citations to top-level parsed
            rag_parsed = parse_response(
                raw_response=rec.get("raw_response", ""),
                strategy=strategy,
                question_type=qtype,
                choices=choices,
                mode="rag",
            )
            rec["parsed"]["citations"] = rag_parsed.citations
        else:
            rag_parsed = parse_response(
                raw_response=rec.get("raw_response", ""),
                strategy=strategy,
                question_type=qtype,
                choices=choices,
                mode="rag",
            )
            rec["parsed"]["citations"] = rag_parsed.citations

    return base_records


# =========================================================================
# Main pipeline
# =========================================================================


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
    """Run RAG inference for a single model x strategy x retriever x noise cell.

    Args:
        model_name: Key from config['models'] or 'MOCK'.
        strategy: One of 'da', 'ras', 'ctl', 'sc'.
        retriever_mode: One of 'bm25', 'dense', 'hybrid'.
        noise_level: Noise level (0.0, 0.2, 0.4, 0.6).
        config: Loaded rag.yaml config.
        split: Dataset split ('dev', 'main_test', 'track_b').
        limit: Max records to process (None = all).
        mock: Use MockLLM instead of real vLLM.
        engine: Pre-loaded vLLM engine (None = create new).
        retriever: Injected retriever instance (for testing).
        noise_pool_paths: Override pool paths for testing.

    Returns:
        Summary dict with total, processed, skipped, parse_rate, duration_s.
    """
    if strategy not in VALID_STRATEGIES:
        raise ValueError(
            f"Invalid strategy: {strategy}. Must be one of {VALID_STRATEGIES}"
        )
    if retriever_mode not in VALID_RETRIEVERS:
        raise ValueError(
            f"Invalid retriever: {retriever_mode}. Must be one of {VALID_RETRIEVERS}"
        )

    start_time = time.time()
    inf_config = config["inference"]
    top_k = inf_config.get("top_k_passages", 10)
    seed = inf_config["seed"]

    # Load noise config
    noise_config = _load_noise_config(config)

    # Noise gate check: skip for mock mode or noise_level==0.0
    if not mock and noise_level > 0.0:
        assert_noise_gate(noise_level, noise_config, project_root=PROJECT_ROOT)

    # Load noise pools
    if noise_pool_paths is not None:
        pools = load_pools(noise_pool_paths)
    else:
        pool_paths = noise_config.get("pools", {})
        # Resolve relative paths against PROJECT_ROOT
        resolved_paths = {k: str(PROJECT_ROOT / v) for k, v in pool_paths.items()}
        pools = load_pools(resolved_paths)

    # Load dataset
    data_path = resolve_split_path(config, split)
    dataset = load_dataset(data_path, split_name=split.replace("_", "-"))

    if limit is not None:
        dataset = dataset[:limit]

    # Check for resume
    output_dir = Path(config["output"]["base_dir"])
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    effective_model = model_name if not mock else "MOCK"
    output_path = get_rag_output_path(
        output_dir,
        effective_model,
        retriever_mode,
        noise_level,
        strategy,
    )
    existing = count_existing_records(output_path)

    if existing >= len(dataset):
        logger.info(
            "Cell %s/%s/%s/noise%.1f already complete (%d/%d records). Skipping.",
            effective_model,
            strategy,
            retriever_mode,
            noise_level,
            existing,
            len(dataset),
        )
        return {
            "model": effective_model,
            "strategy": strategy,
            "retriever": retriever_mode,
            "noise_level": noise_level,
            "total": len(dataset),
            "processed": 0,
            "skipped": existing,
            "parse_rate": 0.0,
            "duration_s": 0.0,
            "status": "skipped",
        }

    if existing > 0:
        logger.info(
            "Resuming %s/%s/%s/noise%.1f from record %d/%d",
            effective_model,
            strategy,
            retriever_mode,
            noise_level,
            existing,
            len(dataset),
        )
        dataset = dataset[existing:]

    # Resolve model config
    model_cfg = get_model_config(config, model_name) if not mock else {}
    effective_max_tokens = model_cfg.get("max_tokens", inf_config["max_tokens"])

    # Create engine
    owns_engine = False
    if engine is None:
        if mock:
            engine = MockLLM(records=dataset)
        else:
            engine = create_engine(model_cfg, seed=seed)
        owns_engine = True

    # Create sampling params
    if mock:
        params = _MockSamplingParams(
            temperature=inf_config["temperature_sc"]
            if strategy == "sc"
            else inf_config["temperature_greedy"],
            max_tokens=effective_max_tokens,
            seed=seed,
            n=inf_config["sc_samples"] if strategy == "sc" else 1,
        )
        params._strategy = strategy
    else:
        params = create_sampling_params(strategy, config, model_cfg)

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

        # --- RAG-specific: retrieve + noise per question ---
        passages_per_record: list[list[dict]] = []
        prompts: list[str] = []

        for record in chunk_records:
            # Step 1: Retrieve passages
            raw_passages = _retrieve_passages(
                retriever,
                retriever_mode,
                record["question"],
                top_k,
            )

            # Step 2: Assemble noisy context
            assembled = assemble_noisy_context(
                real_passages=raw_passages,
                question=record,
                noise_level=noise_level,
                noise_config=noise_config,
                pools=pools,
                seed=seed,
            )
            passages_per_record.append(assembled)

            # Step 3: Build prompt with passages
            choices = get_choices_or_none(record)
            prompt = build_prompt(
                question=record["question"],
                strategy=strategy,
                choices=choices,
                passages=[p["text"] for p in assembled],
            )
            prompts.append(prompt)

        # Truncate prompts that exceed context window
        if not mock and hasattr(engine, "get_tokenizer"):
            max_ml = model_cfg.get("max_model_len", 4096)
            prompts = truncate_long_prompts(
                prompts,
                max_model_len=max_ml,
                max_tokens=effective_max_tokens,
                tokenizer=engine.get_tokenizer(),
            )

        # Run inference
        logger.info(
            "Generating %s/%s/%s/noise%.1f chunk [%d-%d] (%d prompts, n=%s)...",
            effective_model,
            strategy,
            retriever_mode,
            noise_level,
            existing + chunk_start,
            existing + chunk_end,
            len(prompts),
            getattr(params, "n", 1),
        )
        outputs = run_batch(engine, prompts, params)

        # Process outputs with RAG extensions
        result_records = process_rag_outputs(
            records=chunk_records,
            prompts=prompts,
            outputs=outputs,
            strategy=strategy,
            model_name=effective_model,
            passages_per_record=passages_per_record,
            retriever_mode=retriever_mode,
            noise_level=noise_level,
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
        "Cell %s/%s/%s/noise%.1f complete: %d records, %.1f%% parse rate, %.1fs",
        effective_model,
        strategy,
        retriever_mode,
        noise_level,
        total_processed,
        parse_rate,
        duration,
    )

    return {
        "model": effective_model,
        "strategy": strategy,
        "retriever": retriever_mode,
        "noise_level": noise_level,
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
        description="Run RAG inference for a single model x strategy x retriever x noise cell",
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
        "--retriever",
        required=True,
        choices=sorted(VALID_RETRIEVERS),
        help="Retrieval mode",
    )
    parser.add_argument(
        "--noise-level",
        type=float,
        required=True,
        help="Noise level (0.0, 0.2, 0.4, 0.6)",
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
        choices=["dev", "main_test", "track_b"],
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

    summary = run_rag_cell(
        model_name=args.model,
        strategy=args.strategy,
        retriever_mode=args.retriever,
        noise_level=args.noise_level,
        config=config,
        split=args.split,
        limit=args.limit,
        mock=is_mock,
    )

    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
