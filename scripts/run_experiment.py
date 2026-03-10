"""Closed-book experiment orchestrator.

Iterates over the model x strategy matrix, running each cell sequentially.
Minimizes GPU loads by iterating models in the outer loop and strategies
in the inner loop.

Usage:
    python scripts/run_experiment.py --config configs/closed_book.yaml --dry-run
    python scripts/run_experiment.py --config configs/closed_book.yaml --mock
    python scripts/run_experiment.py --config configs/closed_book.yaml --models llama-3.2-3b --strategies da,ctl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_inference import (  # noqa: E402
    count_existing_records,
    get_output_path,
    load_config,
    load_dataset,
    release_engine,
    run_cell,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent


# =========================================================================
# Matrix construction
# =========================================================================

def build_matrix(
    config: dict,
    models: list[str] | None = None,
    strategies: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Build list of (model_name, strategy) tuples for the experiment matrix.

    Outer loop is models, inner loop is strategies - this ordering minimizes
    the number of model loads when executing sequentially.

    Args:
        config: Loaded config dict.
        models: Subset of model names (None = all from config).
        strategies: Subset of strategies (None = all from config).

    Returns:
        Ordered list of (model_name, strategy) tuples.
    """
    model_names = models or list(config["models"].keys())
    strat_list = strategies or config["strategies"]

    # Validate
    for m in model_names:
        if m not in config["models"]:
            raise ValueError(f"Unknown model: {m}. Available: {list(config['models'].keys())}")
    valid = set(config["strategies"])
    for s in strat_list:
        if s not in valid:
            raise ValueError(f"Unknown strategy: {s}. Available: {config['strategies']}")

    matrix: list[tuple[str, str]] = []
    for model_name in model_names:
        for strat in strat_list:
            matrix.append((model_name, strat))

    return matrix


def check_cell_status(
    config: dict,
    model_name: str,
    strategy: str,
    split: str,
    dataset_size: int,
) -> str:
    """Check whether a cell is complete, partial, or pending.

    Args:
        config: Loaded config dict.
        model_name: Model name.
        strategy: Strategy name.
        split: Dataset split.
        dataset_size: Expected number of records.

    Returns:
        'COMPLETE', 'PARTIAL (N/M)', or 'PENDING'.
    """
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    output_path = get_output_path(output_dir, model_name, strategy)
    existing = count_existing_records(output_path)

    if existing >= dataset_size:
        return "COMPLETE"
    if existing > 0:
        return f"PARTIAL ({existing}/{dataset_size})"
    return "PENDING"


# =========================================================================
# Dry run
# =========================================================================

def print_dry_run(
    config: dict,
    matrix: list[tuple[str, str]],
    split: str,
    dataset_size: int,
) -> None:
    """Print the experiment matrix without running inference.

    Args:
        config: Loaded config dict.
        matrix: List of (model, strategy) tuples.
        split: Dataset split.
        dataset_size: Number of questions in the dataset.
    """
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    pending = 0
    complete = 0
    partial = 0

    header = (
        f"{'Cell':<7} {'Model':<22} {'Strategy':<10} "
        f"{'Output':<55} {'Status'}"
    )
    separator = "-" * len(header)

    lines = [
        "",
        f"DRY RUN - Closed-book experiment matrix ({split}, {dataset_size} questions)",
        "",
        header,
        separator,
    ]

    for idx, (model_name, strategy) in enumerate(matrix, 1):
        output_path = get_output_path(output_dir, model_name, strategy)
        rel_path = output_path.relative_to(PROJECT_ROOT) if output_path.is_relative_to(PROJECT_ROOT) else output_path
        status = check_cell_status(config, model_name, strategy, split, dataset_size)

        if status == "COMPLETE":
            complete += 1
        elif status.startswith("PARTIAL"):
            partial += 1
        else:
            pending += 1

        lines.append(
            f"{idx:>2}/{len(matrix):<4} {model_name:<22} {strategy:<10} "
            f"{str(rel_path):<55} {status}"
        )

    lines.append(separator)
    lines.append(
        f"Pending: {pending}  |  Partial: {partial}  |  Complete: {complete}  |  Total: {len(matrix)}"
    )

    # Time estimate
    inf = config["inference"]
    tokens_per_sec = 750  # conservative estimate for 7B AWQ on RTX 4060
    avg_output_tokens = 200
    da_time_min = dataset_size * avg_output_tokens / tokens_per_sec / 60
    sc_time_min = dataset_size * inf["sc_samples"] * avg_output_tokens / tokens_per_sec / 60

    n_greedy = sum(1 for _, s in matrix if s != "sc" and check_cell_status(config, _, s, split, dataset_size) != "COMPLETE")
    n_sc = sum(1 for _, s in matrix if s == "sc" and check_cell_status(config, _, s, split, dataset_size) != "COMPLETE")

    est_hours = (n_greedy * da_time_min + n_sc * sc_time_min) / 60
    lines.append(f"Estimated time for pending cells: ~{est_hours:.1f} hours")
    lines.append(f"  (assuming ~{tokens_per_sec} tok/s, ~{avg_output_tokens} output tokens/question)")
    lines.append("")

    for line in lines:
        logger.info(line)


# =========================================================================
# Orchestration
# =========================================================================

def run_experiment(
    config: dict,
    split: str = "dev",
    models: list[str] | None = None,
    strategies: list[str] | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> list[dict[str, Any]]:
    """Run the full experiment matrix.

    Iterates models in the outer loop (load model once, run all strategies),
    then releases GPU memory before loading the next model.

    Args:
        config: Loaded config dict.
        split: Dataset split.
        models: Subset of model names (None = all).
        strategies: Subset of strategies (None = all).
        limit: Max records per cell.
        mock: Use MockLLM.

    Returns:
        List of summary dicts, one per cell.
    """
    matrix = build_matrix(config, models, strategies)
    logger.info("Experiment matrix: %d cells", len(matrix))

    # Group by model for efficient engine reuse
    model_order: list[str] = []
    model_strategies: dict[str, list[str]] = {}
    for model_name, strategy in matrix:
        if model_name not in model_strategies:
            model_order.append(model_name)
            model_strategies[model_name] = []
        model_strategies[model_name].append(strategy)

    results: list[dict[str, Any]] = []

    for model_idx, model_name in enumerate(model_order):
        strats = model_strategies[model_name]
        logger.info(
            "=== Model %d/%d: %s (strategies: %s) ===",
            model_idx + 1, len(model_order), model_name, ", ".join(strats),
        )

        # Create engine once per model
        engine = None
        if not mock:
            from run_inference import create_engine as _ce, get_model_config as _gmc
            model_cfg = _gmc(config, model_name)
            engine = _ce(model_cfg, seed=config["inference"]["seed"])

        for strat in strats:
            logger.info("--- Cell: %s / %s ---", model_name, strat)
            summary = run_cell(
                model_name=model_name,
                strategy=strat,
                config=config,
                split=split,
                limit=limit,
                mock=mock,
                engine=engine,
            )
            results.append(summary)

        # Release engine after all strategies for this model
        if engine is not None:
            release_engine(engine)
            engine = None

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print formatted summary table of experiment results.

    Args:
        results: List of summary dicts from run_experiment.
    """
    header = (
        f"{'Model':<22} {'Strategy':<10} {'Status':<10} "
        f"{'Processed':<12} {'Parse Rate':<12} {'Duration':<10}"
    )
    separator = "-" * len(header)

    logger.info("")
    logger.info("EXPERIMENT SUMMARY")
    logger.info(header)
    logger.info(separator)

    for r in results:
        logger.info(
            "%s %s %s %s %s %s",
            f"{r['model']:<22}",
            f"{r['strategy']:<10}",
            f"{r['status']:<10}",
            f"{r['processed']:<12}",
            f"{r['parse_rate']:<11.1f}%",
            f"{r['duration_s']:<10}s",
        )

    total_processed = sum(r["processed"] for r in results)
    total_duration = sum(r["duration_s"] for r in results)
    avg_parse = (
        sum(r["parse_rate"] * r["processed"] for r in results) / total_processed
        if total_processed else 0
    )

    logger.info(separator)
    logger.info(
        "Total: %d records processed, %.1f%% avg parse rate, %.1fs total",
        total_processed, avg_parse, total_duration,
    )
    logger.info("")


# =========================================================================
# CLI
# =========================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run closed-book experiment matrix (6 models x 4 strategies)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "closed_book.yaml",
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "main_test"],
        help="Dataset split (default: dev)",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names (default: all)",
    )
    parser.add_argument(
        "--strategies",
        default=None,
        help="Comma-separated strategies (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records per cell (default: all)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLM for testing (no GPU needed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment matrix without running inference",
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

    models_list = args.models.split(",") if args.models else None
    strats_list = args.strategies.split(",") if args.strategies else None

    if args.dry_run:
        matrix = build_matrix(config, models_list, strats_list)
        data_path = PROJECT_ROOT / config["data"][args.split]
        dataset = load_dataset(data_path, split_name=args.split.replace("_", "-"))
        dataset_size = args.limit if args.limit else len(dataset)
        print_dry_run(config, matrix, args.split, dataset_size)
        return

    results = run_experiment(
        config=config,
        split=args.split,
        models=models_list,
        strategies=strats_list,
        limit=args.limit,
        mock=args.mock,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
