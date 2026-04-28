"""RAG experiment orchestrator.

Iterates over the model x strategy x retriever x noise_level matrix,
running each cell sequentially. Minimises GPU loads by iterating models
in the outer loop - one model load covers all (strategy, retriever, noise)
combinations for that model.

Usage:
    python scripts/run_rag_experiment.py --config configs/rag.yaml --dry-run
    python scripts/run_rag_experiment.py --config configs/rag.yaml --mock
    python scripts/run_rag_experiment.py --config configs/rag.yaml \\
        --matrix fractional --models qwen2.5-7b --strategies da,ctl
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from run_inference import (  # noqa: E402
    count_existing_records,
    get_model_config,
    load_config,
    load_dataset,
    release_engine,
)
from run_rag_inference import (  # noqa: E402
    get_rag_output_path,
    resolve_split_path,
    run_rag_cell,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent

# Valid values (mirrored from run_rag_inference to avoid circular imports)
_VALID_RETRIEVERS = {"bm25", "dense", "hybrid"}
_VALID_STRATEGIES = {"da", "ras", "ctl", "sc"}


# =========================================================================
# Matrix construction
# =========================================================================


def build_rag_matrix(
    config: dict,
    mode: str = "full",
    models: list[str] | None = None,
    strategies: list[str] | None = None,
    retrievers: list[str] | None = None,
    noise_levels: list[float] | None = None,
) -> list[tuple[str, str, str, float]]:
    """Build list of (model, strategy, retriever, noise_level) tuples.

    Outer loop is models so that each model is loaded once and all its
    (strategy, retriever, noise) combinations are executed before the next
    model is loaded.

    Args:
        config: Loaded rag.yaml config dict.
        mode: 'full' (all 288 combinations) or 'fractional' (144 deduplicated
            combinations: hybrid x all noise + bm25/dense x noise=0.0 only).
        models: Subset of model names (None = all from config).
        strategies: Subset of strategies (None = all from config).
        retrievers: Subset of retriever modes (None = all from config).
        noise_levels: Subset of noise levels (None = all from config).

    Returns:
        Ordered list of 4-tuples, outer loop = model.

    Raises:
        ValueError: If mode is unknown, or any model/strategy/retriever is not
            in the config.
    """
    if mode not in ("full", "fractional"):
        raise ValueError(f"Unknown mode: {mode!r}. Must be 'full' or 'fractional'.")

    # Resolve lists from config when not overridden
    model_names: list[str] = models or list(config["models"].keys())
    strat_list: list[str] = strategies or list(config["strategies"])
    retriever_list: list[str] = retrievers or list(config["retrievers"])
    noise_list: list[float] = (
        noise_levels if noise_levels is not None else list(config["noise"]["levels"])
    )

    # Validate inputs
    for m in model_names:
        if m not in config["models"]:
            raise ValueError(
                f"Unknown model: {m!r}. Available: {list(config['models'].keys())}"
            )
    valid_strats = set(config["strategies"])
    for s in strat_list:
        if s not in valid_strats:
            raise ValueError(
                f"Unknown strategy: {s!r}. Available: {list(config['strategies'])}"
            )
    valid_retrievers = set(config["retrievers"])
    for r in retriever_list:
        if r not in valid_retrievers:
            raise ValueError(
                f"Unknown retriever: {r!r}. Available: {list(config['retrievers'])}"
            )

    if mode == "full":
        matrix: list[tuple[str, str, str, float]] = []
        for model in model_names:
            for strategy in strat_list:
                for retriever in retriever_list:
                    for noise in noise_list:
                        matrix.append((model, strategy, retriever, float(noise)))
        return matrix

    # mode == "fractional"
    # hybrid gets all noise levels; bm25/dense get noise=0.0 only
    seen: set[tuple[str, str, str, float]] = set()
    matrix = []

    def _add(model: str, strategy: str, retriever: str, noise: float) -> None:
        cell = (model, strategy, retriever, float(noise))
        if cell not in seen:
            seen.add(cell)
            matrix.append(cell)

    for model in model_names:
        for strategy in strat_list:
            # Hybrid: all noise levels that are requested
            if "hybrid" in retriever_list:
                for noise in noise_list:
                    _add(model, strategy, "hybrid", noise)
            # bm25 / dense: noise=0.0 only (if 0.0 is in the requested noise list)
            for retriever in retriever_list:
                if retriever == "hybrid":
                    continue
                for noise in noise_list:
                    if noise == 0.0:
                        _add(model, strategy, retriever, noise)

    return matrix


# =========================================================================
# Cell status helper
# =========================================================================


def check_rag_cell_status(
    config: dict,
    model_name: str,
    strategy: str,
    retriever_mode: str,
    noise_level: float,
    dataset_size: int,
) -> str:
    """Check whether a RAG cell is complete, partial, or pending.

    Args:
        config: Loaded rag.yaml config dict.
        model_name: Model name key.
        strategy: Prompt strategy.
        retriever_mode: Retriever mode.
        noise_level: Noise level float.
        dataset_size: Expected number of records.

    Returns:
        'COMPLETE', 'PARTIAL (N/M)', or 'PENDING'.
    """
    output_dir = PROJECT_ROOT / config["output"]["base_dir"]
    output_path = get_rag_output_path(
        output_dir, model_name, retriever_mode, noise_level, strategy
    )
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
    matrix: list[tuple[str, str, str, float]],
    split: str,
    dataset_size: int,
) -> None:
    """Print the RAG experiment matrix without running inference.

    Args:
        config: Loaded rag.yaml config dict.
        matrix: List of (model, strategy, retriever, noise_level) tuples.
        split: Dataset split name.
        dataset_size: Number of questions in the dataset.
    """
    pending = 0
    complete = 0
    partial = 0

    header = (
        f"{'Cell':<8} {'Model':<22} {'Strategy':<10} {'Retriever':<10} "
        f"{'Noise':<7} {'Status'}"
    )
    separator = "-" * (len(header) + 20)

    lines = [
        "",
        f"DRY RUN - RAG experiment matrix ({split}, {dataset_size} questions)",
        f"Mode: {len(matrix)} cells",
        "",
        header,
        separator,
    ]

    for idx, (model_name, strategy, retriever, noise_level) in enumerate(matrix, 1):
        status = check_rag_cell_status(
            config, model_name, strategy, retriever, noise_level, dataset_size
        )

        if status == "COMPLETE":
            complete += 1
        elif status.startswith("PARTIAL"):
            partial += 1
        else:
            pending += 1

        lines.append(
            f"{idx:>3}/{len(matrix):<4} {model_name:<22} {strategy:<10} "
            f"{retriever:<10} {noise_level:<7.1f} {status}"
        )

    lines.append(separator)
    lines.append(
        f"Pending: {pending}  |  Partial: {partial}  |  Complete: {complete}  "
        f"|  Total: {len(matrix)}"
    )

    # Time estimate (conservative)
    inf = config["inference"]
    tokens_per_sec = 750
    avg_output_tokens = 200
    da_time_min = dataset_size * avg_output_tokens / tokens_per_sec / 60
    sc_time_min = (
        dataset_size * inf["sc_samples"] * avg_output_tokens / tokens_per_sec / 60
    )

    n_greedy = sum(
        1
        for _, s, r, n in matrix
        if s != "sc"
        and check_rag_cell_status(config, _, s, r, n, dataset_size) != "COMPLETE"
    )
    n_sc = sum(
        1
        for _, s, r, n in matrix
        if s == "sc"
        and check_rag_cell_status(config, _, s, r, n, dataset_size) != "COMPLETE"
    )

    est_hours = (n_greedy * da_time_min + n_sc * sc_time_min) / 60
    lines.append(f"Estimated time for pending cells: ~{est_hours:.1f} hours")
    lines.append(
        f"  (assuming ~{tokens_per_sec} tok/s, ~{avg_output_tokens} output tokens/question)"
    )
    lines.append("")

    for line in lines:
        logger.info(line)


# =========================================================================
# Orchestration
# =========================================================================


def run_rag_experiment(
    config: dict,
    split: str = "dev",
    mode: str = "full",
    models: list[str] | None = None,
    strategies: list[str] | None = None,
    retrievers: list[str] | None = None,
    noise_levels: list[float] | None = None,
    limit: int | None = None,
    mock: bool = False,
) -> list[dict[str, Any]]:
    """Run the full RAG experiment matrix.

    Groups matrix by model (outer loop) so the engine is loaded once per model
    and shared across all (strategy, retriever, noise_level) combinations.
    GPU memory is released after each model.

    Args:
        config: Loaded rag.yaml config dict.
        split: Dataset split ('dev', 'main_test', 'track_b').
        mode: Matrix mode ('full' or 'fractional').
        models: Subset of model names (None = all).
        strategies: Subset of strategies (None = all).
        retrievers: Subset of retriever modes (None = all).
        noise_levels: Subset of noise levels (None = all).
        limit: Max records per cell (None = all).
        mock: Use MockLLM - no GPU or real retriever needed.

    Returns:
        List of summary dicts, one per cell, each containing model, strategy,
        retriever, noise_level, total, processed, skipped, parse_rate,
        duration_s, status.
    """
    matrix = build_rag_matrix(
        config,
        mode=mode,
        models=models,
        strategies=strategies,
        retrievers=retrievers,
        noise_levels=noise_levels,
    )
    total_cells = len(matrix)
    logger.info("RAG experiment matrix: %d cells (mode=%s)", total_cells, mode)

    # Determine dataset size upfront for progress logging
    data_path = resolve_split_path(config, split)
    dataset = load_dataset(data_path, split_name=split.replace("_", "-"))
    dataset_size = limit if limit else len(dataset)

    # Group by model, preserving order of first appearance
    model_order: list[str] = []
    model_cells: dict[str, list[tuple[str, str, str, float]]] = {}
    for cell in matrix:
        model_name = cell[0]
        if model_name not in model_cells:
            model_order.append(model_name)
            model_cells[model_name] = []
        model_cells[model_name].append(cell)

    results: list[dict[str, Any]] = []
    cell_idx = 0

    for model_idx, model_name in enumerate(model_order):
        cells = model_cells[model_name]
        logger.info(
            "=== Model %d/%d: %s (%d cells) ===",
            model_idx + 1,
            len(model_order),
            model_name,
            len(cells),
        )

        # Create engine once per model
        engine: Any = None
        if not mock:
            from run_inference import create_engine as _create_engine  # noqa: PLC0415

            model_cfg = get_model_config(config, model_name)
            engine = _create_engine(model_cfg, seed=config["inference"]["seed"])

        # Create retriever once per model
        retriever: Any = None
        if not mock:
            from retriever import Retriever  # noqa: PLC0415

            retriever = Retriever(device="cuda")

        for cell in cells:
            _, strategy, retriever_mode, noise_level = cell
            cell_idx += 1
            cell_start = time.time()

            logger.info(
                "--- [%d/%d] %s / %s / %s / noise=%.1f ---",
                cell_idx,
                total_cells,
                model_name,
                strategy,
                retriever_mode,
                noise_level,
            )

            # Fast-path: skip complete cells without calling run_rag_cell
            existing = count_existing_records(
                get_rag_output_path(
                    PROJECT_ROOT / config["output"]["base_dir"],
                    model_name,
                    retriever_mode,
                    noise_level,
                    strategy,
                )
            )
            if existing >= dataset_size:
                logger.info(
                    "[%d/%d] COMPLETE (skipped) %s/%s/%s/noise=%.1f",
                    cell_idx,
                    total_cells,
                    model_name,
                    strategy,
                    retriever_mode,
                    noise_level,
                )
                results.append(
                    {
                        "model": model_name,
                        "strategy": strategy,
                        "retriever": retriever_mode,
                        "noise_level": noise_level,
                        "total": existing,
                        "processed": 0,
                        "skipped": existing,
                        "parse_rate": 0.0,
                        "duration_s": 0.0,
                        "status": "skipped",
                    }
                )
                continue

            summary = run_rag_cell(
                model_name=model_name,
                strategy=strategy,
                retriever_mode=retriever_mode,
                noise_level=noise_level,
                config=config,
                split=split,
                limit=limit,
                mock=mock,
                engine=engine,
                retriever=retriever,
            )
            results.append(summary)

            elapsed_min = (time.time() - cell_start) / 60
            logger.info(
                "[%d/%d] done: %s / %s / %s / noise=%.1f in %.1f min",
                cell_idx,
                total_cells,
                model_name,
                strategy,
                retriever_mode,
                noise_level,
                elapsed_min,
            )

        # Release engine and GPU memory after all cells for this model
        if engine is not None:
            release_engine(engine)
            engine = None
            try:
                import torch  # noqa: PLC0415

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print formatted summary table of RAG experiment results.

    Args:
        results: List of summary dicts from run_rag_experiment.
    """
    header = (
        f"{'Model':<22} {'Strategy':<10} {'Retriever':<10} {'Noise':<7} "
        f"{'Status':<10} {'Processed':<12} {'Parse Rate':<12} {'Duration':<10}"
    )
    separator = "-" * len(header)

    logger.info("")
    logger.info("RAG EXPERIMENT SUMMARY")
    logger.info(header)
    logger.info(separator)

    for r in results:
        logger.info(
            "%s %s %s %s %s %s %s %s",
            f"{r['model']:<22}",
            f"{r['strategy']:<10}",
            f"{r['retriever']:<10}",
            f"{r['noise_level']:<7.1f}",
            f"{r['status']:<10}",
            f"{r['processed']:<12}",
            f"{r['parse_rate']:<11.1f}%",
            f"{r['duration_s']:<10}s",
        )

    total_processed = sum(r["processed"] for r in results)
    total_duration = sum(r["duration_s"] for r in results)
    avg_parse = (
        sum(r["parse_rate"] * r["processed"] for r in results) / total_processed
        if total_processed
        else 0.0
    )

    logger.info(separator)
    logger.info(
        "Total: %d records processed, %.1f%% avg parse rate, %.1fs total",
        total_processed,
        avg_parse,
        total_duration,
    )
    logger.info("")


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run RAG experiment matrix "
            "(6 models x 4 strategies x 3 retrievers x 4 noise levels)"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "rag.yaml",
        help="Path to rag.yaml config (default: configs/rag.yaml)",
    )
    parser.add_argument(
        "--matrix",
        dest="mode",
        default="full",
        choices=["full", "fractional"],
        help="Matrix mode: full (288) or fractional (144) (default: full)",
    )
    parser.add_argument(
        "--split",
        default="dev",
        choices=["dev", "main_test", "track_b"],
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
        help="Comma-separated strategies: da,ras,ctl,sc (default: all)",
    )
    parser.add_argument(
        "--retrievers",
        default=None,
        help="Comma-separated retrievers: bm25,dense,hybrid (default: all)",
    )
    parser.add_argument(
        "--noise-levels",
        default=None,
        help="Comma-separated noise levels: 0.0,0.2,0.4,0.6 (default: all)",
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
    retriever_list = args.retrievers.split(",") if args.retrievers else None
    noise_list = (
        [float(n) for n in args.noise_levels.split(",")] if args.noise_levels else None
    )

    if args.dry_run:
        matrix = build_rag_matrix(
            config,
            mode=args.mode,
            models=models_list,
            strategies=strats_list,
            retrievers=retriever_list,
            noise_levels=noise_list,
        )
        data_path = resolve_split_path(config, args.split)
        dataset = load_dataset(data_path, split_name=args.split.replace("_", "-"))
        dataset_size = args.limit if args.limit else len(dataset)
        print_dry_run(config, matrix, args.split, dataset_size)
        return

    results = run_rag_experiment(
        config=config,
        split=args.split,
        mode=args.mode,
        models=models_list,
        strategies=strats_list,
        retrievers=retriever_list,
        noise_levels=noise_list,
        limit=args.limit,
        mock=args.mock,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
