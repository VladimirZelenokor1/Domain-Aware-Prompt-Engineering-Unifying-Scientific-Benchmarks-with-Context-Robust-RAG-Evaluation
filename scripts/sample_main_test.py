"""Stratified sampling of the Phase 3 retrievable set to a fixed-size main test.

Reads ``data/sciknoweval/main_test_retrievable.json`` (produced by the
retrievability filter), stratifies by ``domain x level x type`` (up to
120 cells: 4 x 5 x 6), applies three pre-sample stopping criteria, then
samples down to a target size using take-all-small + proportional-scale-
large. Emits both the sampled split and a stratification report.

Stopping criteria (approved 2026-04-21). If ANY of them trip, the script
exits 2 with a pointer to the violations, UNLESS ``--force`` is passed:

  (1) smallest post-filter non-empty cell >= ``MIN_CELL_ITEMS`` items;
  (2) post-filter non-empty cells >= ``MIN_NON_EMPTY_CELLS`` (pre-filter
      has 38 such cells);
  (3) per-domain retention >= ``1 - MAX_DOMAIN_LOSS`` (no domain may lose
      more than 50 % of its items relative to pre-retrievability).

Acceptance invariant (checked post-sample, script exits 2 on failure):
  every non-empty source cell must appear in the sample at least once.

Usage:
    python scripts/sample_main_test.py                     # full run
    python scripts/sample_main_test.py --check-only        # criteria only
    python scripts/sample_main_test.py --target 5000       # bigger sample
    python scripts/sample_main_test.py --force             # bypass criteria
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PREFILTER = PROJECT_ROOT / "data" / "sciknoweval" / "main_test.json"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "sciknoweval" / "main_test_retrievable.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sciknoweval" / "main_test_sampled.json"
DEFAULT_REPORT = PROJECT_ROOT / "outputs" / "main_test_sampled_stratification.json"

SPEC_TYPES = frozenset(
    {
        "mcq-4-choices",
        "mcq-2-choices",
        "open-ended-qa",
        "true_or_false",
        "relation_extraction",
        "filling",
    }
)

# Stopping criteria thresholds. Locked on 2026-04-21.
MIN_CELL_ITEMS = 30
MIN_NON_EMPTY_CELLS = 35
MAX_DOMAIN_LOSS = 0.50


Cell = tuple[str, str, str]


# =========================================================================
# Cross-tabulation
# =========================================================================


def extract_cell(record: dict) -> Cell:
    """Return ``(domain, level, type)`` for a SciKnowEval record."""
    dom = record.get("domain", "?")
    details = record.get("details") or {}
    lvl = details.get("level", "?") if isinstance(details, dict) else "?"
    typ = record.get("type", "?")
    return (dom, lvl, typ)


def cross_tabulate(records: list[dict]) -> dict[Cell, list[int]]:
    """Group record indices by ``(domain, level, type)``."""
    cells: dict[Cell, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        cells[extract_cell(rec)].append(i)
    return dict(cells)


def per_domain_totals(cells: dict[Cell, list[int]]) -> Counter:
    """Return domain -> total record count."""
    totals: Counter = Counter()
    for (dom, _, _), idxs in cells.items():
        totals[dom] += len(idxs)
    return totals


# =========================================================================
# Sampling
# =========================================================================


def effective_sample_quotas(
    cells: dict[Cell, list[int]],
    target: int,
    n_floor: int,
) -> dict[Cell, int]:
    """Per-cell quota under take-all-small + proportional-scale-large.

    Args:
        cells: Output of :func:`cross_tabulate` over the source records.
        target: Desired total sample size.
        n_floor: Cells with ``<= n_floor`` items are taken whole and
            large cells are forced to contribute at least this many.

    Returns:
        Dict ``cell -> quota``. Sum may deviate from ``target`` by a
        small rounding drift; the acceptance invariant is what we
        actually enforce.
    """
    small_cells = {k: v for k, v in cells.items() if len(v) <= n_floor}
    large_cells = {k: v for k, v in cells.items() if len(v) > n_floor}
    floor_sum = sum(len(v) for v in small_cells.values())
    large_sum = sum(len(v) for v in large_cells.values())

    quotas: dict[Cell, int] = {k: len(v) for k, v in small_cells.items()}
    if large_sum == 0 or floor_sum >= target:
        return quotas

    budget = target - floor_sum
    scale = budget / large_sum
    for k, v in large_cells.items():
        quotas[k] = min(len(v), max(n_floor, round(len(v) * scale)))
    return quotas


def sample_stratified(
    records: list[dict],
    target: int,
    n_floor: int,
    seed: int,
) -> list[dict]:
    """Stratified sample, deterministic under ``seed``.

    Args:
        records: Source list (post-retrievability in production).
        target: Desired total sample size.
        n_floor: Small-cell floor.
        seed: RNG seed.

    Returns:
        Sampled list, sorted by ``question_id`` for determinism.
    """
    cells = cross_tabulate(records)
    quotas = effective_sample_quotas(cells, target, n_floor)
    rng = random.Random(seed)
    kept: list[int] = []
    # Sort cells for deterministic iteration order.
    for cell in sorted(cells.keys()):
        idxs = cells[cell]
        quota = quotas.get(cell, 0)
        if quota >= len(idxs):
            kept.extend(idxs)
        else:
            kept.extend(rng.sample(idxs, quota))
    kept.sort()
    out = [records[i] for i in kept]
    out.sort(key=lambda q: q.get("question_id", ""))
    return out


# =========================================================================
# Criteria + invariants
# =========================================================================


def check_stopping_criteria(
    prefilter_cells: dict[Cell, list[int]],
    postfilter_cells: dict[Cell, list[int]],
) -> list[str]:
    """Return a list of stopping-criterion violations (empty if all pass)."""
    violations: list[str] = []

    non_empty_sizes = [len(v) for v in postfilter_cells.values() if v]
    if non_empty_sizes:
        smallest = min(non_empty_sizes)
        if smallest < MIN_CELL_ITEMS:
            violations.append(
                f"smallest non-empty post-filter cell has {smallest} items "
                f"(< {MIN_CELL_ITEMS} threshold)"
            )

    post_non_empty = sum(1 for v in postfilter_cells.values() if v)
    pre_non_empty = sum(1 for v in prefilter_cells.values() if v)
    if post_non_empty < MIN_NON_EMPTY_CELLS:
        violations.append(
            f"non-empty cells dropped to {post_non_empty} "
            f"(< {MIN_NON_EMPTY_CELLS} threshold; pre-filter = {pre_non_empty})"
        )

    pre_dom = per_domain_totals(prefilter_cells)
    post_dom = per_domain_totals(postfilter_cells)
    for dom, pre in pre_dom.items():
        post = post_dom.get(dom, 0)
        loss = 1 - (post / pre) if pre else 0.0
        if loss > MAX_DOMAIN_LOSS:
            violations.append(
                f"domain '{dom}' lost {loss:.1%} of items "
                f"({pre} -> {post}, > {MAX_DOMAIN_LOSS:.0%} threshold)"
            )

    return violations


def verify_every_cell_represented(
    source: list[dict],
    sample: list[dict],
) -> tuple[bool, list[Cell]]:
    """Return ``(accepted, missing_cells)``."""
    src_cells = {k for k, v in cross_tabulate(source).items() if v}
    smp_cells = {k for k, v in cross_tabulate(sample).items() if v}
    missing = sorted(src_cells - smp_cells)
    return (not missing, missing)


# =========================================================================
# Reporting
# =========================================================================


def _cells_summary(records: list[dict]) -> dict:
    cells = cross_tabulate(records)
    sizes = [len(v) for v in cells.values() if v]
    return {
        "total": sum(sizes),
        "non_empty_cells": len(sizes),
        "smallest_cell": min(sizes) if sizes else 0,
        "largest_cell": max(sizes) if sizes else 0,
        "per_domain": dict(per_domain_totals(cells)),
        "cells": sorted(
            [
                {"domain": k[0], "level": k[1], "type": k[2], "count": len(v)}
                for k, v in cells.items()
                if v
            ],
            key=lambda x: (x["domain"], x["level"], x["type"]),
        ),
    }


def build_report(
    *,
    prefilter: list[dict],
    source: list[dict],
    sample: list[dict],
    target: int,
    n_floor: int,
    seed: int,
    input_path: Path,
    output_path: Path,
    accepted: bool,
    missing: list[Cell],
    violations: list[str],
) -> dict:
    """Assemble the stratification report JSON."""
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "seed": seed,
        "target": target,
        "n_floor": n_floor,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "thresholds": {
            "min_cell_items": MIN_CELL_ITEMS,
            "min_non_empty_cells": MIN_NON_EMPTY_CELLS,
            "max_domain_loss": MAX_DOMAIN_LOSS,
        },
        "pre_filter_main_test": _cells_summary(prefilter),
        "post_filter_retrievable": _cells_summary(source),
        "post_sample": _cells_summary(sample),
        "acceptance": {
            "every_source_cell_represented": accepted,
            "missing_cells": [
                {"domain": d, "level": lvl, "type": t} for (d, lvl, t) in missing
            ],
        },
        "stopping_criteria": {
            "violations": violations,
            "passed": not violations,
        },
    }


# =========================================================================
# CLI
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stratified sampling of the Phase 3 retrievable set.",
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--prefilter", type=Path, default=DEFAULT_PREFILTER)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--target", type=int, default=3000)
    p.add_argument("--n-floor", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--force",
        action="store_true",
        help="Bypass stopping criteria and sample anyway. The report "
        "still records violations.",
    )
    p.add_argument(
        "--check-only",
        action="store_true",
        help="Print stopping-criteria status and exit without sampling. "
        "Exits 0 if criteria pass, 2 if any violation.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading pre-filter from %s", args.prefilter)
    with open(args.prefilter, "r", encoding="utf-8") as f:
        prefilter = json.load(f)

    logger.info("Loading post-filter from %s", args.input)
    with open(args.input, "r", encoding="utf-8") as f:
        source = json.load(f)

    prefilter_cells = cross_tabulate(prefilter)
    source_cells = cross_tabulate(source)

    violations = check_stopping_criteria(prefilter_cells, source_cells)
    if violations:
        for v in violations:
            logger.warning("STOP-CRITERION: %s", v)
    else:
        logger.info("All stopping criteria passed.")

    if args.check_only:
        status = {
            "proceed": not violations,
            "violations": violations,
            "prefilter_non_empty_cells": sum(1 for v in prefilter_cells.values() if v),
            "source_non_empty_cells": sum(1 for v in source_cells.values() if v),
            "thresholds": {
                "min_cell_items": MIN_CELL_ITEMS,
                "min_non_empty_cells": MIN_NON_EMPTY_CELLS,
                "max_domain_loss": MAX_DOMAIN_LOSS,
            },
        }
        print(json.dumps(status, indent=2))
        return 0 if not violations else 2

    if violations and not args.force:
        logger.error(
            "Refusing to sample: %d stopping-criterion violation(s). "
            "Report back and use --force to override.",
            len(violations),
        )
        print(
            json.dumps({"proceed": False, "violations": violations}, indent=2),
            file=sys.stderr,
        )
        return 2

    sample = sample_stratified(source, args.target, args.n_floor, args.seed)
    logger.info(
        "Sampled %d / %d (target=%d)",
        len(sample),
        len(source),
        args.target,
    )

    accepted, missing = verify_every_cell_represented(source, sample)
    if not accepted:
        logger.error(
            "Acceptance invariant violated: %d source cells missing from sample",
            len(missing),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    logger.info("Sample written: %s", args.output)

    report = build_report(
        prefilter=prefilter,
        source=source,
        sample=sample,
        target=args.target,
        n_floor=args.n_floor,
        seed=args.seed,
        input_path=args.input,
        output_path=args.output,
        accepted=accepted,
        missing=missing,
        violations=violations,
    )
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Stratification report written: %s", args.report)

    return 0 if accepted else 2


if __name__ == "__main__":
    sys.exit(main())
