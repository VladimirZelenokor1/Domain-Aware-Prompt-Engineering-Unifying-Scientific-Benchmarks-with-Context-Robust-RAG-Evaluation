"""Phase 6 noise-gate: level-aware preconditions for RAG inference startup.

Level 0.0 never requires contradictory passages (composition is empty),
so it can run in parallel with the contradictory-pool manual review.
Levels 0.2 / 0.4 / 0.6 all gate on an approved contradictory pool with
``acceptance_rate >= configs/noise.yaml::noise.contradictory.min_acceptance_rate``.

The module is split out so its behaviour can be unit-tested well before
``run_rag_inference.py`` exists (Phase 6). Phase 6 imports
:func:`assert_noise_gate` at the top of its CLI entry point.

Public API
----------
- :func:`level_requires_contradictory`
- :func:`assert_noise_gate`
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import NoReturn

logger = logging.getLogger(__name__)

# Composition keys that may resolve to a contradictory passage at
# assembly time. Keep in sync with ``noise_assembler._expand_composition``.
RISKY_COMPOSITION_KEYS: frozenset[str] = frozenset(
    {
        "contradictory",
        "contradictory_or_injection",
        "any",
    }
)


def level_requires_contradictory(
    noise_level: float,
    noise_config: dict,
) -> bool:
    """True iff the given level's composition could require contradictory.

    Args:
        noise_level: Level from ``configs/noise.yaml`` (e.g. 0.0, 0.2).
        noise_config: Parsed ``noise`` section of ``noise.yaml``.

    Returns:
        True if ANY of the composition slots for this level could
        resolve to a contradictory passage (directly, or via ``any``
        / ``contradictory_or_injection``).
    """
    levels = noise_config.get("levels", {})
    for key, cfg in levels.items():
        if float(key) == float(noise_level):
            composition = cfg.get("composition") or {}
            return any(k in RISKY_COMPOSITION_KEYS for k in composition)
    return False


def _fail(message: str, code: int = 2) -> NoReturn:
    """Log the failure at ERROR and exit with ``code`` (default 2)."""
    logger.error(message)
    sys.exit(code)


def assert_noise_gate(
    noise_level: float,
    noise_config: dict,
    *,
    project_root: Path | None = None,
    review_stats_path: Path | None = None,
) -> None:
    """Verify preconditions for running Phase 6 at ``noise_level``.

    Checks, in order:

    1. ``irrelevant`` and ``injection`` pool files exist.
    2. If the level can require contradictory:
       a. ``contradictory`` pool file exists;
       b. review stats file exists;
       c. observed ``acceptance_rate`` >= configured
          ``min_acceptance_rate`` (default 0.80).

    On any failure, logs an ERROR and calls ``sys.exit(2)`` so callers
    cannot accidentally proceed. Level 0.0 skips checks (2a)-(2c)
    entirely so it may run in parallel with the manual review.

    Args:
        noise_level: Level from ``configs/noise.yaml``.
        noise_config: Parsed ``noise`` section.
        project_root: Optional explicit root used to resolve relative
            pool paths. Defaults to the current working directory.
        review_stats_path: Optional override for the review-stats file
            (defaults to ``noise.contradictory.review_stats_path``).
    """
    root = Path(project_root) if project_root is not None else Path.cwd()
    pools_section = noise_config.get("pools") or {}
    contra_cfg = noise_config.get("contradictory") or {}
    min_rate = float(contra_cfg.get("min_acceptance_rate", 0.80))

    # ------------------------------------------------------------ always
    for name in ("irrelevant", "injection"):
        rel = pools_section.get(name, f"corpus/noise/{name}_passages.jsonl")
        path = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if not path.exists():
            _fail(
                f"Noise gate: required pool '{name}' missing at {path}. "
                f"Build it via scripts/build_noise.py {name}.",
            )

    # ------------------------------------------------------------ contra gate
    if not level_requires_contradictory(noise_level, noise_config):
        logger.info(
            "Noise gate: level %s has no contradictory slot; passing.",
            noise_level,
        )
        return

    contra_rel = pools_section.get("contradictory")
    if not contra_rel:
        _fail(
            f"Noise gate: level {noise_level} requires 'contradictory' but "
            "it is not declared under noise.pools in the config.",
        )
    contra_path = (
        (root / contra_rel) if not Path(contra_rel).is_absolute() else Path(contra_rel)
    )
    if not contra_path.exists():
        _fail(
            f"Noise gate: level {noise_level} requires contradictory pool at "
            f"{contra_path}, but the file does not exist. Run "
            "scripts/build_noise.py contradictory first.",
        )

    stats_rel = review_stats_path or contra_cfg.get(
        "review_stats_path",
        "outputs/noise_review/contradictory_review_stats.json",
    )
    stats_path = Path(stats_rel)
    if not stats_path.is_absolute():
        stats_path = root / stats_path
    if not stats_path.exists():
        _fail(
            f"Noise gate: level {noise_level} requires review stats at "
            f"{stats_path}, but the file does not exist. Run the 50-item "
            "manual review first; see configs/noise.yaml "
            "'contradictory.lockdown_policy'.",
        )

    try:
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(f"Noise gate: review stats at {stats_path} is not valid JSON: {exc}")

    try:
        observed = float(stats["acceptance_rate"])
    except (KeyError, TypeError, ValueError):
        _fail(
            f"Noise gate: review stats at {stats_path} is missing a numeric "
            "'acceptance_rate' field.",
        )

    if observed < min_rate:
        _fail(
            f"Noise gate: contradictory acceptance_rate={observed:.3f} < "
            f"min_acceptance_rate={min_rate:.3f}. Per noise.yaml "
            "lockdown_policy, the only valid response is to regenerate "
            "with a refined prompt and re-review - NEVER lower the "
            f"threshold. Review stats: {stats_path}.",
        )

    logger.info(
        "Noise gate: level %s passes (acceptance_rate=%.3f >= %.3f).",
        noise_level,
        observed,
        min_rate,
    )
