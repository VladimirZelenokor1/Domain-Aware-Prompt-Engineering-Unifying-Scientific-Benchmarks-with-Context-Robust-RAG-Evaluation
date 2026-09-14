"""Noise assembler for RAG experiments (thesis Section 4.5).

Given a top-k retrieval result and a noise level, replaces a subset of
the retrieved passages with passages drawn from pre-built noise pools
(irrelevant / injection / contradictory). The assembly is deterministic
under a fixed ``seed`` plus ``question_id`` so the same cell of the
experimental matrix is byte-reproducible across runs.

Public API
----------
- ``NoisePool`` -- in-memory pool of noise passages of a single type.
- ``load_pools(config)`` -- construct pools from a noise.yaml config.
- ``assemble_noisy_context(...)`` -- replace n of top_k passages with
  noise according to the composition rules.

Domain alignment
----------------
SciKnowEval question domains use capitalised names ("Biology",
"Chemistry", "Physics", "Material"); the corpus uses lowercase
snake-case ("biology", "chemistry", "physics", "materials_science",
"earth_science"). ``map_question_domain`` normalises the two.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

NoiseType = Literal["irrelevant", "injection", "contradictory"]
ALL_NOISE_TYPES: tuple[NoiseType, ...] = ("irrelevant", "injection", "contradictory")

# SciKnowEval question-domain -> corpus-domain.
QUESTION_TO_CORPUS_DOMAIN: dict[str, str] = {
    "Biology": "biology",
    "Chemistry": "chemistry",
    "Physics": "physics",
    "Material": "materials_science",
}


def map_question_domain(question_domain: str) -> str | None:
    """Return the corpus-side domain label for a SciKnowEval question.

    Args:
        question_domain: Value of ``question["domain"]`` (e.g. "Biology").

    Returns:
        Corpus-side lowercase domain (e.g. "biology"), or ``None`` if
        the question domain is unknown (noise sampling then falls back
        to unconstrained random).
    """
    return QUESTION_TO_CORPUS_DOMAIN.get(question_domain)


# ----------------------------------------------------------------- pools


@dataclass
class NoisePool:
    """In-memory pool of noise passages of a single noise type."""

    noise_type: NoiseType
    records: list[dict]
    # chunk_id -> index into ``records``; used only for fast dedup.
    _id_index: dict[str, int] = field(default_factory=dict, repr=False)
    # domain -> list of indices (only for ``irrelevant``).
    _by_domain: dict[str, list[int]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path, noise_type: NoiseType) -> "NoisePool":
        """Load a JSONL pool from disk.

        Args:
            path: Path to a JSONL file where each line is a noise record
                with at least a ``text`` field.
            noise_type: Pool type; stored for sample-time validation.

        Returns:
            A :class:`NoisePool`. Index structures are populated for
            ``irrelevant`` pools so ``sample_excluding_domain`` runs in
            O(k); for other pool types only the flat list is built.
        """
        records: list[dict] = []
        id_index: dict[str, int] = {}
        by_domain: dict[str, list[int]] = {}
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)
                cid = rec.get("chunk_id") or rec.get("noise_id")
                if cid:
                    id_index[cid] = i
                if noise_type == "irrelevant":
                    by_domain.setdefault(rec.get("domain", "unknown"), []).append(i)

        return cls(
            noise_type=noise_type,
            records=records,
            _id_index=id_index,
            _by_domain=by_domain,
        )

    def __len__(self) -> int:
        return len(self.records)

    def sample(
        self,
        rng: random.Random,
        k: int,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        """Sample ``k`` records uniformly at random from the full pool."""
        if not self.records:
            raise RuntimeError(f"noise pool '{self.noise_type}' is empty")
        return self._sample_from_indices(
            list(range(len(self.records))),
            rng,
            k,
            exclude_ids,
        )

    def sample_excluding_domain(
        self,
        rng: random.Random,
        k: int,
        exclude_domain: str | None,
        exclude_ids: set[str] | None = None,
    ) -> list[dict]:
        """Sample ``k`` records whose ``domain`` differs from ``exclude_domain``.

        Falls back to full-pool sampling if the domain index is absent
        or the exclusion would leave too few candidates.

        Args:
            rng: Random source.
            k: Number of records to return.
            exclude_domain: Corpus-side domain to avoid.
            exclude_ids: Optional set of chunk_ids to skip.
        """
        if self.noise_type != "irrelevant":
            return self.sample(rng, k, exclude_ids=exclude_ids)
        if exclude_domain is None or not self._by_domain:
            return self.sample(rng, k, exclude_ids=exclude_ids)

        candidate_indices: list[int] = []
        for dom, idxs in self._by_domain.items():
            if dom != exclude_domain:
                candidate_indices.extend(idxs)

        if len(candidate_indices) < k:
            logger.warning(
                "irrelevant pool has only %d candidates outside domain %s; "
                "backing off to full pool",
                len(candidate_indices),
                exclude_domain,
            )
            return self.sample(rng, k, exclude_ids=exclude_ids)

        return self._sample_from_indices(
            candidate_indices,
            rng,
            k,
            exclude_ids,
        )

    def _sample_from_indices(
        self,
        candidate_indices: list[int],
        rng: random.Random,
        k: int,
        exclude_ids: set[str] | None,
    ) -> list[dict]:
        if not candidate_indices:
            raise RuntimeError(f"no candidates available in '{self.noise_type}' pool")
        if exclude_ids:
            candidate_indices = [
                i
                for i in candidate_indices
                if (self.records[i].get("chunk_id") or self.records[i].get("noise_id"))
                not in exclude_ids
            ]
        if len(candidate_indices) < k:
            logger.warning(
                "noise pool '%s' cannot satisfy k=%d uniquely after exclusions; "
                "sampling with replacement",
                self.noise_type,
                k,
            )
            picks = [rng.choice(candidate_indices) for _ in range(k)]
        else:
            picks = rng.sample(candidate_indices, k)
        return [self.records[i] for i in picks]


def load_pools(
    paths: dict[str, str | Path],
) -> dict[NoiseType, NoisePool]:
    """Load every pool referenced in the ``pools`` section of noise.yaml.

    Args:
        paths: Mapping ``noise_type -> file path``. Missing files are
            skipped with a warning (useful while the contradictory pool
            has not been built yet).

    Returns:
        Dict ``NoiseType -> NoisePool`` for every file that actually
        existed on disk.
    """
    pools: dict[NoiseType, NoisePool] = {}
    for key, path_str in paths.items():
        if key not in ALL_NOISE_TYPES:
            logger.warning("Skipping unknown pool key in config: %r", key)
            continue
        path = Path(path_str)
        if not path.exists():
            logger.warning("Noise pool %s not found at %s; skipping", key, path)
            continue
        pools[key] = NoisePool.load(path, key)  # type: ignore[arg-type]
        logger.info("Loaded %s pool: %d records", key, len(pools[key]))
    return pools


# ----------------------------------------------------------------- assembly


def _expand_composition(
    composition: dict[str, int],
    rng: random.Random,
) -> list[NoiseType]:
    """Turn a composition dict into a flat list of concrete noise types.

    Recognised keys:
    - ``irrelevant``, ``injection``, ``contradictory``: fixed types.
    - ``contradictory_or_injection``: randomly pick one of the two.
    - ``any``: randomly pick one of the three fixed types.

    Args:
        composition: As specified under ``levels.<lvl>.composition`` in
            noise.yaml.
        rng: Random source.

    Returns:
        A list whose length equals ``sum(composition.values())``, with
        one concrete :data:`NoiseType` per slot.
    """
    slots: list[NoiseType] = []
    for raw_key, count in composition.items():
        key = str(raw_key)
        for _ in range(int(count)):
            if key in ALL_NOISE_TYPES:
                slots.append(key)  # type: ignore[arg-type]
            elif key == "contradictory_or_injection":
                slots.append(rng.choice(("contradictory", "injection")))
            elif key == "any":
                slots.append(rng.choice(ALL_NOISE_TYPES))
            else:
                raise ValueError(
                    f"Unknown composition key: {key!r}. Valid keys are "
                    f"{ALL_NOISE_TYPES + ('contradictory_or_injection', 'any')}"
                )
    return slots


def _stable_seed(base_seed: int, question_id: str) -> int:
    """Deterministic per-question seed derived from ``base_seed`` and id."""
    # Portable Python hash would be fine but salting with the base seed
    # keeps the combined value in int range without builtin ``hash`` salt.
    mixed = f"{base_seed}::{question_id}".encode()
    # FNV-ish fold: use a stable constant algorithm (sha1 truncated).
    import hashlib

    digest = hashlib.sha1(mixed).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def assemble_noisy_context(
    real_passages: list[dict],
    question: dict,
    noise_level: float,
    noise_config: dict,
    pools: dict[NoiseType, NoisePool],
    seed: int,
) -> list[dict]:
    """Build a top-k context that mixes ``real_passages`` with noise.

    Each emitted passage is annotated with ``noise_type`` (either
    ``"real"`` or one of the NoiseType values) for downstream analysis.
    Position within the returned list is shuffled deterministically.

    Args:
        real_passages: The top-k dense/hybrid retrieval result.
        question: The SciKnowEval question record; needs ``domain`` and
            ``question_id`` (or a stable equivalent) for seeding.
        noise_level: One of the keys under ``noise_config["levels"]``
            (e.g. ``0.0``, ``0.2``, ``0.4``, ``0.6``).
        noise_config: Parsed ``configs/noise.yaml``.
        pools: Output of :func:`load_pools`.
        seed: Base seed; combined with ``question_id`` for determinism.

    Returns:
        A list of length ``top_k`` (as declared in the config). The
        first ``top_k - total_noise`` passages are from ``real_passages``
        (in their original rank order prior to final shuffle); the rest
        are drawn from ``pools``. The whole list is then shuffled.
    """
    top_k = int(noise_config["top_k"])
    levels = noise_config["levels"]
    level_key = _normalise_level_key(noise_level, levels)

    level_cfg = levels[level_key]
    composition = level_cfg.get("composition", {})

    if len(real_passages) < top_k:
        logger.warning(
            "real_passages has %d items but top_k=%d; assembling with what's "
            "available.",
            len(real_passages),
            top_k,
        )
    real_truncated = list(real_passages[:top_k])

    rng = random.Random(_stable_seed(seed, str(question.get("question_id", ""))))

    noise_slots = _expand_composition(composition, rng)
    n_noise = len(noise_slots)
    n_real_keep = max(0, top_k - n_noise)

    kept_real = real_truncated[:n_real_keep]
    # Annotate real passages.
    for p in kept_real:
        p.setdefault("noise_type", "real")

    # Fill noise slots, avoiding duplicates inside a single question's context.
    exclude_ids: set[str] = {p.get("chunk_id") for p in kept_real if p.get("chunk_id")}
    noise_records: list[dict] = []
    exclude_domain = map_question_domain(question.get("domain", ""))

    for nt in noise_slots:
        if nt not in pools:
            raise RuntimeError(
                f"Noise level {noise_level} requires a '{nt}' pool but it "
                f"was not loaded. Build it first (scripts/build_noise.py {nt})."
            )
        pool = pools[nt]
        if nt == "irrelevant":
            rec = pool.sample_excluding_domain(
                rng,
                k=1,
                exclude_domain=exclude_domain,
                exclude_ids=exclude_ids,
            )[0]
        else:
            rec = pool.sample(rng, k=1, exclude_ids=exclude_ids)[0]
        rec = dict(rec)
        rec["noise_type"] = nt
        noise_records.append(rec)
        key = rec.get("chunk_id") or rec.get("noise_id")
        if key:
            exclude_ids.add(key)

    combined = kept_real + noise_records
    rng.shuffle(combined)
    return combined[:top_k]


def _normalise_level_key(noise_level: float, levels: dict) -> str:
    """Find the config key matching ``noise_level`` (float vs str tolerated)."""
    # Accept "0.2", "0.20", 0.2 as equivalents.
    target = float(noise_level)
    for k in levels:
        if float(k) == target:
            return k
    raise KeyError(
        f"noise_level={noise_level} not declared in config "
        f"(have {sorted(levels.keys())})"
    )
