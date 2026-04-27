"""Tests for ``scripts/noise_assembler.assemble_noisy_context``.

Covers the four invariants from the work-queue Phase 5 spec:
  - deterministic under a fixed seed + question_id;
  - correct per-level composition counts;
  - positions are shuffled (depend on seed);
  - missing pools raise clearly rather than silently returning junk.

All tests are hermetic: the pools are built from short in-memory
records, no corpus streaming, no GPU, no network.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from noise_assembler import (  # noqa: E402
    ALL_NOISE_TYPES,
    NoisePool,
    _expand_composition,
    _stable_seed,
    assemble_noisy_context,
    map_question_domain,
)


# -------------------------------------------------------------- fixtures


def _make_real_passages(n: int = 10) -> list[dict]:
    return [
        {"chunk_id": f"real{i:03d}", "text": f"real passage {i}", "rank": i + 1}
        for i in range(n)
    ]


def _make_question(domain: str = "Biology", qid: str = "q-0001") -> dict:
    return {
        "question_id": qid,
        "domain": domain,
        "question": "Q?",
        "type": "open-ended-qa",
    }


def _build_pool(noise_type: str, records: list[dict]) -> NoisePool:
    pool = NoisePool(noise_type=noise_type, records=list(records))
    # Populate the per-id / per-domain indices the way ``load`` would.
    for i, r in enumerate(records):
        cid = r.get("chunk_id") or r.get("noise_id")
        if cid:
            pool._id_index[cid] = i
        if noise_type == "irrelevant":
            pool._by_domain.setdefault(r.get("domain", "unknown"), []).append(i)
    return pool


@pytest.fixture
def pools() -> dict:
    irrelevant = [
        {"chunk_id": f"irr_{d}_{i:02d}", "text": f"irr {d} {i}", "domain": d}
        for d in (
            "biology",
            "chemistry",
            "physics",
            "materials_science",
            "earth_science",
        )
        for i in range(20)
    ]
    injection = [
        {"noise_id": f"inj_{i:03d}", "text": f"inj {i}", "template_id": i % 3}
        for i in range(50)
    ]
    contradictory = [
        {"noise_id": f"con_{i:03d}", "text": f"con {i}"} for i in range(50)
    ]
    return {
        "irrelevant": _build_pool("irrelevant", irrelevant),
        "injection": _build_pool("injection", injection),
        "contradictory": _build_pool("contradictory", contradictory),
    }


CONFIG: dict = {
    "seed": 42,
    "top_k": 10,
    "levels": {
        "0.0": {"replaced": 0, "composition": {}},
        "0.2": {
            "replaced": 2,
            "composition": {"irrelevant": 1, "contradictory_or_injection": 1},
        },
        "0.4": {
            "replaced": 4,
            "composition": {
                "irrelevant": 1,
                "contradictory": 1,
                "injection": 1,
                "any": 1,
            },
        },
        "0.6": {
            "replaced": 6,
            "composition": {"irrelevant": 2, "contradictory": 2, "injection": 2},
        },
    },
}


# ---------------------------------------------------------- sanity helpers


def test_map_question_domain_known() -> None:
    assert map_question_domain("Biology") == "biology"
    assert map_question_domain("Material") == "materials_science"


def test_map_question_domain_unknown() -> None:
    assert map_question_domain("Astrology") is None


def test_stable_seed_deterministic() -> None:
    a = _stable_seed(42, "ske-main-test-00001")
    b = _stable_seed(42, "ske-main-test-00001")
    assert a == b


def test_stable_seed_varies_by_id() -> None:
    assert _stable_seed(42, "qA") != _stable_seed(42, "qB")


def test_stable_seed_varies_by_base_seed() -> None:
    assert _stable_seed(1, "q") != _stable_seed(2, "q")


# ---------------------------------------------------------- composition


def test_level_0_returns_real_only(pools: dict) -> None:
    q = _make_question()
    real = _make_real_passages(10)
    out = assemble_noisy_context(real, q, 0.0, CONFIG, pools, seed=42)
    assert len(out) == 10
    assert all(p.get("noise_type", "real") == "real" for p in out)
    # Exactly the 10 real chunk_ids, irrespective of order.
    assert {p["chunk_id"] for p in out} == {p["chunk_id"] for p in real}


def test_level_02_composition(pools: dict) -> None:
    q = _make_question()
    out = assemble_noisy_context(
        _make_real_passages(10), q, 0.2, CONFIG, pools, seed=42
    )
    assert len(out) == 10
    c = Counter(p.get("noise_type", "real") for p in out)
    assert c["real"] == 8
    assert c["irrelevant"] == 1
    # The second slot is contradictory OR injection (random one-of).
    assert c["contradictory"] + c["injection"] == 1


def test_level_04_composition(pools: dict) -> None:
    q = _make_question()
    out = assemble_noisy_context(
        _make_real_passages(10), q, 0.4, CONFIG, pools, seed=42
    )
    assert len(out) == 10
    c = Counter(p.get("noise_type", "real") for p in out)
    assert c["real"] == 6
    assert c["irrelevant"] >= 1
    assert c["contradictory"] >= 1
    assert c["injection"] >= 1
    # The "any" slot contributes one more of any fixed type.
    assert c["irrelevant"] + c["contradictory"] + c["injection"] == 4


def test_level_06_composition(pools: dict) -> None:
    q = _make_question()
    out = assemble_noisy_context(
        _make_real_passages(10), q, 0.6, CONFIG, pools, seed=42
    )
    assert len(out) == 10
    c = Counter(p.get("noise_type", "real") for p in out)
    assert c["real"] == 4
    assert c["irrelevant"] == 2
    assert c["contradictory"] == 2
    assert c["injection"] == 2


# ---------------------------------------------------------- determinism


def test_deterministic_under_fixed_seed(pools: dict) -> None:
    q = _make_question(qid="q-123")
    real = _make_real_passages(10)
    a = assemble_noisy_context(real, q, 0.4, CONFIG, pools, seed=42)
    b = assemble_noisy_context(real, q, 0.4, CONFIG, pools, seed=42)
    assert [p.get("chunk_id") or p.get("noise_id") for p in a] == [
        p.get("chunk_id") or p.get("noise_id") for p in b
    ]


def test_different_seeds_produce_different_orders(pools: dict) -> None:
    q = _make_question(qid="q-123")
    real = _make_real_passages(10)
    a = assemble_noisy_context(real, q, 0.4, CONFIG, pools, seed=1)
    b = assemble_noisy_context(real, q, 0.4, CONFIG, pools, seed=2)
    key_a = [p.get("chunk_id") or p.get("noise_id") for p in a]
    key_b = [p.get("chunk_id") or p.get("noise_id") for p in b]
    assert key_a != key_b, "assembly must depend on seed"


def test_different_question_ids_produce_different_orders(pools: dict) -> None:
    real = _make_real_passages(10)
    a = assemble_noisy_context(
        real,
        _make_question(qid="qA"),
        0.4,
        CONFIG,
        pools,
        seed=42,
    )
    b = assemble_noisy_context(
        real,
        _make_question(qid="qB"),
        0.4,
        CONFIG,
        pools,
        seed=42,
    )
    key_a = [p.get("chunk_id") or p.get("noise_id") for p in a]
    key_b = [p.get("chunk_id") or p.get("noise_id") for p in b]
    assert key_a != key_b, "assembly must depend on question_id"


# ---------------------------------------------------------- shuffling


def test_noise_positions_are_shuffled_not_appended(pools: dict) -> None:
    """Across 16 seeds the noise passages must not always sit at the tail."""
    q = _make_question()
    real = _make_real_passages(10)
    tail_only = 0
    for s in range(16):
        out = assemble_noisy_context(real, q, 0.4, CONFIG, pools, seed=s)
        # True if every noise passage landed in positions 6..9 (the tail).
        tail_positions = {
            i for i, p in enumerate(out) if p.get("noise_type", "real") != "real"
        }
        if tail_positions.issubset({6, 7, 8, 9}):
            tail_only += 1
    assert tail_only < 16, "noise is always at the tail -> shuffle is broken"


# ---------------------------------------------------------- domain control


def test_irrelevant_excludes_question_domain(pools: dict) -> None:
    q = _make_question(domain="Biology")
    real = _make_real_passages(10)
    for s in range(8):
        out = assemble_noisy_context(real, q, 0.6, CONFIG, pools, seed=s)
        irrel = [p for p in out if p.get("noise_type") == "irrelevant"]
        assert len(irrel) == 2
        for p in irrel:
            # Irrelevant distractors must not be from the question's domain.
            assert p.get("domain") != "biology", (
                f"seed={s}: irrelevant distractor leaked in biology: {p}"
            )


# ---------------------------------------------------------- error paths


def test_missing_pool_raises(pools: dict) -> None:
    del pools["contradictory"]
    q = _make_question()
    real = _make_real_passages(10)
    with pytest.raises(RuntimeError, match="contradictory"):
        assemble_noisy_context(real, q, 0.6, CONFIG, pools, seed=42)


def test_unknown_composition_key_raises(pools: dict) -> None:
    cfg = json.loads(json.dumps(CONFIG))
    cfg["levels"]["0.4"]["composition"] = {"bogus": 1, "irrelevant": 1}
    q = _make_question()
    real = _make_real_passages(10)
    with pytest.raises(ValueError, match="Unknown composition key"):
        assemble_noisy_context(real, q, 0.4, cfg, pools, seed=42)


def test_unknown_noise_level_raises(pools: dict) -> None:
    q = _make_question()
    real = _make_real_passages(10)
    with pytest.raises(KeyError, match="0.5"):
        assemble_noisy_context(real, q, 0.5, CONFIG, pools, seed=42)


# ---------------------------------------------------------- composition exp


def test_expand_composition_counts() -> None:
    import random

    rng = random.Random(0)
    slots = _expand_composition(
        {"irrelevant": 2, "contradictory": 1, "any": 3},
        rng,
    )
    assert len(slots) == 6
    assert sum(1 for x in slots if x == "irrelevant") >= 2
    assert all(x in ALL_NOISE_TYPES for x in slots)
