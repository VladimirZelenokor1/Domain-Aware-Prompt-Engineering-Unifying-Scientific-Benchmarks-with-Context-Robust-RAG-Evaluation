"""Tests for ``scripts/build_noise.py contradictory`` (Phase E generator).

Covers:
  - MCQ filtering: only mcq-4-choices / mcq-2-choices with valid answerKey
    survive into the candidate pool.
  - fake_answer selection is deterministic under seed and distinct from
    the gold answer.
  - JSONL output schema: noise_id, noise_type, source_qid, original_answer,
    fake_answer, text, domain, prompt_version, model.
  - End-to-end generation with a deterministic mock LLM produces N
    well-formed records under a fixed seed.
  - Word-count quality filter rejects too-short / too-long generations.
  - Resume: re-running with append=True continues numbering and skips
    already-processed source_qids.
  - contradictory-review samples N rows, verdict column empty.
  - contradictory-stats reads filled CSV, writes acceptance_rate JSON.

All tests are hermetic: no GPU, no vLLM, no network.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_noise import (  # noqa: E402
    CONTRADICTORY_PROMPT_VERSION,
    MAX_WORDS_CONTRADICTORY,
    MIN_WORDS_CONTRADICTORY,
    build_contradictory,
    contradictory_review_export,
    contradictory_review_stats,
    iter_mcq_candidates,
    pick_fake_answer,
)


# ---------------------------------------------------------------- fixtures


def _mcq4(qid: int = 0, domain: str = "Physics", key: str = "C") -> dict:
    return {
        "question": f"Q{qid}: which value is correct?",
        "type": "mcq-4-choices",
        "domain": domain,
        "answerKey": key,
        "answer": "",
        "details": {"id": f"src-{qid:05d}"},
        "choices": {
            "text": [
                f"option-A-{qid}",
                f"option-B-{qid}",
                f"option-C-{qid}",
                f"option-D-{qid}",
            ],
            "label": ["A", "B", "C", "D"],
        },
    }


def _mcq2(qid: int = 0, domain: str = "Chemistry", key: str = "A") -> dict:
    return {
        "question": f"Q{qid}: yes or no?",
        "type": "mcq-2-choices",
        "domain": domain,
        "answerKey": key,
        "answer": "",
        "details": {"id": f"src-{qid:05d}"},
        "choices": {
            "text": [f"yes-text-{qid}", f"no-text-{qid}"],
            "label": ["A", "B"],
        },
    }


def _open(qid: int = 99) -> dict:
    return {
        "question": "Open question",
        "type": "open-ended-qa",
        "domain": "Biology",
        "answerKey": "",
        "answer": "Some narrative answer.",
        "details": {"id": f"src-{qid:05d}"},
        "choices": {"text": [], "label": []},
    }


@pytest.fixture
def main_test_path(tmp_path: Path) -> Path:
    """Mixed dataset: 5 mcq4 + 1 mcq2 + 2 open-ended + 1 mcq4 with bad key."""
    records = (
        [_mcq4(i) for i in range(5)]
        + [_mcq2(10)]
        + [_open(20), _open(21)]
        + [
            {
                **_mcq4(30),
                "answerKey": "Z",  # not in labels -> must be filtered
            }
        ]
    )
    p = tmp_path / "main_test.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


@pytest.fixture
def prompt_template_path(tmp_path: Path) -> Path:
    p = tmp_path / "prompts_noise_contradictory.txt"
    p.write_text(
        "Q: {question}\nA: {fake_answer}\nWrite a paragraph.\n",
        encoding="utf-8",
    )
    return p


# -------------------------- mock LLM -----------------------------------


class _MockGenOutput:
    def __init__(self, text: str) -> None:
        self.text = text


class _MockRequestOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [_MockGenOutput(text)]


class MockGenLLM:
    """Mock vLLM-like engine for contradictory generation tests.

    For each prompt, emits a paragraph that includes the prompt's fake_answer
    so we can verify provenance round-trips, and pads to ~100 words to pass
    the word-count quality filter.
    """

    def __init__(self, *, override_text: str | None = None) -> None:
        self._override = override_text

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any | None = None,
        *,
        use_tqdm: bool = True,
    ) -> list[_MockRequestOutput]:
        results: list[_MockRequestOutput] = []
        for p in prompts:
            if self._override is not None:
                results.append(_MockRequestOutput(self._override))
                continue
            # Extract fake answer marker from prompt body.
            fake = ""
            for line in p.splitlines():
                if line.startswith("A: "):
                    fake = line[3:].strip()
                    break
            padding = " ".join([f"word{i}" for i in range(95)])
            text = (
                f"Established consensus shows {fake} as the correct value. {padding}."
            )
            results.append(_MockRequestOutput(text))
        return results


# ============================================================== iter_mcq


def test_iter_mcq_candidates_keeps_only_mcq_with_valid_key(
    main_test_path: Path,
) -> None:
    cands = list(iter_mcq_candidates(main_test_path))
    qids = {c["source_qid"] for c in cands}
    assert qids == {f"src-{i:05d}" for i in [0, 1, 2, 3, 4, 10]}


def test_iter_mcq_candidates_attaches_original_and_distractors(
    main_test_path: Path,
) -> None:
    cands = list(iter_mcq_candidates(main_test_path))
    rec = next(c for c in cands if c["source_qid"] == "src-00000")
    assert rec["original_answer"] == "option-C-0"
    assert set(rec["distractors"]) == {"option-A-0", "option-B-0", "option-D-0"}
    assert rec["domain"] == "Physics"


# ============================================================ pick_fake


def test_pick_fake_answer_is_distinct_from_gold() -> None:
    import random

    rng = random.Random(42)
    fake = pick_fake_answer(
        original="option-C-0",
        distractors=["option-A-0", "option-B-0", "option-D-0"],
        rng=rng,
    )
    assert fake != "option-C-0"
    assert fake in {"option-A-0", "option-B-0", "option-D-0"}


def test_pick_fake_answer_returns_none_when_no_distractors() -> None:
    import random

    rng = random.Random(0)
    assert pick_fake_answer(original="X", distractors=[], rng=rng) is None


def test_pick_fake_answer_deterministic_under_same_seed() -> None:
    import random

    distractors = ["A", "B", "C"]
    f1 = pick_fake_answer("X", distractors, random.Random(7))
    f2 = pick_fake_answer("X", distractors, random.Random(7))
    assert f1 == f2


# ====================================================== build_contradictory


def test_build_contradictory_writes_target_records(
    tmp_path: Path,
    main_test_path: Path,
    prompt_template_path: Path,
) -> None:
    out = tmp_path / "contradictory_passages.jsonl"
    stats = build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=3,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=2,
    )
    assert stats["written"] == 3
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_build_contradictory_records_have_required_schema(
    tmp_path: Path,
    main_test_path: Path,
    prompt_template_path: Path,
) -> None:
    out = tmp_path / "contradictory_passages.jsonl"
    build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=2,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=8,
    )
    lines = out.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[0])
    required = {
        "noise_id",
        "noise_type",
        "source_qid",
        "domain",
        "original_answer",
        "fake_answer",
        "text",
        "prompt_version",
        "model",
        "generated_at",
    }
    assert required <= set(rec.keys())
    assert rec["noise_type"] == "contradictory"
    assert rec["prompt_version"] == CONTRADICTORY_PROMPT_VERSION
    assert rec["model"] == "qwen2.5-7b"
    assert rec["fake_answer"] != rec["original_answer"]
    assert rec["fake_answer"] in rec["text"]


def test_build_contradictory_filters_short_or_long_generations(
    tmp_path: Path,
    main_test_path: Path,
    prompt_template_path: Path,
) -> None:
    """Generations outside [MIN, MAX] words must be skipped, not written."""
    out = tmp_path / "out.jsonl"
    too_short = MockGenLLM(override_text="too short.")
    stats = build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=3,
        seed=42,
        engine=too_short,
        model_name="qwen2.5-7b",
        batch_size=8,
    )
    assert stats["written"] == 0
    assert stats["rejected_word_count"] >= 1
    assert MIN_WORDS_CONTRADICTORY > 0
    assert MAX_WORDS_CONTRADICTORY > MIN_WORDS_CONTRADICTORY


def test_build_contradictory_resume_appends_and_continues_numbering(
    tmp_path: Path,
    main_test_path: Path,
    prompt_template_path: Path,
) -> None:
    out = tmp_path / "out.jsonl"
    build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=2,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=8,
    )
    first_qids = [
        json.loads(line)["source_qid"] for line in out.read_text().splitlines()
    ]
    build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=4,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=8,
        append=True,
    )
    all_lines = out.read_text(encoding="utf-8").splitlines()
    assert len(all_lines) == 4
    all_qids = [json.loads(line)["source_qid"] for line in all_lines]
    # Original 2 records preserved at the head, no duplicates.
    assert all_qids[:2] == first_qids
    assert len(set(all_qids)) == len(all_qids)
    # Noise IDs must be globally unique and monotonically numbered.
    ids = [json.loads(line)["noise_id"] for line in all_lines]
    assert ids == sorted(ids)
    assert len(set(ids)) == 4


def test_build_contradictory_skips_already_seen_source_qids(
    tmp_path: Path,
    main_test_path: Path,
    prompt_template_path: Path,
) -> None:
    out = tmp_path / "out.jsonl"
    build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=2,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=8,
    )
    seen = {json.loads(line)["source_qid"] for line in out.read_text().splitlines()}
    stats = build_contradictory(
        main_test_path=main_test_path,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=10,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=8,
        append=True,
    )
    new_lines = out.read_text(encoding="utf-8").splitlines()
    new_qids = [json.loads(line)["source_qid"] for line in new_lines]
    # The qids written in the second pass must NOT overlap with the first.
    second_pass_qids = new_qids[2:]
    assert seen.isdisjoint(set(second_pass_qids))
    assert stats["written"] >= 1


# ============================================================ review export


def test_contradictory_review_export_writes_csv(tmp_path: Path) -> None:
    pool = tmp_path / "pool.jsonl"
    records = []
    for i in range(20):
        records.append(
            {
                "noise_id": f"con_{i:05d}",
                "noise_type": "contradictory",
                "source_qid": f"src-{i:05d}",
                "domain": "Physics",
                "original_answer": f"orig-{i}",
                "fake_answer": f"fake-{i}",
                "text": f"paragraph {i}",
                "prompt_version": "v1",
                "model": "qwen2.5-7b",
                "generated_at": "2026-04-30T00:00:00",
            }
        )
    with open(pool, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    csv_out = tmp_path / "review.csv"
    info = contradictory_review_export(
        pool_path=pool, csv_path=csv_out, sample=5, seed=42
    )
    assert info["sampled"] == 5
    with open(csv_out, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert {"noise_id", "source_qid", "fake_answer", "verdict"} <= set(rows[0].keys())
    assert all(row["verdict"] == "" for row in rows)


# ============================================================ review stats


def test_contradictory_review_stats_computes_acceptance_rate(tmp_path: Path) -> None:
    csv_in = tmp_path / "review.csv"
    with open(csv_in, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "noise_id",
                "source_qid",
                "domain",
                "original_answer",
                "fake_answer",
                "text",
                "verdict",
            ],
        )
        w.writeheader()
        for i in range(10):
            w.writerow(
                {
                    "noise_id": f"con_{i:05d}",
                    "source_qid": f"src-{i:05d}",
                    "domain": "Physics",
                    "original_answer": f"orig-{i}",
                    "fake_answer": f"fake-{i}",
                    "text": "x",
                    "verdict": "accept" if i < 8 else "reject",
                }
            )
    stats_out = tmp_path / "stats.json"
    stats = contradictory_review_stats(csv_path=csv_in, stats_path=stats_out)
    assert stats["sample_size"] == 10
    assert stats["acceptance_rate"] == pytest.approx(0.8)
    assert stats["accepted"] == 8
    assert stats["rejected"] == 2

    on_disk = json.loads(stats_out.read_text(encoding="utf-8"))
    assert on_disk["acceptance_rate"] == pytest.approx(0.8)


def test_build_contradictory_handles_braces_in_question(
    tmp_path: Path,
    prompt_template_path: Path,
) -> None:
    """SciKnowEval contains LaTeX/code with literal {} - must not crash."""
    records = [
        {
            "question": (
                "Which method was used for the cultivation of "
                "In\text{2}Te\text{5} {single} crystal?"
            ),
            "type": "mcq-4-choices",
            "domain": "Material",
            "answerKey": "B",
            "answer": "",
            "details": {"id": "src-brace-0"},
            "choices": {
                "text": ["bridg{man}", "{floating} zone", "czoch{ralski}", "vapor"],
                "label": ["A", "B", "C", "D"],
            },
        }
    ]
    main_test = tmp_path / "main_test.json"
    main_test.write_text(json.dumps(records), encoding="utf-8")
    out = tmp_path / "out.jsonl"
    stats = build_contradictory(
        main_test_path=main_test,
        prompt_template_path=prompt_template_path,
        output_path=out,
        target=1,
        seed=42,
        engine=MockGenLLM(),
        model_name="qwen2.5-7b",
        batch_size=1,
    )
    assert stats["written"] == 1


def test_contradictory_review_stats_raises_on_empty_verdicts(tmp_path: Path) -> None:
    csv_in = tmp_path / "review.csv"
    with open(csv_in, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "noise_id",
                "source_qid",
                "domain",
                "original_answer",
                "fake_answer",
                "text",
                "verdict",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "noise_id": "con_00000",
                "source_qid": "x",
                "domain": "Physics",
                "original_answer": "o",
                "fake_answer": "f",
                "text": "t",
                "verdict": "",
            }
        )
    with pytest.raises(ValueError, match="verdict"):
        contradictory_review_stats(csv_path=csv_in, stats_path=tmp_path / "s.json")
