"""Read-only integrity audit for all experiment phases.

Verifies that the closed-book (Phase D), RAG main (Phase F), QASPER
(Phase G) and judge (Phase H) outputs are complete, uncorrupted, and
internally consistent before the final statistics are trusted. Nothing is
modified; the script only reads .jsonl outputs and prints a PASS/WARN/FAIL
report. Exit code is non-zero if any FAIL is recorded.

Usage:
    python scripts/audit_experiments.py
    python scripts/audit_experiments.py --outputs-root outputs
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

PROJECT_ROOT = _SCRIPTS_DIR.parent

# --- Fixed experimental parameters (configs/{rag,global,noise}.yaml) ---------
EXPECTED_MODELS = {
    "llama-3.2-3b",
    "sciphi-mistral-7b",
    "deepseek-r1-qwen-7b",
    "qwen2.5-7b",
    "gemma-2-9b",
    "mistral-nemo-12b",
}
EXPECTED_STRATEGIES = {"da", "ras", "ctl", "sc"}
TOP_K = 10
# noise_level -> number of top-k passages replaced by noise (configs/noise.yaml)
NOISE_REPLACED = {0.0: 0, 0.2: 2, 0.4: 4, 0.6: 6}

RAG_CELL_RE = re.compile(r"^(bm25|dense|hybrid)_noise([0-9.]+)_(da|ras|ctl|sc)$")

# Per-track expectations: (n_cells, is_rag)
TRACKS = {
    "closed_book_main_test": (24, False),
    "rag_main": (144, True),
    "qasper_main": (48, True),
}

# Report accumulator: list of (level, message). level in {PASS, WARN, FAIL}.
_REPORT: list[tuple[str, str]] = []


def record(level: str, message: str) -> None:
    """Append a check result and echo it."""
    _REPORT.append((level, message))
    print(f"[{level}] {message}")


def read_jsonl_safe(path: Path) -> tuple[list[dict], int]:
    """Read a JSONL file, returning (records, n_broken_lines)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                broken += 1
    return records, broken


def predicted_answer(rec: dict) -> str:
    """Mirror compute_metrics.get_predicted_answer (SC vs non-SC)."""
    if rec.get("strategy") == "sc" and rec.get("sc_result"):
        return rec["sc_result"].get("final_answer_normalized", "") or ""
    return rec.get("parsed", {}).get("answer_normalized", "") or ""


def iter_cell_files(track_dir: Path) -> list[Path]:
    """All cell .jsonl files under a track dir (excludes *_metrics.json)."""
    files: list[Path] = []
    for model_dir in sorted(track_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            files.append(jsonl)
    return files


def audit_track(track_dir: Path, n_cells_expected: int, is_rag: bool) -> None:
    """Audit one inference track for completeness and integrity."""
    name = track_dir.name
    if not track_dir.exists():
        record("FAIL", f"{name}: directory missing ({track_dir})")
        return

    files = iter_cell_files(track_dir)
    if len(files) != n_cells_expected:
        record(
            "FAIL",
            f"{name}: {len(files)} cells, expected {n_cells_expected}",
        )
    else:
        record("PASS", f"{name}: {len(files)} cells present")

    models_seen: set[str] = set()
    counts: list[int] = []
    total_broken = 0
    total_dup = 0
    total_empty_pred = 0
    total_records = 0
    empty_gold = 0
    # RAG provenance counters
    retr_mismatch = 0
    noise_mismatch = 0
    bad_topk = 0
    noise_count_mismatch = 0

    for f in files:
        model = f.parent.name
        models_seen.add(model)
        stem = f.stem
        recs, broken = read_jsonl_safe(f)
        total_broken += broken
        counts.append(len(recs))
        total_records += len(recs)

        # duplicate question_id within the cell
        qids = [r.get("question_id") for r in recs]
        dups = len(qids) - len(set(qids))
        total_dup += dups

        # parse the cell coordinates from the filename
        cell_retr = cell_noise = cell_strat = None
        if is_rag:
            m = RAG_CELL_RE.match(stem)
            if m:
                cell_retr, cell_noise, cell_strat = (
                    m.group(1),
                    float(m.group(2)),
                    m.group(3),
                )
            else:
                record("FAIL", f"{name}/{model}/{stem}: filename not parseable")
        else:
            cell_strat = stem

        if cell_strat is not None and cell_strat not in EXPECTED_STRATEGIES:
            record("FAIL", f"{name}/{model}/{stem}: unknown strategy '{cell_strat}'")

        for r in recs:
            if not predicted_answer(r):
                total_empty_pred += 1
            if not str(r.get("gold_answer", "")).strip():
                empty_gold += 1
            if is_rag and cell_retr is not None:
                if r.get("retriever") != cell_retr:
                    retr_mismatch += 1
                if r.get("noise_level") not in (cell_noise, str(cell_noise)):
                    noise_mismatch += 1
                passages = r.get("passages_used") or []
                if len(passages) != TOP_K:
                    bad_topk += 1
                n_noise = sum(1 for p in passages if p.get("noise_type"))
                expected_noise = NOISE_REPLACED.get(cell_noise)
                if expected_noise is not None and n_noise != expected_noise:
                    noise_count_mismatch += 1

    # --- per-track verdicts --------------------------------------------------
    missing_models = EXPECTED_MODELS - models_seen
    if missing_models:
        record("FAIL", f"{name}: missing models {sorted(missing_models)}")
    else:
        record("PASS", f"{name}: all 6 models present")

    if counts:
        cmin, cmax = min(counts), max(counts)
        if cmin == cmax:
            record(
                "PASS", f"{name}: every cell has {cmin} records ({total_records} total)"
            )
        else:
            record(
                "FAIL",
                f"{name}: uneven record counts min={cmin} max={cmax} "
                f"(possible partial/truncated cell)",
            )

    record(
        "FAIL" if total_broken else "PASS",
        f"{name}: {total_broken} corrupted JSON lines",
    )
    record(
        "FAIL" if total_dup else "PASS",
        f"{name}: {total_dup} duplicate question_id within cells",
    )
    record(
        "FAIL" if empty_gold else "PASS",
        f"{name}: {empty_gold} empty gold_answer",
    )
    pct = 100.0 * total_empty_pred / total_records if total_records else 0.0
    record(
        "WARN" if pct > 5 else "PASS",
        f"{name}: {total_empty_pred} empty predicted ({pct:.2f}%)",
    )

    if is_rag:
        record(
            "FAIL" if retr_mismatch else "PASS",
            f"{name}: {retr_mismatch} records whose retriever != filename",
        )
        record(
            "FAIL" if noise_mismatch else "PASS",
            f"{name}: {noise_mismatch} records whose noise_level != filename",
        )
        record(
            "FAIL" if bad_topk else "PASS",
            f"{name}: {bad_topk} records with passages_used != {TOP_K}",
        )
        record(
            "FAIL" if noise_count_mismatch else "PASS",
            f"{name}: {noise_count_mismatch} records whose noise-passage count "
            f"!= config (noise actually applied per design)",
        )


def cell_qid_set(track_dirs: list[Path]) -> dict[str, set[str]]:
    """Map source ``<root>/<model>/<file>`` -> set of question_ids."""
    out: dict[str, set[str]] = {}
    for d in track_dirs:
        if not d.exists():
            continue
        for f in iter_cell_files(d):
            cell = "/".join(f.relative_to(d.parent).parts)
            recs, _ = read_jsonl_safe(f)
            out[cell] = {r.get("question_id") for r in recs}
    return out


def audit_judges(judge_root: Path, source_qids: dict[str, set[str]]) -> None:
    """Audit judge outputs: coverage, value ranges, source join."""
    if not judge_root.exists():
        record("FAIL", f"judge: directory missing ({judge_root})")
        return

    judge_items: dict[str, set[tuple[str, str]]] = {}
    range_bad = 0
    rubric_bad = 0
    orphan = 0  # judged (cell,qid) with no matching source record
    counts_per_judge: dict[str, Counter] = {}

    for jid_dir in sorted(judge_root.iterdir()):
        if not jid_dir.is_dir():
            continue
        jid = jid_dir.name
        items: set[tuple[str, str]] = set()
        cell_counts: Counter = Counter()
        for f in sorted(jid_dir.rglob("*.jsonl")):
            cell = "/".join(f.relative_to(jid_dir).parts)
            recs, _ = read_jsonl_safe(f)
            cell_counts[cell] = len(recs)
            src = source_qids.get(cell)
            for r in recs:
                qid = r.get("question_id")
                items.add((cell, qid))
                if src is not None and qid not in src:
                    orphan += 1
                rb = r.get("rubric")
                if not isinstance(rb, int) or not (0 <= rb <= 5):
                    rubric_bad += 1
                for k in (
                    "faithfulness",
                    "citation_precision",
                    "citation_recall",
                    "coverage",
                    "self_confidence",
                ):
                    v = r.get(k)
                    if v is not None and not (0.0 <= float(v) <= 1.0):
                        range_bad += 1
        judge_items[jid] = items
        counts_per_judge[jid] = cell_counts

    for jid, items in judge_items.items():
        record(
            "PASS",
            f"judge {jid}: {len(items)} judged answers across "
            f"{len(counts_per_judge[jid])} cells",
        )

    if "judge_a" in judge_items and "judge_b" in judge_items:
        a, b = judge_items["judge_a"], judge_items["judge_b"]
        if a == b:
            record(
                "PASS", f"judge a/b cover identical {len(a)} answers (alpha overlap OK)"
            )
        else:
            record(
                "FAIL",
                f"judge a/b coverage differs: a-only={len(a - b)}, b-only={len(b - a)}",
            )

    record(
        "FAIL" if rubric_bad else "PASS",
        f"judge: {rubric_bad} rubric values outside int[0,5]",
    )
    record(
        "FAIL" if range_bad else "PASS",
        f"judge: {range_bad} metric values outside [0,1]",
    )
    record(
        "WARN" if orphan else "PASS",
        f"judge: {orphan} judged answers with no matching source record",
    )


def audit_cross_phase(source_qids: dict[str, set[str]]) -> None:
    """Closed-book and RAG must share the same sampled questions (H1 pairing)."""
    cb = set().union(
        *[v for k, v in source_qids.items() if k.startswith("closed_book_main_test/")]
        or [set()]
    )
    rag = set().union(
        *[v for k, v in source_qids.items() if k.startswith("rag_main/")] or [set()]
    )
    if not cb or not rag:
        record(
            "WARN", "cross-phase: closed-book or RAG qids unavailable for overlap check"
        )
        return
    overlap = len(cb & rag)
    frac = overlap / len(cb)
    record(
        "PASS" if frac > 0.95 else "WARN",
        f"cross-phase: closed-book vs RAG question overlap "
        f"{overlap}/{len(cb)} ({frac:.1%}) - H1 pairing",
    )


def main() -> int:
    """Run the full audit and return a process exit code."""
    parser = argparse.ArgumentParser(
        description="Experiment integrity audit (read-only)"
    )
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    root = args.outputs_root

    print("=" * 70)
    print("EXPERIMENT INTEGRITY AUDIT")
    print("=" * 70)

    for name, (n_cells, is_rag) in TRACKS.items():
        print(f"\n--- {name} ---")
        audit_track(root / name, n_cells, is_rag)

    print("\n--- judges ---")
    source_qids = cell_qid_set(
        [root / "closed_book_main_test", root / "rag_main", root / "qasper_main"]
    )
    audit_judges(root / "judge", source_qids)

    print("\n--- cross-phase ---")
    audit_cross_phase(source_qids)

    # --- summary -------------------------------------------------------------
    levels = Counter(level for level, _ in _REPORT)
    print("\n" + "=" * 70)
    print(
        f"SUMMARY: {levels['PASS']} PASS, {levels['WARN']} WARN, {levels['FAIL']} FAIL"
    )
    print("=" * 70)
    if levels["FAIL"]:
        print("\nFAILURES:")
        for level, msg in _REPORT:
            if level == "FAIL":
                print(f"  - {msg}")
    return 1 if levels["FAIL"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
