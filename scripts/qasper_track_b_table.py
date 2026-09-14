"""Aggregate QASPER (Track B) judge outputs into a rubric table.

Reads the 48 QASPER judge cells from both open-weight judges and reports
mean rubric / faithfulness / coverage broken down by prompting strategy and
by noise level (and overall). This is the quantitative Track-B appendix table
(Exact Match is not meaningful for QASPER's open-ended answers, so quality is
read from the judge rubric).

Usage:
    python scripts/qasper_track_b_table.py
    python scripts/qasper_track_b_table.py --judge-root outputs/judge \\
        --out outputs/chapter5_tables/track_b_qasper.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent

CELL_RE = re.compile(r"^hybrid_noise([0-9.]+)_(da|ras|ctl|sc)$")
JUDGES = ("judge_a", "judge_b")
METRICS = ("rubric", "faithfulness", "coverage")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def collect(judge_root: Path) -> list[dict]:
    """Collect per-record metric rows tagged with strategy and noise level."""
    rows: list[dict] = []
    for jid in JUDGES:
        base = judge_root / jid / "qasper_main"
        if not base.exists():
            continue
        for f in sorted(base.rglob("*.jsonl")):
            m = CELL_RE.match(f.stem)
            if not m:
                continue
            noise, strategy = float(m.group(1)), m.group(2)
            for rec in _read_jsonl(f):
                rows.append(
                    {
                        "judge": jid,
                        "strategy": strategy,
                        "noise_level": noise,
                        "rubric": float(rec.get("rubric", 0)),
                        "faithfulness": float(rec.get("faithfulness", 0.0)),
                        "coverage": float(rec.get("coverage", 0.0)),
                    }
                )
    return rows


def aggregate(rows: list[dict], key: str | None) -> list[dict]:
    """Group rows by `key` (or all together if None) and average the metrics."""
    groups: dict[object, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key] if key else "ALL"].append(r)
    out: list[dict] = []
    for g, recs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = {"group": f"{key or 'overall'}={g}", "n": len(recs)}
        for metric in METRICS:
            row[f"mean_{metric}"] = _mean([r[metric] for r in recs])
        out.append(row)
    return out


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="QASPER Track-B judge rubric table")
    parser.add_argument(
        "--judge-root", type=Path, default=PROJECT_ROOT / "outputs" / "judge"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "chapter5_tables" / "track_b_qasper.csv",
    )
    args = parser.parse_args()

    rows = collect(args.judge_root)
    if not rows:
        print("No QASPER judge records found under", args.judge_root)
        return

    table = (
        aggregate(rows, None)
        + aggregate(rows, "strategy")
        + aggregate(rows, "noise_level")
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "group",
                "n",
                "mean_rubric",
                "mean_faithfulness",
                "mean_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(table)

    print(f"QASPER Track-B rubric table ({len(rows)} judged records) -> {args.out}\n")
    print(f"{'group':24} {'n':>5} {'rubric':>8} {'faith':>8} {'coverage':>9}")
    for r in table:
        print(
            f"{r['group']:24} {r['n']:>5} {r['mean_rubric']:>8.3f} "
            f"{r['mean_faithfulness']:>8.3f} {r['mean_coverage']:>9.3f}"
        )


if __name__ == "__main__":
    main()
