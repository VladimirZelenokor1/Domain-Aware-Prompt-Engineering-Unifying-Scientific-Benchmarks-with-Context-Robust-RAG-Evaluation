"""QASPER (Track B) answer grounding against gold evidence spans (thesis B-4).

The thesis claims QASPER ``highlighted_evidence`` spans provide a ground-truth
check on grounding without human annotation. The spans are collected into the
sample (``split_qasper.py``) but were not consumed downstream. This computes,
per Track B answer, an Evidence-Grounding (EG) rate:

    EG(answer, E) = 1 if max_{e in E} NLI_entail(e, answer) >= 0.5 else 0

over the gold evidence spans E for the answer's question (premise = evidence
span, hypothesis = model answer - the same DeBERTa-v3 NLI direction used for
faithfulness). Records with no usable evidence (e.g. unanswerable questions) or
an empty answer are skipped. Judge-free; needs only the NLI model.

Usage:
    python scripts/qasper_evidence_grounding.py --limit-per-cell 20
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import yaml

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

RAG_CELL_RE = re.compile(r"^(bm25|dense|hybrid)_noise([0-9.]+)_(da|ras|ctl|sc)$")
ENTAIL_THRESHOLD = 0.5


def is_grounded(
    entail_scores: list[float], threshold: float = ENTAIL_THRESHOLD
) -> bool:
    """Whether the answer is entailed by at least one evidence span.

    Args:
        entail_scores: NLI entailment probabilities, one per gold span.
        threshold: Minimum entailment to count a span as supporting.

    Returns:
        True if any span's entailment meets the threshold.
    """
    return any(s >= threshold for s in entail_scores)


def _norm_question(text: str) -> str:
    """Normalise a question for joining (collapse whitespace, lowercase)."""
    return " ".join((text or "").split()).lower()


def load_evidence(sample_path: Path) -> dict[str, list[str]]:
    """Map a normalised QASPER question to its non-empty gold evidence spans.

    Track B inference reassigns ``question_id`` positionally (``ske-track-b-*``),
    so the sample's hash ``question_id`` does not join to the inference outputs.
    Keying on the question *text* (preserved verbatim through inference) joins
    robustly without assuming id space or record order.

    Args:
        sample_path: Path to the QASPER sample JSON (with question, evidence).

    Returns:
        Mapping normalised-question -> list of non-blank evidence strings;
        questions with no usable evidence are omitted.
    """
    with sample_path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    out: dict[str, list[str]] = {}
    for r in rows:
        spans = [
            s for s in (r.get("evidence") or []) if isinstance(s, str) and s.strip()
        ]
        if spans:
            out[_norm_question(r.get("question", ""))] = spans
    return out


def load_evidence_from_raw(raw_dir: Path) -> dict[str, list[str]]:
    """Build a normalised-question -> gold evidence map from raw QASPER files.

    Reads every ``qasper-*-v0.3.json`` in ``raw_dir`` and takes the first
    annotator's ``evidence`` per question (matching ``split_qasper.py``). Covers
    all QASPER questions, so the evaluated Track B questions join by text
    regardless of which sample they were drawn from.

    Args:
        raw_dir: Directory with the raw QASPER split JSON files.

    Returns:
        Mapping normalised-question -> list of non-blank evidence strings;
        questions with no usable evidence are omitted.
    """
    out: dict[str, list[str]] = {}
    for path in sorted(raw_dir.glob("qasper-*.json")):
        with path.open(encoding="utf-8") as fh:
            papers = json.load(fh)
        for paper in papers.values():
            for qa in paper.get("qas", []):
                answers = qa.get("answers", [])
                if not answers:
                    continue
                ev = answers[0].get("answer", {}).get("evidence", []) or []
                spans = [s for s in ev if isinstance(s, str) and s.strip()]
                if spans:
                    out[_norm_question(qa.get("question", ""))] = spans
    return out


def _read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _answer_text(rec: dict) -> str:
    parsed = rec.get("parsed") or {}
    return parsed.get("answer") or rec.get("raw_response") or ""


def collect_cells(
    qasper_dir: Path, limit: int | None
) -> list[tuple[str, float, str, str, list[dict]]]:
    """Return (model, noise, strategy, file, records) for each Track B cell."""
    cells = []
    for f in sorted(glob.glob(str(qasper_dir / "*" / "*.jsonl"))):
        if f.endswith("_metrics.json"):
            continue
        m = RAG_CELL_RE.match(Path(f).stem)
        if not m:
            continue
        _retr, noise, strat = m.group(1), float(m.group(2)), m.group(3)
        model = Path(f).parent.name
        cells.append((model, noise, strat, f, _read_jsonl(f, limit)))
    return cells


def main() -> None:
    """Compute the Evidence-Grounding rate over the Track B judged subset."""
    from run_judge import load_nli_model  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        description="QASPER Track-B evidence grounding (EG)"
    )
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--sample", type=Path, default=PROJECT_ROOT / "data" / "qasper" / "sample.json"
    )
    parser.add_argument(
        # Preferred evidence source: raw QASPER covers every question, so the
        # evaluated Track B questions join by text (the 500-q sample.json does
        # not). Falls back to --sample if the raw dir is absent.
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "qasper" / "raw",
    )
    parser.add_argument("--limit-per-cell", type=int, default=20)
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "qasper_evidence_grounding.json",
    )
    args = parser.parse_args()

    if args.raw_dir.exists() and any(args.raw_dir.glob("qasper-*.json")):
        evidence = load_evidence_from_raw(args.raw_dir)
        src = f"raw QASPER ({args.raw_dir})"
    else:
        evidence = load_evidence(args.sample)
        src = f"sample ({args.sample})"
    print(f"questions with gold evidence: {len(evidence)}  [source: {src}]")

    cells = collect_cells(args.outputs_root / "qasper_main", args.limit_per_cell)
    if not cells:
        print("no Track B cells found under outputs/qasper_main")
        return

    nli_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "judge.yaml")).get(
        "nli", {}
    )
    label_order = nli_cfg.get("label_order", ["contradiction", "entailment", "neutral"])
    ent_idx = label_order.index("entailment")
    nli = load_nli_model(nli_cfg)

    per_cell = []
    skipped_no_evidence = skipped_no_answer = 0
    for model, noise, strat, _f, recs in cells:
        flags: list[float] = []
        for r in recs:
            ans = _answer_text(r).strip()
            spans = evidence.get(_norm_question(r.get("question", "")))
            if not ans:
                skipped_no_answer += 1
                continue
            if not spans:
                skipped_no_evidence += 1
                continue
            scores = nli.predict([(s, ans) for s in spans], apply_softmax=True)
            flags.append(
                1.0 if is_grounded([float(s[ent_idx]) for s in scores]) else 0.0
            )
        per_cell.append(
            {
                "model": model,
                "noise_level": noise,
                "strategy": strat,
                "eg": float(np.mean(flags)) if flags else None,
                "n": len(flags),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(per_cell, args.out.open("w"), indent=2)
    print(
        f"\nwrote {args.out}  "
        f"(skipped: no-evidence={skipped_no_evidence}, no-answer={skipped_no_answer})"
    )

    def agg(key: str) -> dict:
        g = collections.defaultdict(list)
        for c in per_cell:
            if c["eg"] is not None:
                g[c[key]].append(c["eg"])
        return {k: round(float(np.mean(v)), 3) for k, v in sorted(g.items())}

    scored = [c["eg"] for c in per_cell if c["eg"] is not None]
    if scored:
        print(f"\nEG overall: {round(float(np.mean(scored)), 3)}")
    print("EG by noise:", agg("noise_level"))
    print("EG by strategy:", agg("strategy"))
    print("EG by model:", agg("model"))


if __name__ == "__main__":
    main()
