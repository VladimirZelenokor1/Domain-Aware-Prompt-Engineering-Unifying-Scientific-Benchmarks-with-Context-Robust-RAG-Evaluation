"""Two residual validation checks (read-only).

[3] Judge<->source orphan diagnosis: explains the audit WARN about judged
    answers with "no matching source record" - which track/cells, and a sample
    mismatch, so the warning is understood rather than hand-waved.

[4] Retrieval relevance proxy at noise 0: a strict recall@10 is not computable
    on SciKnowEval (no gold passage labels; MCQ gold is a letter), so we use a
    domain-consistency proxy - the fraction of retrieved (non-noise) passages
    whose source chunk domain matches the question domain. Surfaces whether the
    retriever returns topically relevant context (e.g. Computer-Science
    questions have no matching corpus domain).

Usage:
    python scripts/extra_checks.py
    python scripts/extra_checks.py --corpus corpus/all_chunks.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent

# question 'domain' label -> corpus domain label (configs/corpus.yaml)
DOMAIN_ALIAS = {
    "physics": "physics",
    "chemistry": "chemistry",
    "biology": "biology",
    "material": "materials_science",
    "materials": "materials_science",
    "materials_science": "materials_science",
    "cs": "computer_science",
    "computer science": "computer_science",
}


def load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# [3] orphan diagnosis
# ---------------------------------------------------------------------------
def diagnose_orphans(root: Path) -> None:
    """Replicate the audit orphan check with per-track breakdown + a sample."""
    print("=== [3] judge<->source orphan diagnosis ===")
    # Build source question-id sets per cell key "<root>/<model>/<file>".
    src: dict[str, set] = {}
    for d in (root / "closed_book_main_test", root / "rag_main", root / "qasper_main"):
        if not d.exists():
            continue
        for f in d.rglob("*.jsonl"):
            key = "/".join(f.relative_to(d.parent).parts)
            src[key] = {r.get("question_id") for r in load_jsonl(str(f))}

    by_track: collections.Counter = collections.Counter()
    sample = None
    for jid in ("judge_a", "judge_b"):
        jroot = root / "judge" / jid
        if not jroot.exists():
            continue
        for f in jroot.rglob("*.jsonl"):
            cell = "/".join(f.relative_to(jroot).parts)
            track = cell.split("/", 1)[0]
            s = src.get(cell)
            for r in load_jsonl(str(f)):
                qid = r.get("question_id")
                if s is None:
                    by_track[f"{track} (cell not found in source)"] += 1
                    if sample is None:
                        sample = ("CELL-NOT-FOUND", cell, qid, None)
                elif qid not in s:
                    by_track[f"{track} (qid not in source cell)"] += 1
                    if sample is None:
                        sample = ("QID-MISSING", cell, qid, sorted(s)[:3])
    print("orphans by track/reason:", dict(by_track) or "none")
    if sample:
        kind, cell, qid, sample_src = sample
        print(f"sample: reason={kind}  cell={cell}")
        print(f"        judge qid={qid!r}  source sample={sample_src}")
    else:
        print("no orphans - every judged answer matches a source record")


# ---------------------------------------------------------------------------
# [4] retrieval relevance proxy
# ---------------------------------------------------------------------------
def retrieval_domain_proxy(root: Path, corpus_path: Path) -> None:
    """Fraction of non-noise retrieved passages whose domain matches the question."""
    print("\n=== [4] retrieval domain-consistency at noise 0 (proxy for relevance) ===")
    if not corpus_path.exists():
        print(f"corpus not found at {corpus_path}; cannot map chunk -> domain. Skip.")
        return

    # chunk_id -> domain (stream; only keep the domain field)
    chunk_domain: dict[str, str] = {}
    has_domain = False
    for r in load_jsonl(str(corpus_path)):
        cid = r.get("chunk_id")
        dom = r.get("domain")
        if dom is not None:
            has_domain = True
        if cid is not None:
            chunk_domain[cid] = (dom or "").lower()
    if not has_domain:
        print("corpus chunks have no 'domain' field; cannot compute. Skip.")
        return

    # per question-domain: matched / total non-noise passages at noise 0
    matched: collections.Counter = collections.Counter()
    total: collections.Counter = collections.Counter()
    for f in glob.glob(str(root / "rag_main/*/hybrid_noise0.0_da.jsonl")):
        for r in load_jsonl(f):
            qdom = DOMAIN_ALIAS.get(str(r.get("domain", "")).strip().lower())
            for p in r.get("passages_used", []):
                if p.get("noise_type") != "real":
                    continue  # noise passage, skip
                total[qdom] += 1
                if qdom is not None and chunk_domain.get(p.get("chunk_id")) == qdom:
                    matched[qdom] += 1

    print("question-domain   match-rate (retrieved passage domain == question domain)")
    for qdom in sorted(total, key=lambda x: str(x)):
        t = total[qdom]
        rate = matched[qdom] / t if t else 0.0
        label = qdom if qdom is not None else "(unmapped, e.g. CS)"
        print(f"  {str(label):20} {rate:6.1%}  (n={t})")
    print(
        "  note: low match-rate => retriever returns off-domain context "
        "(expected for domains absent from the corpus, e.g. Computer Science)"
    )


def main() -> None:
    """Run the residual checks."""
    parser = argparse.ArgumentParser(
        description="Residual validation checks (read-only)"
    )
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "corpus" / "all_chunks.jsonl"
    )
    args = parser.parse_args()
    diagnose_orphans(args.outputs_root)
    retrieval_domain_proxy(args.outputs_root, args.corpus)


if __name__ == "__main__":
    main()
