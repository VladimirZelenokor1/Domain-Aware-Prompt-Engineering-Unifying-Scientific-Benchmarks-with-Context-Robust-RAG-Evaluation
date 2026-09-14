"""Answer-Context Utility (ACU) and Denoise Rate (DR) for RAG outputs.

Implements the two RAG metrics defined in the thesis (Sec 3.7.2, eq. 9 and 11)
that the inference/judge pipeline did not produce:

    ACU(a, C) = |{relevant passages whose content is reflected in the answer}|
                / |relevant passages|
    DR (a, C) = 1 - |{noise passages reflected in the answer}| / |noise passages|

"Reflected in the answer" is operationalised by NLI entailment of the answer by
the passage (premise = passage text, hypothesis = answer), entailment >= 0.5 -
the same DeBERTa-v3 NLI model used for faithfulness. Relevant passages are those
with noise_type == "real"; noise passages are the rest. DR is undefined at
noise level 0% (no noise passages).

Bounded to the first N records per cell (default 20 = the judged subset) to keep
the NLI cost reasonable. Writes a per-cell summary and prints aggregates.

Usage:
    python scripts/compute_acu_dr.py --limit-per-cell 20
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

# Synthetic noise pools (injection/contradictory carry a ``noise_id`` that is
# NOT in the retrieval corpus; irrelevant distractors carry a real ``chunk_id``).
# Their text must be resolved from these pools, otherwise the noise passages -
# the denominator of the Denoise Rate - are dropped and DR is biased.
_NOISE_POOLS = (
    "irrelevant_distractors.jsonl",
    "injection_passages.jsonl",
    "contradictory_passages.jsonl",
)


def _read_jsonl(path: str, limit: int | None = None) -> list[dict]:
    rows = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        if limit is not None and i >= limit:
            break
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _answer_text(rec: dict) -> str:
    parsed = rec.get("parsed") or {}
    return (
        parsed.get("answer")
        or rec.get("raw_response")
        or parsed.get("answer_normalized")
        or ""
    )


def collect_cells(
    rag_dir: Path, limit: int | None
) -> list[tuple[str, float, str, str, list[dict]]]:
    """Return (model, noise, strategy, file, records) for each RAG cell."""
    cells = []
    for f in sorted(glob.glob(str(rag_dir / "*" / "*.jsonl"))):
        if f.endswith("_metrics.json"):
            continue
        m = RAG_CELL_RE.match(Path(f).stem)
        if not m:
            continue
        retr, noise, strat = m.group(1), float(m.group(2)), m.group(3)
        if retr != "hybrid":
            continue  # noise sweep lives on hybrid; ACU/DR need noise>0 too
        model = Path(f).parent.name
        cells.append((model, noise, strat, f, _read_jsonl(f, limit)))
    return cells


def chunk_text_map(corpus_path: Path, needed: set[str]) -> dict[str, str]:
    """Stream the corpus, keeping text only for the needed chunk_ids."""
    out: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r.get("chunk_id")
            if cid in needed:
                out[cid] = r.get("text", "")
                if len(out) == len(needed):
                    break
    return out


def noise_text_map(noise_dir: Path, needed: set[str]) -> dict[str, str]:
    """Map noise-passage ids to their text from the synthetic noise pools.

    Injection and contradictory passages are referenced in ``passages_used`` by
    their ``noise_id`` (e.g. ``inj_*``/``con_*``), which is absent from the
    retrieval corpus; this recovers their text so they are not dropped from the
    Denoise Rate. Irrelevant distractors carry a real ``chunk_id`` and resolve
    from the corpus, but are included here too for completeness.

    Args:
        noise_dir: Directory holding the noise pool JSONL files.
        needed: Passage ids still missing after the corpus lookup.

    Returns:
        Mapping of ``id -> text`` for ids found in the pools and in ``needed``.
    """
    out: dict[str, str] = {}
    if not needed:
        return out
    for fname in _NOISE_POOLS:
        path = noise_dir / fname
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                nid = r.get("noise_id") or r.get("chunk_id")
                if nid in needed and nid not in out:
                    out[nid] = r.get("text", "")
    return out


def main() -> None:
    """Compute ACU and DR over the (bounded) RAG subset using NLI entailment."""
    from run_judge import load_nli_model  # noqa: PLC0415

    parser = argparse.ArgumentParser(description="ACU + Denoise Rate (NLI-based)")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--corpus", type=Path, default=PROJECT_ROOT / "corpus" / "all_chunks.jsonl"
    )
    parser.add_argument(
        # Matches build_noise.NOISE_DIR (= corpus/noise), where the pools live.
        "--noise-dir",
        type=Path,
        default=PROJECT_ROOT / "corpus" / "noise",
    )
    parser.add_argument("--limit-per-cell", type=int, default=20)
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "outputs" / "acu_dr_summary.json"
    )
    args = parser.parse_args()

    cells = collect_cells(args.outputs_root / "rag_main", args.limit_per_cell)
    if not cells:
        print("no hybrid RAG cells found")
        return
    needed = {
        p.get("chunk_id")
        for _, _, _, _, recs in cells
        for r in recs
        for p in (r.get("passages_used") or [])
    }
    print(f"cells={len(cells)}  unique chunks needed={len(needed)}")
    ctext = chunk_text_map(args.corpus, needed)
    n_corpus = len(ctext)
    # Resolve synthetic-noise ids (inj_*/con_*) the corpus does not contain.
    missing = needed - set(ctext)
    ctext.update(noise_text_map(args.noise_dir, missing))
    n_noise = len(ctext) - n_corpus
    print(
        f"texts resolved: {len(ctext)}/{len(needed)} "
        f"(corpus {n_corpus}, noise pools {n_noise}, "
        f"unresolved {len(needed) - len(ctext)})"
    )

    nli_cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "judge.yaml")).get(
        "nli", {}
    )
    label_order = nli_cfg.get("label_order", ["contradiction", "entailment", "neutral"])
    ent_idx = label_order.index("entailment")
    nli = load_nli_model(nli_cfg)

    per_cell = []
    for model, noise, strat, _f, recs in cells:
        acu_vals, dr_vals = [], []
        for r in recs:
            ans = _answer_text(r)
            passages = r.get("passages_used") or []
            if not ans or not passages:
                continue
            pairs, types = [], []
            for p in passages:
                txt = ctext.get(p.get("chunk_id"), "")
                if txt:
                    pairs.append((txt, ans))
                    types.append(p.get("noise_type"))
            if not pairs:
                continue
            scores = nli.predict(pairs, apply_softmax=True)
            entail = [float(s[ent_idx]) >= ENTAIL_THRESHOLD for s in scores]
            real = [e for e, t in zip(entail, types) if t == "real"]
            noisep = [e for e, t in zip(entail, types) if t != "real"]
            if real:
                acu_vals.append(sum(real) / len(real))
            if noisep:  # DR undefined when no noise passages (noise 0)
                dr_vals.append(1.0 - sum(noisep) / len(noisep))
        per_cell.append(
            {
                "model": model,
                "noise_level": noise,
                "strategy": strat,
                "acu": float(np.mean(acu_vals)) if acu_vals else None,
                "dr": float(np.mean(dr_vals)) if dr_vals else None,
                "n": len(recs),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(per_cell, args.out.open("w"), indent=2)
    print(f"\nwrote {args.out}")

    def agg(key: str, metric: str) -> dict:
        g = collections.defaultdict(list)
        for c in per_cell:
            if c[metric] is not None:
                g[c[key]].append(c[metric])
        return {k: round(float(np.mean(v)), 3) for k, v in sorted(g.items())}

    print("\nACU by noise:", agg("noise_level", "acu"))
    print("ACU by strategy:", agg("strategy", "acu"))
    print("DR  by noise:", agg("noise_level", "dr"))
    print("DR  by strategy:", agg("strategy", "dr"))
    print("DR  by model:", agg("model", "dr"))


if __name__ == "__main__":
    main()
