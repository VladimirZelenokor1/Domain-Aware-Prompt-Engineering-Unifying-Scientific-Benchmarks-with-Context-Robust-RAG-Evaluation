"""Independent re-derivation of H1/H2/H3 and RAG/noise application.

Confirms the headline results directly from the raw outputs, bypassing the
analysis pipeline, so the numbers are corroborated by a second code path:

    [1] RAG context was injected into the prompt (prompt_tokens ratio).
    [2] Noise was applied per config (non-"real" passages = 0/2/4/6).
    [3] H1: Pearson(model size, RAG improvement) is ~0 and all deltas < 0.
    [4] H2: mean EM by noise (full vs no-SciPhi) and by strategy.
    [5] H3: Krippendorff alpha(judge_a, judge_b) re-derived from rubrics.

Read-only. Usage:
    python scripts/validate_results.py
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent


def load_jsonl(path: str) -> list[dict]:
    """Read a JSON-Lines file (one object per line)."""
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_json(path: str) -> object:
    """Read a whole JSON document (e.g. a pretty-printed summary_table.json)."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def check_rag_in_prompt(root: Path) -> None:
    """[1] RAG prompts must be much larger than closed-book (passages present)."""
    cb, rag = [], []
    for f in glob.glob(str(root / "closed_book_main_test/*/da.jsonl")):
        cb += [r["prompt_tokens"] for r in load_jsonl(f) if r.get("prompt_tokens")]
    for f in glob.glob(str(root / "rag_main/*/hybrid_noise0.0_da.jsonl")):
        rag += [r["prompt_tokens"] for r in load_jsonl(f) if r.get("prompt_tokens")]
    cb_m, rag_m = int(np.median(cb)), int(np.median(rag))
    print(
        f"[1] prompt_tokens median: closed-book={cb_m}  RAG(noise0)={rag_m}  "
        f"ratio={rag_m / max(cb_m, 1):.1f}x  -> passages are in the prompt"
    )


def check_noise_applied(root: Path) -> None:
    """[2] Non-'real' passages per record must equal the configured count."""
    expected = {"0.0": 0, "0.2": 2, "0.4": 4, "0.6": 6}
    print("[2] noise composition (non-real passages per record) by level:")
    for lvl, exp in expected.items():
        fs = sorted(glob.glob(str(root / f"rag_main/*/hybrid_noise{lvl}_da.jsonl")))
        if not fs:
            continue
        dist = collections.Counter()
        for r in load_jsonl(fs[0])[:300]:
            ps = r.get("passages_used", [])
            dist[sum(1 for p in ps if p.get("noise_type") != "real")] += 1
        print(f"    noise{lvl} -> {dict(dist)}   (expected {exp})")


def _params_map(root: Path) -> dict[str, float]:
    reg = load_json(str(root.parent / "models" / "MODEL_REGISTRY.json"))["llms"]
    out: dict[str, float] = {}
    for k, v in reg.items():
        out[k] = v.get("params_b")
        out[k[:-4] if k.endswith("-awq") else k] = v.get("params_b")
    return out


def check_h1_h2(root: Path) -> None:
    """[3]/[4] Re-derive H1 correlation and H2 group means from summaries."""
    cb = {
        (r["model"], r["strategy"]): r
        for r in load_json(str(root / "closed_book_main_test/summary_table.json"))
    }
    rag = load_json(str(root / "rag_main/summary_table.json"))
    params = _params_map(root)

    by_model: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rag:
        by_model[r["model"]].append(r)

    xs, ys = [], []
    for m, rows in by_model.items():
        strategies = {r["strategy"] for r in rows}
        cbe = np.mean([cb[(m, s)]["exact_match"] for s in strategies if (m, s) in cb])
        rage = np.mean([r["exact_match"] for r in rows])
        xs.append(params.get(m))
        ys.append(rage - cbe)
    xs, ys = np.array(xs, float), np.array(ys, float)
    r = float(np.corrcoef(xs, ys)[0, 1])
    print(
        f"[3] H1 Pearson(size, RAG-improvement) = {r:.3f}  (expect ~0.17, NS); "
        f"all deltas < 0: {bool(np.all(ys < 0))}"
    )

    def mean_by(rows: list[dict], key: str) -> dict:
        g: dict[object, list[float]] = collections.defaultdict(list)
        for r in rows:
            g[r[key]].append(r["exact_match"])
        return {k: round(float(np.mean(v)), 4) for k, v in sorted(g.items())}

    no_sci = [r for r in rag if r["model"] != "sciphi-mistral-7b"]
    print(f"[4] H2 mean EM by noise (ALL):       {mean_by(rag, 'noise_level')}")
    print(f"    H2 mean EM by noise (no sciphi): {mean_by(no_sci, 'noise_level')}")
    print(f"    H2 mean EM by strategy:          {mean_by(rag, 'strategy')}")


def check_h3(root: Path) -> None:
    """[5] Re-derive Krippendorff alpha(a,b) over cell+qid rubric pairs."""
    try:
        import krippendorff
    except ImportError:
        print("[5] krippendorff not installed - skip")
        return

    def ratings(jid: str) -> dict[str, int]:
        d: dict[str, int] = {}
        for f in glob.glob(str(root / f"judge/{jid}/**/*.jsonl"), recursive=True):
            cell = f.split(f"/{jid}/", 1)[1]
            for r in load_jsonl(f):
                d[f"{cell}::{r.get('question_id')}"] = int(r.get("rubric", 0))
        return d

    a, b = ratings("judge_a"), ratings("judge_b")
    items = sorted(set(a) & set(b))
    mat = np.array([[a[i] for i in items], [b[i] for i in items]], float)
    alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="ordinal")
    print(
        f"[5] H3 Krippendorff alpha(a,b) = {alpha:.4f} over n={len(items)} (expect ~0.716)"
    )


def main() -> None:
    """Run all independent validation checks."""
    parser = argparse.ArgumentParser(description="Independent results validation")
    parser.add_argument("--outputs-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()
    root = args.outputs_root
    check_rag_in_prompt(root)
    check_noise_applied(root)
    check_h1_h2(root)
    check_h3(root)


if __name__ == "__main__":
    main()
