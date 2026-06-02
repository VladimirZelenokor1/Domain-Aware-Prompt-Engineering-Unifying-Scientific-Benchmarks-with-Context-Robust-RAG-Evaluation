"""Generate the perturbation audit set for the H3 robustness test.

Produces three aligned splits from the main_test sample (same N questions,
same order, so they pair by index):

    perturb_base.json    - the unmodified subset (control)
    perturb_class1.json  - class 1 surface perturbations (typos + whitespace);
                           a robust judge/model should NOT change its score
    perturb_class2.json  - class 2 semantic perturbations (rule-based negation
                           of the question stem); a sensitive judge/model SHOULD
                           change its score

Only the `question` text is perturbed; answer/choices/type/domain are
preserved. Class-2 negation is a transparent rule-based approximation (a true
entity-swap protocol would use an LLM); this is stated as a limitation.

The seed-derived, index-based perturbation choice keeps generation fully
reproducible (no RNG state).

Usage:
    python scripts/generate_perturbations.py --n 200 \\
        --split data/sciknoweval/main_test_sampled.json --out-dir data/sciknoweval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent
SEED = 42

# auxiliary/verb negation map for the class-2 (semantic) perturbation
NEGATE = {
    " is ": " is not ",
    " are ": " are not ",
    " was ": " was not ",
    " were ": " were not ",
    " can ": " cannot ",
    " does ": " does not ",
    " do ": " do not ",
    " will ": " will not ",
    " has ": " has not ",
    " have ": " have not ",
}


def _typo(word: str, salt: int) -> str:
    """Deterministically introduce one mild typo into a word (>=4 chars)."""
    if len(word) < 4:
        return word
    i = 1 + (salt % (len(word) - 2))  # interior position, stable per (word, salt)
    if salt % 2 == 0:  # swap adjacent characters
        return word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    return word[:i] + word[i + 1 :]  # drop one character


def perturb_surface(question: str, idx: int) -> str:
    """Class 1: typos in a few words + doubled whitespace (meaning preserved)."""
    words = question.split(" ")
    long_idx = [j for j, w in enumerate(words) if len(w) >= 5]
    # perturb up to 3 long words, chosen deterministically from the index
    for k, j in enumerate(long_idx[:: max(len(long_idx) // 3, 1)][:3]):
        words[j] = _typo(words[j], SEED + idx + k)
    out = " ".join(words)
    return out.replace(". ", ".  ", 1)  # one doubled space


def perturb_semantic(question: str) -> str:
    """Class 2: rule-based negation of the first matched auxiliary verb.

    Transparent approximation of a semantic perturbation (a full protocol
    would swap named entities via an LLM). Returns the original string
    unchanged if no auxiliary is found (caller can filter those out).
    """
    for src, dst in NEGATE.items():
        if src in question:
            return question.replace(src, dst, 1)
    return question


def main() -> None:
    """Generate the three aligned perturbation splits."""
    parser = argparse.ArgumentParser(description="Generate H3 perturbation audit set")
    parser.add_argument(
        "--split",
        type=Path,
        default=PROJECT_ROOT / "data" / "sciknoweval" / "main_test_sampled.json",
    )
    parser.add_argument("--n", type=int, default=200, help="audit subset size")
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "sciknoweval"
    )
    args = parser.parse_args()

    with args.split.open(encoding="utf-8") as fh:
        data = json.load(fh)
    subset = data[: args.n]

    base, c1, c2 = [], [], []
    skipped_c2 = 0
    for idx, rec in enumerate(subset):
        q = rec.get("question", "")
        base.append(dict(rec))
        r1 = dict(rec)
        r1["question"] = perturb_surface(q, idx)
        c1.append(r1)
        r2 = dict(rec)
        neg = perturb_semantic(q)
        if neg == q:
            skipped_c2 += 1
        r2["question"] = neg
        c2.append(r2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("perturb_base", base),
        ("perturb_class1", c1),
        ("perturb_class2", c2),
    ):
        out = args.out_dir / f"{name}.json"
        with out.open("w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
        print(f"wrote {out} ({len(rows)} questions)")
    print(
        f"class2 negation applied to {len(c2) - skipped_c2}/{len(c2)} "
        f"(no auxiliary verb found in {skipped_c2})"
    )


if __name__ == "__main__":
    main()
