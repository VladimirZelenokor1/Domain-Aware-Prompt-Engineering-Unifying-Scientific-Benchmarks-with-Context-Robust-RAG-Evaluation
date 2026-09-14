"""Draft 3.6.4 answer-level Class-2 perturbations for the H3 stability test.

The thesis (Sec 3.6.4) operationalises Class-2 (semantic-degradation)
perturbations on the model OUTPUT - the answer itself - re-judged against the
SAME question, isolating the judge from any model variance. This complements the
question-negation + re-inference Class-2 already produced by
``generate_perturbations.py`` (kept as-is); both are reported.

Two degradations are produced from the base answers (``outputs/perturb_base``):

    perturb_class2pad   - the answer padded with neutral filler (verbosity /
                          dilution); a quality-sensitive judge should not reward
                          (ideally penalise) padded, less-concise answers.
    perturb_class2trunc - the answer truncated to its first half (content
                          removed); a sensitive judge should lower the score.

No model re-inference: the base answer is rewritten and only re-judged. The
per-cell file layout (``<model>/<stem>.jsonl``) is mirrored so the judge outputs
align with ``perturb_base`` and pair by (model, stem, question-index) in
``perturbation_wilcoxon.py``.

Usage:
    python scripts/perturb_answers.py \\
        --base-dir outputs/perturb_base --out-root outputs
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPTS_DIR.parent

logger = logging.getLogger(__name__)

# Neutral, off-topic filler appended for the padding perturbation. It adds
# length and dilutes the answer without contributing correctness.
_FILLER = (
    "Note: the preceding statement is provided as supplementary background "
    "and does not by itself change the substance of the answer."
)

_KINDS = {"pad": "perturb_class2pad", "trunc": "perturb_class2trunc"}


def truncate_answer(answer: str) -> str:
    """Return the first half of the answer's words (at least one).

    Args:
        answer: The model answer text.

    Returns:
        The first ``ceil`` -free half of the whitespace-delimited words; empty
        for blank input.
    """
    words = answer.split()
    if not words:
        return ""
    keep = max(1, len(words) // 2)
    return " ".join(words[:keep])


def pad_answer(answer: str) -> str:
    """Return the answer padded with neutral filler to roughly double length.

    Args:
        answer: The model answer text.

    Returns:
        ``answer`` followed by one or more copies of a fixed filler clause, so
        the result is deterministically longer while preserving the original
        prefix.
    """
    if not answer.strip():
        return answer
    padded = answer
    while len(padded) < 2 * len(answer):
        padded = f"{padded} {_FILLER}"
    return padded


def perturb_record(rec: dict, kind: str) -> dict | None:
    """Rewrite a base inference record's answer with the given degradation.

    The judge reads ``parsed.answer`` (falling back to ``raw_response``); both
    are set to the perturbed text. All other fields are preserved and the input
    record is not mutated.

    Args:
        rec: A base inference record.
        kind: ``"pad"`` or ``"trunc"``.

    Returns:
        A new record with the answer fields rewritten, or ``None`` if the base
        answer is empty (nothing meaningful to degrade).
    """
    parsed = rec.get("parsed") or {}
    answer = (parsed.get("answer") or rec.get("raw_response") or "").strip()
    if not answer:
        return None
    new_answer = truncate_answer(answer) if kind == "trunc" else pad_answer(answer)
    out = dict(rec)
    out_parsed = dict(parsed)
    out_parsed["answer"] = new_answer
    out["parsed"] = out_parsed
    out["raw_response"] = new_answer
    return out


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    """Write padded and truncated answer-perturbation splits from the base."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    parser = argparse.ArgumentParser(
        description="Draft 3.6.4 answer-level Class-2 perturbations (pad/truncate)"
    )
    parser.add_argument(
        "--base-dir", type=Path, default=PROJECT_ROOT / "outputs" / "perturb_base"
    )
    parser.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    base_files = sorted(args.base_dir.glob("*/*.jsonl"))
    if not base_files:
        logger.warning("no base inference files under %s", args.base_dir)
        return

    totals = {k: 0 for k in _KINDS}
    skipped = 0
    for f in base_files:
        rel = f.relative_to(args.base_dir)  # <model>/<stem>.jsonl
        records = _read_jsonl(f)
        for kind, split in _KINDS.items():
            out_path = args.out_root / split / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as out_fh:
                for rec in records:
                    pert = perturb_record(rec, kind)
                    if pert is None:
                        if kind == "trunc":
                            skipped += 1
                        continue
                    out_fh.write(json.dumps(pert, ensure_ascii=False) + "\n")
                    totals[kind] += 1

    for kind, split in _KINDS.items():
        logger.info("wrote %d records to %s/", totals[kind], args.out_root / split)
    logger.info("skipped %d empty-answer records per split", skipped)


if __name__ == "__main__":
    main()
