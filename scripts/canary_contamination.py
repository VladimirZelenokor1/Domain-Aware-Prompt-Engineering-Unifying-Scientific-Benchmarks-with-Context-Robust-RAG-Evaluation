"""Canary-completion contamination probe (thesis section 3.5.1, Layer 2).

Training-data contamination cannot be proven (the models' training corpora are
closed), but it can be probed behaviourally. This truncates each benchmark
question to its opening clause, asks the model to reconstruct the full question
and its reference answer, and measures how often the gold answer is
*regenerated* (exact match or high ROUGE-L overlap). A higher regeneration rate
on one dataset than another is evidence of differential memorisation.

Control design (cross-dataset): SciKnowEval (released 2024) and QASPER (2021)
both fall inside the training windows of all six examinees, so neither is a
clean post-cutoff control. Instead we compare the regeneration rate *between*
the two datasets per model with Fisher's exact test, and report absolute rates.

Caveat (documented for the thesis): on knowledge-QA a capable model can answer
correctly without memorisation, so answer-regeneration is a noisy proxy. The
secondary ``q_verbatim`` signal (does the model reproduce the exact question
wording from a short prefix?) is less capability-confounded, since exact
phrasing cannot be reasoned out.

Usage:
    python scripts/canary_contamination.py --config configs/rag.yaml \\
        --n 200 --seed 42                       # all six models, both datasets
    python scripts/canary_contamination.py --mock --n 5   # pipeline smoke, no GPU
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path
from typing import Any

import yaml
from scipy.stats import fisher_exact

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))
PROJECT_ROOT = _SCRIPTS_DIR.parent

from compute_metrics import compute_exact_match, compute_rouge_l  # noqa: E402

logger = logging.getLogger(__name__)

CLAUSE_DELIMITERS = re.compile(r"[.?!;:,]")
_ANSWER_RE = re.compile(r"ANSWER:\s*(.*)", re.IGNORECASE | re.DOTALL)
_QUESTION_RE = re.compile(
    r"QUESTION:\s*(.*?)(?:\n\s*ANSWER:|$)", re.IGNORECASE | re.DOTALL
)

DEFAULT_TEMPLATE_PATH = PROJECT_ROOT / "prompts" / "contamination" / "canary_completion.txt"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "canary_contamination.json"
SCIKNOWEVAL_PATH = PROJECT_ROOT / "data" / "sciknoweval" / "main_test_sampled.json"
QASPER_PATH = PROJECT_ROOT / "data" / "qasper" / "sample.json"

MC_TYPES = {"mcq-4-choices", "mcq-2-choices", "multiple-choice", "mcq"}
ROUGE_THRESHOLD = 0.75
UNANSWERABLE = {"[unanswerable]", "unanswerable", ""}


# =========================================================================
# Pure helpers
# =========================================================================


def truncate_question(question: str, keep_ratio: float = 0.5) -> str:
    """Truncate a question to its opening clause (the canary prefix).

    Takes the text up to the first clause delimiter (``.?!;:,``). If that clause
    is shorter than three words, falls back to the first ``keep_ratio`` of the
    whitespace tokens (at least three). Whitespace is collapsed first so the
    result is deterministic.

    Args:
        question: Full question text.
        keep_ratio: Fraction of tokens to keep in the fallback prefix.

    Returns:
        The opening-clause prefix, or empty string for empty input.
    """
    q = " ".join((question or "").split())
    if not q:
        return ""
    match = CLAUSE_DELIMITERS.search(q)
    # No delimiter -> no clause: fall back to a ratio prefix so the canary
    # never echoes the whole (answer-bearing) question back.
    clause = q[: match.start()].strip() if match else ""
    if len(clause.split()) >= 3:
        return clause
    tokens = q.split()
    cut = max(3, int(len(tokens) * keep_ratio))
    return " ".join(tokens[:cut])


def build_canary_prompt(template: str, prefix: str) -> str:
    """Fill the canary template's ``{prefix}`` slot with the truncated question."""
    return template.replace("{prefix}", prefix)


def extract_completion_parts(text: str) -> tuple[str, str]:
    """Split a model completion into (reconstructed_question, answer).

    Parses the ``QUESTION: ... ANSWER: ...`` format. If no ``ANSWER:`` label is
    present, the whole text is treated as the answer (question = empty).

    Args:
        text: Raw model completion.

    Returns:
        Tuple of (question_part, answer_part); either may be empty.
    """
    if not text:
        return "", ""
    ans_match = _ANSWER_RE.search(text)
    if not ans_match:
        return "", text.strip()
    answer = ans_match.group(1).strip()
    q_match = _QUESTION_RE.search(text)
    question = q_match.group(1).strip() if q_match else ""
    return question, answer


def is_regenerated(
    output: str, gold: str, qtype: str, rouge_threshold: float = ROUGE_THRESHOLD
) -> bool:
    """Whether a completion regenerates the gold answer.

    Counts as regenerated if (a) the extracted answer exact-matches the gold
    (type-aware normalisation), or (b) for non-MC types, the normalised gold is
    a substring of the answer, or (c) ROUGE-L(answer, gold) meets the threshold
    (verbatim-ish reproduction).

    Args:
        output: Raw model completion (or pre-extracted answer).
        gold: Gold reference answer.
        qtype: Question type (drives normalisation).
        rouge_threshold: Minimum ROUGE-L F1 to count as regenerated.

    Returns:
        True if the gold answer was regenerated.
    """
    if not output or not gold:
        return False
    _, answer = extract_completion_parts(output)
    answer = answer or output
    if compute_exact_match(answer, gold, qtype):
        return True
    # Loose verbatim match (substring or ROUGE-L) only for multi-word, non-MC
    # golds. Short answers (yes/no, single tokens, choice letters) match
    # spuriously inside any longer text, so they are judged by exact match only.
    gold_norm = " ".join(gold.lower().split())
    if qtype not in MC_TYPES and len(gold_norm.split()) >= 3:
        if gold_norm in " ".join(answer.lower().split()):
            return True
        if compute_rouge_l(answer, gold) >= rouge_threshold:
            return True
    return False


def fishers_2x2(
    regen_a: int, n_a: int, regen_b: int, n_b: int
) -> tuple[float, float]:
    """Fisher's exact test on regeneration counts of two datasets.

    Args:
        regen_a: Regenerated count in dataset A.
        n_a: Total scored in dataset A.
        regen_b: Regenerated count in dataset B.
        n_b: Total scored in dataset B.

    Returns:
        Tuple of (odds_ratio, p_value). odds_ratio is inf/nan in degenerate
        tables; p_value is 1.0 when either dataset is empty.
    """
    if n_a == 0 or n_b == 0:
        return float("nan"), 1.0
    table = [[regen_a, n_a - regen_a], [regen_b, n_b - regen_b]]
    odds, p = fisher_exact(table)
    return float(odds), float(p)


# =========================================================================
# Engine-agnostic scoring
# =========================================================================


def generate_completions(
    engine: Any, prompts: list[str], sampling_params: Any
) -> list[str]:
    """Run prompts through a vLLM-like engine and return the first completion text."""
    outputs = engine.generate(prompts, sampling_params)
    return [o.outputs[0].text for o in outputs]


def run_model_canary(
    engine: Any,
    sampling_params: Any,
    records: list[dict],
    template: str,
    rouge_threshold: float = ROUGE_THRESHOLD,
) -> list[dict]:
    """Probe one model over a list of records and score regeneration.

    Args:
        engine: vLLM-like engine exposing ``generate(prompts, sampling_params)``.
        sampling_params: Passed through to the engine (opaque here).
        records: Records with ``question``, ``gold``, ``qtype``, ``dataset``.
        template: Canary prompt template with a ``{prefix}`` slot.
        rouge_threshold: ROUGE-L threshold for regeneration.

    Returns:
        The records, each augmented with ``completion``, ``regenerated`` (bool)
        and ``q_verbatim`` (ROUGE-L of the reconstructed question vs the full
        question - a memorisation signal robust to answer capability).
    """
    prompts = [
        build_canary_prompt(template, truncate_question(r["question"]))
        for r in records
    ]
    completions = generate_completions(engine, prompts, sampling_params)
    scored = []
    for rec, text in zip(records, completions):
        q_recon, _ = extract_completion_parts(text)
        scored.append(
            {
                **rec,
                "completion": text,
                "regenerated": is_regenerated(
                    text, rec["gold"], rec["qtype"], rouge_threshold
                ),
                "q_verbatim": round(compute_rouge_l(q_recon, rec["question"]), 4)
                if q_recon
                else 0.0,
            }
        )
    return scored


# =========================================================================
# Dataset loaders
# =========================================================================


def _sciknoweval_gold(record: dict) -> str:
    """Reference answer for a SciKnowEval record (choice text for MCQ)."""
    qtype = record.get("type", "")
    if qtype in MC_TYPES:
        choices = record.get("choices") or {}
        labels = choices.get("label", [])
        texts = choices.get("text", [])
        key = record.get("answerKey", "")
        if key in labels:
            idx = labels.index(key)
            if idx < len(texts):
                return str(texts[idx]).strip()
        return ""
    return str(record.get("answer") or "").strip()


def load_sciknoweval_sample(path: Path, n: int, seed: int) -> list[dict]:
    """Load and sample SciKnowEval questions normalised for the probe."""
    with path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in sample:
        gold = _sciknoweval_gold(r)
        if gold.lower() in UNANSWERABLE or not r.get("question"):
            continue
        out.append(
            {
                "question": r["question"],
                "gold": gold,
                "qtype": r.get("type", "open-ended-qa"),
                "dataset": "sciknoweval",
            }
        )
    return out


def load_qasper_sample(path: Path, n: int, seed: int) -> list[dict]:
    """Load and sample QASPER questions normalised for the probe."""
    with path.open(encoding="utf-8") as fh:
        rows = json.load(fh)
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))
    out = []
    for r in sample:
        gold = str(r.get("answer_text") or "").strip()
        if gold.lower() in UNANSWERABLE or not r.get("question"):
            continue
        out.append(
            {
                "question": r["question"],
                "gold": gold,
                "qtype": "open-ended-qa",
                "dataset": "qasper",
            }
        )
    return out


# =========================================================================
# Aggregation
# =========================================================================


def _rate(records: list[dict]) -> tuple[int, int, float]:
    regen = sum(1 for r in records if r["regenerated"])
    n = len(records)
    return regen, n, round(regen / n, 4) if n else 0.0


def summarise_model(model: str, scored: list[dict]) -> dict:
    """Build a per-model summary with cross-dataset rates, Fisher, and q_verbatim."""
    qasper = [r for r in scored if r["dataset"] == "qasper"]
    ske = [r for r in scored if r["dataset"] == "sciknoweval"]
    q_regen, q_n, q_rate = _rate(qasper)
    s_regen, s_n, s_rate = _rate(ske)
    odds, p = fishers_2x2(q_regen, q_n, s_regen, s_n)

    def _qv(recs: list[dict]) -> float:
        vals = [r["q_verbatim"] for r in recs]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    return {
        "model": model,
        "qasper": {"regenerated": q_regen, "n": q_n, "rate": q_rate, "q_verbatim": _qv(qasper)},
        "sciknoweval": {"regenerated": s_regen, "n": s_n, "rate": s_rate, "q_verbatim": _qv(ske)},
        "fisher_odds_ratio": odds,
        "fisher_p": round(p, 6),
        "significant_at_0.05": bool(p < 0.05),
    }


# =========================================================================
# vLLM engine (CLI path only)
# =========================================================================


def _create_canary_sampling(max_tokens: int, seed: int) -> Any:
    """Greedy SamplingParams for deterministic completion (CLI path)."""
    from vllm import SamplingParams  # noqa: PLC0415

    return SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed, n=1)


class _MockEngine:
    """No-GPU engine for ``--mock``: echoes the prefix, so nothing regenerates."""

    def generate(self, prompts: list[str], sampling_params: Any) -> list:  # noqa: ARG002
        class _C:
            def __init__(self, t: str) -> None:
                self.text = t

        class _R:
            def __init__(self, t: str) -> None:
                self.outputs = [_C(t)]

        return [_R("QUESTION: \nANSWER: (mock, no regeneration)") for _ in prompts]


# =========================================================================
# CLI
# =========================================================================


def main(argv: list[str] | None = None) -> int:
    """Run the canary-completion contamination probe over all configured models."""
    parser = argparse.ArgumentParser(description="Canary-completion contamination probe")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "rag.yaml")
    parser.add_argument("--sciknoweval", type=Path, default=SCIKNOWEVAL_PATH)
    parser.add_argument("--qasper", type=Path, default=QASPER_PATH)
    parser.add_argument("--n", type=int, default=200, help="Sample size per dataset")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model filter")
    parser.add_argument("--rouge-threshold", type=float, default=ROUGE_THRESHOLD)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mock", action="store_true", help="Run without GPU (smoke test)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    template = args.template.read_text(encoding="utf-8")
    config = yaml.safe_load(args.config.open(encoding="utf-8"))
    models = config["models"]
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        models = {k: v for k, v in models.items() if k in wanted}

    ske_records = load_sciknoweval_sample(args.sciknoweval, args.n, args.seed)
    qasper_records = load_qasper_sample(args.qasper, args.n, args.seed)
    logger.info("Sampled %d SciKnowEval + %d QASPER questions", len(ske_records), len(qasper_records))
    records = qasper_records + ske_records

    summaries = []
    for model_name, model_cfg in models.items():
        logger.info("Probing model: %s", model_name)
        if args.mock:
            engine: Any = _MockEngine()
            sampling: Any = object()
        else:
            from run_inference import create_engine, release_engine  # noqa: PLC0415

            engine = create_engine(model_cfg, seed=args.seed)
            sampling = _create_canary_sampling(args.max_tokens, args.seed)
        try:
            scored = run_model_canary(engine, sampling, records, template, args.rouge_threshold)
        finally:
            if not args.mock:
                release_engine(engine)
        summaries.append(summarise_model(model_name, scored))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {"config": {"n": args.n, "seed": args.seed, "rouge_threshold": args.rouge_threshold}, "per_model": summaries},
        args.out.open("w", encoding="utf-8"),
        indent=2,
    )
    logger.info("Wrote %s", args.out)

    print("\n" + "=" * 78)
    print("CANARY CONTAMINATION PROBE (answer regeneration rate, %)")
    print("=" * 78)
    print(f"{'Model':<22}{'QASPER':>10}{'SciKnow':>10}{'Fisher p':>12}{'sig':>6}")
    print("-" * 78)
    for s in summaries:
        print(
            f"{s['model']:<22}"
            f"{s['qasper']['rate'] * 100:>9.1f}%"
            f"{s['sciknoweval']['rate'] * 100:>9.1f}%"
            f"{s['fisher_p']:>12.4g}"
            f"{('*' if s['significant_at_0.05'] else ''):>6}"
        )
    print("-" * 78)
    print("q_verbatim (reconstructed-question ROUGE-L; less capability-confounded):")
    for s in summaries:
        print(
            f"  {s['model']:<20} QASPER={s['qasper']['q_verbatim']:.3f}  "
            f"SciKnow={s['sciknoweval']['q_verbatim']:.3f}"
        )
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
