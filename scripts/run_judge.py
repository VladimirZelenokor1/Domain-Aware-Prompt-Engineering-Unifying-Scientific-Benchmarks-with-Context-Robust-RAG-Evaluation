"""LLM-judge scorer for Phase H - scores model outputs on 6 metrics.

Metrics:
    - rubric (0-5): LLM-judged quality score
    - faithfulness (0-1): NLI entailment on decomposed claims
    - citation_precision (0-1): fraction of context passages that are relevant
    - citation_recall (0-1): fraction of gold passages that are relevant
    - coverage (0-1): LLM-judged key-point coverage ratio
    - self_confidence (0-1): judge-reported confidence

Usage:
    python scripts/run_judge.py --judge judge_a --target outputs/closed_book_main
    python scripts/run_judge.py --judge judge_b --target outputs/rag_main --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "judge.yaml"
DEFAULT_PROMPTS_DIR = PROJECT_ROOT / "configs" / "judge_prompts"

# NLI label indices: [contradiction=0, entailment=1, neutral=2]
_NLI_ENTAILMENT_IDX = 1


# =========================================================================
# Softmax utility
# =========================================================================


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax over last axis.

    Args:
        logits: Array of raw logits.

    Returns:
        Softmax probabilities with same shape.
    """
    e = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


# =========================================================================
# Prompt loading
# =========================================================================


def load_judge_prompts(prompts_dir: Path) -> dict[str, str]:
    """Load judge prompt templates from configs/judge_prompts/.

    Args:
        prompts_dir: Directory containing rubric_template.txt,
            claims_template.txt, coverage_template.txt.

    Returns:
        Dict mapping template name to template string.

    Raises:
        FileNotFoundError: If prompts_dir or any template file is missing.
    """
    if not prompts_dir.is_dir():
        raise FileNotFoundError(f"Prompts directory not found: {prompts_dir}")

    mapping = {
        "rubric": "rubric_template.txt",
        "claims": "claims_template.txt",
        "coverage": "coverage_template.txt",
    }
    templates: dict[str, str] = {}
    for key, filename in mapping.items():
        path = prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        templates[key] = path.read_text(encoding="utf-8")

    logger.debug(
        "Loaded %d judge prompt templates from %s", len(templates), prompts_dir
    )
    return templates


# =========================================================================
# JSON extraction helper
# =========================================================================


def _extract_json(raw: str) -> str:
    """Strip markdown code fences and extract JSON content.

    Args:
        raw: Raw LLM output that may contain ```json ... ``` fences.

    Returns:
        Cleaned string with fences removed.
    """
    # Strip markdown code fences
    stripped = re.sub(r"```(?:json)?\s*", "", raw)
    stripped = stripped.strip()
    return stripped


# =========================================================================
# Guided-decoding JSON schemas (force valid JSON from the judge LLM)
# =========================================================================

# Length bounds keep guided JSON short so it always closes within the token
# budget (a truncated-but-valid prefix still fails json.loads) and so the
# CPU-bound guided-decoding step stays fast.
RUBRIC_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "rubric": {"type": "integer", "minimum": 0, "maximum": 5},
        "rationale": {"type": "string", "maxLength": 240},
        "self_confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["rubric", "rationale", "self_confidence"],
}

CLAIMS_JSON_SCHEMA = {
    "type": "array",
    "items": {"type": "string", "maxLength": 200},
    "maxItems": 10,
}

COVERAGE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "key_points_total": {"type": "integer", "minimum": 0},
        "key_points_covered": {"type": "integer", "minimum": 0},
        "missing_points": {
            "type": "array",
            "items": {"type": "string", "maxLength": 120},
            "maxItems": 6,
        },
    },
    "required": ["key_points_total", "key_points_covered"],
}

# Per-task output token caps (bounded by the schema lengths above).
_RUBRIC_MAX_TOKENS = 256
_CLAIMS_MAX_TOKENS = 512
_COVERAGE_MAX_TOKENS = 256


def _build_guided_sampling(
    schema: dict,
    temperature: float,
    seed: int,
    max_tokens: int,
    backend: str = "lm-format-enforcer",
) -> Any:
    """Build a vLLM SamplingParams that forces output to match a JSON schema.

    Handles both the newer ``GuidedDecodingParams`` API and the older
    ``guided_json`` kwarg so the judge runs across vLLM versions.

    Args:
        schema: JSON schema the output must conform to.
        temperature: Sampling temperature (0.0 for deterministic judging).
        seed: RNG seed.
        max_tokens: Max output tokens.
        backend: Guided-decoding backend ('lm-format-enforcer' or 'outlines').
            Defaults to lm-format-enforcer (outlines pulls heavy/broken deps
            in some environments).

    Returns:
        Configured vLLM SamplingParams instance.
    """
    from vllm import SamplingParams

    try:
        from vllm.sampling_params import GuidedDecodingParams

        return SamplingParams(
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            n=1,
            guided_decoding=GuidedDecodingParams(json=schema, backend=backend),
        )
    except ImportError:
        return SamplingParams(
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            n=1,
            guided_json=schema,
            guided_decoding_backend=backend,
        )


def build_judge_sampling_set(scoring_cfg: dict) -> dict[str, Any]:
    """Build guided SamplingParams for each judge task.

    Args:
        scoring_cfg: Scoring section from judge.yaml.

    Returns:
        Dict with 'rubric', 'claims', 'coverage' SamplingParams.
    """
    temperature = scoring_cfg.get("judge_temperature", 0.0)
    seed = scoring_cfg.get("judge_seed", 42)
    backend = scoring_cfg.get("guided_backend", "lm-format-enforcer")
    return {
        "rubric": _build_guided_sampling(
            RUBRIC_JSON_SCHEMA, temperature, seed, _RUBRIC_MAX_TOKENS, backend
        ),
        "claims": _build_guided_sampling(
            CLAIMS_JSON_SCHEMA, temperature, seed, _CLAIMS_MAX_TOKENS, backend
        ),
        "coverage": _build_guided_sampling(
            COVERAGE_JSON_SCHEMA, temperature, seed, _COVERAGE_MAX_TOKENS, backend
        ),
    }


# =========================================================================
# Rubric scoring
# =========================================================================


def build_rubric_prompt(template: str, record: dict) -> str:
    """Build rubric prompt from template and record.

    Args:
        template: Rubric template with {question}, {gold_answer},
            {model_answer}, {passages_block} placeholders.
        record: Inference output record.

    Returns:
        Formatted prompt string.
    """
    # Coalesce None -> "" : dict.get(key, default) returns None when the key
    # exists with a null value (e.g. parsed.answer is null on empty outputs),
    # and str.replace() rejects a None argument.
    parsed = record.get("parsed") or {}
    model_answer = parsed.get("answer") or record.get("raw_response") or ""
    passages = record.get("passages_used", [])

    if passages:
        passages_lines = []
        for i, p in enumerate(passages, 1):
            passages_lines.append(f"[Passage {i}] {p.get('text', '')}")
        passages_block = "RETRIEVED PASSAGES:\n" + "\n".join(passages_lines)
    else:
        passages_block = ""

    # Use str.replace() instead of str.format() because templates
    # contain JSON examples with literal curly braces.
    result = template.replace("{question}", record.get("question") or "")
    result = result.replace("{gold_answer}", record.get("gold_answer") or "")
    result = result.replace("{model_answer}", model_answer)
    result = result.replace("{passages_block}", passages_block)
    return result


# Salvage the scored fields from a truncated/malformed rubric response. The
# guided-JSON schema emits "rubric" first, so a response cut off mid-"rationale"
# (e.g. a long answer that exhausts max_tokens) still carries a valid rubric
# integer; recovering it avoids a spurious rubric=0 fallback.
_RUBRIC_SALVAGE_RE = re.compile(r'"rubric"\s*:\s*(\d+)')
_CONF_SALVAGE_RE = re.compile(r'"self_confidence"\s*:\s*([0-9]*\.?[0-9]+)')


def _salvage_rubric(raw: str) -> dict | None:
    """Recover rubric/self_confidence from a truncated rubric JSON, if present.

    Args:
        raw: Raw LLM output string (possibly truncated before the JSON closes).

    Returns:
        A rubric dict if a ``"rubric": N`` field can be read, else ``None``.
    """
    m = _RUBRIC_SALVAGE_RE.search(raw or "")
    if not m:
        return None
    rubric = max(0, min(5, int(m.group(1))))
    cm = _CONF_SALVAGE_RE.search(raw or "")
    conf = max(0.0, min(1.0, float(cm.group(1)))) if cm else 0.0
    return {
        "rubric": rubric,
        "rationale": f"SALVAGED_TRUNCATED: {(raw or '')[:200]}",
        "self_confidence": conf,
    }


def parse_rubric_response(raw: str) -> dict:
    """Parse LLM rubric response JSON.

    Expected format: {"rubric": 0-5, "rationale": "...", "self_confidence": 0.0-1.0}
    On parse failure, salvage the rubric integer from a truncated response if
    possible; otherwise fall back to rubric=0, self_confidence=0.0.

    Args:
        raw: Raw LLM output string.

    Returns:
        Dict with rubric, rationale, self_confidence keys.
    """
    cleaned = _extract_json(raw)
    try:
        data = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        salvaged = _salvage_rubric(raw)
        if salvaged is not None:
            logger.warning(
                "Rubric JSON incomplete (likely truncated); salvaged rubric=%d: %.80s",
                salvaged["rubric"],
                raw,
            )
            return salvaged
        logger.warning("Failed to parse rubric JSON, using fallback: %.100s", raw)
        return {
            "rubric": 0,
            "rationale": f"PARSE_FAILED: {raw[:200]}",
            "self_confidence": 0.0,
        }

    # Clamp values to valid ranges
    rubric = int(data.get("rubric", 0))
    rubric = max(0, min(5, rubric))

    self_confidence = float(data.get("self_confidence", 0.0))
    self_confidence = max(0.0, min(1.0, self_confidence))

    return {
        "rubric": rubric,
        "rationale": str(data.get("rationale", "")),
        "self_confidence": self_confidence,
    }


# =========================================================================
# Claims decomposition
# =========================================================================


def build_claims_prompt(template: str, answer: str) -> str:
    """Build claims decomposition prompt from template and answer text.

    Args:
        template: Claims template with {answer} placeholder.
        answer: Model answer text to decompose.

    Returns:
        Formatted prompt string.
    """
    return template.replace("{answer}", answer or "")


def parse_claims_response(raw: str) -> list[str]:
    """Parse claims JSON array from LLM output.

    Fallback: treat whole answer as a single claim.

    Args:
        raw: Raw LLM output string.

    Returns:
        List of claim strings.
    """
    cleaned = _extract_json(raw)
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(c) for c in data if c]
    except (json.JSONDecodeError, TypeError):
        pass

    logger.warning(
        "Failed to parse claims JSON, using raw text as single claim: %.100s", raw
    )
    return [raw.strip()] if raw.strip() else [""]


# =========================================================================
# Faithfulness (NLI-based)
# =========================================================================


def compute_faithfulness(
    claims: list[str],
    passages: list[dict],
    nli_model: Any,
    threshold: float = 0.5,
) -> float:
    """Fraction of claims entailed by ANY passage.

    For each claim, run NLI(passage_text, claim) for all passages.
    Claim is entailed if max(entailment_prob) >= threshold.
    faithfulness = entailed_count / total_claims.

    Args:
        claims: List of atomic claim strings.
        passages: List of passage dicts with 'text' key.
        nli_model: CrossEncoder-like model with predict() method.
        threshold: Entailment probability threshold.

    Returns:
        Faithfulness score in [0.0, 1.0]. Returns 0.0 if no claims.
    """
    if not claims:
        return 0.0
    if not passages:
        return 0.0

    entailed_count = 0
    for claim in claims:
        # Build NLI pairs: (passage, claim) for each passage
        pairs = [[p.get("text", ""), claim] for p in passages]
        scores = nli_model.predict(pairs, apply_softmax=True)
        scores = np.asarray(scores)

        # Entailment probability is at index 1
        if scores.ndim == 2:
            entailment_probs = scores[:, _NLI_ENTAILMENT_IDX]
        else:
            # Single pair case
            entailment_probs = np.array([scores[_NLI_ENTAILMENT_IDX]])

        max_entailment = float(np.max(entailment_probs))
        if max_entailment >= threshold:
            entailed_count += 1

    return entailed_count / len(claims)


# =========================================================================
# Citation precision / recall
# =========================================================================


def compute_citation_metrics(
    answer_claims: list[str],
    passages: list[dict],
    nli_model: Any,
    threshold: float = 0.5,
) -> dict:
    """Compute citation precision and recall.

    A passage is 'relevant' if max(NLI(passage, claim).entailment
    for claim in claims) >= threshold.
    precision = relevant_passages / total_passages
    recall = relevant_passages_that_are_gold / gold_passages

    Args:
        answer_claims: List of claim strings from the model answer.
        passages: List of passage dicts with 'text' and 'noise_type' keys.
        nli_model: CrossEncoder-like model with predict() method.
        threshold: Entailment probability threshold.

    Returns:
        Dict with citation_precision and citation_recall floats.
    """
    if not passages:
        return {"citation_precision": 0.0, "citation_recall": 0.0}
    if not answer_claims:
        return {"citation_precision": 0.0, "citation_recall": 0.0}

    relevant_flags: list[bool] = []
    for passage in passages:
        passage_text = passage.get("text", "")
        pairs = [[passage_text, claim] for claim in answer_claims]
        scores = nli_model.predict(pairs, apply_softmax=True)
        scores = np.asarray(scores)

        if scores.ndim == 2:
            entailment_probs = scores[:, _NLI_ENTAILMENT_IDX]
        else:
            entailment_probs = np.array([scores[_NLI_ENTAILMENT_IDX]])

        max_ent = float(np.max(entailment_probs))
        relevant_flags.append(max_ent >= threshold)

    total_passages = len(passages)
    relevant_count = sum(relevant_flags)
    precision = relevant_count / total_passages

    # Recall: among gold passages (noise_type="real"), how many are relevant
    gold_indices = [i for i, p in enumerate(passages) if p.get("noise_type") == "real"]
    if not gold_indices:
        recall = 0.0
    else:
        gold_relevant = sum(1 for i in gold_indices if relevant_flags[i])
        recall = gold_relevant / len(gold_indices)

    return {"citation_precision": precision, "citation_recall": recall}


# =========================================================================
# Coverage (LLM-based)
# =========================================================================


def build_coverage_prompt(
    template: str,
    gold_answer: str,
    model_answer: str,
) -> str:
    """Build coverage prompt from template and answers.

    Args:
        template: Coverage template with {gold_answer}, {model_answer} placeholders.
        gold_answer: Reference gold answer.
        model_answer: Model's generated answer.

    Returns:
        Formatted prompt string.
    """
    result = template.replace("{gold_answer}", gold_answer or "")
    result = result.replace("{model_answer}", model_answer or "")
    return result


def parse_coverage_response(raw: str) -> float:
    """Parse coverage JSON, return key_points_covered / key_points_total.

    Args:
        raw: Raw LLM output string.

    Returns:
        Coverage ratio in [0.0, 1.0]. Returns 0.0 on parse failure or zero total.
    """
    cleaned = _extract_json(raw)
    try:
        data = json.loads(cleaned)
        total = int(data.get("key_points_total", 0))
        covered = int(data.get("key_points_covered", 0))
        if total <= 0:
            return 0.0
        return max(0.0, min(1.0, covered / total))
    except (json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Failed to parse coverage JSON, returning 0.0: %.100s", raw)
        return 0.0


# =========================================================================
# Main scoring function
# =========================================================================


def score_record(
    record: dict,
    judge_llm: Any,
    nli_model: Any | None,
    templates: dict[str, str],
    sampling_params: Any,
) -> dict:
    """Score a single output record on all 6 metrics.

    Args:
        record: Inference output record dict.
        judge_llm: vLLM LLM or mock with generate() method.
        nli_model: CrossEncoder NLI model (None for closed-book).
        templates: Dict of prompt templates (rubric, claims, coverage).
        sampling_params: vLLM SamplingParams or mock.

    Returns:
        Dict with rubric, faithfulness, citation_precision,
        citation_recall, coverage, self_confidence, plus rationale.
    """
    model_answer = record.get("parsed", {}).get(
        "answer",
        record.get("raw_response", ""),
    )
    gold_answer = record.get("gold_answer", "")
    passages = record.get("passages_used")
    is_rag = passages is not None and len(passages) > 0

    # --- Step 1: Rubric scoring ---
    rubric_prompt = build_rubric_prompt(templates["rubric"], record)
    rubric_outputs = judge_llm.generate(
        [rubric_prompt], sampling_params, use_tqdm=False
    )
    rubric_raw = rubric_outputs[0].outputs[0].text
    rubric_result = parse_rubric_response(rubric_raw)

    # --- Step 2: Claims decomposition (needed for faithfulness + citation) ---
    faithfulness_score: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None

    if is_rag and nli_model is not None:
        claims_prompt = build_claims_prompt(templates["claims"], model_answer)
        claims_outputs = judge_llm.generate(
            [claims_prompt],
            sampling_params,
            use_tqdm=False,
        )
        claims_raw = claims_outputs[0].outputs[0].text
        claims = parse_claims_response(claims_raw)

        # --- Step 3: Faithfulness ---
        faithfulness_score = compute_faithfulness(
            claims,
            passages,
            nli_model,
            threshold=0.5,
        )

        # --- Step 4: Citation precision/recall ---
        citation_result = compute_citation_metrics(
            claims,
            passages,
            nli_model,
            threshold=0.5,
        )
        citation_precision = citation_result["citation_precision"]
        citation_recall = citation_result["citation_recall"]

    # --- Step 5: Coverage ---
    coverage_prompt = build_coverage_prompt(
        templates["coverage"],
        gold_answer,
        model_answer,
    )
    coverage_outputs = judge_llm.generate(
        [coverage_prompt],
        sampling_params,
        use_tqdm=False,
    )
    coverage_raw = coverage_outputs[0].outputs[0].text
    coverage_score = parse_coverage_response(coverage_raw)

    return {
        "question_id": record.get("question_id", ""),
        "rubric": rubric_result["rubric"],
        "rationale": rubric_result["rationale"],
        "faithfulness": faithfulness_score,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "coverage": coverage_score,
        "self_confidence": rubric_result["self_confidence"],
    }


def score_records(
    records: list[dict],
    judge_llm: Any,
    nli_model: Any | None,
    templates: dict[str, str],
    sampling_set: dict[str, Any],
) -> list[dict]:
    """Score many records with batched judge calls (one generate per task).

    Instead of 3 sequential generate() calls per record, issue one batched
    generate() per task (rubric, claims, coverage) across all records. vLLM
    schedules the whole batch concurrently, which is far faster than the
    per-record path. NLI runs per record (already internally batched).

    Args:
        records: Inference output records for one cell file.
        judge_llm: vLLM LLM (or mock) with a batched generate() method.
        nli_model: CrossEncoder NLI model, or None for closed-book.
        templates: Prompt templates (rubric, claims, coverage).
        sampling_set: Dict of guided SamplingParams per task.

    Returns:
        List of per-record score dicts, aligned with ``records``.
    """

    def _model_answer(rec: dict) -> str:
        parsed = rec.get("parsed") or {}
        return parsed.get("answer") or rec.get("raw_response") or ""

    # --- Pass 1: rubric (all records) ---
    rubric_prompts = [build_rubric_prompt(templates["rubric"], r) for r in records]
    rubric_raw = judge_llm.generate(
        rubric_prompts, sampling_set["rubric"], use_tqdm=False
    )
    rubric_results = [parse_rubric_response(o.outputs[0].text) for o in rubric_raw]

    # --- Pass 2: claims (RAG records only), then NLI per record ---
    rag_idx = [
        i
        for i, r in enumerate(records)
        if r.get("passages_used") and nli_model is not None
    ]
    claims_by_idx: dict[int, list[str]] = {}
    if rag_idx:
        claims_prompts = [
            build_claims_prompt(templates["claims"], _model_answer(records[i]))
            for i in rag_idx
        ]
        claims_raw = judge_llm.generate(
            claims_prompts, sampling_set["claims"], use_tqdm=False
        )
        for i, out in zip(rag_idx, claims_raw):
            claims_by_idx[i] = parse_claims_response(out.outputs[0].text)

    # --- Pass 3: coverage (all records) ---
    coverage_prompts = [
        build_coverage_prompt(
            templates["coverage"], r.get("gold_answer", ""), _model_answer(r)
        )
        for r in records
    ]
    coverage_raw = judge_llm.generate(
        coverage_prompts, sampling_set["coverage"], use_tqdm=False
    )
    coverage_scores = [parse_coverage_response(o.outputs[0].text) for o in coverage_raw]

    # --- Assemble per-record results ---
    results: list[dict] = []
    for i, rec in enumerate(records):
        faithfulness = None
        citation_precision = None
        citation_recall = None
        if i in claims_by_idx:
            claims = claims_by_idx[i]
            passages = rec["passages_used"]
            faithfulness = compute_faithfulness(
                claims, passages, nli_model, threshold=0.5
            )
            cit = compute_citation_metrics(claims, passages, nli_model, threshold=0.5)
            citation_precision = cit["citation_precision"]
            citation_recall = cit["citation_recall"]

        results.append(
            {
                "question_id": rec.get("question_id", ""),
                "rubric": rubric_results[i]["rubric"],
                "rationale": rubric_results[i]["rationale"],
                "faithfulness": faithfulness,
                "citation_precision": citation_precision,
                "citation_recall": citation_recall,
                "coverage": coverage_scores[i],
                "self_confidence": rubric_results[i]["self_confidence"],
            }
        )
    return results


# =========================================================================
# I/O utilities
# =========================================================================


def iter_output_files(target_dirs: list[Path]) -> list[Path]:
    """Discover all JSONL files in target directories (recursive).

    Args:
        target_dirs: List of directories to scan.

    Returns:
        Sorted list of JSONL file paths.
    """
    files: list[Path] = []
    for d in target_dirs:
        if d.is_dir():
            files.extend(d.rglob("*.jsonl"))
    return sorted(files)


def read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into list of dicts.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed record dicts. Empty list if file is empty.
    """
    records: list[dict] = []
    if not path.exists():
        return records

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt line in %s", path)
    return records


def _strip_surrogates(value: Any) -> Any:
    """Recursively replace lone UTF-8 surrogate code points in strings.

    Judge outputs occasionally contain a stray surrogate (e.g. '\\udc0c')
    which raises UnicodeEncodeError when written with encoding='utf-8'.
    Replace such code points so the JSONL always serialises.
    """
    if isinstance(value, str):
        return value.encode("utf-8", "replace").decode("utf-8")
    if isinstance(value, dict):
        return {k: _strip_surrogates(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_surrogates(v) for v in value]
    return value


def _judge_relative_path(source_path: Path) -> Path:
    """Build the judge-output relative path from a source path.

    Preserves the source-root component to avoid filename collisions across
    tracks (e.g. ``rag_main/<model>/bm25_noise0.0_da.jsonl`` and
    ``qasper_main/<model>/bm25_noise0.0_da.jsonl`` share model/filename but
    must not overwrite each other). Uses the last three components
    (source_root/model/filename) when available.

    Args:
        source_path: Original source JSONL path.

    Returns:
        Relative path ``<source_root>/<model>/<filename>`` (or shorter when
        the source path has fewer components).
    """
    parts = source_path.parts
    if len(parts) >= 3:
        return Path(parts[-3]) / parts[-2] / parts[-1]
    if len(parts) == 2:
        return Path(parts[-2]) / parts[-1]
    return Path(source_path.name)


def save_judge_output(
    judge_id: str,
    source_path: Path,
    scored_records: list[dict],
    output_dir: Path,
) -> Path:
    """Save judge outputs to JSONL.

    Path format: {output_dir}/{judge_id}/{source_root}/{model}/{filename}

    Args:
        judge_id: Judge identifier (judge_a, judge_b, judge_c).
        source_path: Original source JSONL path (used for stem).
        scored_records: List of scored record dicts.
        output_dir: Base output directory.

    Returns:
        Path to the written JSONL file.
    """
    # Preserve source_root/model/strategy structure from source path
    # e.g. source: outputs/qasper_main/qwen2.5-7b/bm25_noise0.0_da.jsonl
    # -> judge output: {output_dir}/judge_a/qasper_main/qwen2.5-7b/bm25_noise0.0_da.jsonl
    relative = _judge_relative_path(source_path)

    out_path = output_dir / judge_id / relative
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        for record in scored_records:
            f.write(json.dumps(_strip_surrogates(record), ensure_ascii=False) + "\n")

    logger.info("Saved %d judge records to %s", len(scored_records), out_path)
    return out_path


# =========================================================================
# Engine creation helpers
# =========================================================================


def load_judge_config(config_path: Path) -> dict:
    """Load judge.yaml configuration.

    Args:
        config_path: Path to judge YAML config.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Judge config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_judge_model_config(config: dict, judge_id: str) -> dict:
    """Extract model config for a specific judge from judge.yaml.

    Args:
        config: Parsed judge.yaml config.
        judge_id: One of judge_a, judge_b, judge_c.

    Returns:
        Judge model config dict.

    Raises:
        ValueError: If judge_id is not found in config.
    """
    # Check main_set_judges
    for judge in config.get("main_set_judges", []):
        if judge.get("id") == judge_id:
            return judge

    # Check calibration_judge
    cal = config.get("calibration_judge", {})
    if cal.get("id") == judge_id:
        return cal

    raise ValueError(
        f"Judge '{judge_id}' not found in config. "
        f"Available: {[j['id'] for j in config.get('main_set_judges', [])]}"
        f" + [{config.get('calibration_judge', {}).get('id', 'N/A')}]"
    )


# Rough USD pricing per 1M tokens (input, output) for the cost cap. Only
# used to estimate spend against the hard cap; not billing-accurate.
_API_PRICING_PER_M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4.1": (2.0, 8.0),
    "claude-sonnet-4": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-3-5-sonnet": (3.0, 15.0),
    "claude-3-5-sonnet-latest": (3.0, 15.0),
}
_API_PRICING_DEFAULT = (3.0, 15.0)


class _APICompletion:
    """Mimics one vllm.outputs.CompletionOutput (only ``.text`` is used)."""

    def __init__(self, text: str) -> None:
        self.text = text


class _APIRequestOutput:
    """Mimics a vllm.outputs.RequestOutput (only ``.outputs[0].text`` used)."""

    def __init__(self, text: str) -> None:
        self.outputs = [_APICompletion(text)]


class APIJudge:
    """Proprietary-API judge with a vLLM-compatible ``generate`` interface.

    Returns objects shaped like vLLM ``RequestOutput`` so it drops into
    ``score_records`` unchanged. Enforces a hard USD cost cap and retries
    transient API errors with exponential backoff.

    Args:
        model: API model id (e.g. 'gpt-4o' or 'claude-sonnet-4').
        cost_cap_usd: Hard spend ceiling; ``generate`` raises once exceeded.
        max_retries: Retry attempts per call on transient errors.
        backoff_base_s: Base seconds for exponential backoff.
        max_tokens: Max output tokens per call.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: str,
        *,
        cost_cap_usd: float = 100.0,
        max_retries: int = 5,
        backoff_base_s: float = 2.0,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> None:
        self.model = model
        self.cost_cap_usd = cost_cap_usd
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.spent_usd = 0.0
        self.provider = "anthropic" if "claude" in model.lower() else "openai"
        self._client: Any = None

    def _price(self, in_tokens: int, out_tokens: int) -> float:
        rate_in, rate_out = _API_PRICING_PER_M.get(self.model, _API_PRICING_DEFAULT)
        return (in_tokens * rate_in + out_tokens * rate_out) / 1_000_000.0

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self.provider == "anthropic":
            import anthropic  # noqa: PLC0415

            self._client = anthropic.Anthropic()
        else:
            from openai import OpenAI  # noqa: PLC0415

            self._client = OpenAI()
        return self._client

    def _complete(self, prompt: str) -> tuple[str, int, int]:
        """One API call. Returns (text, input_tokens, output_tokens).

        Separated so tests can monkeypatch it without real network calls.
        """
        client = self._get_client()
        if self.provider == "anthropic":
            resp = client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            return text, resp.usage.input_tokens, resp.usage.output_tokens
        resp = client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or ""
        return text, resp.usage.prompt_tokens, resp.usage.completion_tokens

    def _complete_with_retry(self, prompt: str) -> tuple[str, int, int]:
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self._complete(prompt)
            except Exception as err:  # noqa: BLE001 - API SDKs raise varied types
                last_err = err
                wait = self.backoff_base_s * (2**attempt)
                logger.warning(
                    "API call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries,
                    err,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"API judge call failed after retries: {last_err}")

    def generate(
        self,
        prompts: list[str],
        sampling_params: Any = None,  # noqa: ARG002 - vLLM-compat, unused
        use_tqdm: bool = False,  # noqa: ARG002 - vLLM-compat, unused
    ) -> list[_APIRequestOutput]:
        """Score each prompt via the API, enforcing the cost cap."""
        results: list[_APIRequestOutput] = []
        for prompt in prompts:
            if self.spent_usd >= self.cost_cap_usd:
                raise RuntimeError(
                    f"judge_c cost cap ${self.cost_cap_usd:.2f} reached "
                    f"(spent ${self.spent_usd:.2f})"
                )
            text, in_tok, out_tok = self._complete_with_retry(prompt)
            self.spent_usd += self._price(in_tok, out_tok)
            results.append(_APIRequestOutput(text))
        return results


def create_judge_engine(judge_cfg: dict, scoring_cfg: dict) -> Any:
    """Create the judge backend: vLLM for open-weight, API for proprietary.

    Args:
        judge_cfg: Judge model config from judge.yaml.
        scoring_cfg: Scoring section from judge.yaml.

    Returns:
        A vLLM ``LLM`` (open-weight judges) or an :class:`APIJudge`
        (proprietary calibration judge). Both expose ``generate``.
    """
    if judge_cfg.get("role") == "proprietary_calibration":
        api_cfg = judge_cfg.get("api", {})
        return APIJudge(
            model=judge_cfg["model"],
            cost_cap_usd=float(judge_cfg.get("cost_cap_usd", 100.0)),
            max_retries=int(api_cfg.get("max_retries", 5)),
        )

    from run_inference import create_engine

    # Build model config dict compatible with create_engine
    model_path = f"models/{judge_cfg['model']}"
    model_cfg = {
        "path": model_path,
        "quantization": "awq",
        "gpu_memory_utilization": 0.85,
        "max_model_len": 4096,
        "enforce_eager": False,
        "dtype": "auto",
    }

    return create_engine(model_cfg, seed=scoring_cfg.get("judge_seed", 42))


def create_judge_sampling_params(scoring_cfg: dict) -> Any:
    """Create SamplingParams for judge LLM.

    Args:
        scoring_cfg: Scoring section from judge.yaml.

    Returns:
        vLLM SamplingParams instance.
    """
    from vllm import SamplingParams

    return SamplingParams(
        temperature=scoring_cfg.get("judge_temperature", 0.0),
        max_tokens=1024,
        seed=scoring_cfg.get("judge_seed", 42),
        n=1,
    )


def load_nli_model(nli_cfg: dict) -> Any:
    """Load NLI CrossEncoder model.

    Args:
        nli_cfg: NLI section from judge.yaml.

    Returns:
        sentence_transformers.CrossEncoder instance.
    """
    from sentence_transformers import CrossEncoder

    model_name = nli_cfg.get("model", "cross-encoder/nli-deberta-v3-large")
    # Check local path first
    local_path = PROJECT_ROOT / "models" / "nli-deberta-v3-large"
    if local_path.is_dir():
        model_path = str(local_path)
    else:
        model_path = model_name

    logger.info("Loading NLI model from %s", model_path)
    return CrossEncoder(model_path)


# =========================================================================
# Main pipeline
# =========================================================================


def run_judge_pipeline(
    judge_id: str,
    target_dirs: list[Path],
    config: dict,
    judge_llm: Any,
    nli_model: Any | None,
    sampling_set: dict[str, Any],
    templates: dict[str, str],
    limit_per_file: int | None = None,
    limit_total: int | None = None,
) -> dict[str, Any]:
    """Run judge scoring on all JSONL files in target directories.

    Uses batched scoring (one generate() per task across all records of a
    file). Resumable: skips files that already have complete judge output.

    Args:
        judge_id: Judge identifier.
        target_dirs: Directories containing inference JSONL outputs.
        config: Parsed judge.yaml config.
        judge_llm: vLLM LLM or mock.
        nli_model: NLI CrossEncoder or None.
        sampling_set: Dict of guided SamplingParams per task (rubric,
            claims, coverage).
        templates: Judge prompt templates.
        limit_per_file: If set, judge only the first N records of each cell
            (stratified subset; same N questions across cells). None = all.
        limit_total: If set, stop after this many newly scored records across
            all files (calibration cap for the API judge). None = no cap.

    Returns:
        Summary dict with files_processed, records_scored, files_skipped.
    """
    output_dir = PROJECT_ROOT / "outputs" / "judge"
    source_files = iter_output_files(target_dirs)
    logger.info("Found %d JSONL files to judge", len(source_files))

    files_processed = 0
    files_skipped = 0
    total_scored = 0

    for source_path in source_files:
        if limit_total is not None and total_scored >= limit_total:
            logger.info("Reached limit_total=%d, stopping.", limit_total)
            break
        records = read_jsonl(source_path)
        if not records:
            logger.warning("Empty file, skipping: %s", source_path)
            continue

        if limit_per_file is not None:
            records = records[:limit_per_file]

        if limit_total is not None:
            remaining = limit_total - total_scored
            if remaining <= 0:
                break
            records = records[:remaining]

        # Check if already judged (resumable)
        relative = _judge_relative_path(source_path)
        judge_out = output_dir / judge_id / relative

        if judge_out.exists():
            existing = read_jsonl(judge_out)
            if len(existing) >= len(records):
                logger.info(
                    "Already judged (%d records), skipping: %s",
                    len(existing),
                    source_path,
                )
                files_skipped += 1
                continue

        logger.info("Judging %d records from %s ...", len(records), source_path.name)
        scored = score_records(
            records=records,
            judge_llm=judge_llm,
            nli_model=nli_model,
            templates=templates,
            sampling_set=sampling_set,
        )
        for result in scored:
            result["judge_id"] = judge_id

        save_judge_output(judge_id, source_path, scored, output_dir)
        files_processed += 1
        total_scored += len(scored)
        logger.info("Done %s (%d scored)", source_path.name, len(scored))

    return {
        "judge_id": judge_id,
        "files_processed": files_processed,
        "files_skipped": files_skipped,
        "total_scored": total_scored,
    }


# =========================================================================
# CLI
# =========================================================================


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="LLM-judge scorer for Phase H",
    )
    parser.add_argument(
        "--judge",
        required=True,
        choices=["judge_a", "judge_b", "judge_c"],
        help="Judge identifier",
    )
    parser.add_argument(
        "--target",
        nargs="+",
        type=Path,
        required=True,
        help="One or more directories containing inference JSONL outputs",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to judge config YAML (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--calibration-subset",
        type=int,
        default=None,
        help="Max items for judge_c calibration subset",
    )
    parser.add_argument(
        "--cost-cap-usd",
        type=float,
        default=None,
        help="Hard cost cap in USD for judge_c",
    )
    parser.add_argument(
        "--limit-per-file",
        type=int,
        default=None,
        help="Judge only the first N records of each cell file "
        "(stratified subset; same N questions across cells). Default: all.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    config = load_judge_config(args.config)
    scoring_cfg = config.get("scoring", {})
    nli_cfg = config.get("nli", {})

    # Load prompt templates
    prompts_dir = PROJECT_ROOT / config.get("prompt_templates", {}).get(
        "dir",
        "configs/judge_prompts",
    )
    templates = load_judge_prompts(prompts_dir)

    # Get judge config
    judge_cfg = _get_judge_model_config(config, args.judge)

    # CLI cost cap overrides the config for the API judge.
    if args.cost_cap_usd is not None:
        judge_cfg["cost_cap_usd"] = args.cost_cap_usd

    # Create judge engine + guided-JSON sampling params (per task)
    judge_llm = create_judge_engine(judge_cfg, scoring_cfg)
    sampling_set = build_judge_sampling_set(scoring_cfg)

    # Load NLI model (only needed if any target has RAG outputs)
    nli_model = load_nli_model(nli_cfg)

    # Resolve target directories
    target_dirs = [d if d.is_absolute() else PROJECT_ROOT / d for d in args.target]

    # Calibration cap: the API judge (judge_c) scores a bounded subset.
    limit_total = None
    if judge_cfg.get("role") == "proprietary_calibration":
        limit_total = args.calibration_subset or judge_cfg.get("max_items", 1000)

    summary = run_judge_pipeline(
        judge_id=args.judge,
        target_dirs=target_dirs,
        config=config,
        judge_llm=judge_llm,
        nli_model=nli_model,
        sampling_set=sampling_set,
        templates=templates,
        limit_per_file=args.limit_per_file,
        limit_total=limit_total,
    )

    if isinstance(judge_llm, APIJudge):
        logger.info("API judge spend: $%.2f", judge_llm.spent_usd)

    logger.info("Judge pipeline complete: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
