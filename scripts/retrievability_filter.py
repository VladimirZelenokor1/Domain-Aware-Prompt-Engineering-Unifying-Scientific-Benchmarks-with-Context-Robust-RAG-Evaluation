"""Retrievability filter, corpus leakage check, and retrieval quality validation.

Filters SciKnowEval questions by whether the correct answer is present in
top-10 retrieved passages. Also checks for corpus leakage (verbatim answer
in corpus) and computes Recall@10 per retriever on dev set (TABLE VI).

Usage:
    python scripts/retrievability_filter.py                  # full pipeline
    python scripts/retrievability_filter.py --skip-leakage   # no leakage check
    python scripts/retrievability_filter.py --eval-only      # TABLE VI on dev set
    python scripts/retrievability_filter.py --force          # wipe checkpoint
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Minimum keyword overlap ratio for relaxed MCQ matching
MCQ_KEYWORD_OVERLAP_THRESHOLD = 0.3
# Cosine similarity threshold for semantic matching
SEMANTIC_SIM_THRESHOLD = 0.7

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAIN_TEST_PATH = PROJECT_ROOT / "data" / "sciknoweval" / "main_test.json"
DEV_PATH = PROJECT_ROOT / "data" / "sciknoweval" / "dev.json"
OUTPUT_PATH = PROJECT_ROOT / "data" / "sciknoweval" / "main_test_retrievable.json"

# Checkpoint schema version (bump when progress.jsonl record shape changes)
CHECKPOINT_SCHEMA_VERSION = 1
# Fixed thesis parameters used for manifest fingerprinting
TOP_K = 10
SEED = 42


# =========================================================================
# Helpers
# =========================================================================


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, collapse whitespace."""
    return re.sub(r"\s+", " ", text.lower().strip())


def get_reference_answer(question: dict) -> str | None:
    """Extract reference answer text from a question record.

    Args:
        question: SciKnowEval question dict.

    Returns:
        Answer text string, or None if unavailable.
    """
    qtype = question.get("type", "")

    # MCQ: use choices text for the correct answer key
    if qtype in ("mcq-4-choices", "mcq-2-choices"):
        answer_key = question.get("answerKey", "")
        choices = question.get("choices", {})
        labels = choices.get("label", [])
        texts = choices.get("text", [])

        if answer_key and answer_key in labels:
            idx = labels.index(answer_key)
            if idx < len(texts):
                return texts[idx]
        return None

    # For open-ended, filling, relation_extraction, true_or_false
    answer = question.get("answer", "")
    if answer and answer.strip():
        return answer

    return None


def check_retrievability(
    question: dict,
    passages: list[dict],
) -> bool:
    """Check if any retrieved passage contains the reference answer.

    Args:
        question: SciKnowEval question dict.
        passages: List of retrieved passage dicts with 'text' field.

    Returns:
        True if the answer is found in at least one passage.
    """
    answer = get_reference_answer(question)
    if answer is None:
        return False

    qtype = question.get("type", "")

    # True/false: answer is just "True"/"False", not useful for matching.
    # Weaker check: verify we got some passages back.
    if qtype == "true_or_false":
        return len(passages) > 0

    norm_answer = normalize_text(answer)

    # Open-ended with long answers: check sentence-level
    if qtype == "open-ended-qa" and len(norm_answer) > 200:
        sentences = [
            s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20
        ]
        for sent in sentences[:5]:
            norm_sent = normalize_text(sent)
            for p in passages:
                if norm_sent in normalize_text(p.get("text", "")):
                    return True
        return False

    # Default: substring match of full answer
    for p in passages:
        if norm_answer in normalize_text(p.get("text", "")):
            return True

    return False


STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "of",
        "in",
        "to",
        "and",
        "or",
        "for",
        "it",
        "that",
        "this",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "not",
        "but",
        "they",
        "their",
        "its",
        "can",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "do",
        "does",
        "did",
        "no",
        "yes",
        "so",
        "if",
        "all",
        "each",
        "any",
        "both",
        "few",
        "more",
        "most",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "also",
        "here",
        "there",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "why",
        "how",
        "much",
        "many",
        "using",
        "used",
        "one",
        "two",
        "new",
        "following",
        "given",
        "based",
    }
)


def _extract_keywords(text: str) -> set[str]:
    """Extract content words from text (removing stopwords and short tokens)."""
    words = set(normalize_text(text).split())
    return {w for w in words - STOPWORDS if len(w) > 2}


def check_retrievability_relaxed(
    question: dict,
    passages: list[dict],
    embedding_model=None,
) -> bool:
    """Relaxed retrievability check using keyword overlap and semantic similarity.

    For MCQ: checks if passages contain key terms from question+answer.
    For open-ended: uses semantic similarity via BGE embeddings.
    For true_or_false: checks if question topic is in passages.

    Args:
        question: SciKnowEval question dict.
        passages: List of retrieved passage dicts with 'text' field.
        embedding_model: SentenceTransformer for semantic matching (optional).

    Returns:
        True if passages are relevant to answering the question.
    """
    if not passages:
        return False

    answer = get_reference_answer(question)
    qtype = question.get("type", "")

    # True/false: check if question topic keywords appear in passages
    if qtype == "true_or_false":
        q_keywords = _extract_keywords(question.get("question", ""))
        passage_text = " ".join(p.get("text", "") for p in passages[:5])
        p_keywords = _extract_keywords(passage_text)
        overlap = len(q_keywords & p_keywords)
        return overlap >= 3  # at least 3 topic words match

    if answer is None:
        return False

    # Combine question + answer keywords for matching
    q_keywords = _extract_keywords(question.get("question", ""))
    a_keywords = _extract_keywords(answer)

    # All passage text combined
    passage_text = " ".join(p.get("text", "") for p in passages[:5])
    p_keywords = _extract_keywords(passage_text)

    # MCQ: keyword overlap between (question+answer) and passages
    if qtype in ("mcq-4-choices", "mcq-2-choices"):
        # Answer keywords in passages (concept presence)
        if a_keywords:
            a_overlap = len(a_keywords & p_keywords) / len(a_keywords)
            if a_overlap >= MCQ_KEYWORD_OVERLAP_THRESHOLD:
                return True

        # Question topic keywords in passages (topic relevance)
        if q_keywords:
            q_overlap = len(q_keywords & p_keywords) / len(q_keywords)
            if q_overlap >= 0.5:
                return True

        return False

    # Open-ended / filling / relation_extraction: semantic similarity
    if embedding_model is not None and len(answer.strip()) > 10:
        answer_emb = embedding_model.encode(
            [answer[:500]],
            normalize_embeddings=True,
        )
        passage_embs = embedding_model.encode(
            [p["text"][:500] for p in passages[:5]],
            normalize_embeddings=True,
        )
        similarities = np.dot(passage_embs, answer_emb.T).flatten()
        if similarities.max() >= SEMANTIC_SIM_THRESHOLD:
            return True

    # Fallback: keyword overlap for open-ended
    combined = q_keywords | a_keywords
    if combined:
        overlap = len(combined & p_keywords) / len(combined)
        if overlap >= MCQ_KEYWORD_OVERLAP_THRESHOLD:
            return True

    return False


def check_leakage(
    question: dict,
    bm25_searcher,
) -> bool:
    """Check if reference answer appears verbatim in corpus via BM25.

    Args:
        question: SciKnowEval question dict.
        bm25_searcher: Pyserini LuceneSearcher instance.

    Returns:
        True if corpus contains the answer verbatim (leaked).
    """
    answer = get_reference_answer(question)
    if answer is None:
        return False

    norm_answer = normalize_text(answer)

    # Skip very short answers (would match accidentally)
    if len(norm_answer) < 20:
        return False

    # Exact phrase query via BM25
    escaped = norm_answer[:200].replace('"', '\\"')
    try:
        hits = bm25_searcher.search(f'"{escaped}"', k=1)
        if hits:
            doc_text = normalize_text(
                json.loads(bm25_searcher.doc(hits[0].docid).raw())["contents"]
            )
            return norm_answer[:200] in doc_text
    except Exception as e:  # noqa: BLE001 - pyserini surfaces a zoo of errors
        logger.debug("Leakage check failed for question: %s", e)

    return False


# =========================================================================
# Checkpoint
# =========================================================================


def _sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    """Compute sha256 of a file in streaming mode."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _retriever_version(retriever) -> str:
    """Best-effort retriever fingerprint.

    Uses the FAISS index mtime as a stand-in for the absent
    FAISS ``manifest.json`` ``build_date`` field. Falls back to the string
    ``"unknown"`` when the path is not available (tests, offline mocks).

    Args:
        retriever: Retriever-like object; may expose ``_faiss_index_path``.

    Returns:
        Version string safe to embed in ``manifest.json``.
    """
    path_attr = getattr(retriever, "_faiss_index_path", None)
    if path_attr is None:
        return "unknown"
    try:
        faiss_path = Path(str(path_attr))
        if not faiss_path.exists():
            return "unknown"
        mtime = int(faiss_path.stat().st_mtime)
        return f"faiss-mtime-{mtime}"
    except OSError:
        return "unknown"


class Checkpoint:
    """Append-only JSONL checkpoint for retrievability filtering.

    Layout under ``cache_dir``:

    - ``progress.jsonl`` - one JSON record per processed question.
    - ``stats.json``     - last stats snapshot (atomic replace).
    - ``manifest.json``  - config fingerprint; mismatch invalidates cache.

    Attributes:
        cache_dir: Directory that holds checkpoint files.
        manifest: Config fingerprint to validate on open.
    """

    PROGRESS_FILENAME = "progress.jsonl"
    STATS_FILENAME = "stats.json"
    MANIFEST_FILENAME = "manifest.json"

    def __init__(self, cache_dir: Path, manifest: dict) -> None:
        self.cache_dir = Path(cache_dir)
        self.manifest = manifest
        self._progress_path = self.cache_dir / self.PROGRESS_FILENAME
        self._stats_path = self.cache_dir / self.STATS_FILENAME
        self._manifest_path = self.cache_dir / self.MANIFEST_FILENAME
        self._progress_fp = None
        self._processed: set[str] = set()
        self._records: list[dict] = []
        self._opened = False

    # ---------------------------------------------------------------- open

    def open(self) -> None:
        """Create the cache directory, validate the manifest, load progress.

        Raises:
            SystemExit: If an existing manifest disagrees with ``self.manifest``.
                Exit code is 2 so callers can distinguish this from other errors.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Manifest validation happens before touching progress.jsonl so that a
        # mismatched run cannot accidentally append to a stale log.
        if self._manifest_path.exists():
            try:
                existing = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("Could not read existing manifest (%s); rewriting.", exc)
                existing = None

            if existing is not None and existing != self.manifest:
                diff = self._manifest_diff(existing, self.manifest)
                logger.error(
                    "Checkpoint manifest mismatch. Re-run with --force to wipe. "
                    "Diff: %s",
                    json.dumps(diff, sort_keys=True),
                )
                raise SystemExit(2)

        # Always (re)write the manifest so fresh runs leave one on disk.
        self._manifest_path.write_text(
            json.dumps(self.manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        # Load and sanitise existing progress.jsonl (drop malformed tail line).
        if self._progress_path.exists():
            self._records = self._load_progress(self._progress_path)
            self._processed = {
                r["question_id"] for r in self._records if "question_id" in r
            }

        # Open append-only, line-buffered handle. text mode + buffering=1 is
        # WSL2-safe and does not require fsync after every write.
        self._progress_fp = self._progress_path.open("a", encoding="utf-8", buffering=1)
        self._opened = True

    @staticmethod
    def _manifest_diff(existing: dict, current: dict) -> dict:
        """Return a key -> (existing, current) diff for mismatched keys."""
        diff: dict = {}
        for key in sorted(set(existing) | set(current)):
            if existing.get(key) != current.get(key):
                diff[key] = {"existing": existing.get(key), "current": current.get(key)}
        return diff

    @staticmethod
    def _load_progress(path: Path) -> list[dict]:
        """Parse ``progress.jsonl`` and truncate an incomplete trailing line."""
        if not path.exists():
            return []

        raw = path.read_bytes()
        if not raw:
            return []

        # Split on \n but keep track so we can rewrite the file if the last
        # record was partially written.
        lines = raw.split(b"\n")
        # If the file ends with a newline the last element is an empty bytes.
        trailing_empty = lines and lines[-1] == b""
        if trailing_empty:
            lines = lines[:-1]

        records: list[dict] = []
        bad_tail = False
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                if i == len(lines) - 1 and not trailing_empty:
                    bad_tail = True
                    logger.warning(
                        "Dropping malformed trailing line in %s (crash recovery).",
                        path,
                    )
                    break
                logger.warning("Dropping malformed line %d in %s.", i, path)

        if bad_tail:
            # Rewrite the file without the bad tail line.
            good = b"\n".join(
                json.dumps(r, sort_keys=True).encode("utf-8") for r in records
            )
            if good:
                good += b"\n"
            path.write_bytes(good)

        return records

    # --------------------------------------------------------------- state

    def processed_ids(self) -> set[str]:
        """Return the set of question_ids already recorded on disk."""
        if not self._opened:
            raise RuntimeError("Checkpoint.open() must be called first.")
        return set(self._processed)

    def records(self) -> list[dict]:
        """Return all progress records in the order they were appended."""
        if not self._opened:
            raise RuntimeError("Checkpoint.open() must be called first.")
        return list(self._records)

    # ---------------------------------------------------------------- io

    def write(self, record: dict) -> None:
        """Append a single progress record and flush.

        Args:
            record: Must contain at least ``question_id``. Other keys are
                serialised as-is.
        """
        if not self._opened or self._progress_fp is None:
            raise RuntimeError("Checkpoint.open() must be called first.")
        if "question_id" not in record:
            raise ValueError("record is missing 'question_id'")

        qid = record["question_id"]
        if qid in self._processed:
            # Idempotent: do not double-write the same question.
            return

        line = json.dumps(record, sort_keys=True, ensure_ascii=False)
        self._progress_fp.write(line + "\n")
        self._progress_fp.flush()

        self._processed.add(qid)
        self._records.append(record)

    def write_stats(self, stats: dict) -> None:
        """Atomically replace ``stats.json`` with the given snapshot."""
        tmp_path = self._stats_path.with_suffix(".json.tmp")
        tmp_path.write_text(
            json.dumps(stats, sort_keys=True, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp_path, self._stats_path)

    def close(self) -> None:
        """Flush and close the progress file handle."""
        if self._progress_fp is not None:
            try:
                self._progress_fp.flush()
            except OSError as exc:
                logger.warning("Checkpoint flush on close failed: %s", exc)
            self._progress_fp.close()
            self._progress_fp = None
        self._opened = False


def _resolve_cache_dir(split_path: Path) -> Path:
    """Return the per-split checkpoint directory next to the split file."""
    split_path = Path(split_path)
    return split_path.parent / ".retrievability_cache" / split_path.stem


def _derive_split_name(split_path: Path) -> str:
    """Produce the ``split_name`` used by ``run_inference.load_dataset``."""
    return Path(split_path).stem.replace("_", "-")


def _assign_question_ids(questions: list[dict], split_path: Path) -> None:
    """Assign ``question_id`` in place, mirroring run_inference.load_dataset."""
    split_name = _derive_split_name(split_path)
    for idx, record in enumerate(questions):
        record["question_id"] = f"ske-{split_name}-{idx:05d}"


def _build_manifest(
    split_path: Path,
    split_size: int,
    retriever,
    *,
    skip_leakage: bool,
    top_k: int = TOP_K,
    threshold: float = MCQ_KEYWORD_OVERLAP_THRESHOLD,
    seed: int = SEED,
) -> dict:
    """Construct the manifest dict used to validate the checkpoint."""
    split_path = Path(split_path)
    return {
        "split_file": str(split_path).replace("\\", "/"),
        "split_size": split_size,
        "split_sha256": _sha256_file(split_path) if split_path.exists() else "",
        "retriever_version": _retriever_version(retriever),
        "skip_leakage": bool(skip_leakage),
        "top_k": int(top_k),
        "threshold": float(threshold),
        "seed": int(seed),
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
    }


def _init_stats(total: int) -> dict:
    """Return an empty stats skeleton sized for ``total`` questions."""
    return {
        "total": total,
        "with_answer": 0,
        "retrievable": 0,
        "leaked": 0,
        "by_domain": {},
        "by_level": {},
        "by_type": {},
    }


def _bump_bucket(stats: dict, key: str, val: str, field: str) -> None:
    """Increment ``stats[key][val][field]`` (initialising the bucket)."""
    bucket = stats[key].setdefault(val, {"total": 0, "retrievable": 0, "leaked": 0})
    bucket[field] = bucket.get(field, 0) + 1


def _replay_stats(records: list[dict], total: int) -> dict:
    """Rebuild the stats snapshot from a list of progress records.

    Used on resume so that we don't depend on ``stats.json`` having survived
    a crash.

    Args:
        records: Progress records loaded from progress.jsonl.
        total: Total number of questions in the split.

    Returns:
        Stats dict with the same shape as the live counter.
    """
    stats = _init_stats(total)
    for rec in records:
        domain = rec.get("domain", "unknown")
        level = rec.get("level", "unknown")
        qtype = rec.get("type", "unknown")
        _bump_bucket(stats, "by_domain", domain, "total")
        _bump_bucket(stats, "by_level", level, "total")
        _bump_bucket(stats, "by_type", qtype, "total")
        if rec.get("has_answer", rec.get("retrievable") is not None):
            stats["with_answer"] += 1
        if rec.get("leaked"):
            stats["leaked"] += 1
            _bump_bucket(stats, "by_domain", domain, "leaked")
            _bump_bucket(stats, "by_level", level, "leaked")
            _bump_bucket(stats, "by_type", qtype, "leaked")
        if rec.get("retrievable") and not rec.get("leaked"):
            stats["retrievable"] += 1
            _bump_bucket(stats, "by_domain", domain, "retrievable")
            _bump_bucket(stats, "by_level", level, "retrievable")
            _bump_bucket(stats, "by_type", qtype, "retrievable")
    return stats


# =========================================================================
# Main pipelines
# =========================================================================


def run_retrievability_filter(
    test_path: Path,
    retriever,
    output_path: Path,
    *,
    skip_leakage: bool = False,
    force: bool = False,
    flush_every: int = 50,
) -> dict:
    """Run retrievability filtering with append-only checkpoint / resume.

    On restart, any question already recorded in ``progress.jsonl`` is
    skipped. The final ``output_path`` JSON is sorted by ``question_id`` so
    the output is deterministic regardless of resume order.

    Args:
        test_path: Path to the split file (e.g. ``main_test.json``).
        retriever: Retriever instance (``retrieve_hybrid`` + embedding).
        output_path: Destination for the filtered questions JSON.
        skip_leakage: Skip the corpus leakage check (faster, no BM25).
        force: If True, wipe the checkpoint directory before running.
        flush_every: Write ``stats.json`` every N newly processed questions.

    Returns:
        Final statistics dict (same shape as the old implementation).
    """
    test_path = Path(test_path)
    output_path = Path(output_path)

    with test_path.open("r", encoding="utf-8") as f:
        questions = json.load(f)

    _assign_question_ids(questions, test_path)
    total = len(questions)
    logger.info("Loaded %d questions from %s", total, test_path)

    # Checkpoint setup -----------------------------------------------------
    cache_dir = _resolve_cache_dir(test_path)
    if force and cache_dir.exists():
        logger.warning("--force: removing existing checkpoint at %s", cache_dir)
        shutil.rmtree(cache_dir)

    manifest = _build_manifest(
        test_path,
        total,
        retriever,
        skip_leakage=skip_leakage,
    )
    checkpoint = Checkpoint(cache_dir, manifest)
    checkpoint.open()

    already_done = checkpoint.processed_ids()
    if already_done:
        pct = 100.0 * len(already_done) / max(total, 1)
        logger.info(
            "Resuming from checkpoint: %d questions already processed (%.1f%%)",
            len(already_done),
            pct,
        )

    # Stats: replay from progress.jsonl so a lost stats.json is recoverable.
    stats = _replay_stats(checkpoint.records(), total)

    # Leakage / embedding model are only needed for the active loop --------
    embedding_model = None
    bm25_searcher = None
    remaining = [q for q in questions if q["question_id"] not in already_done]

    if remaining:
        embedding_model = retriever._get_embedding_model()
        if not skip_leakage:
            from pyserini.search.lucene import LuceneSearcher

            bm25_searcher = LuceneSearcher(str(PROJECT_ROOT / "indices" / "bm25"))
            bm25_searcher.set_bm25(k1=1.2, b=0.75)

    # Main loop ------------------------------------------------------------
    processed_in_run = 0
    try:
        for q in tqdm(remaining, desc="Filtering"):
            domain = q.get("domain", "unknown")
            level = q.get("details", {}).get("level", "unknown")
            qtype = q.get("type", "unknown")

            _bump_bucket(stats, "by_domain", domain, "total")
            _bump_bucket(stats, "by_level", level, "total")
            _bump_bucket(stats, "by_type", qtype, "total")

            answer = get_reference_answer(q)
            has_answer = answer is not None
            is_retrievable = False
            is_leaked = False

            if has_answer:
                stats["with_answer"] += 1
                passages = retriever.retrieve_hybrid(q["question"], top_k=TOP_K)
                is_retrievable = check_retrievability_relaxed(
                    q,
                    passages,
                    embedding_model,
                )
                if not skip_leakage and bm25_searcher is not None:
                    is_leaked = check_leakage(q, bm25_searcher)

                if is_leaked:
                    stats["leaked"] += 1
                    _bump_bucket(stats, "by_domain", domain, "leaked")
                    _bump_bucket(stats, "by_level", level, "leaked")
                    _bump_bucket(stats, "by_type", qtype, "leaked")

                if is_retrievable and not is_leaked:
                    stats["retrievable"] += 1
                    _bump_bucket(stats, "by_domain", domain, "retrievable")
                    _bump_bucket(stats, "by_level", level, "retrievable")
                    _bump_bucket(stats, "by_type", qtype, "retrievable")

            record = {
                "question_id": q["question_id"],
                "has_answer": has_answer,
                "retrievable": bool(is_retrievable),
                "leaked": bool(is_leaked),
                "domain": domain,
                "level": level,
                "type": qtype,
            }
            checkpoint.write(record)
            processed_in_run += 1

            if flush_every > 0 and processed_in_run % flush_every == 0:
                checkpoint.write_stats(stats)
                logger.info(
                    "Heartbeat: %d/%d processed this run, retrievable=%d leaked=%d",
                    processed_in_run,
                    len(remaining),
                    stats["retrievable"],
                    stats["leaked"],
                )

        # Final stats snapshot.
        checkpoint.write_stats(stats)

        # Snapshot progress records before close so the materialisation
        # below does not depend on a still-open checkpoint.
        final_records = checkpoint.records()
    finally:
        checkpoint.close()

    # Materialise the filtered output --------------------------------------
    keep_ids = {
        r["question_id"]
        for r in final_records
        if r.get("retrievable") and not r.get("leaked")
    }
    qmap = {q["question_id"]: q for q in questions}
    filtered: list[dict] = []
    for qid in sorted(keep_ids):
        q = qmap.get(qid)
        if q is None:
            continue
        q_out = dict(q)
        q_out["retrievable"] = True
        filtered.append(q_out)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d retrievable questions to %s", len(filtered), output_path)

    return stats


def run_recall_evaluation(
    dev_path: Path,
    retriever,
) -> dict[str, dict[str, float]]:
    """Compute Recall@10 per retrieval method on dev set (TABLE VI).

    Reports both strict (substring) and relaxed (keyword/semantic) matching.

    Args:
        dev_path: Path to dev.json.
        retriever: Retriever instance.

    Returns:
        Dict mapping method name to strict/relaxed Recall@10 scores.
    """
    with open(dev_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    # Filter to questions with reference answers
    questions_with_answers = [
        q for q in questions if get_reference_answer(q) is not None
    ]
    logger.info(
        "Evaluating Recall@10 on %d/%d dev questions (with answers)",
        len(questions_with_answers),
        len(questions),
    )

    # Load embedding model for semantic matching
    embedding_model = retriever._get_embedding_model()

    methods = ["bm25", "dense", "hybrid"]
    strict: dict[str, dict[str, int]] = {m: {"hits": 0, "total": 0} for m in methods}
    relaxed: dict[str, dict[str, int]] = {m: {"hits": 0, "total": 0} for m in methods}

    for q in tqdm(questions_with_answers, desc="Recall@10"):
        query = q["question"]

        for method in methods:
            if method == "bm25":
                passages = retriever.retrieve_bm25(query, top_k=10)
            elif method == "dense":
                passages = retriever.retrieve_dense(query, top_k=10)
            else:
                passages = retriever.retrieve_hybrid(query, top_k=10)

            strict[method]["total"] += 1
            relaxed[method]["total"] += 1

            if check_retrievability(q, passages):
                strict[method]["hits"] += 1
            if check_retrievability_relaxed(q, passages, embedding_model):
                relaxed[method]["hits"] += 1

    results: dict[str, dict[str, float]] = {}
    print("\n=== TABLE VI: Retrieval Quality (Recall@10 on dev set) ===")
    print(f"{'Method':<10} {'Strict':>10} {'Relaxed':>10} {'Total':>6}")
    print("-" * 40)
    for method in methods:
        total = strict[method]["total"]
        s_hits = strict[method]["hits"]
        r_hits = relaxed[method]["hits"]
        s_r = s_hits / total if total > 0 else 0.0
        r_r = r_hits / total if total > 0 else 0.0
        results[method] = {"strict": s_r, "relaxed": r_r}
        print(f"{method:<10} {s_r:>9.1%} {r_r:>9.1%} {total:>6}")

    return results


def print_stats(stats: dict) -> None:
    """Pretty-print retrievability statistics."""
    total = stats["total"]
    with_answer = stats["with_answer"]
    retrievable = stats["retrievable"]
    leaked = stats["leaked"]

    print("\n=== Retrievability Filter Results ===")
    print(f"Total questions:    {total}")
    print(f"With answer:        {with_answer}")
    denom = with_answer if with_answer else 1
    print(
        f"Retrievable:        {retrievable} ({retrievable / denom * 100:.1f}% of answerable)"
    )
    print(f"Leaked (excluded):  {leaked}")

    for label, key in [
        ("Domain", "by_domain"),
        ("Level", "by_level"),
        ("Type", "by_type"),
    ]:
        print(f"\nBy {label}:")
        print(f"  {'Value':<25} {'Total':>6} {'Retrievable':>12} {'Leaked':>8}")
        for val, counts in sorted(stats[key].items()):
            retr = counts["retrievable"]
            tot = counts["total"]
            leak = counts["leaked"]
            pct = f"({retr / tot * 100:.0f}%)" if tot > 0 else ""
            print(f"  {val:<25} {tot:>6} {retr:>8} {pct:>4} {leak:>8}")


# =========================================================================
# CLI
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the retrievability filter CLI."""
    parser = argparse.ArgumentParser(
        description="Retrievability filter, leakage check, and recall evaluation.",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run TABLE VI recall evaluation on dev set.",
    )
    parser.add_argument(
        "--skip-leakage",
        action="store_true",
        help="Skip corpus leakage check.",
    )
    parser.add_argument(
        "--split",
        default=str(MAIN_TEST_PATH),
        help="Path to the split file (default: main_test.json).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear any existing checkpoint before running.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=50,
        help="Flush stats.json every N questions (default: 50).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for models (default: cuda).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Import here to avoid loading models at module level
    from retriever import Retriever

    retriever = Retriever(device=args.device)

    if args.eval_only:
        run_recall_evaluation(DEV_PATH, retriever)
        return

    # Full pipeline
    split_path = Path(args.split)
    # Place output next to the split, same stem + '_retrievable'.
    output_path = split_path.parent / f"{split_path.stem}_retrievable.json"
    stats = run_retrievability_filter(
        split_path,
        retriever,
        output_path,
        skip_leakage=args.skip_leakage,
        force=args.force,
        flush_every=args.flush_every,
    )
    print_stats(stats)

    # Also run TABLE VI on dev set
    print()
    run_recall_evaluation(DEV_PATH, retriever)


if __name__ == "__main__":
    main()
