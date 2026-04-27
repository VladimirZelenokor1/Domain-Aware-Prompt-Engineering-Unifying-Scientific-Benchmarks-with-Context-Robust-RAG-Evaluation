# Based on prototype by analyst. Scaled for production.
"""Shared chunking and domain classification utilities.

Core chunking logic preserved from analyst's build_pubmed_corpus_hf.py prototype.
Provides token-based sliding-window chunking, chunk ID generation,
keyword-based domain classification, and streaming JSONL output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import IO, Generator

import tiktoken
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from analyst prototype - DO NOT CHANGE)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 256
CHUNK_OVERLAP = 64
TOKENIZER_NAME = "cl100k_base"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_CONFIG_PATH = PROJECT_ROOT / "configs" / "corpus.yaml"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def init_tokenizer() -> tiktoken.Encoding:
    """Initialize the tiktoken tokenizer."""
    return tiktoken.get_encoding(TOKENIZER_NAME)


# ---------------------------------------------------------------------------
# Chunking (preserved from analyst's build_pubmed_corpus_hf.py)
# ---------------------------------------------------------------------------

def _chunk_tokens(
    tokens: list[int],
    chunk_size: int,
    overlap: int,
) -> Generator[tuple[int, list[int]], None, None]:
    """Yield (index, token_chunk) pairs using a sliding window.

    Args:
        tokens: Full list of token IDs.
        chunk_size: Number of tokens per chunk.
        overlap: Number of overlapping tokens between consecutive chunks.

    Yields:
        Tuple of (chunk_index, token_chunk).
    """
    start = 0
    idx = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        yield idx, chunk
        start += chunk_size - overlap
        idx += 1
        if start >= len(tokens):
            break


def generate_chunk_id(source: str, source_id: str, idx: int) -> str:
    """Generate a deterministic chunk ID via MD5 hash.

    Args:
        source: Data source name (e.g. 'wikipedia', 'pubmed', 'openstax').
        source_id: Article/document identifier within the source.
        idx: Chunk index within the document.

    Returns:
        16-character hex string.
    """
    raw = f"{source}_{source_id}_{idx}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def chunk_text(
    text: str,
    source: str,
    source_id: str,
    tokenizer: tiktoken.Encoding,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict]:
    """Split text into overlapping token-based chunks with metadata.

    Args:
        text: Raw text to chunk.
        source: Data source name.
        source_id: Document identifier within the source.
        tokenizer: Tiktoken encoding instance.
        chunk_size: Tokens per chunk.
        overlap: Overlapping tokens between chunks.

    Returns:
        List of chunk dicts with keys: chunk_id, text, source, source_id.
    """
    if not text or not text.strip():
        return []

    tokens = tokenizer.encode(text)
    chunks = []

    for idx, token_chunk in _chunk_tokens(tokens, chunk_size, overlap):
        chunk_text_decoded = tokenizer.decode(token_chunk)
        chunk_id = generate_chunk_id(source, source_id, idx)
        chunks.append({
            "chunk_id": chunk_id,
            "text": chunk_text_decoded,
            "source": source,
            "source_id": source_id,
        })

    return chunks


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

_domain_keywords: dict[str, list[str]] | None = None
_domain_patterns: dict[str, re.Pattern] | None = None


def load_domain_keywords(
    config_path: Path | None = None,
) -> dict[str, list[str]]:
    """Load domain keyword lists from corpus config YAML.

    Args:
        config_path: Path to corpus.yaml. Defaults to configs/corpus.yaml.

    Returns:
        Dict mapping domain name to list of keyword strings.
    """
    path = config_path or CORPUS_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config["domains"]


def _get_domain_patterns(
    config_path: Path | None = None,
) -> dict[str, re.Pattern]:
    """Build and cache compiled regex patterns for each domain.

    Keywords are sorted longest-first so multi-word phrases match before
    their single-word components.
    """
    global _domain_keywords, _domain_patterns

    if _domain_patterns is not None:
        return _domain_patterns

    _domain_keywords = load_domain_keywords(config_path)
    _domain_patterns = {}

    for domain, keywords in _domain_keywords.items():
        sorted_kw = sorted(keywords, key=len, reverse=True)
        escaped = [re.escape(kw) for kw in sorted_kw]
        pattern = re.compile(
            r"\b(?:" + "|".join(escaped) + r")\b",
            re.IGNORECASE,
        )
        _domain_patterns[domain] = pattern

    return _domain_patterns


def classify_domain(
    text: str,
    title: str = "",
    config_path: Path | None = None,
) -> str:
    """Classify text into a scientific domain by keyword matching.

    Checks title first (fast path), then falls back to first 500 chars
    of text body. Returns the domain with the most keyword hits.

    Priority order for ties: biology > chemistry > physics >
    materials_science > earth_science.

    Args:
        text: Document body text.
        title: Document title (checked first).
        config_path: Optional path to corpus.yaml.

    Returns:
        Domain string, or 'general_science' if no match.
    """
    patterns = _get_domain_patterns(config_path)

    # Check title first (fast, most reliable)
    title_scores: dict[str, int] = {}
    if title:
        title_lower = title.lower()
        for domain, pattern in patterns.items():
            matches = pattern.findall(title_lower)
            if matches:
                title_scores[domain] = len(matches)

    if title_scores:
        return max(title_scores, key=title_scores.get)  # type: ignore[arg-type]

    # Fallback: check first 500 chars of text body
    text_sample = text[:2000].lower()  # ~500 words
    text_scores: dict[str, int] = {}
    for domain, pattern in patterns.items():
        matches = pattern.findall(text_sample)
        if matches:
            text_scores[domain] = len(matches)

    if text_scores:
        return max(text_scores, key=text_scores.get)  # type: ignore[arg-type]

    return "general_science"


# ---------------------------------------------------------------------------
# JSONL I/O
# ---------------------------------------------------------------------------

def write_chunks_jsonl(chunks: list[dict], fh: IO[str]) -> int:
    """Write chunk dicts as JSONL to an open file handle.

    Args:
        chunks: List of chunk dicts to write.
        fh: Open file handle in write/append mode.

    Returns:
        Number of chunks written.
    """
    for chunk in chunks:
        fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return len(chunks)


def load_corpus_config(config_path: Path | None = None) -> dict:
    """Load the full corpus configuration.

    Args:
        config_path: Path to corpus.yaml. Defaults to configs/corpus.yaml.

    Returns:
        Parsed config dict.
    """
    path = config_path or CORPUS_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
