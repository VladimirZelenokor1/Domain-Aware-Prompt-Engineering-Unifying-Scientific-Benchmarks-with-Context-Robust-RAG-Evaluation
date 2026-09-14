"""Build QASPER corpus: download, chunk papers, and optionally build indices.

Downloads the QASPER dataset from HuggingFace, chunks all paper texts
(abstracts + sections), writes a corpus JSONL, and optionally builds
BM25 + FAISS search indices for Track B evaluation.

Usage:
    python scripts/build_qasper_corpus.py                   # full build
    python scripts/build_qasper_corpus.py --skip-indices     # corpus only
    python scripts/build_qasper_corpus.py --cache-dir /tmp   # custom HF cache
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from chunking import chunk_text, init_tokenizer, write_chunks_jsonl  # noqa: E402

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "qasper"
DEFAULT_INDICES_DIR = PROJECT_ROOT / "indices"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-en-v1.5"


# =========================================================================
# Dataset loading
# =========================================================================


def load_dataset(
    cache_dir: Path | None = None,
) -> dict[str, list[dict]]:
    """Load QASPER dataset from HuggingFace.

    Args:
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        Dict mapping split name to list of paper dicts.

    Raises:
        ImportError: If the ``datasets`` library is not installed.
    """
    try:
        from datasets import load_dataset as hf_load_dataset
    except ImportError:
        logger.error(
            "The 'datasets' library is required. Install with: pip install datasets"
        )
        raise

    logger.info("Loading QASPER dataset from HuggingFace...")
    ds = hf_load_dataset(
        "allenai/qasper",
        cache_dir=str(cache_dir) if cache_dir else None,
    )

    result: dict[str, list[dict]] = {}
    for split_name in ds:
        result[split_name] = list(ds[split_name])
    return result


# =========================================================================
# Corpus building
# =========================================================================


def build_qasper_corpus(
    output_dir: Path,
    cache_dir: Path | None = None,
) -> dict:
    """Download QASPER, chunk all papers, write corpus JSONL.

    Args:
        output_dir: Where to write corpus.jsonl and papers.json.
        cache_dir: HuggingFace cache directory.

    Returns:
        Stats dict with paper_count, chunk_count.
    """
    ds = load_dataset(cache_dir)

    tokenizer = init_tokenizer()
    papers: list[dict] = []

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "corpus.jsonl"
    total_chunks = 0

    with open(corpus_path, "w", encoding="utf-8") as fh:
        for split_name in ds:
            for paper in ds[split_name]:
                paper_id = paper["id"]
                title = paper["title"]

                # Chunk abstract
                if paper.get("abstract") and paper["abstract"].strip():
                    abstract_chunks = chunk_text(
                        paper["abstract"],
                        source="qasper",
                        source_id=f"{paper_id}_abstract",
                        tokenizer=tokenizer,
                    )
                    write_chunks_jsonl(abstract_chunks, fh)
                    total_chunks += len(abstract_chunks)

                # Chunk each section's paragraphs
                full_text = paper.get("full_text", {})
                sections = full_text.get("section_name", [])
                paragraphs_list = full_text.get("paragraphs", [])

                for sec_idx, (sec_name, paragraphs) in enumerate(
                    zip(sections, paragraphs_list)
                ):
                    for para_idx, para in enumerate(paragraphs):
                        if not para or not para.strip():
                            continue
                        chunks = chunk_text(
                            para,
                            source="qasper",
                            source_id=f"{paper_id}_s{sec_idx}_p{para_idx}",
                            tokenizer=tokenizer,
                        )
                        # Add section metadata to each chunk
                        for c in chunks:
                            c["paper_id"] = paper_id
                            c["section"] = sec_name or ""
                        write_chunks_jsonl(chunks, fh)
                        total_chunks += len(chunks)

                # Count questions
                qas = paper.get("qas", {})
                question_list = qas.get("question", [])

                papers.append(
                    {
                        "paper_id": paper_id,
                        "title": title,
                        "split": split_name,
                        "n_questions": len(question_list),
                    }
                )

    # Save papers manifest
    papers_path = output_dir / "papers.json"
    with open(papers_path, "w", encoding="utf-8") as f:
        json.dump(papers, f, indent=2, ensure_ascii=False)

    logger.info(
        "QASPER corpus built: %d papers, %d chunks -> %s",
        len(papers),
        total_chunks,
        corpus_path,
    )

    return {"paper_count": len(papers), "chunk_count": total_chunks}


# =========================================================================
# Index building
# =========================================================================


def build_qasper_indices(corpus_path: Path, indices_dir: Path) -> None:
    """Build BM25 and FAISS indices for QASPER corpus.

    Args:
        corpus_path: Path to corpus.jsonl.
        indices_dir: Root directory for index output.
    """
    from build_indices import (  # noqa: F811
        build_bm25_index,
        build_faiss_index,
        prepare_pyserini_input,
    )

    bm25_dir = indices_dir / "qasper_bm25"
    faiss_dir = indices_dir / "qasper_faiss"
    pyserini_input = indices_dir / "qasper_pyserini_input"

    # BM25
    logger.info("Building BM25 index for QASPER...")
    prepare_pyserini_input(corpus_path, pyserini_input)
    build_bm25_index(pyserini_input, bm25_dir)

    # FAISS (requires GPU for embeddings)
    logger.info("Building FAISS index for QASPER...")
    build_faiss_index(
        corpus_path,
        faiss_dir,
        EMBEDDING_MODEL_PATH,
        cleanup_embeddings=True,
    )

    logger.info("QASPER indices built at %s", indices_dir)


# =========================================================================
# CLI
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build QASPER corpus and search indices for Track B.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for corpus.jsonl and papers.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--indices-dir",
        type=Path,
        default=DEFAULT_INDICES_DIR,
        help=f"Output directory for search indices (default: {DEFAULT_INDICES_DIR})",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="HuggingFace dataset cache directory.",
    )
    parser.add_argument(
        "--skip-indices",
        action="store_true",
        help="Build corpus only, skip BM25 and FAISS index building.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for QASPER corpus builder."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    logger.info("=== Building QASPER corpus ===")
    stats = build_qasper_corpus(args.output_dir, cache_dir=args.cache_dir)
    logger.info(
        "Corpus stats: %d papers, %d chunks",
        stats["paper_count"],
        stats["chunk_count"],
    )

    if not args.skip_indices:
        corpus_path = args.output_dir / "corpus.jsonl"
        logger.info("=== Building QASPER indices ===")
        build_qasper_indices(corpus_path, args.indices_dir)
    else:
        logger.info("Skipping index build (--skip-indices)")

    logger.info("=== QASPER corpus build complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
