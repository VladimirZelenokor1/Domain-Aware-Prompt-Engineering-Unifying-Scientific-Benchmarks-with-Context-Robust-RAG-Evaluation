# Based on prototype by analyst. Scaled for production.
"""PubMed corpus builder.

Loads PubMed abstracts/articles from HuggingFace (ccdv/pubmed-summarization),
classifies domains by keyword matching on abstracts, and chunks full article
text into overlapping token windows.

Usage:
    python scripts/build_pubmed.py --max-articles 20
    python scripts/build_pubmed.py --max-articles 400000 --output corpus/pubmed_chunks.jsonl
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from chunking import (  # noqa: E402
    chunk_text,
    classify_domain,
    init_tokenizer,
    write_chunks_jsonl,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "corpus" / "pubmed_chunks.jsonl"

# PubMed dataset config
DATASET_NAME = "ccdv/pubmed-summarization"
DATASET_SPLIT = "train"


# =========================================================================
# Builder
# =========================================================================

def build(
    max_articles: int = 400_000,
    output_path: Path | None = None,
) -> dict:
    """Build PubMed corpus chunks.

    Args:
        max_articles: Maximum number of articles to process.
        output_path: Path for output JSONL. Defaults to corpus/pubmed_chunks.jsonl.

    Returns:
        Stats dict with articles_count, chunks_count, domain_distribution.
    """
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s (split=%s, streaming)...", DATASET_NAME, DATASET_SPLIT)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    tokenizer = init_tokenizer()

    articles_count = 0
    chunks_count = 0
    skipped_empty = 0
    domain_dist: dict[str, int] = {}

    with open(out, "w", encoding="utf-8") as fh:
        progress = tqdm(dataset, desc="PubMed articles", unit="art")
        for idx, article in enumerate(progress):
            if articles_count >= max_articles:
                break

            # Source ID: use article_id if available, else enumeration index
            article_id = article.get("article_id", None)
            if not article_id or article_id.startswith("unknown"):
                article_id = f"pubmed_{idx}"

            # Use full article text for chunking (abstracts are too short)
            text = article.get("article", "")
            abstract = article.get("abstract", "")

            if not text or not text.strip():
                # Fallback to abstract if article text is missing
                text = abstract

            if not text or not text.strip():
                skipped_empty += 1
                continue

            # Classify domain by abstract keywords
            # PubMed is predominantly biomedical, default to biology
            domain = classify_domain(abstract or text, title="")
            if domain == "general_science":
                domain = "biology"

            # Chunk
            chunks = chunk_text(text, "pubmed", article_id, tokenizer)

            # Attach domain
            for c in chunks:
                c["domain"] = domain

            n = write_chunks_jsonl(chunks, fh)
            chunks_count += n
            articles_count += 1
            domain_dist[domain] = domain_dist.get(domain, 0) + 1

            if articles_count % 1000 == 0:
                progress.set_postfix(
                    articles=articles_count,
                    chunks=chunks_count,
                )

    stats = {
        "articles_count": articles_count,
        "chunks_count": chunks_count,
        "skipped_empty": skipped_empty,
        "domain_distribution": domain_dist,
    }
    logger.info(
        "PubMed done: %d articles, %d chunks, %d empty skipped",
        articles_count, chunks_count, skipped_empty,
    )
    logger.info("Domain distribution: %s", domain_dist)

    return stats


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PubMed science corpus chunks.",
    )
    parser.add_argument(
        "--max-articles", type=int, default=400_000,
        help="Maximum number of articles to process (default: 400000).",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    build(max_articles=args.max_articles, output_path=args.output)


if __name__ == "__main__":
    main()
