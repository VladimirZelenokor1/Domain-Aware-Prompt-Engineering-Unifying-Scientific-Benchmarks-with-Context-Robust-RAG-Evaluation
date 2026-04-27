# Based on prototype by analyst. Scaled for production.
"""Wikipedia corpus builder.

Loads English Wikipedia from HuggingFace (wikimedia/wikipedia 20231101.en),
filters science articles by keyword matching, classifies domains, and chunks
into overlapping token windows.

Usage:
    python scripts/build_wikipedia.py --max-articles 20
    python scripts/build_wikipedia.py --max-articles 200000 --output corpus/wikipedia_chunks.jsonl
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
DEFAULT_OUTPUT = PROJECT_ROOT / "corpus" / "wikipedia_chunks.jsonl"

# Wikipedia dataset config
DATASET_NAME = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.en"
MIN_ARTICLE_WORDS = 200


# =========================================================================
# Builder
# =========================================================================

def build(
    max_articles: int = 200_000,
    output_path: Path | None = None,
) -> dict:
    """Build Wikipedia corpus chunks.

    Args:
        max_articles: Maximum number of science articles to process.
        output_path: Path for output JSONL. Defaults to corpus/wikipedia_chunks.jsonl.

    Returns:
        Stats dict with articles_count, chunks_count, domain_distribution.
    """
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s/%s (streaming)...", DATASET_NAME, DATASET_CONFIG)
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, streaming=True, split="train")

    tokenizer = init_tokenizer()

    articles_count = 0
    chunks_count = 0
    skipped_stubs = 0
    skipped_non_science = 0
    domain_dist: dict[str, int] = {}

    with open(out, "w", encoding="utf-8") as fh:
        progress = tqdm(dataset, desc="Wikipedia articles", unit="art")
        for article in progress:
            if articles_count >= max_articles:
                break

            title = article.get("title", "")
            text = article.get("text", "")

            # Skip stubs
            if len(text.split()) < MIN_ARTICLE_WORDS:
                skipped_stubs += 1
                continue

            # Classify domain
            domain = classify_domain(text, title=title)
            if domain == "general_science":
                skipped_non_science += 1
                continue

            # Chunk
            source_id = str(article.get("id", title))
            chunks = chunk_text(text, "wikipedia", source_id, tokenizer)

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
                    stubs=skipped_stubs,
                )

    stats = {
        "articles_count": articles_count,
        "chunks_count": chunks_count,
        "skipped_stubs": skipped_stubs,
        "skipped_non_science": skipped_non_science,
        "domain_distribution": domain_dist,
    }
    logger.info(
        "Wikipedia done: %d articles, %d chunks, %d stubs skipped, %d non-science skipped",
        articles_count, chunks_count, skipped_stubs, skipped_non_science,
    )
    logger.info("Domain distribution: %s", domain_dist)

    return stats


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Wikipedia science corpus chunks.",
    )
    parser.add_argument(
        "--max-articles", type=int, default=200_000,
        help="Maximum number of science articles to process (default: 200000).",
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
