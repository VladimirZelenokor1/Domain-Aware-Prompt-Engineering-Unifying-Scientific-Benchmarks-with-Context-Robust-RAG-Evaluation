# Based on prototype by analyst. Scaled for production.
"""OpenStax textbook corpus builder.

Loads OpenStax textbook paragraphs from HuggingFace, filters to 12 target
science textbooks, concatenates section paragraphs, and chunks into
overlapping token windows.

Usage:
    python scripts/build_openstax.py
    python scripts/build_openstax.py --output corpus/openstax_chunks.jsonl
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
    init_tokenizer,
    write_chunks_jsonl,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "corpus" / "openstax_chunks.jsonl"

# OpenStax dataset config
DATASET_NAME = "HuggingFaceTB/openstax_paragraphs"

# Target textbooks -> domain mapping (approved list of 12)
OPENSTAX_BOOKS: dict[str, str] = {
    "Biology 2e": "biology",
    "Concepts of Biology": "biology",
    "Microbiology": "biology",
    "Anatomy and Physiology 2e": "biology",
    "Chemistry 2e": "chemistry",
    "Chemistry: Atoms First 2e": "chemistry",
    "Organic Chemistry": "chemistry",
    "University Physics Volume 1": "physics",
    "University Physics Volume 2": "physics",
    "University Physics Volume 3": "physics",
    "College Physics 2e": "physics",
    "Astronomy 2e": "earth_science",
}


# =========================================================================
# Builder
# =========================================================================

def _process_sections(
    sections: list[dict],
    book_title: str,
    path_prefix: str,
    domain: str,
    tokenizer,
    fh,
) -> tuple[int, int]:
    """Process a list of sections, chunking their paragraph content.

    Args:
        sections: List of section dicts with 'title' and 'paragraph' keys.
        book_title: Book title for source_id.
        path_prefix: Parent path (e.g. "Biology 2e::Chapter 1").
        domain: Domain label.
        tokenizer: Tiktoken encoding instance.
        fh: Open file handle for JSONL output.

    Returns:
        Tuple of (sections_count, chunks_count).
    """
    sections_count = 0
    chunks_count = 0

    for section in sections:
        section_title = section.get("title", "untitled_section")

        paragraphs = section.get("paragraph") or []
        if isinstance(paragraphs, str):
            section_text = paragraphs
        elif isinstance(paragraphs, list):
            section_text = "\n\n".join(
                p for p in paragraphs if isinstance(p, str) and p.strip()
            )
        else:
            continue

        if not section_text or not section_text.strip():
            continue

        source_id = f"{path_prefix}::{section_title}"
        chunks = chunk_text(section_text, "openstax", source_id, tokenizer)

        for c in chunks:
            c["domain"] = domain

        n = write_chunks_jsonl(chunks, fh)
        chunks_count += n
        sections_count += 1

    return sections_count, chunks_count


def _walk_node(
    node: dict,
    book_title: str,
    path: str,
    domain: str,
    tokenizer,
    fh,
) -> tuple[int, int]:
    """Recursively walk the chapter/sub-chapter tree, processing sections.

    The OpenStax dataset has a recursive structure: each node can have
    'chapters' (sub-chapters) and 'sections' (leaf content).

    Args:
        node: A chapter/sub-chapter dict.
        book_title: Book title for source_id.
        path: Accumulated path (e.g. "Biology 2e::The Cell::Cell Structure").
        domain: Domain label.
        tokenizer: Tiktoken encoding instance.
        fh: Open file handle for JSONL output.

    Returns:
        Tuple of (sections_count, chunks_count).
    """
    sections_count = 0
    chunks_count = 0

    # Process sections at this level
    sections = node.get("sections") or []
    if sections:
        sc, cc = _process_sections(
            sections, book_title, path, domain, tokenizer, fh,
        )
        sections_count += sc
        chunks_count += cc

    # Recurse into sub-chapters
    sub_chapters = node.get("chapters") or []
    for sub_ch in sub_chapters:
        sub_title = sub_ch.get("title", "untitled")
        sub_path = f"{path}::{sub_title}"
        sc, cc = _walk_node(
            sub_ch, book_title, sub_path, domain, tokenizer, fh,
        )
        sections_count += sc
        chunks_count += cc

    return sections_count, chunks_count


def _process_book(
    book: dict,
    domain: str,
    tokenizer,
    fh,
) -> tuple[int, int]:
    """Process one OpenStax book into chunks via recursive tree traversal.

    Args:
        book: Book record from HF dataset with recursive chapters structure.
        domain: Pre-assigned domain for this book.
        tokenizer: Tiktoken encoding instance.
        fh: Open file handle for JSONL output.

    Returns:
        Tuple of (sections_count, chunks_count).
    """
    book_title = book.get("book_title", "unknown")
    chapters = book.get("chapters") or []
    sections_count = 0
    chunks_count = 0

    for chapter in chapters:
        chapter_title = chapter.get("title", "untitled_chapter")
        path = f"{book_title}::{chapter_title}"
        sc, cc = _walk_node(
            chapter, book_title, path, domain, tokenizer, fh,
        )
        sections_count += sc
        chunks_count += cc

    return sections_count, chunks_count


def build(
    output_path: Path | None = None,
) -> dict:
    """Build OpenStax corpus chunks.

    Args:
        output_path: Path for output JSONL. Defaults to corpus/openstax_chunks.jsonl.

    Returns:
        Stats dict with books_found, sections_count, chunks_count, etc.
    """
    out = output_path or DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset %s...", DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split="train")

    tokenizer = init_tokenizer()

    books_found: list[str] = []
    books_missing: list[str] = []
    total_sections = 0
    total_chunks = 0
    domain_dist: dict[str, int] = {}

    # Index available books by title
    available_titles = {row["book_title"]: i for i, row in enumerate(dataset)}
    logger.info("Available books in dataset: %d", len(available_titles))
    logger.info("Available titles: %s", list(available_titles.keys()))

    with open(out, "w", encoding="utf-8") as fh:
        for book_title, domain in tqdm(OPENSTAX_BOOKS.items(), desc="OpenStax books"):
            if book_title not in available_titles:
                # Try fuzzy match (case-insensitive, partial)
                matched = None
                for avail_title in available_titles:
                    if book_title.lower() in avail_title.lower():
                        matched = avail_title
                        break
                if matched:
                    logger.info("Fuzzy matched '%s' -> '%s'", book_title, matched)
                    row_idx = available_titles[matched]
                else:
                    logger.warning("Book not found: '%s'", book_title)
                    books_missing.append(book_title)
                    continue
            else:
                row_idx = available_titles[book_title]

            book = dataset[row_idx]
            sections, chunks = _process_book(book, domain, tokenizer, fh)
            total_sections += sections
            total_chunks += chunks
            books_found.append(book_title)
            domain_dist[domain] = domain_dist.get(domain, 0) + chunks

            logger.info(
                "  %s: %d sections, %d chunks (domain=%s)",
                book_title, sections, chunks, domain,
            )

    stats = {
        "books_found": books_found,
        "books_missing": books_missing,
        "books_count": len(books_found),
        "sections_count": total_sections,
        "chunks_count": total_chunks,
        "domain_distribution": domain_dist,
    }
    logger.info(
        "OpenStax done: %d/%d books, %d sections, %d chunks",
        len(books_found), len(OPENSTAX_BOOKS), total_sections, total_chunks,
    )
    if books_missing:
        logger.warning("Missing books: %s", books_missing)

    return stats


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OpenStax textbook corpus chunks.",
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
    build(output_path=args.output)


if __name__ == "__main__":
    main()
