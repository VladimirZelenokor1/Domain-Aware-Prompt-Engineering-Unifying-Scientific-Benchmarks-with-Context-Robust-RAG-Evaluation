# Based on prototype by analyst. Scaled for production.
"""Corpus build orchestrator.

Runs all three builders (Wikipedia, PubMed, OpenStax) sequentially, merges
the outputs, performs exact and near-duplicate deduplication, and generates
a manifest with corpus statistics.

Usage:
    python scripts/build_corpus.py --max-articles 20
    python scripts/build_corpus.py --skip-build
    python scripts/build_corpus.py --skip-dedup
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import re
import sys
from pathlib import Path

from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

import build_openstax  # noqa: E402
import build_pubmed  # noqa: E402
import build_wikipedia  # noqa: E402
from chunking import load_corpus_config  # noqa: E402

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"


# =========================================================================
# Merge
# =========================================================================

def merge_jsonl(
    source_files: list[Path],
    output_path: Path,
) -> int:
    """Merge multiple JSONL files into one.

    Args:
        source_files: List of input JSONL file paths.
        output_path: Path for merged output.

    Returns:
        Total number of records written.
    """
    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for src in source_files:
            if not src.exists():
                logger.warning("Source file not found, skipping: %s", src)
                continue
            with open(src, encoding="utf-8") as f:
                for line in f:
                    out.write(line)
                    total += 1
            logger.info("Merged %s", src.name)

    logger.info("Total records merged: %d", total)
    return total


# =========================================================================
# Deduplication
# =========================================================================

def _normalize_text(text: str) -> str:
    """Normalize text for dedup comparison: lowercase, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _text_hash(text: str) -> str:
    """SHA-256 hash of normalized text."""
    return hashlib.sha256(_normalize_text(text).encode()).hexdigest()


def _text_shingles(text: str, k: int = 3) -> set[str]:
    """Generate word k-shingles from text.

    Args:
        text: Input text.
        k: Shingle size (number of words).

    Returns:
        Set of shingle strings.
    """
    words = _normalize_text(text).split()
    if len(words) < k:
        return {" ".join(words)}
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def dedup_exact(input_path: Path, output_path: Path) -> tuple[int, int]:
    """Remove exact duplicate chunks by text hash.

    Args:
        input_path: Input JSONL path.
        output_path: Deduplicated output JSONL path.

    Returns:
        Tuple of (records_kept, duplicates_removed).
    """
    seen_hashes: set[str] = set()
    kept = 0
    removed = 0

    with open(input_path, encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc="Exact dedup"):
            record = json.loads(line)
            h = _text_hash(record["text"])
            if h in seen_hashes:
                removed += 1
                continue
            seen_hashes.add(h)
            fout.write(line)
            kept += 1

    logger.info("Exact dedup: kept %d, removed %d", kept, removed)
    return kept, removed


def dedup_minhash(
    input_path: Path,
    output_path: Path,
    num_perm: int = 128,
    threshold: float = 0.9,
    shingle_size: int = 3,
) -> tuple[int, int]:
    """Remove near-duplicate chunks using MinHash LSH.

    For each cluster of similar chunks, keeps the longest one.

    Args:
        input_path: Input JSONL path (already exact-deduped).
        output_path: Deduplicated output JSONL path.
        num_perm: Number of MinHash permutations.
        threshold: Jaccard similarity threshold for near-duplicates.
        shingle_size: Word k-shingle size.

    Returns:
        Tuple of (records_kept, duplicates_removed).
    """
    logger.info("Building MinHash signatures (num_perm=%d, threshold=%.2f)...", num_perm, threshold)

    # Phase 1: Build MinHash for each record
    records: list[dict] = []
    minhashes: list[MinHash] = []

    with open(input_path, encoding="utf-8") as f:
        for line in tqdm(f, desc="MinHash signatures"):
            record = json.loads(line)
            records.append(record)

            shingles = _text_shingles(record["text"], shingle_size)
            m = MinHash(num_perm=num_perm)
            for s in shingles:
                m.update(s.encode("utf-8"))
            minhashes.append(m)

    logger.info("Built %d MinHash signatures", len(minhashes))

    # Phase 2: LSH to find near-duplicate clusters
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for i, m in enumerate(tqdm(minhashes, desc="LSH indexing")):
        try:
            lsh.insert(str(i), m)
        except ValueError:
            # Duplicate key (already inserted via a near-duplicate)
            pass

    # Phase 3: Find clusters and pick the longest chunk from each
    to_remove: set[int] = set()
    for i in range(len(records)):
        if i in to_remove:
            continue
        result = lsh.query(minhashes[i])
        cluster = [int(r) for r in result if int(r) != i and int(r) not in to_remove]
        if not cluster:
            continue

        # Pick the longest from the cluster (including current)
        candidates = [i] + cluster
        best = max(candidates, key=lambda idx: len(records[idx]["text"]))
        for idx in candidates:
            if idx != best:
                to_remove.add(idx)

    # Phase 4: Write survivors
    kept = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for i, record in enumerate(records):
            if i not in to_remove:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1

    removed = len(to_remove)
    logger.info("MinHash dedup: kept %d, removed %d", kept, removed)
    return kept, removed


# =========================================================================
# Manifest
# =========================================================================

def generate_manifest(
    corpus_dir: Path,
    wiki_stats: dict | None,
    pubmed_stats: dict | None,
    openstax_stats: dict | None,
    total_before_dedup: int,
    exact_removed: int,
    minhash_removed: int,
    total_after_dedup: int,
) -> dict:
    """Generate corpus manifest with build statistics.

    Args:
        corpus_dir: Directory for manifest output.
        wiki_stats: Stats from Wikipedia builder (or None if skipped).
        pubmed_stats: Stats from PubMed builder (or None if skipped).
        openstax_stats: Stats from OpenStax builder (or None if skipped).
        total_before_dedup: Total chunks before deduplication.
        exact_removed: Exact duplicates removed.
        minhash_removed: Near-duplicates removed.
        total_after_dedup: Final chunk count.

    Returns:
        Manifest dict.
    """
    # Aggregate domain distribution from final corpus
    domain_dist: dict[str, int] = {}
    final_path = corpus_dir / "all_chunks.jsonl"
    if final_path.exists():
        with open(final_path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                domain = record.get("domain", "unknown")
                domain_dist[domain] = domain_dist.get(domain, 0) + 1

    manifest = {
        "build_date": datetime.date.today().isoformat(),
        "chunk_size_tokens": 256,
        "chunk_overlap_tokens": 64,
        "tokenizer": "tiktoken_cl100k_base",
        "wikipedia_dump_date": "20231101",
        "wikipedia_articles_count": wiki_stats["articles_count"] if wiki_stats else 0,
        "wikipedia_chunks_count": wiki_stats["chunks_count"] if wiki_stats else 0,
        "pubmed_abstracts_count": pubmed_stats["articles_count"] if pubmed_stats else 0,
        "pubmed_chunks_count": pubmed_stats["chunks_count"] if pubmed_stats else 0,
        "openstax_textbooks_count": openstax_stats["books_count"] if openstax_stats else 0,
        "openstax_textbooks_list": openstax_stats["books_found"] if openstax_stats else [],
        "openstax_passages_count": openstax_stats["chunks_count"] if openstax_stats else 0,
        "total_chunks_before_dedup": total_before_dedup,
        "total_chunks_after_dedup": total_after_dedup,
        "duplicates_removed_exact": exact_removed,
        "duplicates_removed_minhash": minhash_removed,
        "domain_distribution": domain_dist,
    }

    manifest_path = corpus_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info("Manifest written to %s", manifest_path)
    return manifest


# =========================================================================
# Orchestrator
# =========================================================================

def run(
    max_articles: int = 200_000,
    skip_build: bool = False,
    skip_dedup: bool = False,
) -> None:
    """Run full corpus build pipeline.

    Args:
        max_articles: Max articles for Wikipedia and PubMed builders.
        skip_build: If True, skip builders and only merge/dedup existing files.
        skip_dedup: If True, skip deduplication step.
    """
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    wiki_out = CORPUS_DIR / "wikipedia_chunks.jsonl"
    pubmed_out = CORPUS_DIR / "pubmed_chunks.jsonl"
    openstax_out = CORPUS_DIR / "openstax_chunks.jsonl"

    wiki_stats = None
    pubmed_stats = None
    openstax_stats = None

    # Step 1: Build
    if not skip_build:
        logger.info("=" * 60)
        logger.info("Step 1/3: Building Wikipedia corpus...")
        logger.info("=" * 60)
        wiki_stats = build_wikipedia.build(
            max_articles=max_articles, output_path=wiki_out,
        )

        logger.info("=" * 60)
        logger.info("Step 2/3: Building PubMed corpus...")
        logger.info("=" * 60)
        pubmed_stats = build_pubmed.build(
            max_articles=max_articles, output_path=pubmed_out,
        )

        logger.info("=" * 60)
        logger.info("Step 3/3: Building OpenStax corpus...")
        logger.info("=" * 60)
        openstax_stats = build_openstax.build(output_path=openstax_out)
    else:
        logger.info("Skipping build step (--skip-build)")

    # Step 2: Merge
    logger.info("=" * 60)
    logger.info("Merging corpus files...")
    logger.info("=" * 60)
    merged_path = CORPUS_DIR / "all_chunks_raw.jsonl"
    total_before = merge_jsonl([wiki_out, pubmed_out, openstax_out], merged_path)

    # Step 3: Dedup
    exact_removed = 0
    minhash_removed = 0
    total_after = total_before

    if not skip_dedup:
        logger.info("=" * 60)
        logger.info("Deduplicating...")
        logger.info("=" * 60)

        # Exact dedup
        exact_deduped_path = CORPUS_DIR / "all_chunks_exact_deduped.jsonl"
        kept_exact, exact_removed = dedup_exact(merged_path, exact_deduped_path)

        # Near-duplicate dedup (MinHash)
        config = load_corpus_config()
        dedup_cfg = config.get("dedup", {}).get("near_duplicate", {})
        num_perm = dedup_cfg.get("num_perm", 128)
        threshold = dedup_cfg.get("threshold", 0.9)
        shingle_size = dedup_cfg.get("shingle_size", 3)

        final_path = CORPUS_DIR / "all_chunks.jsonl"
        kept_minhash, minhash_removed = dedup_minhash(
            exact_deduped_path, final_path,
            num_perm=num_perm, threshold=threshold, shingle_size=shingle_size,
        )
        total_after = kept_minhash

        # Clean intermediate files
        exact_deduped_path.unlink(missing_ok=True)
    else:
        logger.info("Skipping dedup (--skip-dedup)")
        # Just copy merged as final
        final_path = CORPUS_DIR / "all_chunks.jsonl"
        if merged_path.exists():
            final_path.write_bytes(merged_path.read_bytes())
        total_after = total_before

    # Clean raw merged file
    if merged_path.exists() and (CORPUS_DIR / "all_chunks.jsonl").exists():
        merged_path.unlink(missing_ok=True)

    # Step 4: Manifest
    logger.info("=" * 60)
    logger.info("Generating manifest...")
    logger.info("=" * 60)
    manifest = generate_manifest(
        CORPUS_DIR,
        wiki_stats, pubmed_stats, openstax_stats,
        total_before, exact_removed, minhash_removed, total_after,
    )
    logger.info("Final corpus: %d chunks (removed %d exact + %d near-dup)",
                total_after, exact_removed, minhash_removed)
    logger.info("Domain distribution: %s", manifest["domain_distribution"])


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, merge, and deduplicate the full science corpus.",
    )
    parser.add_argument(
        "--max-articles", type=int, default=200_000,
        help="Max articles for Wikipedia/PubMed builders (default: 200000).",
    )
    parser.add_argument(
        "--skip-build", action="store_true",
        help="Skip builders, only merge and dedup existing JSONL files.",
    )
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="Skip deduplication step.",
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
    run(
        max_articles=args.max_articles,
        skip_build=args.skip_build,
        skip_dedup=args.skip_dedup,
    )


if __name__ == "__main__":
    main()
