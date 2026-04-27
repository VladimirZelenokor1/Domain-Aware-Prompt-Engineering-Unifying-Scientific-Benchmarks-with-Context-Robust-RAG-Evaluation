"""Build BM25 and FAISS search indices over the corpus.

Converts corpus/all_chunks.jsonl into Pyserini Lucene index (BM25) and
FAISS IVF-PQ index (dense retrieval). RAM-safe: embeddings computed and
stored in batches to stay within 16 GB constraint.

Usage:
    python scripts/build_indices.py                      # build both
    python scripts/build_indices.py --only bm25           # BM25 only
    python scripts/build_indices.py --only faiss          # FAISS only
    python scripts/build_indices.py --cleanup-embeddings  # delete .npy after FAISS
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import subprocess
import sys
from pathlib import Path

import torch
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "corpus" / "all_chunks.jsonl"
CORPUS_TOTAL = 3_939_826  # from manifest

INDICES_DIR = PROJECT_ROOT / "indices"
BM25_INDEX_DIR = INDICES_DIR / "bm25"
PYSERINI_INPUT_DIR = INDICES_DIR / "pyserini_input"
FAISS_DIR = INDICES_DIR / "faiss"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-en-v1.5"

# FAISS IVF-PQ parameters
EMBEDDING_DIM = 768
NLIST = 2048
M_PQ = 48
NBITS = 8
TRAIN_SAMPLE_SIZE = 200_000
EMBEDDING_BATCH_SIZE = 100_000
ENCODE_BATCH_SIZE = 64  # optimal for RTX 4060 with 256-token chunks

# Known pre-existing chunk_id collisions in the corpus (different texts
# that hash to the same 64-bit chunk_id). The integrity check tolerates
# these and logs them; any NEW duplicate triggers a loud assertion.
KNOWN_CORPUS_CID_COLLISIONS: frozenset[str] = frozenset(
    {
        "ac930078ab952c30",  # two OpenStax A&P "Gas Exchange" chunks
    }
)


# =========================================================================
# Corpus integrity
# =========================================================================


def count_unique_corpus_cids(corpus_path: Path) -> int:
    """Return the count of unique ``chunk_id`` values in a corpus JSONL.

    Streams the file once (memory-safe for multi-GB corpora).

    Args:
        corpus_path: Path to ``corpus/all_chunks.jsonl``.

    Returns:
        Number of distinct ``chunk_id`` strings across all records.
    """
    seen: set[str] = set()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen.add(json.loads(line)["chunk_id"])
    return len(seen)


def assert_faiss_integrity(
    index: "faiss.Index",
    id_map_list: list[str],
    corpus_unique_cids: int,
    known_collisions: frozenset[str] = KNOWN_CORPUS_CID_COLLISIONS,
) -> None:
    """Loudly assert that a FAISS index and its id_map are consistent.

    Two invariants must hold:

    1. ``index.ntotal == len(id_map_list)``
       -- every FAISS vector has a positional chunk_id.
    2. ``len(set(id_map_list)) == corpus_unique_cids``
       -- every unique chunk_id in the corpus is represented exactly
       once in the id_map (duplicates are only permitted for the
       chunk_ids listed in ``known_collisions``).

    The number of duplicates in ``id_map_list`` must equal the number
    of known collisions actually present. Any extra duplicate is
    treated as an unknown corruption and fails the check.

    Args:
        index: A FAISS index (``faiss.Index`` or subclass).
        id_map_list: Positional list of chunk_ids aligned with the
            FAISS index (``id_map_list[i]`` = chunk_id of vector ``i``).
        corpus_unique_cids: Result of ``count_unique_corpus_cids``.
        known_collisions: Pre-approved chunk_ids that are allowed to
            appear more than once (corpus-side hash collisions).

    Raises:
        AssertionError: If any invariant is violated. The message
            enumerates every failure found.
    """
    errors: list[str] = []

    n_total = int(getattr(index, "ntotal", -1))
    n_map = len(id_map_list)
    unique = len(set(id_map_list))

    if n_total != n_map:
        errors.append(
            f"FAISS ntotal ({n_total}) != id_map length ({n_map}); "
            f"{n_map - n_total} positional slots have no vector."
        )

    if unique != corpus_unique_cids:
        errors.append(
            f"id_map unique chunk_ids ({unique}) != corpus unique "
            f"chunk_ids ({corpus_unique_cids}); "
            f"{corpus_unique_cids - unique} corpus chunk(s) missing from "
            f"id_map or {unique - corpus_unique_cids} extra(s)."
        )

    # Enumerate duplicate chunk_ids in id_map and compare against the
    # known-collisions allowlist.
    dup_counts: dict[str, int] = {}
    for cid in id_map_list:
        dup_counts[cid] = dup_counts.get(cid, 0) + 1
    observed_dups = {cid: c for cid, c in dup_counts.items() if c > 1}

    unknown_dups = {
        cid: c for cid, c in observed_dups.items() if cid not in known_collisions
    }
    if unknown_dups:
        sample = ", ".join(f"{cid}(x{c})" for cid, c in list(unknown_dups.items())[:5])
        errors.append(
            f"{len(unknown_dups)} id_map chunk_id(s) duplicated without being "
            f"in the known-collisions allowlist. Sample: {sample}"
        )

    if observed_dups:
        logger.warning(
            "id_map contains %d known-collision duplicate(s): %s",
            len(observed_dups),
            sorted(observed_dups.keys()),
        )

    if errors:
        raise AssertionError(
            "FAISS index integrity check failed:\n  - " + "\n  - ".join(errors)
        )

    logger.info(
        "FAISS integrity OK: ntotal=%d, id_map=%d (%d unique), "
        "corpus_unique=%d, known_collisions=%d",
        n_total,
        n_map,
        unique,
        corpus_unique_cids,
        len(observed_dups),
    )


# =========================================================================
# BM25
# =========================================================================


def prepare_pyserini_input(
    corpus_path: Path,
    output_dir: Path,
    docs_per_file: int = 100_000,
) -> int:
    """Convert corpus JSONL to Pyserini JsonCollection format.

    Args:
        corpus_path: Path to corpus/all_chunks.jsonl.
        output_dir: Directory for Pyserini input files.
        docs_per_file: Documents per output file.

    Returns:
        Total document count.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    doc_count = 0
    fh = None

    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Preparing Pyserini input", total=CORPUS_TOTAL):
                if doc_count % docs_per_file == 0:
                    if fh is not None:
                        fh.close()
                    fname = output_dir / f"docs_{file_idx:02d}.jsonl"
                    fh = open(fname, "w", encoding="utf-8")
                    file_idx += 1

                rec = json.loads(line)
                pyserini_doc = {"id": rec["chunk_id"], "contents": rec["text"]}
                fh.write(json.dumps(pyserini_doc, ensure_ascii=False) + "\n")
                doc_count += 1
    finally:
        if fh is not None:
            fh.close()

    logger.info("Prepared %d docs in %d files at %s", doc_count, file_idx, output_dir)
    return doc_count


def build_bm25_index(
    input_dir: Path,
    index_dir: Path,
    threads: int = 4,
) -> None:
    """Build Lucene BM25 index via Pyserini CLI.

    Args:
        input_dir: Directory with Pyserini input JSONL files.
        index_dir: Output directory for Lucene index.
        threads: Number of indexing threads.
    """
    index_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(input_dir),
        "--index",
        str(index_dir),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    logger.info("Building BM25 index: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)

    if result.returncode != 0:
        logger.error(
            "BM25 index build failed:\nSTDOUT: %s\nSTDERR: %s",
            result.stdout[-2000:],
            result.stderr[-2000:],
        )
        raise RuntimeError(f"Pyserini index build failed with code {result.returncode}")

    logger.info("BM25 index built at %s", index_dir)


def verify_bm25(index_dir: Path) -> None:
    """Quick BM25 search verification."""
    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=1.2, b=0.75)

    hits = searcher.search("What is photosynthesis?", k=10)
    logger.info("BM25 verification: %d hits for 'What is photosynthesis?'", len(hits))

    for i, hit in enumerate(hits[:3]):
        doc = json.loads(searcher.doc(hit.docid).raw())
        logger.info(
            "  Hit %d: score=%.4f id=%s text=%.80s...",
            i + 1,
            hit.score,
            hit.docid,
            doc["contents"],
        )

    if len(hits) < 10:
        logger.warning("Expected >= 10 hits, got %d", len(hits))


# =========================================================================
# FAISS
# =========================================================================


def build_faiss_index(
    corpus_path: Path,
    faiss_dir: Path,
    model_path: Path,
    cleanup_embeddings: bool = False,
    encode_batch_size: int = ENCODE_BATCH_SIZE,
    embedding_batch_size: int = EMBEDDING_BATCH_SIZE,
) -> None:
    """Build FAISS IVF-PQ index with batched embedding computation.

    Resumable: skips batches whose .npy files already exist on disk.
    Memory-safe: writes id_map incrementally to a text file.

    Args:
        corpus_path: Path to corpus/all_chunks.jsonl.
        faiss_dir: Output directory for FAISS index and id_map.
        model_path: Path to BGE-base-en-v1.5 model directory.
        cleanup_embeddings: Delete temporary .npy files after building.
        encode_batch_size: GPU batch size for encoding (lower = less resource usage).
        embedding_batch_size: Chunks per embedding batch file (lower = less RAM).
    """
    faiss_dir.mkdir(parents=True, exist_ok=True)
    id_map_txt = faiss_dir / "id_map.txt"

    # Check existing batches for resume (use id_map.txt for accurate count)
    existing_batches = sorted(faiss_dir.glob("embeddings_*.npy"))
    start_batch = len(existing_batches)
    skip_lines = 0
    if start_batch > 0 and id_map_txt.exists():
        with open(id_map_txt, "r") as f:
            skip_lines = sum(1 for line in f if line.strip())
        logger.info(
            "Resuming: found %d existing batches, %d IDs in map, skipping %d chunks",
            start_batch,
            skip_lines,
            skip_lines,
        )

    # Phase 1: Compute embeddings in batches
    logger.info("Loading embedding model from %s (fp16)", model_path)
    model = SentenceTransformer(
        str(model_path),
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )

    batch_texts: list[str] = []
    batch_ids: list[str] = []
    batch_idx = start_batch
    total_embedded = skip_lines
    lines_read = 0

    logger.info(
        "Phase 1: Computing embeddings (batch_size=%d, encode_bs=%d)",
        embedding_batch_size,
        encode_batch_size,
    )

    # Open id_map.txt in append mode for resume safety
    with (
        open(id_map_txt, "a", encoding="utf-8") as id_fh,
        open(corpus_path, "r", encoding="utf-8") as f,
    ):
        for line in tqdm(f, desc="Reading corpus", total=CORPUS_TOTAL):
            lines_read += 1
            if lines_read <= skip_lines:
                continue

            rec = json.loads(line)
            batch_texts.append(rec["text"])
            batch_ids.append(rec["chunk_id"])

            if len(batch_texts) >= embedding_batch_size:
                embeddings = model.encode(
                    batch_texts,
                    batch_size=encode_batch_size,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                )

                emb_path = faiss_dir / f"embeddings_{batch_idx:04d}.npy"
                np.save(str(emb_path), embeddings.astype(np.float32))

                # Write chunk_ids incrementally (one per line)
                for cid in batch_ids:
                    id_fh.write(cid + "\n")
                id_fh.flush()

                total_embedded += len(batch_texts)
                logger.info(
                    "Batch %d: %d embeddings saved (%d/%d total)",
                    batch_idx,
                    len(batch_texts),
                    total_embedded,
                    CORPUS_TOTAL,
                )

                batch_texts.clear()
                batch_ids.clear()
                del embeddings
                gc.collect()
                torch.cuda.empty_cache()
                batch_idx += 1

    # Process remaining
    if batch_texts:
        embeddings = model.encode(
            batch_texts,
            batch_size=encode_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        emb_path = faiss_dir / f"embeddings_{batch_idx:04d}.npy"
        np.save(str(emb_path), embeddings.astype(np.float32))

        with open(id_map_txt, "a", encoding="utf-8") as id_fh:
            for cid in batch_ids:
                id_fh.write(cid + "\n")

        total_embedded += len(batch_texts)
        logger.info("Final batch %d: %d embeddings saved", batch_idx, len(batch_texts))
        del embeddings
        gc.collect()
        batch_idx += 1

    # Unload model to free GPU
    del model
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        "Phase 1 complete: %d embeddings in %d batches", total_embedded, batch_idx
    )

    # Phase 2: Train quantizer on sample
    logger.info("Phase 2: Training IVF-PQ quantizer on %d samples", TRAIN_SAMPLE_SIZE)

    train_files = sorted(faiss_dir.glob("embeddings_*.npy"))
    train_parts: list[np.ndarray] = []
    loaded = 0
    for tf in train_files:
        if loaded >= TRAIN_SAMPLE_SIZE:
            break
        batch = np.load(str(tf))
        need = TRAIN_SAMPLE_SIZE - loaded
        train_parts.append(batch[:need])
        loaded += min(len(batch), need)
        del batch

    train_matrix = np.vstack(train_parts).astype(np.float32)
    del train_parts
    gc.collect()

    logger.info(
        "Training on %d vectors (shape %s)", len(train_matrix), train_matrix.shape
    )

    quantizer = faiss.IndexFlatIP(EMBEDDING_DIM)
    index = faiss.IndexIVFPQ(quantizer, EMBEDDING_DIM, NLIST, M_PQ, NBITS)
    index.train(train_matrix)

    del train_matrix
    gc.collect()
    logger.info("Quantizer trained")

    # Phase 3: Add embeddings in batches
    logger.info("Phase 3: Adding embeddings to index")

    for emb_file in tqdm(
        sorted(faiss_dir.glob("embeddings_*.npy")), desc="Adding to FAISS"
    ):
        batch = np.load(str(emb_file)).astype(np.float32)
        index.add(batch)
        del batch
        gc.collect()

    logger.info("FAISS index contains %d vectors", index.ntotal)

    # Phase 4: Save
    index_path = faiss_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    # Convert id_map.txt -> id_map.json
    id_map_path = faiss_dir / "id_map.json"
    with open(faiss_dir / "id_map.txt", "r", encoding="utf-8") as f:
        id_map = [line.strip() for line in f if line.strip()]
    with open(id_map_path, "w", encoding="utf-8") as f:
        json.dump(id_map, f)
    logger.info("ID map saved to %s (%d entries)", id_map_path, len(id_map))

    # Phase 5: Integrity assertion (was silent in the original build -
    # the cause of the 100 K Wikipedia gap. Make it loud.)
    logger.info("Phase 5: Verifying index integrity against corpus")
    corpus_unique = count_unique_corpus_cids(corpus_path)
    assert_faiss_integrity(index, id_map, corpus_unique)

    # Cleanup temporary files
    if cleanup_embeddings:
        for emb_file in faiss_dir.glob("embeddings_*.npy"):
            emb_file.unlink()
        logger.info("Temporary embedding files deleted")


def verify_faiss(faiss_dir: Path, model_path: Path, bm25_index_dir: Path) -> None:
    """Quick FAISS search verification."""
    from pyserini.search.lucene import LuceneSearcher

    index = faiss.read_index(str(faiss_dir / "index.faiss"))
    index.nprobe = 128

    with open(faiss_dir / "id_map.json", "r") as f:
        id_map = json.load(f)

    model = SentenceTransformer(
        str(model_path),
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    query_vec = model.encode(
        ["What is photosynthesis?"],
        normalize_embeddings=True,
    ).astype(np.float32)

    scores, ids = index.search(query_vec, 10)

    logger.info(
        "FAISS verification: %d hits for 'What is photosynthesis?'", len(ids[0])
    )

    searcher = LuceneSearcher(str(bm25_index_dir))
    for i, (score, idx) in enumerate(zip(scores[0][:3], ids[0][:3])):
        if idx == -1:
            continue
        chunk_id = id_map[idx]
        doc = searcher.doc(chunk_id)
        text = json.loads(doc.raw())["contents"][:80] if doc else "NOT FOUND"
        logger.info(
            "  Hit %d: score=%.4f id=%s text=%s...", i + 1, score, chunk_id, text
        )

    del model
    gc.collect()


# =========================================================================
# CLI
# =========================================================================


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build BM25 and FAISS search indices over the corpus.",
    )
    parser.add_argument(
        "--only",
        choices=["bm25", "faiss"],
        help="Build only one index type.",
    )
    parser.add_argument(
        "--cleanup-embeddings",
        action="store_true",
        help="Delete temporary .npy embedding files after FAISS build.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Threads for BM25 indexing (default: 4).",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=ENCODE_BATCH_SIZE,
        help="GPU batch size for encoding (default: 64, use 8-16 for low resource).",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=EMBEDDING_BATCH_SIZE,
        help="Chunks per .npy file (default: 100000, use 50000 for low RAM).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    do_bm25 = args.only is None or args.only == "bm25"
    do_faiss = args.only is None or args.only == "faiss"

    if do_bm25:
        logger.info("=== Building BM25 index ===")
        prepare_pyserini_input(CORPUS_PATH, PYSERINI_INPUT_DIR)
        build_bm25_index(PYSERINI_INPUT_DIR, BM25_INDEX_DIR, threads=args.threads)
        verify_bm25(BM25_INDEX_DIR)

    if do_faiss:
        logger.info("=== Building FAISS index ===")
        build_faiss_index(
            CORPUS_PATH,
            FAISS_DIR,
            EMBEDDING_MODEL_PATH,
            cleanup_embeddings=args.cleanup_embeddings,
            encode_batch_size=args.encode_batch_size,
            embedding_batch_size=args.embedding_batch_size,
        )
        if do_bm25 or BM25_INDEX_DIR.exists():
            verify_faiss(FAISS_DIR, EMBEDDING_MODEL_PATH, BM25_INDEX_DIR)

    logger.info("=== Index building complete ===")


if __name__ == "__main__":
    main()
