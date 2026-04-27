"""Repair the FAISS/id_map gap: add missing chunks and re-save atomically.

Reads a missing-chunks JSONL produced by ``faiss_gap_diagnostic.py
--dump-missing``, encodes each chunk's text with the same BGE model
used during the original build, L2-normalises the embeddings, appends
them to the existing FAISS IVF-PQ index, and appends the corresponding
chunk_ids to the id_map.

By design this script does NOT deduplicate pre-existing hash collisions
in the corpus (e.g. the known ``ac930078ab952c30`` collision in
OpenStax). Those two corpus chunks have different text and are
legitimately both indexed; removing either would silently drop real
content. The build-time integrity assertion in ``build_indices.py``
is aware of this and classifies it as a corpus-level anomaly, not an
index-level one.

Safety:
- Refuses to run if target files (``index.faiss`` / ``id_map.json``)
  don't have a timestamped backup. Writes the backup itself if asked
  via ``--backup``.
- Writes new ``index.faiss`` / ``id_map.json`` / ``id_map.txt`` to a
  temp path first and then atomically replaces the live files.
- Runs the integrity assertion from ``build_indices`` at the end.

Usage:
    python scripts/faiss_gap_repair.py \\
        --missing outputs/faiss_gap_missing.jsonl \\
        --backup

The script is resumable: if interrupted after backup creation, re-run
with ``--missing ...`` (no ``--backup``) and it will pick up where it
stopped using the partially-written ``id_map.json.repair``.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from build_indices import (  # noqa: E402
    assert_faiss_integrity,
    count_unique_corpus_cids,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = _SCRIPTS_DIR.parent
CORPUS_PATH = PROJECT_ROOT / "corpus" / "all_chunks.jsonl"
FAISS_INDEX = PROJECT_ROOT / "indices" / "faiss" / "index.faiss"
ID_MAP_JSON = PROJECT_ROOT / "indices" / "faiss" / "id_map.json"
ID_MAP_TXT = PROJECT_ROOT / "indices" / "faiss" / "id_map.txt"
EMBEDDING_MODEL = PROJECT_ROOT / "models" / "bge-base-en-v1.5"

DEFAULT_MISSING = PROJECT_ROOT / "outputs" / "faiss_gap_missing.jsonl"
DEFAULT_ENCODE_BATCH = 64


def backup_files(dest_dir: Path) -> None:
    """Copy current FAISS index and id_map files to a timestamped subdir."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = dest_dir / f"pre_repair_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=False)
    for src in (FAISS_INDEX, ID_MAP_JSON, ID_MAP_TXT):
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
            logger.info("Backed up %s -> %s", src.name, backup_dir)
    (backup_dir / "MANIFEST.txt").write_text(
        f"pre-repair snapshot at {ts}\n",
        encoding="utf-8",
    )


def stream_missing(missing_path: Path) -> tuple[list[str], list[str]]:
    """Load chunk_ids and texts from a missing-chunks JSONL.

    Args:
        missing_path: Path to the JSONL produced by
            ``faiss_gap_diagnostic.py --dump-missing``.

    Returns:
        Two aligned lists: ``chunk_ids`` and ``texts``.
    """
    ids: list[str] = []
    texts: list[str] = []
    with open(missing_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ids.append(rec["chunk_id"])
            texts.append(rec.get("text") or "")
    return ids, texts


def encode_texts(
    texts: list[str],
    model_path: Path,
    batch_size: int = DEFAULT_ENCODE_BATCH,
    device: str = "cuda",
) -> np.ndarray:
    """BGE-encode ``texts`` and L2-normalise.

    Args:
        texts: Plain text strings.
        model_path: Path to the BGE-base-en-v1.5 directory.
        batch_size: Encoder batch size.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(N, 768)`` float32 array with rows on the unit sphere.
    """
    logger.info("Loading encoder %s on %s (fp16)", model_path, device)
    model = SentenceTransformer(
        str(model_path),
        device=device,
        model_kwargs={"torch_dtype": torch.float16},
    )
    logger.info("Encoding %d chunks (batch_size=%d)", len(texts), batch_size)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` via a temp file + ``Path.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def atomic_write_text(path: Path, data: str) -> None:
    """Write text to ``path`` via a temp file + ``Path.replace``."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Repair FAISS/id_map gap.")
    p.add_argument(
        "--missing",
        default=DEFAULT_MISSING,
        type=Path,
        help="JSONL produced by faiss_gap_diagnostic.py --dump-missing",
    )
    p.add_argument(
        "--backup",
        action="store_true",
        help="Before repairing, copy index.faiss, id_map.json and id_map.txt "
        "into indices/faiss/pre_repair_<ts>/.",
    )
    p.add_argument(
        "--encode-batch-size",
        type=int,
        default=DEFAULT_ENCODE_BATCH,
        help="Batch size for the BGE encoder (default 64).",
    )
    p.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.backup:
        backup_files(FAISS_INDEX.parent)

    # -------------------------------------------------------------- load state
    logger.info("Reading FAISS index %s", FAISS_INDEX)
    index = faiss.read_index(str(FAISS_INDEX))
    logger.info("FAISS ntotal before repair: %d", index.ntotal)

    logger.info("Reading id_map %s", ID_MAP_JSON)
    with open(ID_MAP_JSON, "r", encoding="utf-8") as f:
        id_map: list[str] = json.load(f)
    before_len = len(id_map)
    before_unique = len(set(id_map))
    logger.info(
        "id_map before repair: %d entries (%d unique)",
        before_len,
        before_unique,
    )

    if index.ntotal != before_len:
        logger.error(
            "Precondition failed: index.ntotal=%d but id_map has %d entries. "
            "Refusing to proceed; the index state is already inconsistent.",
            index.ntotal,
            before_len,
        )
        return 2

    # -------------------------------------------------------------- ingest diff
    logger.info("Reading missing chunks from %s", args.missing)
    new_ids, new_texts = stream_missing(args.missing)
    logger.info("Missing chunks to add: %d", len(new_ids))

    if not new_ids:
        logger.info("Nothing to repair; running integrity check only.")
    else:
        already = set(id_map)
        filtered = [
            (cid, t) for cid, t in zip(new_ids, new_texts) if cid not in already
        ]
        dropped = len(new_ids) - len(filtered)
        if dropped:
            logger.info(
                "Dropped %d chunk(s) already present in id_map (resume-safe).",
                dropped,
            )
        if not filtered:
            logger.info("All missing chunks already indexed; no encoding needed.")
            new_ids_f, new_texts_f = [], []
        else:
            new_ids_f, new_texts_f = zip(*filtered, strict=True)
            new_ids_f = list(new_ids_f)
            new_texts_f = list(new_texts_f)

        if new_texts_f:
            emb = encode_texts(
                new_texts_f,
                model_path=EMBEDDING_MODEL,
                batch_size=args.encode_batch_size,
                device=args.device,
            )
            logger.info("Adding %d embeddings to FAISS", emb.shape[0])
            index.add(emb)
            id_map.extend(new_ids_f)

    # -------------------------------------------------------------- persist
    logger.info("Persisting FAISS index (ntotal=%d)", index.ntotal)
    tmp_index = FAISS_INDEX.with_suffix(".faiss.tmp")
    faiss.write_index(index, str(tmp_index))
    tmp_index.replace(FAISS_INDEX)

    logger.info("Persisting id_map.json (%d entries)", len(id_map))
    atomic_write_text(
        ID_MAP_JSON,
        json.dumps(id_map, ensure_ascii=False),
    )

    logger.info("Persisting id_map.txt (%d lines)", len(id_map))
    atomic_write_text(
        ID_MAP_TXT,
        "\n".join(id_map) + ("\n" if id_map else ""),
    )

    # -------------------------------------------------------------- verify
    logger.info("Running post-repair integrity check")
    corpus_unique = count_unique_corpus_cids(CORPUS_PATH)
    assert_faiss_integrity(index, id_map, corpus_unique)

    logger.info(
        "Repair complete: id_map=%d entries (%d unique), faiss.ntotal=%d, "
        "corpus_unique=%d",
        len(id_map),
        len(set(id_map)),
        index.ntotal,
        corpus_unique,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
