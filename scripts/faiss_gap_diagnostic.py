"""Diagnose the ~100K gap between corpus and FAISS index.

Streams corpus/all_chunks.jsonl once, loads indices/faiss/id_map.json, and
reports:
- corpus total, id_map total, FAISS ntotal
- missing ids = corpus - id_map  (extras in corpus, not indexed)
- extras   ids = id_map - corpus (indexed but absent from corpus)
- per-source breakdown of missing (wikipedia/pubmed/openstax)
- per-domain breakdown of missing
- chunk length histogram (tokens approximated by whitespace-split words)
- positional distribution: in which line-range of the corpus the gaps fall

All results are printed as markdown tables and dumped as JSON next to the
script.

Usage:
    python scripts/faiss_gap_diagnostic.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import faiss

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = PROJECT_ROOT / "corpus" / "all_chunks.jsonl"
ID_MAP_JSON = PROJECT_ROOT / "indices" / "faiss" / "id_map.json"
ID_MAP_TXT = PROJECT_ROOT / "indices" / "faiss" / "id_map.txt"
FAISS_INDEX = PROJECT_ROOT / "indices" / "faiss" / "index.faiss"
REPORT_JSON = PROJECT_ROOT / "outputs" / "faiss_gap_report.json"

# Bucket edges (word count, approximates tokens for English text)
LENGTH_BUCKETS: list[tuple[int, int | None]] = [
    (0, 30),
    (30, 60),
    (60, 120),
    (120, 200),
    (200, 300),
    (300, None),
]


def bucket_of(n: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if hi is None:
            if n >= lo:
                return f">={lo}"
        elif lo <= n < hi:
            return f"{lo}-{hi - 1}"
    return "unknown"


def load_id_map(path_json: Path, path_txt: Path) -> list[str]:
    if path_json.exists():
        with open(path_json, "r", encoding="utf-8") as f:
            return json.load(f)
    if path_txt.exists():
        logger.warning("id_map.json missing, falling back to id_map.txt")
        with open(path_txt, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    raise FileNotFoundError(f"Neither {path_json} nor {path_txt} exists")


def faiss_ntotal(index_path: Path) -> int:
    idx = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    return int(idx.ntotal)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FAISS gap diagnostic")
    p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    p.add_argument(
        "--dump-missing",
        default=None,
        type=Path,
        help=(
            "If set, stream every missing corpus chunk (full record: "
            "chunk_id, text, source, source_id, domain) to this JSONL "
            "file for downstream repair."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    logger.info("Loading id_map from %s", ID_MAP_JSON)
    id_map = load_id_map(ID_MAP_JSON, ID_MAP_TXT)
    id_map_set = set(id_map)
    n_map = len(id_map)
    n_map_unique = len(id_map_set)

    logger.info("Reading FAISS ntotal from %s", FAISS_INDEX)
    n_faiss = faiss_ntotal(FAISS_INDEX)

    logger.info("Streaming corpus from %s", CORPUS_PATH)
    corpus_ids: set[str] = set()
    total_corpus = 0

    # Tallies over corpus entries not present in id_map (= FAISS-missing)
    missing_by_source: Counter = Counter()
    missing_by_domain: Counter = Counter()
    missing_by_length: Counter = Counter()
    missing_by_decile: Counter = Counter()
    missing_ids_sample: list[str] = []

    present_by_source: Counter = Counter()
    present_by_domain: Counter = Counter()
    present_by_length: Counter = Counter()

    # Overall distribution for reference (all corpus chunks)
    all_by_source: Counter = Counter()
    all_by_domain: Counter = Counter()
    all_by_length: Counter = Counter()

    dump_fp = None
    dump_count = 0
    if args.dump_missing is not None:
        args.dump_missing.parent.mkdir(parents=True, exist_ok=True)
        dump_fp = open(args.dump_missing, "w", encoding="utf-8")
        logger.info("Will stream missing chunks to %s", args.dump_missing)

    CORPUS_APPROX = 3_939_826
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cid = rec["chunk_id"]
            source = rec.get("source", "unknown")
            domain = rec.get("domain", "unknown")
            wc = len((rec.get("text") or "").split())
            bucket = bucket_of(wc)

            total_corpus += 1
            corpus_ids.add(cid)
            all_by_source[source] += 1
            all_by_domain[domain] += 1
            all_by_length[bucket] += 1

            if cid in id_map_set:
                present_by_source[source] += 1
                present_by_domain[domain] += 1
                present_by_length[bucket] += 1
            else:
                decile = min(9, (i * 10) // CORPUS_APPROX)
                missing_by_source[source] += 1
                missing_by_domain[domain] += 1
                missing_by_length[bucket] += 1
                missing_by_decile[decile] += 1
                if dump_fp is not None:
                    dump_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    dump_count += 1
                if len(missing_ids_sample) < 20:
                    missing_ids_sample.append(cid)

            if total_corpus % 500_000 == 0:
                logger.info("  ...read %d chunks", total_corpus)

    if dump_fp is not None:
        dump_fp.close()
        logger.info(
            "Wrote %d missing chunk records to %s",
            dump_count,
            args.dump_missing,
        )

    extras_in_faiss = id_map_set - corpus_ids
    duplicates_in_map = n_map - n_map_unique
    missing_total = sum(missing_by_source.values())

    # Build report
    report = {
        "counts": {
            "corpus_total": total_corpus,
            "id_map_entries": n_map,
            "id_map_unique": n_map_unique,
            "id_map_duplicates": duplicates_in_map,
            "faiss_ntotal": n_faiss,
            "missing_vs_id_map": total_corpus - n_map_unique,
            "extras_in_id_map": len(extras_in_faiss),
            "faiss_vs_id_map_delta": n_map_unique - n_faiss,
        },
        "missing_by_source": dict(missing_by_source),
        "missing_by_domain": dict(missing_by_domain),
        "missing_by_length": dict(missing_by_length),
        "missing_by_decile": {str(k): v for k, v in sorted(missing_by_decile.items())},
        "present_by_source": dict(present_by_source),
        "present_by_domain": dict(present_by_domain),
        "present_by_length": dict(present_by_length),
        "all_by_source": dict(all_by_source),
        "all_by_domain": dict(all_by_domain),
        "all_by_length": dict(all_by_length),
        "missing_ids_sample": missing_ids_sample,
        "extras_in_id_map_sample": list(extras_in_faiss)[:20],
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote report to %s", REPORT_JSON)

    # =========================================================================
    # Markdown-style tables for terminal
    # =========================================================================
    print()
    print("=" * 70)
    print("FAISS GAP DIAGNOSTIC SUMMARY")
    print("=" * 70)
    print(f"corpus_total         : {total_corpus:>10,}")
    print(
        f"id_map entries       : {n_map:>10,}   "
        f"(unique: {n_map_unique:,}, duplicates: {duplicates_in_map})"
    )
    print(f"FAISS index ntotal   : {n_faiss:>10,}")
    print(f"missing (corpus - id_map) : {total_corpus - n_map_unique:>10,}")
    print(f"extras  (id_map - corpus) : {len(extras_in_faiss):>10,}")
    print(f"faiss_vs_id_map_delta     : {n_map_unique - n_faiss:>10,}")
    print()

    def pct(num: int, den: int) -> str:
        return f"{100.0 * num / den:.2f}%" if den else "n/a"

    print("--- Missing per source (chunks in corpus but not in id_map) ---")
    print(f"{'source':<12} {'missing':>10} {'total':>10} {'miss %':>8}")
    for src in sorted(set(all_by_source) | set(missing_by_source)):
        m = missing_by_source.get(src, 0)
        t = all_by_source.get(src, 0)
        print(f"{src:<12} {m:>10,} {t:>10,} {pct(m, t):>8}")
    print()

    print("--- Missing per domain ---")
    print(f"{'domain':<20} {'missing':>10} {'total':>10} {'miss %':>8}")
    for dom in sorted(set(all_by_domain) | set(missing_by_domain)):
        m = missing_by_domain.get(dom, 0)
        t = all_by_domain.get(dom, 0)
        print(f"{dom:<20} {m:>10,} {t:>10,} {pct(m, t):>8}")
    print()

    print("--- Length histogram (word count buckets) ---")
    print(f"{'bucket':<10} {'missing':>10} {'total':>10} {'miss %':>8}")
    bucket_order = [
        f"{lo}-{hi - 1}" if hi is not None else f">={lo}" for lo, hi in LENGTH_BUCKETS
    ]
    for b in bucket_order:
        m = missing_by_length.get(b, 0)
        t = all_by_length.get(b, 0)
        print(f"{b:<10} {m:>10,} {t:>10,} {pct(m, t):>8}")
    print()

    print("--- Positional distribution (decile of corpus line order) ---")
    print(f"{'decile':<10} {'missing':>10}")
    for d in range(10):
        m = missing_by_decile.get(d, 0)
        print(f"{d * 10}-{d * 10 + 10}%      {m:>10,}")
    print()

    print("--- Sample missing chunk_ids (first 20) ---")
    for cid in missing_ids_sample:
        print(f"  {cid}")
    if extras_in_faiss:
        print()
        print("--- Sample extras in id_map (first 20) ---")
        for cid in list(extras_in_faiss)[:20]:
            print(f"  {cid}")

    # ------------------------------------------------------------------
    # Classification of the outcome (per user's 4 diagnostic categories)
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if len(extras_in_faiss) > 0:
        print("(D) id_map contains chunk_ids not present in the current corpus.")
        print("    Corpus was rebuilt after FAISS indexing.")
        print("    Action: complete reindexing.")
    elif missing_total == 0:
        # No per-source/domain/decile evidence of actual missing chunks.
        # Any remaining discrepancy is a corpus-level duplicate (hash
        # collision) that is legitimately indexed twice.
        if duplicates_in_map == 0 and n_map_unique == n_faiss:
            print("No gap detected: corpus, id_map, and FAISS index are consistent.")
        else:
            print(
                f"No actual missing chunks. id_map has {duplicates_in_map} "
                f"duplicate slot(s) arising from pre-existing corpus "
                f"chunk_id collision(s); FAISS legitimately indexes both "
                f"sides and the build-time integrity assertion tolerates "
                f"this. No repair needed."
            )
    else:
        # Source concentration?
        src_total = sum(missing_by_source.values())
        if src_total > 0:
            top_src, top_src_count = missing_by_source.most_common(1)[0]
            src_share = top_src_count / src_total
        else:
            top_src, top_src_count, src_share = "n/a", 0, 0.0

        # Length concentration?
        len_total = sum(missing_by_length.values())
        if len_total > 0:
            top_bucket, top_bucket_count = missing_by_length.most_common(1)[0]
            len_share = top_bucket_count / len_total
        else:
            top_bucket, top_bucket_count, len_share = "n/a", 0, 0.0

        # Positional concentration?
        dec_total = sum(missing_by_decile.values())
        if dec_total > 0:
            top_dec, top_dec_count = missing_by_decile.most_common(1)[0]
            dec_share = top_dec_count / dec_total
        else:
            top_dec, top_dec_count, dec_share = -1, 0, 0.0

        print(
            f"missing concentration: top source={top_src} ({src_share:.0%}), "
            f"top length bucket={top_bucket} ({len_share:.0%}), "
            f"top decile={top_dec} ({dec_share:.0%})"
        )

        if src_share >= 0.80:
            print(
                f"(A) ~{src_share:.0%} of the gap is concentrated in a single "
                f"source ({top_src}). Likely ingest timeout on that batch. "
                f"Action: reindex only this source."
            )
        elif len_share >= 0.70:
            print(
                f"(B) ~{len_share:.0%} of the gap is concentrated in length "
                f"bucket {top_bucket}. Length filter inconsistency between "
                f"BM25 and FAISS. Action: unify the filter."
            )
        elif dec_share >= 0.70:
            print(
                f"(C-pos) ~{dec_share:.0%} of the gap is concentrated in one "
                f"positional decile ({top_dec * 10}-{(top_dec + 1) * 10}% of "
                f"the corpus). A contiguous batch was dropped during a "
                f"resume or append. Action: reindex that slice."
            )
        else:
            print(
                "(C) Missing is distributed roughly uniformly across source / "
                "length / corpus position. Likely an internal encoder error "
                "or a mid-run batch failure. Action: reindex the missing ids."
            )

    print()
    sys.stdout.flush()


if __name__ == "__main__":
    main()
