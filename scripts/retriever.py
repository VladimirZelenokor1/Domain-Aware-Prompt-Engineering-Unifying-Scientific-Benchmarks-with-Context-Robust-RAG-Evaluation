"""Unified retriever for BM25, dense, and hybrid search.

Provides three retrieval modes over the indexed corpus:
- BM25 via Pyserini/Lucene (k1=1.2, b=0.75)
- Dense via FAISS IVF-PQ + BGE-base-en-v1.5 embeddings
- Hybrid: BM25 + Dense -> RRF fusion -> BGE-reranker-v2-m3 reranking

Usage:
    python scripts/retriever.py --method bm25 --query "What is photosynthesis?"
    python scripts/retriever.py --method dense --query "What is photosynthesis?"
    python scripts/retriever.py --method hybrid --query "What is photosynthesis?"
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default paths
BM25_INDEX_DIR = PROJECT_ROOT / "indices" / "bm25"
FAISS_INDEX_PATH = PROJECT_ROOT / "indices" / "faiss" / "index.faiss"
FAISS_ID_MAP_PATH = PROJECT_ROOT / "indices" / "faiss" / "id_map.json"
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-en-v1.5"
RERANKER_MODEL_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3"

# Fixed parameters from thesis
BM25_K1 = 1.2
BM25_B = 0.75
RRF_K = 60
NPROBE = 128


class Retriever:
    """Unified retriever supporting BM25, dense, and hybrid modes.

    All resources are lazy-loaded on first use per method. The BM25 index
    (with --storeRaw) serves as the canonical text store for all modes.

    Args:
        bm25_index_dir: Path to Pyserini Lucene index directory.
        faiss_index_path: Path to FAISS index file.
        faiss_id_map_path: Path to id_map.json (FAISS int -> chunk_id).
        embedding_model_path: Path to BGE-base-en-v1.5 model directory.
        reranker_model_path: Path to BGE-reranker-v2-m3 model directory.
        device: PyTorch device for models ("cuda" or "cpu").
    """

    def __init__(
        self,
        bm25_index_dir: Path | str = BM25_INDEX_DIR,
        faiss_index_path: Path | str = FAISS_INDEX_PATH,
        faiss_id_map_path: Path | str = FAISS_ID_MAP_PATH,
        embedding_model_path: Path | str = EMBEDDING_MODEL_PATH,
        reranker_model_path: Path | str = RERANKER_MODEL_PATH,
        device: str = "cuda",
    ) -> None:
        self._bm25_index_dir = str(bm25_index_dir)
        self._faiss_index_path = str(faiss_index_path)
        self._faiss_id_map_path = str(faiss_id_map_path)
        self._embedding_model_path = str(embedding_model_path)
        self._reranker_model_path = str(reranker_model_path)
        self._device = device

        # Lazy-loaded resources
        self._bm25_searcher = None
        self._faiss_index = None
        self._id_map: list[str] | None = None
        self._embedding_model = None
        self._reranker = None

    def _get_bm25_searcher(self):
        if self._bm25_searcher is None:
            from pyserini.search.lucene import LuceneSearcher

            self._bm25_searcher = LuceneSearcher(self._bm25_index_dir)
            self._bm25_searcher.set_bm25(k1=BM25_K1, b=BM25_B)
            logger.info("BM25 searcher loaded from %s", self._bm25_index_dir)
        return self._bm25_searcher

    def _get_faiss_index(self):
        if self._faiss_index is None:
            self._faiss_index = faiss.read_index(self._faiss_index_path)
            self._faiss_index.nprobe = NPROBE
            logger.info(
                "FAISS index loaded (%d vectors, nprobe=%d)",
                self._faiss_index.ntotal, NPROBE,
            )
        return self._faiss_index

    def _get_id_map(self) -> list[str]:
        if self._id_map is None:
            with open(self._faiss_id_map_path, "r") as f:
                self._id_map = json.load(f)
            logger.info("ID map loaded (%d entries)", len(self._id_map))
        return self._id_map

    def _get_embedding_model(self):
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self._embedding_model_path, device=self._device,
                model_kwargs={"torch_dtype": torch.float16},
            )
            logger.info("Embedding model loaded on %s", self._device)
        return self._embedding_model

    def _get_reranker(self):
        if self._reranker is None:
            self._reranker = CrossEncoder(
                self._reranker_model_path, device=self._device,
            )
            logger.info("Reranker loaded on %s", self._device)
        return self._reranker

    def _lookup_text(self, chunk_id: str) -> str:
        """Look up chunk text from BM25 index (canonical text store)."""
        searcher = self._get_bm25_searcher()
        doc = searcher.doc(chunk_id)
        if doc is None:
            return ""
        return json.loads(doc.raw())["contents"]

    def retrieve_bm25(self, query: str, top_k: int = 10) -> list[dict]:
        """BM25 retrieval via Pyserini.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with chunk_id, text, score, rank.
        """
        searcher = self._get_bm25_searcher()
        hits = searcher.search(query, k=top_k)

        results = []
        for rank, hit in enumerate(hits):
            doc = json.loads(searcher.doc(hit.docid).raw())
            results.append({
                "chunk_id": hit.docid,
                "text": doc["contents"],
                "score": float(hit.score),
                "rank": rank + 1,
            })
        return results

    def retrieve_dense(self, query: str, top_k: int = 10) -> list[dict]:
        """Dense retrieval via FAISS + BGE embeddings.

        Args:
            query: Search query string.
            top_k: Number of results to return.

        Returns:
            List of dicts with chunk_id, text, score, rank.
        """
        model = self._get_embedding_model()
        index = self._get_faiss_index()
        id_map = self._get_id_map()

        query_vec = model.encode(
            [query], normalize_embeddings=True,
        ).astype(np.float32)
        scores, ids = index.search(query_vec, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0])):
            if idx == -1:
                continue
            chunk_id = id_map[idx]
            text = self._lookup_text(chunk_id)
            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "score": float(score),
                "rank": rank + 1,
            })
        return results

    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = RRF_K,
    ) -> list[dict]:
        """Hybrid retrieval: BM25 + Dense -> RRF -> reranking.

        Args:
            query: Search query string.
            top_k: Number of results to return.
            rrf_k: RRF constant (default: 60).

        Returns:
            List of dicts with chunk_id, text, score, rank.
        """
        # Step 1: Get candidates from both retrievers
        bm25_results = self.retrieve_bm25(query, top_k=20)
        dense_results = self.retrieve_dense(query, top_k=20)

        # Step 2: RRF fusion
        rrf_scores: dict[str, float] = {}
        all_docs: dict[str, dict] = {}

        for result in bm25_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])
            all_docs[cid] = result

        for result in dense_results:
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (rrf_k + result["rank"])
            if cid not in all_docs:
                all_docs[cid] = result

        # Step 3: Top-20 by RRF score
        top_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:20]
        candidates = []
        for cid, rrf_score in top_rrf:
            doc = all_docs[cid].copy()
            doc["rrf_score"] = rrf_score
            candidates.append(doc)

        # Step 4: Rerank with CrossEncoder
        reranker = self._get_reranker()
        pairs = [(query, doc["text"]) for doc in candidates]
        rerank_scores = reranker.predict(pairs)

        for doc, score in zip(candidates, rerank_scores):
            doc["score"] = float(score)

        # Step 5: Sort by reranker score, return top-k
        candidates.sort(key=lambda x: x["score"], reverse=True)
        for rank, doc in enumerate(candidates[:top_k]):
            doc["rank"] = rank + 1

        return candidates[:top_k]


# =========================================================================
# CLI
# =========================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search the indexed corpus.",
    )
    parser.add_argument(
        "--method", required=True,
        choices=["bm25", "dense", "hybrid"],
        help="Retrieval method.",
    )
    parser.add_argument(
        "--query", required=True,
        help="Search query.",
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of results (default: 10).",
    )
    parser.add_argument(
        "--device", default="cuda",
        choices=["cuda", "cpu"],
        help="Device for embedding/reranker models.",
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

    retriever = Retriever(device=args.device)

    if args.method == "bm25":
        results = retriever.retrieve_bm25(args.query, top_k=args.top_k)
    elif args.method == "dense":
        results = retriever.retrieve_dense(args.query, top_k=args.top_k)
    else:
        results = retriever.retrieve_hybrid(args.query, top_k=args.top_k)

    # Pretty-print results
    print(f"\n{'Rank':<5} {'Score':<10} {'Chunk ID':<18} {'Text Preview'}")
    print("-" * 100)
    for r in results:
        preview = r["text"][:80].replace("\n", " ")
        print(f"{r['rank']:<5} {r['score']:<10.4f} {r['chunk_id']:<18} {preview}...")


if __name__ == "__main__":
    main()
